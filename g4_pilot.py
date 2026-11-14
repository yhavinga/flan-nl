#!/usr/bin/env python3
"""The Gemma 4 pilot of FLAN-nl (PILOT_GEMMA4.md): the sample, the measures of Gemma 4 and TranslateGemma on it, the
blind material and the report. The translation itself is translate_gemma4.py.

    python g4_pilot.py draw --plan plan2 --english en/w0.parquet ... --out pilot
    python g4_pilot.py gather --pilot pilot --shards DIR --out pilot/tg               (TranslateGemma's; pilot/g4 Gemma 4's)
    python g4_pilot.py measure --pilot pilot --templates-nl templates_nl.json
    python g4_pilot.py blind --pilot pilot --out DIR --key key.json
    python g4_pilot.py checkset --pilot pilot --out check.md
    python g4_pilot.py report --pilot pilot --qe qe-w*.parquet --logs DIR [--check check.json] --out flan-g4-pilot.md
    python g4_pilot.py decide --report flan-g4-pilot.json --key key.json --judgments DIR

draw: the sets S, E and T of PILOT_GEMMA4.md from the plan (seed SEED) -> sample.parquet (one row per segment of the
pilot, with its sets, cell and weight), segments-w<k>.parquet (what each worker translates), plan/ (the examples of
E, their segments and English, plan.json: a plan that flan_nl.py assemble takes), sample.json (counts, also of the
whole plan per class and cell).
gather: the shards of one model, the pilot's segments only, each exactly once -> <out>/translations.parquet (id,
text_nl, ended, tokens_in, tokens_out).
measure: per segment and model the measures (af, loops, chatter, executed and its reasons, assemble's checks and
repairs, length ratios) -> measures.parquet; flan_nl.py assemble on plan/ with each model -> assemble-<model>/; the
labels and options of E -> examples.parquet; the pairs for MetricX-QE -> qe/pairs-w<k>.parquet.
blind: 100 segments of S, A and B in a seeded order, in portions of 25 with the rubric; the key apart.
checkset: segments for the hand check of the detector `executed`: flagged ones of each model and unflagged
instruction-like ones; check.json beside it is filled by hand ({"<model>:<id>": true|false}, true = executed).
report: everything in Dutch (Markdown) and JSON, with the decision rule's parts that need no reviewer.
decide: the blind judgements with the key -> parts 1 and 2 of the rule and the verdict.
"""

import argparse
import glob
import hashlib
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import flan_nl as fn
import publish as pb

SEED = "flan-g4-pilot/0"
MODELS = ("g4", "tg")
NAMES = {"g4": "Gemma 4 31B-it", "tg": "TranslateGemma 27B"}
CLASSES = {"tiny": 32, "small": 128, "medium": 256, "long": 384, "xlong": 512}  # translate.py's and models.yml's
# The answer kinds of the examples that are translated; the plan's empty kind is that of the examples it does not
# translate (dropped tasks, Dutch tasks, a segment too long)
KINDS = ("label", "option", "text")
WORKERS = 4
PER_CELL = 67
PER_TASK_IN_CELL = 2
EXAMPLES_PER_KIND = 134
THROUGHPUT = {"tiny": 2048, "small": 1024, "medium": 512, "long": 384, "xlong": 384}  # per worker
BLIND_PER_CLASS = 20
BLIND_PORTION = 25

# A segment that looks like an instruction, a question, a request to translate, or code
INSTRUCTION_LIKE = re.compile(
    r"^\s*(?:write|generate|answer|translate|classify|choose|select|pick|read|given|find|identify|determine|create|"
    r"summari[sz]e|describe|explain|provide|rewrite|complete|fill|list|name|tell|decide|predict|output|return|label|"
    r"rate|evaluate|judge|indicate|state|extract|count|compute|calculate|solve|convert|check|construct|compose|make|"
    r"come up|please|you (?:are|will|need|must|should)|your task|in this task|the task)\b"
    r"|\?[\s\"'”’)\]]*$|\btranslat(?:e|ion)\b", re.IGNORECASE)
CODE_LIKE = re.compile(r"[{}<>]|==|=>|\bdef \w+\(|\breturn\b|\bfunction\b|\bSELECT\b|\bprint\(|;\s*$|\$\w+|\\[a-z]+",
                       re.MULTILINE)

# chatter: bt_gemma4.py's introduction and note (github.com/yhavinga/nl-backtranslation, gemma4_bt/bt_gemma4.py)
INTRO = re.compile(r"^\s*(?:\*\*)?(?:(?:sure|certainly|of course)[,!.]?\s+)?(?:here(?:'s| is)\b[^\n]{0,60}\btranslation|"
                   r"(?:english |dutch )?translation\s*:|hier is\b[^\n]{0,60}\bvertaling|vertaling\s*:)", re.IGNORECASE)
NOTE = re.compile(r"\n\s*(?:\*\*)?(?:note|translator'?s note|opmerking|noot|n\.b\.)\s*:", re.IGNORECASE)

# executed: the parts of the detector (PILOT_GEMMA4.md)
_QUESTION_END = re.compile(r"\?[\s\"'”’)\]*_]*$")
ANSWER_MARKER = re.compile(r"^\s*(?:\*\*)?(?:antwoord|answer|oplossing|solution|uitleg|explanation|uitvoer|output|"
                           r"resultaat|result|redenering|reasoning)\s*(?:\*\*)?\s*:|\b(?:het (?:juiste )?antwoord is|"
                           r"the (?:correct )?answer is)\b", re.IGNORECASE | re.MULTILINE)
ASSISTANT_OPENING = re.compile(r"^\s*(?:\*\*)?(?:zeker|natuurlijk|graag|sure|certainly|of course|hier (?:is|zijn)|"
                               r"here (?:is|are))\b", re.IGNORECASE)
ASSISTANT_SOURCE = re.compile(r"^\s*(?:sure|certainly|surely|of course|naturally|gladly|here|definitely|absolutely)\b",
                              re.IGNORECASE)


# ------------------------------------------------------------------------------------------------------ measures

def class_of(tokens):
    for name, high in CLASSES.items():
        if tokens <= high:
            return name
    raise ValueError(f"a segment of {tokens} tokens is longer than every class")


def instruction_like(text):
    return bool(INSTRUCTION_LIKE.search(text) or CODE_LIKE.search(text))


def lines(text):
    return sum(1 for line in text.split("\n") if line.strip())


def executed_reasons(source, output):
    """The parts of the detector `executed` that a translation meets: the segment carried out, answered, completed or
    continued instead of translated"""
    s, o = source.strip(), output.strip()
    reasons = []
    if _QUESTION_END.search(s) and not _QUESTION_END.search(o):
        reasons.append("question_lost")
    if len(o) > 2.2 * len(s) + 20:
        reasons.append("too_long")
    if len(s) >= 40 and len(o) < 0.4 * len(s):
        reasons.append("too_short")
    if lines(o) >= lines(s) + 2:
        reasons.append("added_lines")
    if len(ANSWER_MARKER.findall(o)) > len(ANSWER_MARKER.findall(s)):
        reasons.append("answer_marker")
    if ASSISTANT_OPENING.search(o) and not ASSISTANT_SOURCE.search(s):
        reasons.append("assistant_opening")
    return reasons


def chatter(source, output):
    """An introduction or a note that the source does not have, or assemble's commentary check"""
    if INTRO.search(output) or (NOTE.search(output) and not NOTE.search(source)):
        return True
    match = fn._COMMENTARY.search(output)
    return bool(match and not fn._COMMENTARY.search(source) and "translat" not in source.lower())


def words(text):
    return len(text.split())


def segment_measures(source, output, ended, language):
    """The measures of one translation. language: langid's code of the repaired output (assemble's check)."""
    repaired, repairs = fn.repair_translation(source, output)
    loop = pb.decode_loop(output, source)
    reasons = executed_reasons(source, output)
    return {"ended": bool(ended), "loop": bool(loop), "af": bool(ended) and not loop, "chatter": chatter(source, output),
            "executed": bool(reasons), "executed_reasons": ",".join(reasons),
            "check": fn.check_translation(source, repaired, ended, lambda _: language) or "",
            "repairs": ",".join(repairs), "english_left": bool(pb.english_left(output, source)),
            "chars_ratio": round(len(output.strip()) / max(1, len(source.strip())), 4),
            "words_ratio": round(words(output) / max(1, words(source)), 4), "identical_to_source": output.strip() == source.strip()}


def normalized(text):
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".!?;:, ")


def options_collapsed(inputs_en, instruction_pieces, segment_text, translation):
    """Two different English options of an example with the same Dutch: the options as flan_nl.find_options finds them
    in the English instruction, each option's Dutch the translation of the segment with its text (an option without
    words stays as it is)"""
    ids = {segment_text[v]: v for kind, v in instruction_pieces if kind == "S"}
    seen = {}
    for _, _, start, stop in fn.find_options(inputs_en):
        english = inputs_en[start:stop].strip()
        dutch = translation[ids[english]] if english in ids else english
        key = normalized(dutch)
        if key in seen and seen[key] != normalized(english):
            return True
        seen.setdefault(key, normalized(english))
    return False


def weighted_rate(values, weights):
    total = sum(weights)
    return sum(w for v, w in zip(values, weights) if v) / total if total else None


def weighted_mean(values, weights):
    pairs = [(v, w) for v, w in zip(values, weights) if v is not None]
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total if total else None


def weighted_quantile(values, weights, q):
    pairs = sorted((v, w) for v, w in zip(values, weights) if v is not None)
    if not pairs:
        return None
    total, run = sum(w for _, w in pairs), 0.0
    for v, w in pairs:
        run += w
        if run >= q * total:
            return v
    return pairs[-1][0]


# ---------------------------------------------------------------------------------------------------------- draw

def rng(*parts):
    return random.Random(int(hashlib.sha256("/".join((SEED,) + tuple(map(str, parts))).encode()).hexdigest()[:16], 16))


def task_name(source):
    return fn.task_of(source)


def homes(examples):
    """segment id -> (answer kind, task, role) of its home example: the first example with status translate that uses
    it, in a seeded order. role: instruction or answer (where the home example has it)."""
    order = [k for k, status in enumerate(examples["status"]) if status == "translate"]
    rng("homes").shuffle(order)
    home = {}
    for k in order:
        kind, task = examples["answer_kind"][k], task_name(examples["source"][k])
        for role in ("instruction", "answer"):
            for piece_kind, value in json.loads(examples[role][k]):
                if piece_kind == "S" and value not in home:
                    home[value] = (kind, task, role)
    return home


def pick_spread(ids, task_of_id, n, per_task, generator):
    """n of ids in a seeded order, at most per_task of one task"""
    ids = sorted(ids)
    generator.shuffle(ids)
    taken, per = [], Counter()
    for i in ids:
        if len(taken) == n:
            break
        if per[task_of_id(i)] < per_task:
            taken.append(i)
            per[task_of_id(i)] += 1
    return taken


def draw_sets(segments, examples, template_ids):
    """(S: {id: cell}, cell population, E: example rows, T: {id: worker}). segments: id -> (text, tokens); examples:
    columns of examples.parquet."""
    home = homes(examples)
    cells = defaultdict(list)
    for i, (text, tokens) in segments.items():
        if i in template_ids or i not in home:
            continue
        kind, task, _ = home[i]
        cells[(class_of(tokens), "instr" if instruction_like(text) else "other", kind)].append(i)
    population = {cell: len(ids) for cell, ids in cells.items()}
    s = {}
    for cell in sorted(cells):
        for i in pick_spread(cells[cell], lambda i: home[i][1], PER_CELL, PER_TASK_IN_CELL, rng("S", *cell)):
            s[i] = cell
    e = []
    for kind in KINDS:
        rows = [k for k, (status, kd) in enumerate(zip(examples["status"], examples["answer_kind"]))
                if status == "translate" and kd == kind]
        e += pick_spread(rows, lambda k: task_name(examples["source"][k]), EXAMPLES_PER_KIND, 1, rng("E", kind))
    used = set(s) | set(template_ids)
    for k in e:
        used |= {v for role in ("instruction", "answer") for piece_kind, v in json.loads(examples[role][k]) if piece_kind == "S"}
    t = {}
    for name in CLASSES:
        pool = sorted(i for i, (_, tokens) in segments.items() if class_of(tokens) == name and i not in used)
        rng("T", name).shuffle(pool)
        for w in range(WORKERS):
            for i in pool[w * THROUGHPUT[name]:(w + 1) * THROUGHPUT[name]]:
                t[i] = w
    return s, population, sorted(e), t


def draw(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(args.out)
    (out / "plan").mkdir(parents=True, exist_ok=True)
    plan_info = json.loads(Path(args.plan, "plan.json").read_text())
    template_ids = set(plan_info["template_segments"].values())
    segments = {}
    for path in sorted(glob.glob(f"{args.plan}/segments-w*.parquet")):
        table = pq.read_table(path, columns=["id", "text", "tokens"]).to_pydict()
        segments.update(zip(table["id"], zip(table["text"], table["tokens"])))
    examples = pq.read_table(f"{args.plan}/examples.parquet").to_pydict()
    s, population, e, t = draw_sets(segments, examples, template_ids)
    e_ids = set(template_ids)
    for k in e:
        e_ids |= {v for role in ("instruction", "answer") for kind, v in json.loads(examples[role][k]) if kind == "S"}
    home = homes(examples)
    ids = sorted(set(s) | e_ids | set(t))
    rows = {"id": ids, "text": [segments[i][0] for i in ids], "tokens": [segments[i][1] for i in ids],
            "class": [class_of(segments[i][1]) for i in ids], "in_s": [i in s for i in ids],
            "in_e": [i in e_ids for i in ids], "in_t": [i in t for i in ids],
            "template": [i in template_ids for i in ids],
            "cell": ["/".join(s[i]) if i in s else "" for i in ids],
            "weight": [population[s[i]] / sum(1 for j in s if s[j] == s[i]) if i in s else 0.0 for i in ids],
            "instruction_like": [instruction_like(segments[i][0]) for i in ids],
            "kind": [home[i][0] if i in home else "" for i in ids], "task": [home[i][1] if i in home else "" for i in ids],
            "role": [home[i][2] if i in home else "template" for i in ids]}
    # Each worker translates its T and a quarter of the rest, dealt by id
    rest = [i for i in ids if i not in t]
    worker = dict(t)
    worker.update({i: k % WORKERS for k, i in enumerate(rest)})
    rows["worker"] = [worker[i] for i in ids]
    pq.write_table(pa.table(rows), out / "sample.parquet", compression="zstd")
    for w in range(WORKERS):
        mine = [k for k, i in enumerate(ids) if worker[i] == w]
        pq.write_table(pa.table({"id": [ids[k] for k in mine], "text": [rows["text"][k] for k in mine],
                                 "tokens": [rows["tokens"][k] for k in mine]}), out / f"segments-w{w}.parquet",
                       compression="zstd")
    # The plan of E for assemble: its examples, its segments (and the template strings), its English, plan.json
    pq.write_table(pa.table({c: [examples[c][k] for k in e] for c in examples}), out / "plan" / "examples.parquet",
                   compression="zstd")
    e_sorted = sorted(e_ids)
    pq.write_table(pa.table({"id": e_sorted, "text": [segments[i][0] for i in e_sorted],
                             "tokens": [segments[i][1] for i in e_sorted]}), out / "plan" / "segments-w0.parquet",
                   compression="zstd")
    (out / "plan" / "plan.json").write_text(Path(args.plan, "plan.json").read_text())
    wanted = {(examples["host"][k], examples["position"][k]) for k in e}
    english = {"host": [], "position": [], "inputs": [], "targets": [], "source": []}
    for path in args.english:
        table = pq.read_table(path, columns=["host", "position", "inputs", "targets", "source"]).to_pydict()
        for k, key in enumerate(zip(table["host"], table["position"])):
            if key in wanted:
                for c in english:
                    english[c].append(table[c][k])
    if len(english["host"]) != len(wanted):
        raise SystemExit(f"{len(wanted) - len(english['host'])} examples of E have no English row in {args.english}")
    pq.write_table(pa.table(english), out / "plan" / "en.parquet", compression="zstd")
    plan_classes = Counter(class_of(tokens) for _, tokens in segments.values())
    summary = {"seed": SEED, "segments": len(ids), "S": len(s), "E_examples": len(e), "E_segments": len(e_ids),
               "T": len(t), "templates": len(template_ids), "per_worker": dict(Counter(worker.values())),
               "plan_segments": len(segments), "plan_classes": {c: plan_classes[c] for c in CLASSES},
               "plan_class_tokens": {c: sum(tokens for _, tokens in segments.values() if class_of(tokens) == c) for c in CLASSES},
               "cells": {"/".join(cell): {"plan": population[cell], "drawn": sum(1 for j in s if s[j] == cell)}
                         for cell in sorted(population)},
               "E_kinds": dict(Counter(examples["answer_kind"][k] for k in e)),
               "T_per_worker": {c: THROUGHPUT[c] for c in CLASSES}}
    (out / "sample.json").write_text(json.dumps(summary, indent=1))
    print(f"DRAW {json.dumps({k: summary[k] for k in ('segments', 'S', 'E_examples', 'E_segments', 'T', 'per_worker')})}", flush=True)


# -------------------------------------------------------------------------------------------------------- gather

def gather(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    wanted = set(pq.read_table(Path(args.pilot) / "sample.parquet", columns=["id"]).column("id").to_pylist())
    rows = {c: [] for c in ("id", "text_nl", "ended", "tokens_in", "tokens_out")}
    seen = set()
    for path in sorted(glob.glob(f"{args.shards}/**/*.parquet", recursive=True)):
        table = pq.read_table(path, columns=list(rows)).to_pydict()
        for k, i in enumerate(table["id"]):
            if i in wanted:
                if i in seen:
                    raise SystemExit(f"segment {i} translated twice ({path})")
                seen.add(i)
                for c in rows:
                    rows[c].append(table[c][k])
    missing = wanted - seen
    if missing and not args.allow_missing:
        raise SystemExit(f"{len(missing)} of {len(wanted)} segments of the pilot have no translation in {args.shards}, "
                         f"e.g. {sorted(missing)[:5]}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), Path(args.out) / "translations.parquet", compression="zstd")
    print(f"GATHER {json.dumps({'shards': args.shards, 'segments': len(seen), 'missing': len(missing)})}", flush=True)


# ------------------------------------------------------------------------------------------------------- measure

def _measure_chunk(items):
    return [segment_measures(*item) for item in items]


def measure(args):
    import concurrent.futures

    import pyarrow as pa
    import pyarrow.parquet as pq

    pilot = Path(args.pilot)
    sample = pq.read_table(pilot / "sample.parquet").to_pydict()
    text = dict(zip(sample["id"], sample["text"]))
    out = {m: pq.read_table(pilot / m / "translations.parquet").to_pydict() for m in MODELS}
    translations = {m: {i: (t, e) for i, t, e in zip(out[m]["id"], out[m]["text_nl"], out[m]["ended"])} for m in MODELS}
    for m in MODELS:
        missing = set(sample["id"]) - set(translations[m])
        if missing:
            raise SystemExit(f"{m}: {len(missing)} segments of the pilot have no translation, e.g. {sorted(missing)[:5]}")
    ids = sample["id"]
    rows = defaultdict(list)
    with concurrent.futures.ProcessPoolExecutor(args.processes) as pool:
        for m in MODELS:
            outputs = [fn.repair_translation(text[i], translations[m][i][0])[0] for i in ids]
            chunk = 1000
            languages = [lang for part in pool.map(fn._identify, [outputs[k:k + chunk] for k in range(0, len(ids), chunk)])
                         for lang in part]
            items = [(text[i], translations[m][i][0], translations[m][i][1], lang) for i, lang in zip(ids, languages)]
            measured = [r for part in pool.map(_measure_chunk, [items[k:k + chunk] for k in range(0, len(items), chunk)])
                        for r in part]
            for i, r in zip(ids, measured):
                rows["id"].append(i)
                rows["model"].append(m)
                rows["text_nl"].append(translations[m][i][0])
                for key, value in r.items():
                    rows[key].append(value)
    pq.write_table(pa.table(dict(rows)), pilot / "measures.parquet", compression="zstd")

    # assemble on E with each model's translations: its checks, repairs, template variants, the rows it keeps
    plan = pilot / "plan"
    for m in MODELS:
        (pilot / f"assemble-{m}").mkdir(exist_ok=True)
        fn.main(["assemble", "--english", str(plan / "en.parquet"), "--plan", str(plan), "--translations",
                 str(pilot / m), "--out", str(pilot / f"assemble-{m}"), "--world", str(WORKERS),
                 "--processes", str(args.processes), "--templates-nl", args.templates_nl])
    examples = pq.read_table(plan / "examples.parquet").to_pydict()
    english = pq.read_table(plan / "en.parquet").to_pydict()
    english = {(h, p): x for h, p, x in zip(english["host"], english["position"], english["inputs"])}
    kept = {}
    for m in MODELS:
        kept[m] = set()
        for h in range(WORKERS):
            table = pq.read_table(pilot / f"assemble-{m}" / f"w{h}.parquet", columns=["position"]).to_pydict()
            kept[m] |= {(h, p) for p in table["position"]}
    rows = defaultdict(list)
    for k in range(len(examples["host"])):
        key = (examples["host"][k], examples["position"][k])
        instruction = [(kind, v) for kind, v in json.loads(examples["instruction"][k])]
        rows["host"].append(key[0])
        rows["position"].append(key[1])
        rows["task"].append(task_name(examples["source"][k]))
        rows["answer_kind"].append(examples["answer_kind"][k])
        rows["options"].append(examples["options"][k])
        for m in MODELS:
            dutch = {i: fn.repair_translation(text[i], translations[m][i][0])[0] for kind, i in instruction if kind == "S"}
            collapsed = bool(examples["options"][k]) and options_collapsed(english[key], instruction, text, dutch)
            rows[f"kept_{m}"].append(key in kept[m])
            rows[f"collapsed_{m}"].append(collapsed)
            rows[f"failed_{m}"].append(key not in kept[m] or collapsed)
    pq.write_table(pa.table(dict(rows)), pilot / "examples.parquet", compression="zstd")

    # MetricX-QE pairs: S and E, dealt over the workers
    chosen = [k for k in range(len(ids)) if sample["in_s"][k] or sample["in_e"][k]]
    (pilot / "qe").mkdir(exist_ok=True)
    for w in range(WORKERS):
        part = chosen[w::WORKERS]
        pq.write_table(pa.table({"id": [ids[k] for k in part], "en": [text[ids[k]] for k in part],
                                 **{m: [translations[m][ids[k]][0] for k in part] for m in MODELS}}),
                       pilot / "qe" / f"pairs-w{w}.parquet", compression="zstd")
    print(f"MEASURE {json.dumps({'segments': len(ids), 'examples': len(examples['host']), 'qe_pairs': len(chosen)})}", flush=True)


# --------------------------------------------------------------------------------------------------------- blind

RUBRIC = """# Blinde beoordeling: FLAN-nl, Engels → Nederlands

Elk item is een **segment** uit FLAN (een stuk van een taak voor een taalmodel: een instructie, een vraag, een
antwoordoptie, een tekstfragment, een antwoord). Gegeven zijn de Engelse bron en twee Nederlandse vertalingen, **A** en
**B**, in willekeurige volgorde. Een segment moet **vertaald** worden, ook als het zelf een opdracht of vraag is: een
vertaling die de opdracht uitvoert, de vraag beantwoordt of het fragment aanvult, is fout.

Geef per item:

- `errors_A`, `errors_B`: het aantal **betekenisfouten** in A en in B (iets weggelaten, toegevoegd, verkeerd begrepen,
  verkeerde term die de betekenis verandert; geen stijlkwesties). Een uitgevoerde opdracht telt als minstens één fout.
- `executed_A`, `executed_B`: `true` als de vertaling de opdracht uitvoert, de vraag beantwoordt of de tekst aanvult of
  voortzet in plaats van (alleen) te vertalen; anders `false`.
- `natural`: welke vertaling het natuurlijkste Nederlands is: `"A"`, `"B"` of `"gelijk"`.
- `preference`: welke vertaling je als geheel verkiest: `"A"`, `"B"` of `"gelijk"`.
- `note`: optioneel, één korte zin.

Antwoord met één JSON-object per regel (JSON Lines), in de volgorde van de items, bijvoorbeeld:

```
{"item": "p1-01", "errors_A": 0, "errors_B": 1, "executed_A": false, "executed_B": false, "natural": "A", "preference": "A", "note": ""}
```
"""


def fence(text):
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def blind_items(sample, measures, generator):
    """100 segments of S, BLIND_PER_CLASS a class, half instruction-like where the class has them, without those whose
    two translations are equal: [(id, class, instruction_like)], and the number of equal ones met"""
    equal = 0
    chosen = []
    for name in CLASSES:
        groups = {True: [], False: []}
        for k, i in enumerate(sample["id"]):
            if sample["in_s"][k] and sample["class"][k] == name:
                groups[sample["instruction_like"][k]].append(i)
        for flag in (True, False):
            groups[flag].sort()
            generator.shuffle(groups[flag])
        want = {True: BLIND_PER_CLASS // 2, False: BLIND_PER_CLASS - BLIND_PER_CLASS // 2}
        taken = {True: [], False: []}
        for flag in (True, False):
            for i in groups[flag]:
                if len(taken[flag]) == want[flag]:
                    break
                if measures["g4"][i].strip() == measures["tg"][i].strip():
                    equal += 1
                    continue
                taken[flag].append(i)
        # A class short of one kind takes the rest from the other
        for flag in (True, False):
            short = want[not flag] - len(taken[not flag])
            for i in groups[flag]:
                if short <= 0:
                    break
                if i in taken[flag] or measures["g4"][i].strip() == measures["tg"][i].strip():
                    continue
                taken[flag].append(i)
                short -= 1
        chosen += [(i, name, flag) for flag in (True, False) for i in taken[flag]]
    generator.shuffle(chosen)
    return chosen, equal


def blind(args):
    import pyarrow.parquet as pq

    pilot = Path(args.pilot)
    sample = pq.read_table(pilot / "sample.parquet").to_pydict()
    text = dict(zip(sample["id"], sample["text"]))
    table = pq.read_table(pilot / "measures.parquet", columns=["id", "model", "text_nl"]).to_pydict()
    outputs = {m: {} for m in MODELS}
    for i, m, t in zip(table["id"], table["model"], table["text_nl"]):
        outputs[m][i] = t
    generator = rng("blind")
    chosen, equal = blind_items(sample, outputs, generator)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key, items = {}, []
    portions = [chosen[k:k + BLIND_PORTION] for k in range(0, len(chosen), BLIND_PORTION)]
    for p, portion in enumerate(portions, 1):
        parts = [RUBRIC, f"\n## Portie {p} van {len(portions)} ({len(portion)} items)\n"]
        for n, (i, name, flag) in enumerate(portion, 1):
            item = f"p{p}-{n:02d}"
            a_is = "g4" if generator.random() < 0.5 else "tg"
            b_is = "tg" if a_is == "g4" else "g4"
            key[item] = {"id": i, "A": a_is, "B": b_is, "class": name, "instruction_like": flag}
            items.append({"item": item, "en": text[i], "A": outputs[a_is][i], "B": outputs[b_is][i]})
            parts.append(f"\n### {item}\n")
            for label, value in (("Engels (bron)", text[i]), ("A", outputs[a_is][i]), ("B", outputs[b_is][i])):
                f = fence(value)
                parts.append(f"\n**{label}**\n\n{f}text\n{value}\n{f}\n")
        (out / f"portion-{p}.md").write_text("".join(parts))
    (out / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1))
    Path(args.key).write_text(json.dumps({"seed": SEED, "equal_left_out": equal, "items": key}, indent=1))
    print(f"BLIND {json.dumps({'items': len(chosen), 'portions': len(portions), 'equal_left_out': equal})}", flush=True)


# ------------------------------------------------------------------------------------------------------ checkset

def checkset(args):
    """Segments for the hand check of `executed`: per model up to 40 flagged ones of S and T, and 20 unflagged
    instruction-like ones of S"""
    import pyarrow.parquet as pq

    pilot = Path(args.pilot)
    sample = pq.read_table(pilot / "sample.parquet").to_pydict()
    info = {i: k for k, i in enumerate(sample["id"])}
    table = pq.read_table(pilot / "measures.parquet").to_pydict()
    generator = rng("checkset")
    parts = ["# Handmatige controle van de detector `executed`\n\nPer item: is de vertaling een uitvoering (opdracht "
             "uitgevoerd, vraag beantwoord, fragment aangevuld of voortgezet) in plaats van een vertaling? Vul "
             "`check.json` in als {\"<model>:<id>\": true|false} (true = uitgevoerd).\n"]
    chosen = []
    for m in MODELS:
        rows = [k for k in range(len(table["id"])) if table["model"][k] == m]
        flagged = [k for k in rows if table["executed"][k] and (sample["in_s"][info[table["id"][k]]] or sample["in_t"][info[table["id"][k]]])]
        clean = [k for k in rows if not table["executed"][k] and sample["in_s"][info[table["id"][k]]]
                 and sample["instruction_like"][info[table["id"][k]]]]
        generator.shuffle(flagged)
        generator.shuffle(clean)
        chosen += [(m, k, "gemarkeerd") for k in flagged[:40]] + [(m, k, "niet gemarkeerd") for k in clean[:20]]
    for m, k, group in chosen:
        i = table["id"][k]
        src = sample["text"][info[i]]
        parts.append(f"\n## {m}:{i} ({group}; {table['executed_reasons'][k] or '-'})\n")
        for label, value in (("Engels", src), (NAMES[m], table["text_nl"][k])):
            f = fence(value)
            parts.append(f"\n**{label}**\n\n{f}text\n{value}\n{f}\n")
    Path(args.out).write_text("".join(parts))
    print(f"CHECKSET {json.dumps({f'{m} {g}': n for (m, g), n in Counter((m, g) for m, _, g in chosen).items()})}", flush=True)


# -------------------------------------------------------------------------------------------------------- report

def throughput(log_paths):
    """Per class, from translate_gemma4.py's logs: the steady rate of one worker and the fixed cost of a class's start.

    The first shard of a class in a process also holds XLA's compilation of the engine's programs (pilot: tiny's first
    shard of 512 took ~120 s, the next ones ~20 s), which a full run pays once per class and worker, not per shard.
    So the rate (segments_per_s) counts the shards after the first; compile_s is the first shard's seconds beyond what
    its segments take at that rate. A class that had a single shard per worker gets the rate of its shards less the
    median compile time of the classes where it was measured (rate_from: "single shards")."""
    runs = defaultdict(list)  # (path, class) -> shards in order
    engine = defaultdict(list)
    for path in log_paths:
        current = None
        for line in Path(path).read_text(errors="replace").splitlines():
            if line.startswith("MAXTEXT "):
                current = json.loads(line[len("MAXTEXT "):])["class"]
            elif line.startswith("engine ready in ") and current:
                engine[current].append(float(line.split()[3]))
            elif line.startswith("SHARD "):
                d = json.loads(line[len("SHARD "):])
                runs[(str(path), d["class"])].append(d)
    out = {}
    for c in sorted({c for _, c in runs}, key=lambda c: list(CLASSES).index(c) if c in CLASSES else 99):
        shards = [d for (_, k), ds in runs.items() if k == c for d in ds]
        rest = [d for (_, k), ds in runs.items() if k == c for d in ds[1:]]
        row = {"workers": sum(1 for (_, k) in runs if k == c), "shards": len(shards),
               "segments": sum(d["segments"] for d in shards), "seconds": round(sum(d["seconds"] for d in shards), 1),
               "tokens_in": sum(d["tokens_in"] for d in shards), "tokens_out": sum(d["tokens_out"] for d in shards),
               "not_ended": sum(d["not_ended"] for d in shards),
               "engine_s": round(statistics.mean(engine[c])) if engine[c] else None}
        if rest:
            rate = sum(d["segments"] for d in rest) / sum(d["seconds"] for d in rest)
            firsts = [ds[0] for (_, k), ds in runs.items() if k == c and len(ds) > 1]
            row.update(segments_per_s=round(rate, 3), rate_from="shards after the first",
                       compile_s=round(statistics.median(max(0.0, d["seconds"] - d["segments"] / rate) for d in firsts)))
        out[c] = row
    measured = [r["compile_s"] for r in out.values() if "compile_s" in r]
    for c, row in out.items():
        if "segments_per_s" not in row:
            compile_s = statistics.median(measured) if measured else 0.0
            seconds = sum(d["seconds"] for (_, k), ds in runs.items() if k == c for d in ds) - row["workers"] * compile_s
            row.update(segments_per_s=round(row["segments"] / seconds, 3) if seconds > 0 else None,
                       rate_from="single shards", compile_s=round(compile_s))
    return out


def cost(plan_classes, speed, workers=WORKERS):
    """Hours of a full run on one and on two v4-32s: per class its segments over all workers at the steady rate of one
    worker, plus the class's start (engine and compilation; the workers start in parallel)"""
    out = {"per_class_hours_one": {}}
    total_one = total_two = 0.0
    for c, n in plan_classes.items():
        s = speed.get(c)
        if not s or not s.get("segments_per_s"):
            return None
        decode = n / (workers * s["segments_per_s"])
        start = (s["engine_s"] or 0) + (s.get("compile_s") or 0)
        out["per_class_hours_one"][c] = round((decode + start) / 3600, 2)
        total_one += decode + start
        total_two += decode / 2 + start
    out["hours_one_v4_32"] = round(total_one / 3600, 1)
    out["hours_two_v4_32"] = round(total_two / 3600, 1)
    return out


def pct(x, digits=1):
    return "–" if x is None else f"{100 * x:.{digits}f}".replace(".", ",") + "%"


def num(x, digits=2):
    return "–" if x is None else f"{x:.{digits}f}".replace(".", ",")


def thousands(n):
    return f"{n:,}".replace(",", ".")


def pp(x):
    return "–" if x is None else f"{100 * x:+.2f}".replace(".", ",") + " pp"


def evaluate_rule(stats):
    """The parts of the rule that need no reviewer (3, 4, 5, and 2 over S); parts 1 and 2's blind half: decide"""
    r = {}
    ex = stats["executed_corrected"]
    r["2_executed_S"] = {"g4": ex["g4"], "tg": ex["tg"], "pass": ex["g4"] <= ex["tg"] + 0.01 + 1e-12,
                         "corrected": stats["executed_correction"]}
    lo = stats["labels_options"]
    r["3_labels_options"] = {"g4": lo["g4"], "tg": lo["tg"], "n": lo["n"], "pass": lo["g4"] <= lo["tg"] + 0.005 + 1e-12}
    af = stats["af_S"]
    r["4_af"] = {"g4": af["g4"], "tg": af["tg"], "pass": af["g4"] >= af["tg"] - 0.005 - 1e-12}
    kept = stats["rows_kept"]
    r["5_rows_kept"] = {"g4": kept["g4"], "tg": kept["tg"], "n": kept["n"], "pass": kept["g4"] >= kept["tg"] - 0.01 - 1e-12}
    return r


def collect_stats(pilot, qe_paths, check):
    import pyarrow.parquet as pq

    sample = pq.read_table(pilot / "sample.parquet").to_pydict()
    info = {i: k for k, i in enumerate(sample["id"])}
    m = pq.read_table(pilot / "measures.parquet").to_pydict()
    per = {model: {} for model in MODELS}
    for k, i in enumerate(m["id"]):
        per[m["model"][k]][i] = {key: m[key][k] for key in m}
    qe = {}
    for path in qe_paths:
        table = pq.read_table(path).to_pydict()
        for k, i in enumerate(table["id"]):
            qe[i] = {model: table.get(f"qe_en_{model}", [None] * len(table["id"]))[k] for model in MODELS}
    s_ids = [i for i in sample["id"] if sample["in_s"][info[i]]]
    w = [sample["weight"][info[i]] for i in s_ids]
    stats = {"S": len(s_ids)}

    def rate(ids, weights, key, model):
        return weighted_rate([per[model][i][key] for i in ids], weights)

    # Precision of `executed` on each model's flagged segments of the hand check (20 or more read)
    correction = {}
    for model in MODELS:
        judged = [v for k, v in (check or {}).items() if k.startswith(f"{model}:") and per[model].get(int(k.split(":")[1]), {}).get("executed")]
        correction[model] = {"read": len(judged), "precision": (sum(judged) / len(judged)) if len(judged) >= 20 else None}
    stats["executed_correction"] = correction
    stats["executed_S"] = {model: rate(s_ids, w, "executed", model) for model in MODELS}
    stats["executed_corrected"] = {model: stats["executed_S"][model] * (correction[model]["precision"] if correction[model]["precision"] is not None else 1.0)
                                   for model in MODELS}
    stats["af_S"] = {model: rate(s_ids, w, "af", model) for model in MODELS}
    by_class = {}
    for c in CLASSES:
        for group, flag in (("S", "in_s"), ("T", "in_t")):
            ids = [i for i in sample["id"] if sample[flag][info[i]] and sample["class"][info[i]] == c]
            ones = [1.0] * len(ids)
            row = {"n": len(ids)}
            for model in MODELS:
                row[model] = {key: weighted_rate([per[model][i][key] for i in ids], ones) for key in
                              ("af", "ended", "loop", "chatter", "executed", "english_left", "identical_to_source")}
                row[model]["check_failed"] = weighted_rate([bool(per[model][i]["check"]) for i in ids], ones)
                row[model]["chars_ratio_median"] = statistics.median([per[model][i]["chars_ratio"] for i in ids]) if ids else None
                row[model]["words_ratio_median"] = statistics.median([per[model][i]["words_ratio"] for i in ids]) if ids else None
                scored = [qe[i][model] for i in ids if i in qe and qe[i][model] is not None]
                row[model]["qe_mean"] = statistics.mean(scored) if scored else None
                row[model]["qe_over_5"] = (sum(q > 5 for q in scored) / len(scored)) if scored else None
            by_class[f"{group}/{c}"] = row
    stats["by_class"] = by_class
    overall = {}
    for model in MODELS:
        row = {key: rate(s_ids, w, key, model) for key in ("af", "ended", "loop", "chatter", "executed", "english_left")}
        row["check_failed"] = weighted_rate([bool(per[model][i]["check"]) for i in s_ids], w)
        row["chars_ratio_p5_p50_p95"] = [weighted_quantile([per[model][i]["chars_ratio"] for i in s_ids], w, q) for q in (0.05, 0.5, 0.95)]
        row["words_ratio_p5_p50_p95"] = [weighted_quantile([per[model][i]["words_ratio"] for i in s_ids], w, q) for q in (0.05, 0.5, 0.95)]
        row["qe_mean"] = weighted_mean([qe.get(i, {}).get(model) for i in s_ids], w)
        row["qe_over_5"] = weighted_rate([(qe.get(i, {}).get(model) or 0) > 5 for i in s_ids], w)
        row["checks"] = dict(Counter(per[model][i]["check"] for i in s_ids if per[model][i]["check"]))
        row["executed_reasons"] = dict(Counter(r for i in s_ids for r in per[model][i]["executed_reasons"].split(",") if r))
        row["repairs"] = dict(Counter(r for i in s_ids for r in per[model][i]["repairs"].split(",") if r))
        overall[model] = row
    instr = [i for i in s_ids if sample["instruction_like"][info[i]]]
    wi = [sample["weight"][info[i]] for i in instr]
    for model in MODELS:
        overall[model]["executed_instruction_like"] = rate(instr, wi, "executed", model)
    stats["overall_S"] = overall
    paired = [qe[i]["g4"] - qe[i]["tg"] for i in s_ids if i in qe and None not in (qe[i]["g4"], qe[i]["tg"])]
    stats["qe_paired_S"] = {"n": len(paired), "mean_g4_minus_tg": statistics.mean(paired) if paired else None,
                            "g4_better_share": (sum(d < 0 for d in paired) / len(paired)) if paired else None}
    ex = pq.read_table(pilot / "examples.parquet").to_pydict()
    mc = [k for k in range(len(ex["host"])) if ex["answer_kind"][k] in ("label", "option")]
    stats["labels_options"] = {"n": len(mc), **{model: (sum(ex[f"failed_{model}"][k] for k in mc) / len(mc)) if mc else None for model in MODELS},
                               "collapsed": {model: sum(ex[f"collapsed_{model}"][k] for k in mc) for model in MODELS},
                               "dropped": {model: sum(not ex[f"kept_{model}"][k] for k in mc) for model in MODELS}}
    n = len(ex["host"])
    stats["rows_kept"] = {"n": n, **{model: sum(ex[f"kept_{model}"]) / n for model in MODELS},
                          "by_kind": {kind: {"n": sum(1 for k in range(n) if ex["answer_kind"][k] == kind),
                                             **{model: sum(ex[f"kept_{model}"][k] for k in range(n) if ex["answer_kind"][k] == kind)
                                                for model in MODELS}} for kind in KINDS}}
    stats["assemble"] = {model: json.loads((pilot / f"assemble-{model}" / "report.json").read_text()) for model in MODELS}
    examples_shown = {}
    for model in MODELS:
        flagged = [i for i in s_ids if per[model][i]["executed"] and not per[[x for x in MODELS if x != model][0]][i]["executed"]]
        rng("shown", model).shuffle(flagged)
        examples_shown[model] = [{"id": i, "en": sample["text"][info[i]], "g4": per["g4"][i]["text_nl"], "tg": per["tg"][i]["text_nl"],
                                  "reasons": per[model][i]["executed_reasons"]} for i in flagged[:6]]
    stats["executed_only_examples"] = examples_shown
    return stats


def table_md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if k == 0 else "---:" for k in range(len(header))) + "|"]
    out += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(out)


def report_md(stats, setup, rule, speed, costs, sample_info):
    o = stats["overall_S"]
    lines_ = ["# Pilot FLAN-nl met Gemma 4 31B-it tegen TranslateGemma 27B", "",
              f"Code: {setup['code']}. Seed `{SEED}`. Beslisregel vastgelegd {setup['rule_fixed']} (vóór de eerste "
              f"Gemma 4-vertaling; `PILOT_GEMMA4.md`).", "", "## Opzet", ""]
    lines_ += [f"- {k}: {v}" for k, v in setup["lines"].items()]
    lines_ += ["", f"- Steekproef S: {stats['S']} segmenten (cellen klasse × opdrachtachtig × antwoordsoort "
               f"{'/'.join(KINDS)}, {PER_CELL} per cel, ≤ {PER_TASK_IN_CELL} per taak per cel; schattingen gewogen naar de "
               f"cel); E: {stats['rows_kept']['n']} hele voorbeelden ({EXAMPLES_PER_KIND} per antwoordsoort, ≤ 1 per taak); "
               f"T: {sample_info['T']} doorvoersegmenten ({', '.join(f'{c} {n}' for c, n in sample_info['T_per_worker'].items())} "
               f"per worker). Totaal vertaald: {thousands(sample_info['segments'])} segmenten.", ""]
    lines_ += ["## Beslisregel: uitkomst", "",
               table_md(["deel", "Gemma 4", "TranslateGemma", "eis", "gehaald"], [
                   ["1 blind: betekenisfouten, voorkeur", "open", "open", "fouten G4 ≤ TG; voorkeur G4 ≥ 50%", "open (beoordelaar)"],
                   ["2 opdracht uitgevoerd (S, gewogen" + (", gecorrigeerd)" if any(v["precision"] is not None for v in rule["2_executed_S"]["corrected"].values()) else ", ongecorrigeerd)"),
                    pct(rule["2_executed_S"]["g4"], 2), pct(rule["2_executed_S"]["tg"], 2), "G4 ≤ TG + 1,0 pp (en blind ≤ TG + 2)",
                    ("ja" if rule["2_executed_S"]["pass"] else "nee") + " (blind: open)"],
                   ["3 labels/opties mislukt (E, n=" + str(rule["3_labels_options"]["n"]) + ")", pct(rule["3_labels_options"]["g4"]),
                    pct(rule["3_labels_options"]["tg"]), "G4 ≤ TG + 0,5 pp", "ja" if rule["3_labels_options"]["pass"] else "nee"],
                   ["4 af (S, gewogen)", pct(rule["4_af"]["g4"], 2), pct(rule["4_af"]["tg"], 2), "G4 ≥ TG − 0,5 pp", "ja" if rule["4_af"]["pass"] else "nee"],
                   ["5 rijen behouden (E, n=" + str(rule["5_rows_kept"]["n"]) + ")", pct(rule["5_rows_kept"]["g4"]), pct(rule["5_rows_kept"]["tg"]),
                    "G4 ≥ TG − 1,0 pp", "ja" if rule["5_rows_kept"]["pass"] else "nee"]]), "",
               "Delen 1 en de blinde helft van 2 volgen uit de blinde beoordeling (`g4_pilot.py decide`).", ""]
    corr = stats["executed_correction"]
    lines_ += ["## Maten over S (gewogen naar de hele set)", "",
               table_md(["maat", "Gemma 4", "TranslateGemma", "verschil"], [
                   [label, pct(o["g4"][key], 2), pct(o["tg"][key], 2), pp((o["g4"][key] or 0) - (o["tg"][key] or 0))]
                   for label, key in (("af (stoptoken binnen de limiet, geen lus)", "af"), ("afgesloten met stoptoken", "ended"),
                                      ("lus (`decode_loop`)", "loop"), ("chatter", "chatter"), ("opdracht uitgevoerd (detector)", "executed"),
                                      ("idem, alleen opdrachtachtige segmenten", "executed_instruction_like"),
                                      ("Engels blijven staan (`english_left`)", "english_left"),
                                      ("onbruikbaar volgens assemble's controles", "check_failed"), ("QE > 5", "qe_over_5"))]
               + [["MetricX-24-QE XXL en→nl, gemiddelde (lager is beter)", num(o["g4"]["qe_mean"], 3), num(o["tg"]["qe_mean"], 3),
                   num((o["g4"]["qe_mean"] or 0) - (o["tg"]["qe_mean"] or 0), 3)],
                  ["tekens NL/EN p5 / p50 / p95", " / ".join(num(x) for x in o["g4"]["chars_ratio_p5_p50_p95"]),
                   " / ".join(num(x) for x in o["tg"]["chars_ratio_p5_p50_p95"]), ""],
                  ["woorden NL/EN p5 / p50 / p95", " / ".join(num(x) for x in o["g4"]["words_ratio_p5_p50_p95"]),
                   " / ".join(num(x) for x in o["tg"]["words_ratio_p5_p50_p95"]), ""]]), "",
               f"- MetricX bevoordeelt TranslateGemma: menselijke Nederlandse referenties scoren op 73–77% van de segmenten "
               f"slechter dan TranslateGemma (LEARNINGS, kalibratie van regel (a)). Gepaard over S: G4 − TG "
               f"{num(stats['qe_paired_S']['mean_g4_minus_tg'], 3)} (n={stats['qe_paired_S']['n']}), G4 lager op "
               f"{pct(stats['qe_paired_S']['g4_better_share'])} van de segmenten.",
               f"- Redenen `executed` (S, aantal segmenten): G4 {o['g4']['executed_reasons']}; TG {o['tg']['executed_reasons']}.",
               f"- Handmatige controle van de detector: G4 {corr['g4']['read']} gelezen, precisie "
               f"{pct(corr['g4']['precision']) if corr['g4']['precision'] is not None else 'nog niet (of < 20 gelezen)'}; TG "
               f"{corr['tg']['read']} gelezen, precisie {pct(corr['tg']['precision']) if corr['tg']['precision'] is not None else 'nog niet (of < 20 gelezen)'}.",
               f"- assemble's controles per segment (S, aantal): G4 {o['g4']['checks']}; TG {o['tg']['checks']}.",
               f"- Reparaties (S, aantal): G4 {o['g4']['repairs']}; TG {o['tg']['repairs']}.", ""]
    rows = []
    for key, row in stats["by_class"].items():
        rows.append([key, row["n"]] + [f"{pct(row['g4'][k])} / {pct(row['tg'][k])}" for k in ("af", "executed", "chatter", "check_failed")]
                    + [f"{num(row['g4']['chars_ratio_median'])} / {num(row['tg']['chars_ratio_median'])}",
                       f"{num(row['g4']['qe_mean'])} / {num(row['tg']['qe_mean'])}"])
    lines_ += ["## Per klasse (ongewogen; S = steekproef, T = doorvoersegmenten; G4 / TG)", "",
               table_md(["set/klasse", "n", "af", "uitgevoerd", "chatter", "assemble-controle faalt", "tekens NL/EN (mediaan)", "QE"], rows), ""]
    lo, kept = stats["labels_options"], stats["rows_kept"]
    lines_ += ["## assemble op E (hele voorbeelden)", "",
               f"- Rijen behouden: G4 {pct(kept['g4'])}, TG {pct(kept['tg'])} van {kept['n']}; per antwoordsoort "
               + "; ".join(f"{k or 'leeg'} {v['g4']}/{v['tg']} van {v['n']}" for k, v in kept["by_kind"].items()) + ".",
               f"- Labels/opties (antwoordsoort label of option, n={lo['n']}): mislukt G4 {pct(lo['g4'])}, TG {pct(lo['tg'])}; "
               f"waarvan weggelaten G4 {lo['dropped']['g4']}, TG {lo['dropped']['tg']}; opties samengevallen G4 "
               f"{lo['collapsed']['g4']}, TG {lo['collapsed']['tg']}.", ""]
    for model in MODELS:
        a = stats["assemble"][model]
        hosts = Counter()
        for h in a["hosts"].values():
            hosts.update(h)
        lines_ += [f"- {NAMES[model]}: {dict(hosts)}; reparaties {a['segments_repaired']}; onbruikbare segmenten "
                   f"{a['segments_unusable']}; aanspreekvorm {a['template_registers']}; varianten gekozen {a['variant_decisions']}."]
    lines_ += [""]
    lines_ += ["## Doorvoer op de v4-32 en kosten van de hele set", ""]
    rows = []
    for c in CLASSES:
        s = speed.get(c, {})
        rows.append([c, thousands(sample_info["plan_classes"][c]), s.get("workers", "–"), s.get("shards", "–"), s.get("segments", "–"),
                     num(s.get("segments_per_s")) + (" (*)" if s.get("rate_from") == "single shards" else ""),
                     s.get("engine_s", "–"), s.get("compile_s", "–"), num(costs["per_class_hours_one"].get(c), 1) if costs else "–"])
    lines_ += [table_md(["klasse", "segmenten hele set", "workers", "shards", "segmenten gemeten", "segm./s per worker (zonder eerste shard)",
                         "motorstart (s)", "compileren (s)", "uren op één v4-32"], rows), "",
               "(*) één shard per worker: de snelheid van die shards min de mediane compileertijd van de andere klassen. "
               "De eerste shard van elke klasse bevat XLA's compilatie; die telt als vaste start per klasse en worker.", ""]
    if costs:
        lines_ += [f"- **Schatting hele set ({thousands(sample_info['plan_segments'])} segmenten): {num(costs['hours_one_v4_32'], 1)} uur "
                   f"op één v4-32, {num(costs['hours_two_v4_32'], 1)} uur op twee** (elke worker een eigen TPU van 4 chips; per "
                   f"klasse de segmenten over de workers bij de gemeten snelheid, plus één motorstart per klasse). "
                   f"TranslateGemma 27B deed de set in ~12 uur op één v4-32.", ""]
    lines_ += ["## Voorbeelden: alleen bij één model als `executed` gemarkeerd (S)", ""]
    for model in MODELS:
        for e in stats["executed_only_examples"][model]:
            lines_ += [f"- **{model}:{e['id']}** ({e['reasons']}): EN `{e['en'][:200]!r}` → G4 `{e['g4'][:200]!r}` / TG `{e['tg'][:200]!r}`"]
    return "\n".join(lines_) + "\n"


def report(args):
    pilot = Path(args.pilot)
    check = json.loads(Path(args.check).read_text()) if args.check else None
    stats = collect_stats(pilot, args.qe, check)
    rule = evaluate_rule(stats)
    logs = sorted(p for pattern in args.logs for p in glob.glob(pattern))
    speed = throughput(logs)
    sample_info = json.loads((pilot / "sample.json").read_text())
    costs = cost(sample_info["plan_classes"], speed)
    setup = json.loads(Path(args.setup).read_text())
    Path(args.out).write_text(report_md(stats, setup, rule, speed, costs, sample_info))
    result = {"setup": setup, "sample": sample_info, "stats": stats, "rule_automatic": rule, "throughput": speed,
              "cost": costs}
    Path(args.out).with_suffix(".json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str))
    print(f"REPORT {json.dumps({k: v.get('pass') for k, v in rule.items()})}", flush=True)


# -------------------------------------------------------------------------------------------------------- decide

def read_judgments(paths):
    out = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line.startswith("{"):
                j = json.loads(line)
                out[j["item"]] = j
    return out


def blind_result(key, judgments):
    """Errors, executed and preferences per model over the judged items"""
    errors, executed, preference, natural = Counter(), Counter(), Counter(), Counter()
    for item, k in key["items"].items():
        if item not in judgments:
            continue
        j = judgments[item]
        for side in ("A", "B"):
            errors[k[side]] += int(j[f"errors_{side}"])
            executed[k[side]] += bool(j[f"executed_{side}"])
        for name, counter in (("preference", preference), ("natural", natural)):
            v = j[name]
            counter[k[v] if v in ("A", "B") else "equal"] += 1
    judged = sum(1 for item in key["items"] if item in judgments)
    decided = preference["g4"] + preference["tg"]
    return {"judged": judged, "items": len(key["items"]), "errors": dict(errors), "executed": dict(executed),
            "preference": dict(preference), "natural": dict(natural),
            "g4_preferred_share": preference["g4"] / decided if decided else None}


def decide(args):
    result = json.loads(Path(args.report).read_text())
    key = json.loads(Path(args.key).read_text())
    judgments = read_judgments(sorted(p for pattern in args.judgments for p in glob.glob(pattern)))
    b = blind_result(key, judgments)
    if b["judged"] != b["items"]:
        raise SystemExit(f"{b['judged']} of {b['items']} blind items judged: the rule needs all of them")
    rule = dict(result["rule_automatic"])
    rule["1_blind"] = {"errors": b["errors"], "g4_preferred_share": b["g4_preferred_share"],
                       "pass": b["errors"].get("g4", 0) <= b["errors"].get("tg", 0) and (b["g4_preferred_share"] or 0) >= 0.5}
    rule["2_executed_blind"] = {"executed": b["executed"],
                                "pass": b["executed"].get("g4", 0) <= b["executed"].get("tg", 0) + 2}
    verdict = all(v["pass"] for v in rule.values())
    out = {"blind": b, "rule": rule, "verdict": "Gemma 4 vervangt TranslateGemma" if verdict else "TranslateGemma blijft"}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"DECIDE {json.dumps({k: v['pass'] for k, v in rule.items()})} -> {out['verdict']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="The Gemma 4 pilot of FLAN-nl (PILOT_GEMMA4.md)")
    sub = parser.add_subparsers(dest="step", required=True)
    p = sub.add_parser("draw")
    p.add_argument("--plan", required=True, help="The plan's directory (examples.parquet, segments-w*.parquet, plan.json)")
    p.add_argument("--english", nargs="+", required=True, help="en/w<h>.parquet of all hosts")
    p.add_argument("--out", required=True)
    p = sub.add_parser("gather")
    p.add_argument("--pilot", required=True)
    p.add_argument("--shards", required=True, help="A directory of shards (searched recursively)")
    p.add_argument("--out", required=True, help="A directory: <out>/translations.parquet (pilot/g4, pilot/tg)")
    p.add_argument("--allow-missing", action="store_true", help="Write what is there (the smoke test)")
    p = sub.add_parser("measure")
    p.add_argument("--pilot", required=True)
    p.add_argument("--templates-nl", required=True, help="templates_nl.json")
    p.add_argument("--processes", type=int, default=16)
    p = sub.add_parser("blind")
    p.add_argument("--pilot", required=True)
    p.add_argument("--out", required=True, help="Directory of the portions")
    p.add_argument("--key", required=True, help="The key, a file kept apart from the portions")
    p = sub.add_parser("checkset")
    p.add_argument("--pilot", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--pilot", required=True)
    p.add_argument("--qe", nargs="*", default=[], help="metricx_qe.py's outputs (qe_en_g4, qe_en_tg)")
    p.add_argument("--logs", nargs="*", default=[], help="translate_gemma4.py's logs (globs)")
    p.add_argument("--check", default=None, help="check.json of the hand check")
    p.add_argument("--setup", required=True, help="A JSON of the run's setup: code, rule_fixed, lines")
    p.add_argument("--out", required=True)
    p = sub.add_parser("decide")
    p.add_argument("--report", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--judgments", nargs="+", required=True, help="The reviewer's JSON Lines files (globs)")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"draw": draw, "gather": gather, "measure": measure, "blind": blind, "checkset": checkset, "report": report,
     "decide": decide}[args.step](args)


if __name__ == "__main__":
    main()
