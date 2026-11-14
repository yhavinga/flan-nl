#!/usr/bin/env python3
"""The whole of FLAN-nl translated with Gemma 4 31B-it (after the pilot, PILOT_GEMMA4.md): the plan's segments in
units that any worker of any TPU VM claims in the bucket, their status, and the checks of the whole set.

    python g4_full.py units --plan plan2 --out units.json [--tokenizer DIR]
    python g4_full.py status --units units.json (--out gs://<dir of the shards> | --listing FILE)
    python g4_full.py check --plan plan2 --english en/w0.parquet ... --shards DIR --units units.json \\
        --templates-nl templates_nl.json --nl nl-g4 --out report/flan-g4-full.md [--tg-report nl/report.json]

units: per segment file of the plan and length class (models.yml; a segment's class by the plan's token count, as in
translate_gemma4.py), the class's segments in file order cut into units of UNIT_SIZE[class] -> units.json. The classes
go longest first (ORDER): every worker takes the next free unit, so the units at the end of the run are the shortest
and the workers finish within a few minutes of each other. With --tokenizer every prompt is measured first; one longer
than its class's largest prefill length stops the planning, not a worker in the middle of the run.
status: per class the units and segments stored and pending, the hours left at the pilot's speeds; COMPLETE or
PENDING <n> on the last line.
check: the pilot's measures (g4_pilot.segment_measures: af, loop, chatter, executed, assemble's checks and repairs,
english_left, length ratios) on every segment, per class; flan_nl.py assemble into --nl (the rows kept and why the
others were left out); the labels and options of the examples whose answer is one (left out, options collapsed); the
report in Dutch (Markdown) and JSON beside it, and measures.parquet (one row per segment).

The translation is translate_gemma4.py --units: for each unit of its class a worker claims it (claim(): an object in
the claims directory, created only if absent; a claim older than the lease, or of another run on the same VM, is taken
over, as in nl-backtranslation's gemma4_bt/bt_gemma4.py), translates it and stores <out>/<class>/<unit>.parquet with
the columns of translate.py's shards. A stored unit is skipped, so a restart and a second VM go on where the work is.
"""

import argparse
import glob
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

ORDER = ("xlong", "long", "medium", "small", "tiny")
# Units of a few minutes each at the pilot's speeds (~2 min tiny and small, ~3 medium, ~5.5 long, ~9 xlong); the lease
# of a claim (LEASE_SECONDS) holds the longest with the engine's start and compilation
UNIT_SIZE = {"xlong": 512, "long": 512, "medium": 512, "small": 1024, "tiny": 2048}
LEASE_SECONDS = 30 * 60
# Segments a second per worker (a TPU of 4 chips of a v4-32), measured in the pilot without each class's first shard
PILOT_SPEED = {"tiny": 18.89, "small": 10.07, "medium": 3.02, "long": 1.55, "xlong": 0.98}


def now():
    return time.time()


# ------------------------------------------------------------------------------------------------------- units

def plan_units(files, bounds, sizes=None, order=ORDER):
    """files: {file name: (ids, tokens)} of the plan's segment files; bounds: {class: (low, high)} (a segment of n
    tokens is in the class when low < n <= high). The units: per class (in order) and file, the class's segments in
    file order cut into pieces of sizes[class]: {name, class, segments, index, n, first, last, tokens}."""
    sizes = sizes or UNIT_SIZE
    units = []
    for c in order:
        low, high = bounds[c]
        for name in sorted(files):
            ids, tokens = files[name]
            chosen = [k for k, n in enumerate(tokens) if low < n <= high]
            size = sizes[c]
            for index, start in enumerate(range(0, len(chosen), size)):
                part = chosen[start:start + size]
                units.append({"name": f"{c}-{Path(name).stem.replace('segments-', '')}-{index:04d}", "class": c,
                              "segments": name, "index": index, "n": len(part), "first": ids[part[0]],
                              "last": ids[part[-1]], "tokens": sum(tokens[k] for k in part)})
    return units


def unit_rows(unit, size, ids, texts, tokens, low, high):
    """(ids, texts) of a unit, from its segment file's columns; checked against the plan's unit"""
    chosen = [k for k, n in enumerate(tokens) if low < n <= high][unit["index"] * size:][:unit["n"]]
    if len(chosen) != unit["n"] or ids[chosen[0]] != unit["first"] or ids[chosen[-1]] != unit["last"]:
        raise SystemExit(f"{unit['name']}: the segment file does not hold the unit of units.json")
    return [ids[k] for k in chosen], [texts[k] for k in chosen]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def units(args):
    import pyarrow.parquet as pq

    import gemma4_engine as ge
    import translate_gemma4 as tg4

    paths = sorted(Path(args.plan).glob("segments-w*.parquet"))
    if not paths:
        raise SystemExit(f"no segments-w*.parquet in {args.plan}")
    bounds = {c: tg4.class_bounds(args.model, c) for c in ORDER}
    largest = max(high for _, high in bounds.values())
    files, texts = {}, {}
    for path in paths:
        table = pq.read_table(path, columns=["id", "text", "tokens"]).to_pydict()
        if max(table["tokens"]) > largest:
            raise SystemExit(f"{path} has a segment of {max(table['tokens'])} tokens, more than the classes allow ({largest})")
        files[path.name] = (table["id"], table["tokens"])
        texts[path.name] = table["text"]
    planned = plan_units(files, bounds)
    model = ge.MODELS[args.model]
    longest = {}
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        codec = ge.TurnCodec(tokenizer, model, model["prompt"], "en", "nl")
        for c in ORDER:
            low, high = bounds[c]
            prompts = [codec.text(t) for name in files for t, n in zip(texts[name], files[name][1]) if low < n <= high]
            lengths = [len(x) for x in tokenizer(prompts, add_special_tokens=False)["input_ids"]] if prompts else [0]
            limit = model["length_classes"][c]["prefill_lengths"][-1]
            longest[c] = max(lengths)
            if longest[c] > limit:
                raise SystemExit(f"{c}: a prompt of {longest[c]} tokens is longer than the class's largest prefill "
                                 f"length {limit}")
    per_class = {c: {"low": bounds[c][0], "high": bounds[c][1], "size": UNIT_SIZE[c],
                     "units": sum(u["class"] == c for u in planned), "segments": sum(u["n"] for u in planned if u["class"] == c),
                     "tokens": sum(u["tokens"] for u in planned if u["class"] == c), "longest_prompt": longest.get(c)}
                 for c in ORDER}
    info = {"model": args.model, "prompt": model["prompt"], "order": list(ORDER), "classes": per_class,
            "files": {p.name: {"segments": len(files[p.name][0]), "sha256": sha256(p)} for p in paths},
            "segments": sum(len(v[0]) for v in files.values()), "units": planned}
    if sum(u["n"] for u in planned) != info["segments"]:
        raise SystemExit("the units do not hold every segment of the plan once")
    Path(args.out).write_text(json.dumps(info, indent=1))
    print(f"UNITS {json.dumps({'segments': info['segments'], 'units': len(planned), 'classes': {c: {k: v for k, v in d.items() if k in ('units', 'segments', 'longest_prompt')} for c, d in per_class.items()}})}", flush=True)


# ------------------------------------------------------------------------------------------------------ claims

class GcsClaims:
    """A bucket directory (gs://bucket/prefix) of small text objects through the Python client: create and replace
    with generation preconditions, so that of two workers claiming one unit only one succeeds"""

    def __init__(self, root, client=None):
        import gemma4_engine as ge

        self.client = client or ge.gcs_client()
        name, prefix = ge.split_gs(root.rstrip("/"))
        self.bucket, self.prefix = self.client.bucket(name), prefix.rstrip("/")

    def _blob(self, key):
        return self.bucket.blob(f"{self.prefix}/{key}")

    def read(self, key):
        """(text, generation), or None when absent or changed while being read"""
        from google.api_core.exceptions import NotFound, PreconditionFailed

        blob = self._blob(key)
        try:
            blob.reload()
            return blob.download_as_text(if_generation_match=blob.generation), blob.generation
        except (NotFound, PreconditionFailed):
            return None

    def _write(self, key, text, generation):
        from google.api_core.exceptions import NotFound, PreconditionFailed

        try:
            self._blob(key).upload_from_string(text, if_generation_match=generation)
            return True
        except (PreconditionFailed, NotFound):
            return False

    def create(self, key, text):
        return self._write(key, text, 0)

    def replace(self, key, text, generation):
        return self._write(key, text, generation)


class MemoryClaims:
    """The same in memory, for the tests"""

    def __init__(self):
        self.objects, self.generation = {}, 0

    def read(self, key):
        return (self.objects[key][0], self.objects[key][1]) if key in self.objects else None

    def create(self, key, text):
        if key in self.objects:
            return False
        self.generation += 1
        self.objects[key] = (text, self.generation)
        return True

    def replace(self, key, text, generation):
        if key not in self.objects or self.objects[key][1] != generation:
            return False
        self.generation += 1
        self.objects[key] = (text, self.generation)
        return True


def claim(store, key, me, lease=LEASE_SECONDS, t=None):
    """Claim key for me ({owner: <vm>/w<k>, run}): created if absent; taken over when the claim is older than lease
    seconds, or of another run of the same VM (a chain started again). Returns whether it is mine. As claim() of
    nl-backtranslation's gemma4_bt/bt_gemma4.py."""
    t = now() if t is None else t
    body = json.dumps({**me, "time": t})
    if store.create(key, body):
        return True
    found = store.read(key)
    if found is None:  # gone in between: the next pass tries again
        return False
    text, generation = found
    old = json.loads(text)
    mine = old.get("owner") == me["owner"] and old.get("run") == me["run"]
    same_vm = old.get("owner", "").split("/")[0] == me["owner"].split("/")[0] and old.get("run") != me["run"]
    if mine or same_vm or t - old.get("time", 0) > lease:
        return store.replace(key, body, generation)
    return False


def work_units(units, stored, claim_unit, exists, translate, store, max_units=None):
    """For each unit in order that is not stored: claim it, translate it (translate(unit) -> (rows, seconds)) and
    store it (store(unit, rows, seconds)). stored() -> the names stored (a listing, at most a minute old);
    claim_unit(name) -> whether the unit is now ours; exists(name) -> stored since the listing. A unit is stored whole
    or not at all. Returns the names done here."""
    done = []
    for u in units:
        if max_units is not None and len(done) == max_units:
            break
        if u["name"] in stored() or not claim_unit(u["name"]) or exists(u["name"]):
            continue
        rows, seconds = translate(u)
        store(u, rows, seconds)
        done.append(u["name"])
    return done


# ------------------------------------------------------------------------------------------------------ status

def status_of(info, stored_names, workers=(4, 8)):
    """Per class the units and segments stored and pending, and the hours left at the pilot's speeds"""
    per = {}
    for c in info["order"]:
        mine = [u for u in info["units"] if u["class"] == c]
        left = [u for u in mine if u["name"] not in stored_names]
        per[c] = {"units": len(mine), "stored": len(mine) - len(left), "segments": sum(u["n"] for u in mine),
                  "segments_left": sum(u["n"] for u in left)}
    pending = sum(p["units"] - p["stored"] for p in per.values())
    total = sum(p["segments"] for p in per.values())
    done = total - sum(p["segments_left"] for p in per.values())
    hours = {f"hours_left_{w}_workers": round(sum(per[c]["segments_left"] / PILOT_SPEED[c] for c in per) / w / 3600, 2)
             for w in workers}
    return {"classes": per, "pending": pending, "segments_done": done, "segments": total,
            "share_done": round(done / max(total, 1), 4), **hours}


def stored_listing(names):
    """Unit names from object names or URLs of <out>/<class>/<unit>.parquet"""
    return {Path(n.strip()).stem for n in names if n.strip().endswith(".parquet")}


def status(args):
    info = json.loads(Path(args.units).read_text())
    if args.listing:
        names = stored_listing(Path(args.listing).read_text().split())
    else:
        import gemma4_engine as ge

        bucket, prefix = ge.split_gs(args.out.rstrip("/") + "/")
        names = stored_listing(b.name for b in ge.gcs_client().list_blobs(bucket, prefix=prefix))
    unknown = names - {u["name"] for u in info["units"]}
    if unknown:
        raise SystemExit(f"{len(unknown)} stored shards are no units of {args.units}, e.g. {sorted(unknown)[:3]}")
    s = status_of(info, names)
    print(f"STATUS {json.dumps(s)}")
    print("COMPLETE" if s["pending"] == 0 else f"PENDING {s['pending']}", flush=True)


# ------------------------------------------------------------------------------------------------------- check

def _measure(items):
    import g4_pilot as gp

    return [dict(gp.segment_measures(*item), instruction_like=gp.instruction_like(item[0])) for item in items]


def quantiles(values, qs=(0.05, 0.5, 0.95)):
    if not values:
        return [None] * len(qs)
    ordered = sorted(values)
    return [round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 3) for q in qs]


def class_summary(rows):
    """The measures of one class's segments (or all) as rates and counts"""
    n = len(rows)
    il = [r for r in rows if r["instruction_like"]]

    def rate(key, subset=rows):
        return round(sum(bool(r[key]) for r in subset) / max(len(subset), 1), 5)

    return {"segments": n, "af": rate("af"), "ended": rate("ended"), "loop": rate("loop"), "chatter": rate("chatter"),
            "executed": rate("executed"), "instruction_like": len(il), "executed_instruction_like": rate("executed", il),
            "english_left": rate("english_left"), "identical_to_source": rate("identical_to_source"),
            "unusable": rate("check"), "unusable_reasons": dict(Counter(r["check"] for r in rows if r["check"]).most_common()),
            "executed_reasons": dict(Counter(x for r in rows for x in r["executed_reasons"].split(",") if x).most_common()),
            "repairs": dict(Counter(x for r in rows for x in r["repairs"].split(",") if x).most_common()),
            "chars_ratio_p5_p50_p95": quantiles([r["chars_ratio"] for r in rows]),
            "words_ratio_p5_p50_p95": quantiles([r["words_ratio"] for r in rows]),
            "tokens_out_mean": round(statistics.fmean(r["tokens_out"] for r in rows), 1) if rows else None}


def rows_kept(report):
    """From assemble's report.json: the examples to translate, those kept, and why the others were left out"""
    hosts = report["hosts"].values()
    reasons = Counter()
    for h in hosts:
        for k, v in h.items():
            if k.startswith("dropped:segment_") or k in ("dropped:answer_not_in_instruction", "dropped:empty"):
                reasons[k] += v
    kept = sum(h.get("translated", 0) for h in hosts)
    total = kept + sum(reasons.values())
    return {"to_translate": total, "kept": kept, "kept_share": round(kept / max(total, 1), 5),
            "dropped": dict(reasons.most_common()), "flan_rows": sum(h.get("flan_rows", 0) for h in hosts),
            "kept_dutch_task": sum(h.get("kept_dutch_task", 0) for h in hosts)}


def check(args):
    import concurrent.futures

    import pyarrow as pa
    import pyarrow.parquet as pq

    import flan_nl as fn
    import g4_pilot as gp

    info = json.loads(Path(args.units).read_text())
    segments, klass = {}, {}
    for path in sorted(glob.glob(f"{args.plan}/segments-w*.parquet")):
        table = pq.read_table(path, columns=["id", "text", "tokens"]).to_pydict()
        for i, text, n in zip(table["id"], table["text"], table["tokens"]):
            segments[i] = text
            klass[i] = gp.class_of(n)
    if len(segments) != info["segments"]:
        raise SystemExit(f"the plan has {len(segments):,} segments, units.json {info['segments']:,}")
    translated = {}
    for path in sorted(glob.glob(f"{args.shards}/**/*.parquet", recursive=True)):
        table = pq.read_table(path, columns=["id", "text_nl", "ended", "tokens_out"]).to_pydict()
        for i, out, ended, n in zip(table["id"], table["text_nl"], table["ended"], table["tokens_out"]):
            if i in translated:
                raise SystemExit(f"segment {i} translated twice ({path})")
            translated[i] = (out, ended, n)
    missing = set(segments) - set(translated)
    if missing:
        raise SystemExit(f"{len(missing):,} of {len(segments):,} segments have no translation, e.g. {sorted(missing)[:5]}")
    ids = sorted(segments)
    repaired = {i: fn.repair_translation(segments[i], translated[i][0])[0] for i in ids}
    chunk = 2000
    with concurrent.futures.ProcessPoolExecutor(args.processes) as pool:
        outputs = [repaired[i] for i in ids]
        languages = [x for part in pool.map(fn._identify, [outputs[k:k + chunk] for k in range(0, len(ids), chunk)]) for x in part]
        items = [(segments[i], translated[i][0], translated[i][1], lang) for i, lang in zip(ids, languages)]
        measured = [r for part in pool.map(_measure, [items[k:k + chunk] for k in range(0, len(items), chunk)]) for r in part]
    rows = [dict(r, id=i, length_class=klass[i], tokens_out=translated[i][2]) for i, r in zip(ids, measured)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = defaultdict(list)
    for r in rows:
        for key, value in r.items():
            columns[key].append(value)
    pq.write_table(pa.table(dict(columns)), out.with_name("measures.parquet"), compression="zstd")
    per_class = {c: class_summary([r for r in rows if r["length_class"] == c]) for c in gp.CLASSES}
    per_class["all"] = class_summary(rows)
    del rows, items, measured, columns
    print(f"MEASURED {json.dumps({c: {k: v for k, v in s.items() if k in ('segments', 'af', 'executed', 'unusable')} for c, s in per_class.items()})}", flush=True)

    # assemble: the rows kept, why the others were left out
    Path(args.nl).mkdir(parents=True, exist_ok=True)
    fn.main(["assemble", "--english", *args.english, "--plan", args.plan, "--translations", args.shards,
             "--out", args.nl, "--processes", str(args.processes), "--templates-nl", args.templates_nl])
    assembled = json.loads((Path(args.nl) / "report.json").read_text())
    kept_rows = rows_kept(assembled)
    tg_rows = rows_kept(json.loads(Path(args.tg_report).read_text())) if args.tg_report else None

    # labels and options: an example whose answer is a label or an option fails when it is left out or two of its
    # English options get one Dutch text (the pilot's part 3)
    kept = set()
    for path in sorted(Path(args.nl).glob("w*.parquet")):
        h = int(path.stem[1:])
        kept |= {(h, p) for p in pq.read_table(path, columns=["position"]).column("position").to_pylist()}
    english = {}
    for path in args.english:
        table = pq.read_table(path, columns=["host", "position", "inputs"]).to_pydict()
        english.update({(h, p): x for h, p, x in zip(table["host"], table["position"], table["inputs"])})
    examples = pq.read_table(Path(args.plan) / "examples.parquet",
                             columns=["host", "position", "source", "status", "instruction", "options", "answer_kind"]).to_pydict()
    kinds = defaultdict(Counter)
    for k in range(len(examples["host"])):
        if examples["status"][k] != "translate" or fn.task_drop_reason(fn.task_of(examples["source"][k])) is not None:
            continue
        key, kind = (examples["host"][k], examples["position"][k]), examples["answer_kind"][k]
        c = kinds[kind]
        c["examples"] += 1
        c["kept"] += key in kept
        if kind in ("label", "option"):
            instruction = [(t, v) for t, v in json.loads(examples["instruction"][k])]
            collapsed = bool(examples["options"][k]) and gp.options_collapsed(english[key], instruction, segments, repaired)
            c["options_collapsed"] += collapsed
            c["failed"] += key not in kept or collapsed
    labels = {kind: dict(c, failed_share=round(c["failed"] / max(c["examples"], 1), 5) if kind in ("label", "option") else None,
                         kept_share=round(c["kept"] / max(c["examples"], 1), 5)) for kind, c in sorted(kinds.items())}
    both = [kinds[k] for k in ("label", "option")]
    labels["label+option"] = {"examples": sum(c["examples"] for c in both), "failed": sum(c["failed"] for c in both),
                              "failed_share": round(sum(c["failed"] for c in both) / max(sum(c["examples"] for c in both), 1), 5)}

    result = {"setup": json.loads(Path(args.setup).read_text()) if args.setup else {}, "units": len(info["units"]),
              "segments": len(ids), "per_class": per_class, "rows": kept_rows, "rows_translategemma": tg_rows,
              "labels_options": labels, "assemble": {k: v for k, v in assembled.items() if k != "hosts"},
              "assemble_hosts": assembled["hosts"]}
    out.with_suffix(".json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    out.write_text(report_md(result))
    print(f"CHECK {json.dumps({'segments': len(ids), 'af': per_class['all']['af'], 'executed': per_class['all']['executed'], 'rows_kept': kept_rows['kept_share'], 'labels_options_failed': labels['label+option']['failed_share']})}", flush=True)


def pct(x, digits=2):
    return "–" if x is None else f"{100 * x:.{digits}f}%".replace(".", ",")


def thousands(n):
    return f"{n:,}".replace(",", ".")


def report_md(r):
    s = r["setup"]
    lines = ["# FLAN-nl met Gemma 4 31B-it: controles over de hele set", ""]
    for key, value in s.get("lines", {}).items():
        lines.append(f"- {key}: {value}")
    lines += [f"- Segmenten: {thousands(r['segments'])} in {thousands(r['units'])} units", "",
              "Geen MetricX over de hele set (de pilot vergeleek MetricX-QE op de steekproef). De maten zijn die van de",
              "pilot (`g4_pilot.segment_measures`); ongewogen, per klasse en over alles.", "",
              "## Per klasse", "",
              "| klasse | segmenten | af | stoptoken | lus | chatter | uitgevoerd | idem, opdrachtachtig (n) | Engels blijven staan | onbruikbaar (assemble) | gelijk aan bron | tekens NL/EN p5 / p50 / p95 | uitvoertokens gem. |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for c, x in r["per_class"].items():
        ratio = " / ".join("–" if v is None else f"{v:.2f}".replace(".", ",") for v in x["chars_ratio_p5_p50_p95"])
        lines.append(f"| {c} | {thousands(x['segments'])} | {pct(x['af'])} | {pct(x['ended'])} | {pct(x['loop'])} | "
                     f"{pct(x['chatter'])} | {pct(x['executed'])} | {pct(x['executed_instruction_like'])} "
                     f"({thousands(x['instruction_like'])}) | {pct(x['english_left'])} | {pct(x['unusable'])} | "
                     f"{pct(x['identical_to_source'])} | {ratio} | {x['tokens_out_mean']} |")
    a = r["per_class"]["all"]
    lines += ["", f"- Redenen `executed` (alle segmenten): {a['executed_reasons']}",
              f"- assemble's controles, onbruikbaar: {a['unusable_reasons']}", f"- Reparaties: {a['repairs']}"]
    for c in r["per_class"]:
        if c != "all":
            lines.append(f"- {c}: onbruikbaar {r['per_class'][c]['unusable_reasons']}; executed {r['per_class'][c]['executed_reasons']}")
    k, t = r["rows"], r["rows_translategemma"]
    lines += ["", "## assemble: rijen behouden", "",
              "| | Gemma 4 31B-it | TranslateGemma 27B (gepubliceerd) |", "|---|---:|---:|",
              f"| voorbeelden te vertalen | {thousands(k['to_translate'])} | {thousands(t['to_translate']) if t else '–'} |",
              f"| behouden | {thousands(k['kept'])} ({pct(k['kept_share'])}) | {(thousands(t['kept']) + ' (' + pct(t['kept_share']) + ')') if t else '–'} |"]
    for reason in sorted(set(k["dropped"]) | set(t["dropped"] if t else {})):
        lines.append(f"| {reason} | {thousands(k['dropped'].get(reason, 0))} | {thousands(t['dropped'].get(reason, 0)) if t else '–'} |")
    lines += [f"| Nederlandse taken (niet vertaald, behouden) | {thousands(k['kept_dutch_task'])} | {thousands(t['kept_dutch_task']) if t else '–'} |",
              f"| FLAN-rijen in het plan | {thousands(k['flan_rows'])} | {thousands(t['flan_rows']) if t else '–'} |", "",
              "## Labels en opties", "",
              "Een voorbeeld met een label of optie als antwoord faalt als assemble het weglaat of als twee Engelse opties",
              "dezelfde Nederlandse tekst krijgen (deel 3 van de beslisregel van de pilot).", "",
              "| antwoordsoort | voorbeelden | behouden | opties samengevallen | mislukt |", "|---|---:|---:|---:|---:|"]
    for kind, x in r["labels_options"].items():
        if kind == "label+option":
            lines.append(f"| label + optie | {thousands(x['examples'])} | | | {thousands(x['failed'])} ({pct(x['failed_share'])}) |")
        else:
            failed = f"{thousands(x['failed'])} ({pct(x['failed_share'])})" if x["failed_share"] is not None else "–"
            lines.append(f"| {kind or '(leeg)'} | {thousands(x['examples'])} | {thousands(x['kept'])} ({pct(x['kept_share'])}) | "
                         f"{thousands(x.get('options_collapsed', 0))} | {failed} |")
    lines += ["", "## assemble: varianten en aanspreekvorm", "",
              f"- Aanspreekvorm: {r['assemble'].get('template_registers')}",
              f"- Varianten gekozen: {r['assemble'].get('variant_decisions')}",
              f"- Segmenten gerepareerd: {r['assemble'].get('segments_repaired')}; onbruikbaar: {r['assemble'].get('segments_unusable')}"]
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description="FLAN-nl with Gemma 4: units, status, checks of the whole set")
    sub = parser.add_subparsers(dest="step", required=True)
    p = sub.add_parser("units", help="the plan's segments in units -> units.json")
    p.add_argument("--plan", required=True, help="Directory with the plan's segments-w*.parquet")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="gemma4-31b-flan")
    p.add_argument("--tokenizer", default=None, help="Gemma 4's tokenizer: measure every prompt against its class")
    p = sub.add_parser("status", help="units stored and pending")
    p.add_argument("--units", required=True)
    p.add_argument("--out", help="gs:// directory of the shards (<out>/<class>/<unit>.parquet)")
    p.add_argument("--listing", help="Instead of --out: a file with the object names of the stored shards")
    p = sub.add_parser("check", help="the checks of the whole set, assemble and the report")
    p.add_argument("--plan", required=True)
    p.add_argument("--english", nargs="+", required=True)
    p.add_argument("--shards", required=True, help="Local directory of the shards (searched recursively)")
    p.add_argument("--units", required=True)
    p.add_argument("--templates-nl", required=True)
    p.add_argument("--nl", required=True, help="assemble's output directory (w<N>.parquet, report.json)")
    p.add_argument("--out", required=True, help="The report (.md; .json and measures.parquet beside it)")
    p.add_argument("--tg-report", default=None, help="assemble's report.json of the published TranslateGemma run")
    p.add_argument("--setup", default=None, help="A JSON file with {lines: {name: text}} for the report's head")
    p.add_argument("--processes", type=int, default=64)
    args = parser.parse_args(argv)
    if args.step == "status" and not (args.out or args.listing):
        parser.error("status needs --out or --listing")
    {"units": units, "status": status, "check": check}[args.step](args)


if __name__ == "__main__":
    main()
