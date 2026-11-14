"""translate_gemma4.py, models.yml and g4_pilot.py on made-up FLAN rows and made-up translators: the classes and the
prompt, the decode loop of the vendored engine, the pilot's draw, measures, assemble on its examples, blind material,
report and decision. No model, no network and no TPU."""

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_flan_nl import FakeDevice  # noqa: E402

import flan_nl as fn  # noqa: E402
import g4_pilot as gp  # noqa: E402
import gemma4_engine as ge  # noqa: E402
import translate as tr  # noqa: E402
import translate_gemma4 as tg4  # noqa: E402

MODEL = ge.MODELS[tg4.DEFAULT_MODEL]
PROMPT_TOKENS = 166  # flan-v1 beside the segment in Gemma 4's turn, measured with its tokenizer on 14 Nov


def test_the_classes_are_flan_nls_and_keep_the_decode_limit_within_the_window():
    classes = MODEL["length_classes"]
    assert list(classes) == tr.CLASS_ORDER == list(gp.CLASSES)
    for name, shapes in classes.items():
        assert shapes["max_source"] == tr.LENGTH_CLASSES[name]["max_source"] == gp.CLASSES[name]
        assert shapes["max_decode"] <= 1024
        assert shapes["max_source"] + PROMPT_TOKENS <= shapes["prefill_lengths"][-1]
        assert shapes["prefill_lengths"] == sorted(shapes["prefill_lengths"])
        slot_tokens = MODEL["slots"][name] * (shapes["prefill_lengths"][-1] + shapes["max_decode"])
        assert slot_tokens <= 47_000, (name, slot_tokens)


def test_the_prompt_is_natural_v1_fmt_with_the_sentences_against_carrying_out_the_task():
    plain = ge.PROMPTS[MODEL["prompt"]]["plain"]
    assert plain.startswith("Translate the following {source} text into natural, idiomatic {target}")
    assert "do not carry it out, answer it, complete it or continue it" in plain
    assert plain.endswith("Output only the translation, without explanation or commentary.\n\n\n{text}")
    user = ge.user_text(ge.PROMPTS["flan-v1"], "  What is 2+2?  ", "en", "nl")
    assert user.startswith("Translate the following English text into natural, idiomatic Dutch")
    assert user.endswith("\n\n\nWhat is 2+2?")


def test_segments_are_chosen_by_the_plans_token_count():
    assert tg4.class_bounds(tg4.DEFAULT_MODEL, "tiny") == (0, 32)
    assert tg4.class_bounds(tg4.DEFAULT_MODEL, "medium") == (128, 256)
    ids, texts = tg4.select([1, 2, 3, 4], ["a", "b", "c", "d"], [32, 33, 128, 129], 32, 128)
    assert (ids, texts) == ([2, 3], ["b", "c"])


class FakeCodec:
    def __init__(self, lengths):
        self.lengths = lengths

    def encode(self, text):
        return [0] * self.lengths[text]


def test_a_prompt_longer_than_the_largest_prefill_stops_before_the_engine():
    shapes = MODEL["length_classes"]["tiny"]
    assert tg4.check_prompts(FakeCodec({"a": 10, "b": 256}), ["a", "b"], shapes) == 256
    with pytest.raises(SystemExit, match="longer than the class's largest prefill length"):
        tg4.check_prompts(FakeCodec({"a": 10, "b": 257}), ["a", "b"], shapes)


class FakeBlob:
    def __init__(self, store, name):
        self.store, self.name = store, name

    def upload_from_filename(self, path):
        self.store[self.name] = Path(path).read_bytes()


class FakeClient:
    """google.cloud.storage.Client's list_blobs, bucket and blob, on a dict of (bucket, name) -> bytes"""

    def __init__(self):
        self.objects = {}

    def list_blobs(self, bucket, prefix):
        return [FakeBlob(None, n) for (b, n) in self.objects if b == bucket and n.startswith(prefix)]

    def bucket(self, bucket):
        client = self

        class Bucket:
            def blob(self, name):
                return FakeBlob(_Store(client.objects, bucket), name)
        return Bucket()


class _Store(dict):
    def __init__(self, objects, bucket):
        super().__init__()
        self.objects, self.bucket_name = objects, bucket

    def __setitem__(self, name, value):
        self.objects[(self.bucket_name, name)] = value


def test_shards_are_stored_and_found_through_the_python_client(tmp_path):
    client = FakeClient()
    local = tmp_path / "shard-00003.parquet"
    local.write_bytes(b"x")
    tg4.upload(client, local, "gs://bkt/out/w0/small/shard-00003.parquet")
    tg4.upload(client, local, "gs://bkt/out/w0/smaller/shard-00009.parquet")
    assert client.objects[("bkt", "out/w0/small/shard-00003.parquet")] == b"x"
    assert tg4.stored_shards(client, "gs://bkt/out/w0/small") == {"shard-00003.parquet"}
    assert tg4.stored_shards(client, "gs://bkt/out/w0/medium/") == set()


@pytest.mark.parametrize("slots, sync_every", [(1, 1), (5, 8)])
def test_the_vendored_decode_loop_stops_at_any_of_gemma_4s_stop_tokens(slots, sync_every):
    prompts = [[a, a % 7] for a in range(1, 20)]
    outputs = ge.decode_all(prompts, slots, FakeDevice(slots), max_decode=40, sync_every=sync_every, eos_ids=[1, 106, 50])
    for prompt, (tokens, ended) in zip(prompts, outputs):
        assert (tokens, ended) == (FakeDevice.completion(prompt)[:-1], True)


# ---------------------------------------------------------------------------------------------------- the detector

@pytest.mark.parametrize("source, output, reasons", [
    ("What is the capital of France?", "Wat is de hoofdstad van Frankrijk?", []),
    ("What is the capital of France?", "Parijs.", ["question_lost"]),
    ("Write a story about a dog.", "Schrijf een verhaal over een hond.", []),
    ("Write a story about a dog.", "Er was eens een hond die Max heette. Hij woonde in een klein huis aan de rand van het "
     "bos en speelde elke dag met de kinderen.", ["too_long"]),
    ("Premise: the man sleeps. Hypothesis: the man is awake. Does the premise entail it?", "Nee", ["question_lost", "too_short"]),
    ("Is the sky blue?", "Is de lucht blauw?\n\nAntwoord: ja", ["question_lost", "answer_marker"]),
    ("Name a colour.", "Noem een kleur.\nRood\nBlauw", ["added_lines"]),
    ("Is the sky blue?", "Zeker! De lucht is blauw.", ["question_lost", "assistant_opening"]),
    ("Sure, I will come.", "Zeker, ik kom.", []),
    ("Answer: yes", "Antwoord: ja", []),
])
def test_executed_reasons(source, output, reasons):
    assert gp.executed_reasons(source, output) == reasons


def test_chatter_is_an_introduction_a_note_or_assembles_commentary():
    assert gp.chatter("Good morning", "Hier is de vertaling: Goedemorgen")
    assert gp.chatter("Good morning", "Goedemorgen\nNoot: informeel")
    assert not gp.chatter("Note: be careful\nGo", "Let op: wees voorzichtig\nGa")
    assert not gp.chatter("Good morning", "Goedemorgen")


def test_options_collapse_when_two_english_options_get_one_dutch_text():
    inputs = "Is it big?\nOPTIONS:\n- large\n- big\n- small"
    pieces = [("S", 1), ("L", "\nOPTIONS:\n- "), ("S", 2), ("L", "\n- "), ("S", 3), ("L", "\n- "), ("S", 4)]
    text = {1: "Is it big?", 2: "large", 3: "big", 4: "small"}
    assert gp.options_collapsed(inputs, pieces, text, {1: "Is het groot?", 2: "groot", 3: "Groot.", 4: "klein"})
    assert not gp.options_collapsed(inputs, pieces, text, {1: "Is het groot?", 2: "omvangrijk", 3: "groot", 4: "klein"})


def test_weighted_estimates():
    assert gp.weighted_rate([True, False, False], [2, 1, 1]) == 0.5
    assert gp.weighted_mean([1.0, None, 3.0], [1, 5, 3]) == 2.5
    assert gp.weighted_quantile([3, 1, 2], [1, 1, 2], 0.5) == 2


# ------------------------------------------------------------------------------------------- a made-up plan, run

TASKS = [f"flan/t0_zsopt_data/task{k}" for k in range(12)]


def made_up_rows():
    """(host, position, source, inputs, targets): questions with options and a label or option answer, and free
    answers, short enough (fewer than 5 words a segment) that assemble's language check does not apply"""
    rows = []
    for k in range(60):
        task = TASKS[k % len(TASKS)]
        h, p = k % 4, k
        if k % 3 == 0:
            rows.append((h, p, task, f"Q: Is item {k} red?\nOPTIONS:\n- yes it is\n- no it is not\nA:", "yes it is"))
        elif k % 3 == 1:
            rows.append((h, p, task, f"Q: Which colour is item {k}?\n(A) red colour (B) blue colour\nA:", "(A)"))
        else:
            rows.append((h, p, task, f"Write a word for {k} things.\nOutput:", f"many {k} things"))
    return rows


def word_count(text):
    return len(text.split())


def write_plan(tmp_path):
    """A plan as flan_nl.py plan writes it (examples.parquet, segments-w*.parquet, plan.json), with word counts as
    token counts, and the English extracts"""
    keys = fn.load_templates([str(ROOT / "templates_nl.json")])
    examples = {k: [] for k in ("host", "position", "source", "status", "instruction", "answer", "options", "answer_kind", "splits")}
    counted = {t: word_count(t) for t in keys}
    planned = []
    for h, p, source, inputs, targets in made_up_rows():
        instruction, answer, info = fn.plan_example(inputs, targets, word_count, templates=keys)
        for kind, v in instruction + answer:
            if kind == "S":
                counted[v] = word_count(v) * 9  # spread the segments over the classes
        planned.append((h, p, source, instruction, answer, info))
    texts = sorted(counted, key=lambda t: (min(counted[t], 512), t))
    ids = {t: i for i, t in enumerate(texts)}
    for h, p, source, instruction, answer, info in planned:
        for key, value in (("host", h), ("position", p), ("source", source), ("status", "translate"),
                           ("instruction", json.dumps([[k, ids[v] if k == "S" else v] for k, v in instruction])),
                           ("answer", json.dumps([[k, ids[v] if k == "S" else v] for k, v in answer])),
                           ("options", info["options"]), ("answer_kind", info["answer"]), ("splits", info["splits"])):
            examples[key].append(value)
    plan = tmp_path / "plan2"
    plan.mkdir()
    pq.write_table(pa.table(examples), plan / "examples.parquet")
    for w in range(4):
        pq.write_table(pa.table({"id": list(range(w, len(texts), 4)), "text": texts[w::4],
                                 "tokens": [min(counted[t], 512) for t in texts[w::4]]}), plan / f"segments-w{w}.parquet")
    (plan / "plan.json").write_text(json.dumps({"template_segments": {t: ids[t] for t in keys}}))
    en = tmp_path / "en"
    en.mkdir()
    for w in range(4):
        rows = [r for r in made_up_rows() if r[0] == w]
        pq.write_table(pa.table({"host": [r[0] for r in rows], "position": [r[1] for r in rows], "source": [r[2] for r in rows],
                                 "inputs": [r[3] for r in rows], "targets": [r[4] for r in rows]}), en / f"w{w}.parquet")
    return plan, sorted(str(p) for p in en.glob("*.parquet")), texts


DUTCH = {"is": "is", "item": "voorwerp", "red": "rood", "yes": "ja", "it": "het", "no": "nee", "not": "niet",
         "which": "welke", "colour": "kleur", "blue": "blauw", "write": "schrijf", "a": "een", "word": "woord",
         "for": "voor", "things": "dingen", "many": "veel"}


def dutch(text):
    return " ".join(DUTCH.get(w.lower().strip("?.:"), w) + ("?" if w.endswith("?") else "") for w in text.split())


def write_shards(directory, texts, translate, ids):
    directory.mkdir(parents=True)
    pq.write_table(pa.table({"id": ids, "text_nl": [translate(texts[i]) for i in ids], "ended": [True] * len(ids),
                             "tokens_in": [5] * len(ids), "tokens_out": [5] * len(ids)}), directory / "shard-00000.parquet")


def test_the_pilot_from_draw_to_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(gp, "PER_CELL", 3)
    monkeypatch.setattr(gp, "EXAMPLES_PER_KIND", 4)
    monkeypatch.setattr(gp, "THROUGHPUT", {c: 1 for c in gp.CLASSES})
    monkeypatch.setattr(gp, "BLIND_PER_CLASS", 2)
    monkeypatch.setattr(gp, "BLIND_PORTION", 3)
    plan, english, texts = write_plan(tmp_path)
    pilot = tmp_path / "pilot"
    gp.main(["draw", "--plan", str(plan), "--english", *english, "--out", str(pilot)])
    sample = pq.read_table(pilot / "sample.parquet").to_pydict()
    info = json.loads((pilot / "sample.json").read_text())
    assert info["S"] == sum(sample["in_s"]) and info["E_examples"] == 12 and set(info["E_kinds"]) == set(gp.KINDS)
    # the same draw twice
    gp.main(["draw", "--plan", str(plan), "--english", *english, "--out", str(tmp_path / "again")])
    assert pq.read_table(tmp_path / "again" / "sample.parquet").to_pydict() == sample
    # every segment of the pilot is translated by exactly one worker
    per_worker = [pq.read_table(pilot / f"segments-w{w}.parquet").column("id").to_pylist() for w in range(4)]
    assert sorted(i for ids in per_worker for i in ids) == sample["id"]
    # S's weights add up to the segments of its cells
    assert sum(sample["weight"]) == pytest.approx(sum(c["plan"] for c in info["cells"].values() if c["drawn"]))

    # TranslateGemma translates every segment; Gemma 4 answers the questions instead
    all_ids = list(range(len(texts)))
    write_shards(tmp_path / "tg-shards" / "w0", texts, dutch, all_ids)
    write_shards(tmp_path / "g4-shards" / "w1" / "tiny", texts, lambda t: "ja" if t.endswith("?") else dutch(t), sample["id"])
    gp.main(["gather", "--pilot", str(pilot), "--shards", str(tmp_path / "tg-shards"), "--out", str(pilot / "tg")])
    gp.main(["gather", "--pilot", str(pilot), "--shards", str(tmp_path / "g4-shards"), "--out", str(pilot / "g4")])
    with pytest.raises(SystemExit, match="have no translation"):
        gp.main(["gather", "--pilot", str(pilot), "--shards", str(tmp_path / "g4-shards" / "nothing"), "--out", str(tmp_path / "x")])
    gp.main(["measure", "--pilot", str(pilot), "--templates-nl", str(ROOT / "templates_nl.json"), "--processes", "2"])
    measures = pq.read_table(pilot / "measures.parquet").to_pydict()
    flagged = {(m, i) for m, i, e in zip(measures["model"], measures["id"], measures["executed"]) if e}
    questions = {i for i, t in zip(sample["id"], sample["text"]) if t.endswith("?")}
    assert questions and {("g4", i) for i in questions} <= flagged and not any(m == "tg" for m, _ in flagged)
    examples = pq.read_table(pilot / "examples.parquet").to_pydict()
    assert all(examples["kept_tg"])
    pairs = [pq.read_table(pilot / "qe" / f"pairs-w{w}.parquet").to_pydict() for w in range(4)]
    assert sum(len(p["id"]) for p in pairs) == sum(1 for s, e in zip(sample["in_s"], sample["in_e"]) if s or e)

    gp.main(["blind", "--pilot", str(pilot), "--out", str(tmp_path / "blind"), "--key", str(tmp_path / "key.json")])
    key = json.loads((tmp_path / "key.json").read_text())
    portions = sorted((tmp_path / "blind").glob("portion-*.md"))
    assert portions and all("Blinde beoordeling" in p.read_text() for p in portions)
    blind_text = "".join(p.read_text() for p in portions)
    assert '"g4"' not in blind_text and "Gemma" not in blind_text and "TranslateGemma" not in blind_text
    for item, k in key["items"].items():
        assert {k["A"], k["B"]} == {"g4", "tg"} and item in blind_text
    gp.main(["checkset", "--pilot", str(pilot), "--out", str(tmp_path / "check.md")])
    assert "g4:" in (tmp_path / "check.md").read_text()

    # the report, with made-up QE and a made-up log
    for w in range(4):
        p = pairs[w]
        pq.write_table(pa.table({"id": p["id"], "qe_en_g4": [3.0] * len(p["id"]), "qe_en_tg": [2.0] * len(p["id"])}),
                       tmp_path / f"qe-w{w}.parquet")
    log = tmp_path / "translate-w0.log"
    log.write_text("\n".join([f"MAXTEXT {json.dumps({'class': c})}\nengine ready in 150 s\n"
                              f"SHARD {json.dumps({'class': c, 'segments': 100, 'seconds': 10.0, 'tokens_in': 1000, 'tokens_out': 800, 'not_ended': 0})}"
                              for c in gp.CLASSES]))
    setup = tmp_path / "setup.json"
    setup.write_text(json.dumps({"code": "test", "rule_fixed": "now", "lines": {"VM": "none"}}))
    check = {f"g4:{i}": True for i in questions}
    (tmp_path / "check.json").write_text(json.dumps(check))
    out = tmp_path / "report" / "flan-g4-pilot.md"
    out.parent.mkdir()
    gp.main(["report", "--pilot", str(pilot), "--qe", *[str(tmp_path / f"qe-w{w}.parquet") for w in range(4)],
             "--logs", str(tmp_path / "translate-*.log"), "--check", str(tmp_path / "check.json"), "--setup", str(setup),
             "--out", str(out)])
    result = json.loads(out.with_suffix(".json").read_text())
    assert result["rule_automatic"]["2_executed_S"]["pass"] is False
    assert result["cost"]["hours_one_v4_32"] > 0 and "## Beslisregel: uitkomst" in out.read_text()

    # the blind judgements: TranslateGemma preferred and fewer errors -> TranslateGemma stays
    judgments = tmp_path / "review.jsonl"
    judgments.write_text("\n".join(json.dumps({"item": item, "errors_A": int(k["A"] == "g4"), "errors_B": int(k["B"] == "g4"),
                                               "executed_A": k["A"] == "g4", "executed_B": k["B"] == "g4", "natural": "gelijk",
                                               "preference": "A" if k["A"] == "tg" else "B"})
                                   for item, k in key["items"].items()))
    gp.main(["decide", "--report", str(out.with_suffix(".json")), "--key", str(tmp_path / "key.json"),
             "--judgments", str(judgments), "--out", str(tmp_path / "decision.json")])
    decision = json.loads((tmp_path / "decision.json").read_text())
    assert decision["rule"]["1_blind"]["pass"] is False and decision["verdict"] == "TranslateGemma blijft"
    assert decision["blind"]["g4_preferred_share"] == 0.0


def test_throughput_and_cost_from_the_logs(tmp_path):
    def shard(c, segments, seconds):
        return f"SHARD {json.dumps({'class': c, 'segments': segments, 'seconds': seconds, 'tokens_in': 1, 'tokens_out': 2, 'not_ended': 0})}"

    for w in range(2):
        (tmp_path / f"w{w}.log").write_text("\n".join([
            f"MAXTEXT {json.dumps({'class': 'tiny'})}", "engine ready in 100 s",
            shard("tiny", 50, 105.0), shard("tiny", 50, 5.0), shard("tiny", 50, 5.0),  # 100 s of compilation first
            f"MAXTEXT {json.dumps({'class': 'xlong'})}", "engine ready in 120 s", shard("xlong", 20, 120.0)]))
    speed = gp.throughput(sorted(tmp_path.glob("*.log")))
    assert speed["tiny"]["segments_per_s"] == 10.0 and speed["tiny"]["compile_s"] == 100 and speed["tiny"]["workers"] == 2
    assert speed["tiny"]["engine_s"] == 100 and speed["tiny"]["rate_from"] == "shards after the first"
    # xlong: one shard a worker, its 120 s less tiny's 100 s of compilation
    assert speed["xlong"]["segments_per_s"] == 1.0 and speed["xlong"]["rate_from"] == "single shards"
    costs = gp.cost({"tiny": 144_000, "xlong": 4_000}, speed)
    one = (144_000 / 40 + 200) + (4_000 / 4 + 220)
    two = (144_000 / 80 + 200) + (4_000 / 8 + 220)
    assert costs["hours_one_v4_32"] == pytest.approx(one / 3600, abs=0.05)
    assert costs["hours_two_v4_32"] == pytest.approx(two / 3600, abs=0.05)
    assert gp.cost({"tiny": 1, "small": 1}, speed) is None
    assert np.isclose(gp.weighted_rate([True], [1.0]), 1.0)
