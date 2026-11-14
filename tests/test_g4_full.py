"""g4_full.py and translate_gemma4.py --units on made-up data: the units, the claims, the work of several workers on
one set of units, the status and the checks of the whole set. No model, no network and no TPU."""

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_gemma4 import dutch, write_plan  # noqa: E402

import g4_full as g4f  # noqa: E402
import translate_gemma4 as tg4  # noqa: E402

BOUNDS = {c: tg4.class_bounds(tg4.DEFAULT_MODEL, c) for c in g4f.ORDER}


def made_up_files():
    tokens = [5, 40, 600 // 2, 31, 33, 129, 257, 385, 512, 1, 32, 128, 256, 384, 6, 7, 8]
    return {"segments-w0.parquet": (list(range(0, 2 * len(tokens), 2)), tokens),
            "segments-w1.parquet": (list(range(1, 2 * len(tokens), 2)), tokens[::-1])}


def test_units_hold_every_segment_once_longest_class_first():
    files = made_up_files()
    sizes = {"xlong": 1, "long": 2, "medium": 2, "small": 2, "tiny": 3}
    units = g4f.plan_units(files, BOUNDS, sizes)
    assert units == g4f.plan_units(files, BOUNDS, sizes)
    assert [u["class"] for u in units] == sorted((u["class"] for u in units), key=g4f.ORDER.index)
    assert len({u["name"] for u in units}) == len(units)
    covered = []
    for u in units:
        ids, tokens = files[u["segments"]]
        low, high = BOUNDS[u["class"]]
        got, _ = g4f.unit_rows(u, sizes[u["class"]], ids, [f"t{i}" for i in ids], tokens, low, high)
        assert len(got) == u["n"] <= sizes[u["class"]] and all(low < tokens[ids.index(i)] <= high for i in got)
        covered += got
    assert sorted(covered) == sorted(i for ids, _ in files.values() for i in ids)
    assert units[0]["name"] == "xlong-w0-0000"
    ids, tokens = files["segments-w0.parquet"]
    with pytest.raises(SystemExit, match="does not hold the unit"):
        g4f.unit_rows(dict(units[0], first=-1), 1, ids, [""] * len(ids), tokens, *BOUNDS["xlong"])


def test_a_claim_is_one_owners_until_its_lease_ends_or_its_vm_starts_a_new_run():
    store = g4f.MemoryClaims()
    a = {"owner": "mbert-corpus/w0", "run": "r1"}
    assert g4f.claim(store, "u", a, lease=60, t=0)
    assert g4f.claim(store, "u", a, lease=60, t=10)  # its own run again (a failed pass tried once more)
    assert not g4f.claim(store, "u", {"owner": "mbert-mix/w0", "run": "m1"}, lease=60, t=20)
    assert not g4f.claim(store, "u", {"owner": "mbert-corpus/w1", "run": "r1"}, lease=60, t=20)
    assert g4f.claim(store, "u", {"owner": "mbert-corpus/w1", "run": "r2"}, lease=60, t=20)  # the chain started again
    assert g4f.claim(store, "u", {"owner": "mbert-mix/w3", "run": "m1"}, lease=60, t=200)  # the lease ended
    assert json.loads(store.read("u")[0])["owner"] == "mbert-mix/w3"


def test_workers_of_two_vms_translate_every_unit_once():
    units = [{"name": f"tiny-w0-{k:04d}"} for k in range(7)]
    claims, stored, translated = g4f.MemoryClaims(), set(), []

    def worker(owner, run, max_units=None, t=0):
        me = {"owner": owner, "run": run}

        def translate(u):
            translated.append((owner, u["name"]))
            return {"id": [u["name"]]}, 1.0

        return g4f.work_units(units, lambda: stored, lambda n: g4f.claim(claims, n, me, 60, t), lambda n: n in stored,
                              translate, lambda u, rows, s: stored.add(u["name"]), max_units)

    assert worker("mbert-corpus/w0", "r1", max_units=1) == ["tiny-w0-0000"]
    # a unit claimed and not stored (a worker that is still at it): the other VM leaves it and takes the rest
    assert g4f.claim(claims, "tiny-w0-0001", {"owner": "mbert-corpus/w1", "run": "r1"}, 60, 0)
    assert worker("mbert-mix/w0", "m1", max_units=2) == ["tiny-w0-0002", "tiny-w0-0003"]
    assert worker("mbert-corpus/w2", "r1") == ["tiny-w0-0004", "tiny-w0-0005", "tiny-w0-0006"]
    assert worker("mbert-mix/w1", "m1") == []  # nothing free: no translation, so no engine
    assert worker("mbert-mix/w1", "m1", t=100) == ["tiny-w0-0001"]  # the lease of the claim ended
    assert sorted(n for _, n in translated) == [u["name"] for u in units]


def test_status_counts_units_and_segments_per_class(tmp_path, capsys):
    files = made_up_files()
    units = g4f.plan_units(files, BOUNDS)
    info = {"order": list(g4f.ORDER), "segments": 34, "units": units}
    s = g4f.status_of(info, set())
    assert s["pending"] == len(units) and s["segments"] == 34
    assert s["hours_left_4_workers"] >= s["hours_left_8_workers"] >= 0
    listing = tmp_path / "listing.txt"
    listing.write_text("\n".join(f"gs://b/t/{u['class']}/{u['name']}.parquet" for u in units) + "\n")
    (tmp_path / "units.json").write_text(json.dumps(info))
    g4f.main(["status", "--units", str(tmp_path / "units.json"), "--listing", str(listing)])
    assert capsys.readouterr().out.strip().splitlines()[-1] == "COMPLETE"
    listing.write_text("gs://b/t/tiny/tiny-w9-0000.parquet\n")
    with pytest.raises(SystemExit, match="no units"):
        g4f.main(["status", "--units", str(tmp_path / "units.json"), "--listing", str(listing)])


def test_units_mode_needs_its_claims_owner_and_run():
    with pytest.raises(SystemExit, match="--units needs"):
        tg4.main(["--segments", "d", "--out", "gs://b/o", "--ckpt", "c", "--tokenizer", "t", "--length-class", "tiny",
                  "--units", "units.json"])


def test_the_checks_of_the_whole_set_and_assemble(tmp_path):
    plan, english, texts = write_plan(tmp_path)
    g4f.main(["units", "--plan", str(plan), "--out", str(tmp_path / "units.json")])
    info = json.loads((tmp_path / "units.json").read_text())
    assert info["segments"] == len(texts) and sum(u["n"] for u in info["units"]) == len(texts)
    # every unit stored as translate_gemma4.py --units stores it; questions answered instead of translated
    shards = tmp_path / "shards"
    for u in info["units"]:
        table = pq.read_table(plan / u["segments"]).to_pydict()
        low, high = BOUNDS[u["class"]]
        ids, unit_texts = g4f.unit_rows(u, info["classes"][u["class"]]["size"], table["id"], table["text"],
                                        table["tokens"], low, high)
        (shards / u["class"]).mkdir(parents=True, exist_ok=True)
        out = ["ja" if t.endswith("?") else dutch(t) for t in unit_texts]
        pq.write_table(pa.table({"id": ids, "text_nl": out, "ended": [True] * len(ids), "tokens_in": [9] * len(ids),
                                 "tokens_out": [len(t.split()) for t in out]}), shards / u["class"] / f"{u['name']}.parquet")
    tg_report = tmp_path / "tg-report.json"
    tg_report.write_text(json.dumps({"hosts": {"0": {"flan_rows": 60, "translated": 59, "dropped:segment_truncated": 1}}}))
    out = tmp_path / "report" / "flan-g4-full.md"
    g4f.main(["check", "--plan", str(plan), "--english", *english, "--shards", str(shards), "--units",
              str(tmp_path / "units.json"), "--templates-nl", str(ROOT / "templates_nl.json"), "--nl", str(tmp_path / "nl"),
              "--out", str(out), "--tg-report", str(tg_report), "--processes", "2"])
    result = json.loads(out.with_suffix(".json").read_text())
    assert result["segments"] == len(texts) and result["per_class"]["all"]["segments"] == len(texts)
    questions = sum(t.endswith("?") for t in texts)
    assert questions and result["per_class"]["all"]["executed"] > 0
    assert result["rows"]["to_translate"] == 60 and result["rows_translategemma"]["kept"] == 59
    assert set(result["labels_options"]) >= {"label", "option", "text", "label+option"}
    assert sorted(p.name for p in (tmp_path / "nl").glob("w*.parquet")) == [f"w{h}.parquet" for h in range(4)]
    measures = pq.read_table(out.with_name("measures.parquet")).to_pydict()
    assert sorted(measures["id"]) == list(range(len(texts)))
    text = out.read_text()
    assert "## Per klasse" in text and "## Labels en opties" in text and "TranslateGemma 27B (gepubliceerd)" in text
    # a segment without a translation stops the check
    first = info["units"][0]
    (shards / first["class"] / f"{first['name']}.parquet").unlink()
    with pytest.raises(SystemExit, match="have no translation"):
        g4f.main(["check", "--plan", str(plan), "--english", *english, "--shards", str(shards), "--units",
                  str(tmp_path / "units.json"), "--templates-nl", str(ROOT / "templates_nl.json"),
                  "--nl", str(tmp_path / "nl2"), "--out", str(tmp_path / "r2" / "x.md"), "--processes", "2"])
