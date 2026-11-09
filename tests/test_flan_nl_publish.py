"""publish.py on made-up shards, extracts and assembly outputs: the link of extracted rows to their rows in
Open-Orca/FLAN, the folder for Hugging Face and its card, and the upload call. No network: the shards come from a fake
source and huggingface_hub is replaced by a fake module."""

import json
import sys
import types
from collections import deque
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import publish as pub  # noqa: E402

FOLDERS = pub.FLAN_FOLDERS
QUIET = {"log": lambda *a, **k: None}


def shard(rows, template_type="zs_opt"):
    """A shard of Open-Orca/FLAN from (task, inputs, targets) rows; _template_idx is the row's index times 10."""
    return pa.table({"inputs": [x for _, x, _ in rows], "targets": [y for _, _, y in rows],
                     "_task_name": [t for t, _, _ in rows], "_template_idx": [10 * k for k in range(len(rows))],
                     "_task_source": ["Src"] * len(rows), "_template_type": [template_type] * len(rows)})


class FakeSource:
    """{folder: [table, ...]} as the shards of the revision; remembers which it handed out."""

    def __init__(self, folders):
        self.folders = folders
        self.opened = []

    def shards(self, folder):
        return [f"{folder}/part.{k}.parquet" for k in range(len(self.folders.get(folder, [])))]

    def tables(self, names):
        for name in names:
            folder, part = name.split("/")
            self.opened.append(name)
            yield name, self.folders[folder][int(part.split(".")[1])]


def template_type(folder):
    return "zs_opt" if "_zsopt_" in folder else "zs_noopt"


def manifest(host, build, folders, world=4):
    """F's manifest of one host: {folder: (shards, shards_read)}; folders not named have one shard F did not need."""
    report = {}
    for folder in FOLDERS:
        n, read = folders.get(folder, (1, 1))
        report[folder] = {"shards": n, "shards_read": read, "dropped": {"too_long": 2}, "excluded_tasks": {"sick_x": "dumb_leak"}}
    return {"build": build, "config": {"rank": host, "world": world, "flan": f"{pub.FLAN_REPO}@{pub.FLAN_REVISION}",
                                       "folders": list(FOLDERS), "cap": 2000, "max_length": 1024, "flan_share": 0.1},
            "flan": {"per_host_cap": 500, "pool_examples": 10, "examples": 5, "folders": report}}


def extract(path, host, rows, build="b0"):
    """An English extract (flan_nl.py extract) from (position, folder, task, inputs, targets) rows."""
    table = pa.table({"inputs": [r[3] for r in rows], "targets": [r[4] for r in rows],
                      "source": [f"flan/{r[1]}/{r[2]}" for r in rows], "position": pa.array([r[0] for r in rows], pa.int64()),
                      "host": pa.array([host] * len(rows), pa.int32())})
    pq.write_table(table.replace_schema_metadata({b"flan_decay_build": build.encode()}), path)
    return path


def empty_folders(**given):
    """Every folder with one shard of an unrelated row, except those given."""
    return {f: given.get(f, [shard([("other", "x", "y")], template_type(f))]) for f in FOLDERS}


def test_identical_rows_take_identical_source_rows_one_by_one():
    needed = {("t", "same", "a"): deque([7, 3]), ("t", "other", "b"): deque([5])}
    table = shard([("t", "same", "a"), ("u", "same", "a"), ("t", "other", "b"), ("t", "same", "a"), ("t", "same", "a")])
    got = pub.match_shard(needed, table)
    # The first id in line takes the first equal row; a third equal row is given to nobody; another task never matches
    assert [(i, row) for i, row, *_ in got] == [(7, 0), (5, 2), (3, 3)]
    assert got[0][1:] == (0, 0, "Src", "t", "zs_opt")
    assert not any(needed.values())


def test_a_host_of_a_folder_with_fewer_shards_than_hosts_matches_only_its_own_rows():
    needed = {("t", "x", "y"): deque([1])}
    table = shard([("t", "x", "y")] * 6)
    assert [row for _, row, *_ in pub.match_shard(needed, table, residue=2, world=4)] == [2]


def test_link_host_gives_every_row_its_shard_and_row(tmp_path):
    t0 = "flan_zsopt_data"
    rows = [(10, t0, "t", "q1", "a"), (11, t0, "t", "q1", "a"), (12, t0, "t", "q2", "b"), (13, "cot_zsopt_data", "c", "q3", "c")]
    path = extract(tmp_path / "w1.parquet", 1, rows)
    flan = [shard([("t", "q9", "z")]), shard([("t", "q1", "a"), ("t", "q2", "b")]), shard([("t", "q0", "z")]),
            shard([("t", "q1", "a")])]
    cot = shard([("c", "q3", "c")] * 3)
    # World 2: host 1 reads shards 1 and 3 of flan_zsopt_data (F's manifest: both), and rows 1, 3, ... of cot's one shard
    m = manifest(1, "b0", {t0: (4, 2), "cot_zsopt_data": (1, 1)}, world=2)
    table, report = pub.link_host(1, path, m, FakeSource(empty_folders(flan_zsopt_data=flan, cot_zsopt_data=[cot])), **QUIET)
    assert table.column_names == pub.LINK_COLUMNS
    got = {p: (f, r) for p, f, r in zip(*(table.column(c).to_pylist() for c in ("position", "flan_file", "flan_row")))}
    assert got == {10: (f"{t0}/part.1.parquet", 0), 11: (f"{t0}/part.3.parquet", 0), 12: (f"{t0}/part.1.parquet", 1),
                   13: ("cot_zsopt_data/part.0.parquet", 1)}
    assert table.column("template_idx").to_pylist() == [0, 0, 10, 10]
    assert table.column("task_name").to_pylist() == ["t", "t", "t", "c"]
    assert report["rows"] == report["linked"] == 4 and report["rows_sharing_content"] == 2
    assert report["folders"][t0]["shards_opened"] == 2


def test_a_row_without_a_source_row_in_the_shards_f_read_fails(tmp_path):
    t0 = "flan_zsopt_data"
    path = extract(tmp_path / "w0.parquet", 0, [(1, t0, "t", "q1", "a"), (2, t0, "t", "q1", "a")])
    flan = [shard([("t", "q1", "a")]), shard([("t", "q1", "a")])]
    source = FakeSource(empty_folders(flan_zsopt_data=flan))
    # Host 0 of 1 has units 0 and 1, but F read only the first: the second identical row is not there
    with pytest.raises(SystemExit, match=r"1 rows of F have no source row in the 1 shards F read.*\(2, 't', 'q1', 'a'\)"):
        pub.link_host(0, path, manifest(0, "b0", {t0: (2, 1)}, world=1), source, **QUIET)
    assert f"{t0}/part.1.parquet" not in source.opened


def test_link_refuses_an_extract_of_another_build(tmp_path):
    path = extract(tmp_path / "w0.parquet", 0, [(1, "flan_zsopt_data", "t", "q", "a")], build="other")
    with pytest.raises(SystemExit, match="build other"):
        pub.link_host(0, path, manifest(0, "b0", {}), FakeSource(empty_folders()), **QUIET)


def test_link_host_links_an_extract_from_the_hub_to_its_source_rows(tmp_path):
    def words(texts, **kwargs):
        return {"input_ids": [text.split() for text in texts]}

    niv2 = {"task001_en": {"input": ["English"], "output": ["English"], "instruction": ["English"]}}
    # Three shards a folder, cot one (each host reads its own rows of it); in every shard rows k and k + 7 are the same
    folders = {}
    for f in FOLDERS:
        task = "task001_en" if f.startswith("niv2_") else f"{f}_task"
        folders[f] = [shard([(task, f"q {f} {s} {k % 7}", f"a {k % 7}") for k in range(12)], template_type(f))
                      for s in range(1 if f == "cot_zsopt_data" else 3)]
    world = 2
    for host in range(world):
        table, manifest = pub.flan_nl.hub_extract(host, FakeSource(folders), words, niv2, world, cap=40, max_length=64,
                                                  seed=7)
        path = tmp_path / f"w{host}.parquet"
        pq.write_table(table, path)
        # As link reads it from the file extract writes beside the extract
        manifest = json.loads(json.dumps(manifest))
        links, report = pub.link_host(host, path, manifest, FakeSource(folders), **QUIET)
        assert report["rows"] == report["linked"] == report["f_pool"] == table.num_rows > 0
        assert report["f_flan_share"] is None and report["rows_sharing_content"] > 0
        extracted = {p: (s, x, y) for p, s, x, y in zip(*(table.column(c).to_pylist()
                                                          for c in ("position", "source", "inputs", "targets")))}
        sources = set()
        for p, name, row in zip(*(links.column(c).to_pylist() for c in ("position", "flan_file", "flan_row"))):
            folder, part = name.split("/")
            source = folders[folder][int(part.split(".")[1])].slice(row, 1).to_pylist()[0]
            assert extracted[p] == (f"flan/{folder}/{source['_task_name']}", source["inputs"], source["targets"])
            if folder == "cot_zsopt_data":
                assert row % world == host
            sources.add((name, row))
        assert len(sources) == table.num_rows


# ------------------------------------------------------------------------------------------------------------ build

ENGLISH = {(0, 5): ("Q: What is the capital of France?\nA:", "Paris is the capital"),
           (0, 9): ("Premise: The dog runs.\nOPTIONS:\n- yes\n- no", "yes"),
           (1, 2): ("Translate nothing here, just answer the question please.", "The answer is no"),
           (1, 4): ("Q: How many?\nA:", "Three")}
DUTCH = {(0, 5): ("Vraag: Wat is de hoofdstad van Frankrijk?\nAntwoord:", "Parijs is de hoofdstad"),
         (0, 9): ("Premisse: De hond rent.\nOPTIES:\n- ja\n- nee", "ja"),
         (1, 2): ("Gegeven: een zin.", "Antwoord: een zin."),  # a Dutch task, kept as it was
         (1, 4): ("Vraag: Hoeveel?\nAntwoord:", "Drie")}
SOURCES = {(0, 5): "flan/t0_zsopt_data/t0_task", (0, 9): "flan/flan_zsopt_data/rte:2.0.0",
           (1, 2): "flan/niv2_zsopt_data/task1133_xcsr_nl_commonsense_mc_classification", (1, 4): "flan/t0_zsopt_data/t0_task"}
LINKS = {(0, 5): ("t0_zsopt_data/part.3.parquet", 17, 4, "T0", "t0_task", "zs_opt"),
         (0, 9): ("flan_zsopt_data/part.0.parquet", 8, 2, "Flan2021", "rte:2.0.0", "zs_opt"),
         (1, 2): ("niv2_zsopt_data/part.1.parquet", 3, 0, "NIv2", "task1133_xcsr_nl_commonsense_mc_classification", "zs_opt"),
         (1, 4): ("t0_zsopt_data/part.3.parquet", 2, 1, "T0", "t0_task", "zs_opt")}
STATUS = {(0, 5): "translate", (0, 9): "translate", (0, 7): "task:word_game", (1, 2): "dutch_task", (1, 4): "translate",
          (1, 6): "translate"}
NL_REPORT = {
    "segments": 1234, "segments_repaired": {"danda": 3, "final_period": 41},
    "segments_unusable": {"length_ratio": 7, "not_dutch": 2},
    "hosts": {"0": {"flan_rows": 3, "translated": 2, "dropped:task:word_game": 1},
              "1": {"flan_rows": 3, "translated": 1, "kept_dutch_task": 1, "dropped:answer_not_in_instruction": 1}},
    "template_variants": {"Q:": {"Vraag:": 2}, "A:": {"Antwoord:": 2}},
    "template_registers": {"u (mt)": 1, "je (hash)": 1, "rows with a variant outside the register": 0},
    "head_tail_pairs": {}, "variant_decisions": {"seeded": 1},
}


CODE_URL = "https://example.org/flan-nl"


def assembled(tmp_path, links=LINKS, dutch=DUTCH, english=ENGLISH):
    """nl/, links/, plan2/ and en/ as the assembly, link, plan and extract leave them."""
    nl, linked, plan, en = tmp_path / "nl", tmp_path / "links", tmp_path / "plan2", tmp_path / "en"
    for d in (nl, linked, plan, en):
        d.mkdir()
    for h in (0, 1):
        keys = sorted(k for k in dutch if k[0] == h)
        pq.write_table(pa.table({"position": [p for _, p in keys], "source": [SOURCES[k] for k in keys],
                                 "inputs": [dutch[k][0] for k in keys], "targets": [dutch[k][1] for k in keys]}),
                       nl / f"w{h}.parquet")
        keys = sorted(k for k in english if k[0] == h)
        pq.write_table(pa.table({"host": pa.array([h] * len(keys), pa.int32()), "position": [p for _, p in keys],
                                 "source": [SOURCES[k] for k in keys], "inputs": [english[k][0] for k in keys],
                                 "targets": [english[k][1] for k in keys]}), en / f"w{h}.parquet")
        keys = sorted(k for k in links if k[0] == h)
        pq.write_table(pa.table({"host": [h] * len(keys), "position": [p for _, p in keys],
                                 **{c: [links[k][i] for k in keys] for i, c in enumerate(pub.LINK_COLUMNS[2:])}}),
                       linked / f"w{h}.parquet")
    (nl / "report.json").write_text(json.dumps(NL_REPORT))
    folder = {"rows": 1, "rows_sharing_content": 0, "shards": 4, "shards_read_by_f": 1, "shards_opened": 1,
              "f_dropped": {"too_long": 2, "language": 5}}
    host = {"f_build": "b", "rows": 3, "linked": 3, "f_world": 4, "f_cap": 2000, "f_cap_per_host": 500, "f_max_length": 1024,
            "f_flan_share": 0.1, "f_pool": 9, "f_examples": 3, "folders": {f: folder for f in FOLDERS}}
    (linked / "report.json").write_text(json.dumps({"rows": 6, "linked": 6, "hosts": {"0": host, "1": host},
                                                    "f_excluded_tasks": {"dumb_leak": ["task1612_sick_label_classification"],
                                                                         "language": ["wmt16_translate/de-en"]}}))
    keys = sorted(STATUS)
    pq.write_table(pa.table({"host": [h for h, _ in keys], "position": [p for _, p in keys],
                             "status": [STATUS[k] for k in keys]}), plan / "examples.parquet")
    return ["--nl", str(nl), "--links", str(linked), "--plan", str(plan), "--english", str(en), "--repo", "someone/flan-nl",
            "--code-url", CODE_URL, "--processes", "1"]


def test_build_writes_the_columns_sorted_by_source_file_and_row(tmp_path):
    out = tmp_path / "hf"
    pub.main(["build", *assembled(tmp_path), "--out", str(out)])
    files = sorted((out / "data").glob("train-*.parquet"))
    assert [f.name for f in files] == ["train-00000-of-00001.parquet"]
    table = pq.read_table(files[0])
    assert table.column_names == pub.DATA_COLUMNS
    assert table.schema.field("translated").type == pa.bool_() and table.schema.field("flan_row").type == pa.int64()
    rows = table.to_pylist()
    assert [(r["flan_file"], r["flan_row"]) for r in rows] == sorted((v[0], v[1]) for v in LINKS.values())
    assert [r["id"] for r in rows] == ["w0-9", "w1-2", "w1-4", "w0-5"]
    by_id = {r["id"]: r for r in rows}
    assert by_id["w1-2"]["translated"] is False and by_id["w0-5"]["translated"] is True
    assert (by_id["w0-5"]["inputs"], by_id["w0-5"]["targets"]) == DUTCH[(0, 5)]
    assert by_id["w0-9"] | {"inputs": None, "targets": None} == {
        "id": "w0-9", "inputs": None, "targets": None, "translated": True, "task_name": "rte:2.0.0", "task_source": "Flan2021",
        "folder": "flan_zsopt_data", "template_type": "zs_opt", "template_idx": 2, "flan_revision": pub.FLAN_REVISION,
        "flan_file": "flan_zsopt_data/part.0.parquet", "flan_row": 8}


def test_the_output_has_no_english_text(tmp_path):
    out = tmp_path / "hf"
    pub.main(["build", *assembled(tmp_path), "--out", str(out)])
    written = b"".join(p.read_bytes() for p in (out / "data").glob("*.parquet")).decode("latin-1")
    table = pq.read_table(next((out / "data").glob("*.parquet")))
    text = " ".join(table.column("inputs").to_pylist() + table.column("targets").to_pylist())
    card = (out / "README.md").read_text()
    for inputs, targets in ENGLISH.values():
        for english in (inputs, targets):
            assert english not in text and english not in card
            assert english.encode().decode("latin-1") not in written


def test_the_card_has_the_numbers_of_the_data_and_the_reports(tmp_path):
    out = tmp_path / "hf"
    pub.main(["build", *assembled(tmp_path), "--out", str(out), "--repo", "someone/flan-nl"])
    card = (out / "README.md").read_text()
    header = card.split("---")[1]
    for line in ("license: cc-by-4.0", "- nl", "- Open-Orca/FLAN", "- text-generation",
                 "- n<1K", "pretty_name: FLAN-nl", "path: data/train-*"):
        assert line in header
    # Not on the Hub's official list of task categories
    assert "text2text-generation" not in header
    for fact in ("6 English rows, of which 4 are in this dataset (3 translated from English, 1 of",
                 "| translated from English (`translated = true`) | 3 |",
                 "| Dutch tasks, not translated (`translated = false`) | 1 |",
                 "| total | 4 |",
                 "| `t0_zsopt_data` | `zs_opt` | 2 | 1 | 2 of 4 |",
                 "(1 task: `task1612_sick_label_classification`) | 0 |",
                 # too_long: 2 per folder and host, 7 folders, 2 hosts
                 "rows of more than 1,024 tokens (instruction and answer, in the experiment's tokenizer) | 28 |",
                 "at most 500 rows per (folder, task) per host, so at most 2,000",
                 f"`{pub.FLAN_REVISION}`",
                 "(1,234 segments)", "(3 segments)", "(41 segments)",
                 "| `length_ratio` |", "| 7 |",
                 "`task1133_xcsr_nl_commonsense_mc_classification`",
                 'load_dataset("someone/flan-nl"',
                 "Google claims no rights in Outputs you generate using Gemma.",
                 "used them until FLAN made up 10% of\n   the tokens of its training mix: 6 rows.",
                 f"assembled and published these rows: <{CODE_URL}> (`flan_nl.py`"):
        assert fact in card, fact
    word_games = [line for line in card.splitlines() if line.startswith("| word games")]
    assert word_games and word_games[0].endswith(f"| {len(pub.flan_nl.DROPPED_TASKS['word_game'])} | 1 |")
    assert [line for line in card.splitlines() if line.startswith("| extractive answers")][0].endswith("| 1 |")


def test_a_dutch_row_without_a_link_fails(tmp_path):
    links = {k: v for k, v in LINKS.items() if k != (1, 4)}
    with pytest.raises(SystemExit, match=r"1 Dutch rows have no link.*\(1, 4, 'flan/t0_zsopt_data/t0_task'\)"):
        pub.main(["build", *assembled(tmp_path, links), "--out", str(tmp_path / "hf")])


def test_a_link_of_another_task_fails(tmp_path):
    links = dict(LINKS)
    links[(0, 5)] = ("t0_zsopt_data/part.3.parquet", 17, 4, "T0", "another_task", "zs_opt")
    with pytest.raises(SystemExit, match="1 Dutch rows disagree with their link"):
        pub.main(["build", *assembled(tmp_path, links), "--out", str(tmp_path / "hf")])


def test_a_report_that_does_not_add_up_fails_before_anything_is_written(tmp_path):
    args = assembled(tmp_path)
    report = json.loads(json.dumps(NL_REPORT))
    report["hosts"]["0"]["flan_rows"] = 4
    (tmp_path / "nl" / "report.json").write_text(json.dumps(report))
    with pytest.raises(SystemExit, match="nl/report.json counts 7 FLAN rows of F, links/report.json 6"):
        pub.main(["build", *args, "--out", str(tmp_path / "hf")])
    assert not (tmp_path / "hf" / "data").exists() and not (tmp_path / "hf" / "README.md").exists()


def test_build_refuses_a_folder_with_old_files(tmp_path):
    (tmp_path / "hf" / "data").mkdir(parents=True)
    (tmp_path / "hf" / "data" / "train-00000-of-00009.parquet").write_bytes(b"old")
    with pytest.raises(SystemExit, match="not empty"):
        pub.main(["build", *assembled(tmp_path), "--out", str(tmp_path / "hf")])


def test_build_needs_the_dataset_repo_and_the_code_url(tmp_path, capsys):
    args = assembled(tmp_path)
    for flag in ("--repo", "--code-url"):
        k = args.index(flag)
        with pytest.raises(SystemExit):
            pub.main(["build", *args[:k], *args[k + 2:], "--out", str(tmp_path / "hf")])
        assert f"the following arguments are required: {flag}" in capsys.readouterr().err
    assert not (tmp_path / "hf").exists()


def test_the_card_of_an_extract_from_the_hub_says_how_many_rows_were_taken(tmp_path):
    args = assembled(tmp_path)
    report = json.loads((tmp_path / "links" / "report.json").read_text())
    for host in report["hosts"].values():
        host["f_flan_share"] = None  # extract from Open-Orca/FLAN with --rows
    (tmp_path / "links" / "report.json").write_text(json.dumps(report))
    pub.main(["build", *args, "--out", str(tmp_path / "hf")])
    card = (tmp_path / "hf" / "README.md").read_text()
    assert ("3. Each host shuffled its rows with a fixed seed and took the first rows of that order (all of them, or a "
            "given number per host): 6 rows. These are the rows that were translated.") in card
    assert "training mix: 6 rows" not in card


# ------------------------------------------------------------------------------------ checks for this publication
# The patterns of the first build's samples (9 Nov), shortened: w2-497773, w3-976648 and w3-1242456 (decode loops),
# w0-51286 (English left, which langid took for Dutch), w1-1353724 (a samsum dialogue, fine)

DIALOGUE_NL = ("Susan: Weet je nog dat het morgen Megan’s verjaardag is?\nMark: Natuurlijk... :P\n"
               "Susan: Ik weet niet wat ik haar moet geven.\nMark: En wanneer is het precies?\nSusan: Morgen\n"
               "Mark: Oeps, ik dacht dat het dinsdag was.\nSusan: Nee, morgen.\n"
               "Mark: Oké, dan neem ik een fles rode wijn mee, en verder niets, denk ik.\nSusan: Ja, wijn kan wel.")
DIALOGUE_EN = ("Susan: Do you remember about Megans bday?\nMark: Of course... :P\nSusan: I dont know what to buy her\n"
               "Mark: And when is it?\nSusan: Tomorrow\nMark: Oops, I thought its on tuesday \nSusan: Nooo, no, tomorrow\n"
               "Mark: Ok, so a bottle of red wine and nothing more, I guess\nSusan: Yeah, can be wine")
PLOT_EN = ("Rahul has a novel strategy to rise up the ladder of success. He provides his flat to his superiors at the "
           "office, where they can safely bring their mistresses, girlfriends or prostitutes, in return for "
           "recommendation for his promotion. Ranjeet uses the flat to meet Neha. Amidst one of their escapades in "
           "Rahul's flat, Neha asks Ranjeet about their future, which leads to an argument.")


@pytest.mark.parametrize("nl, en", [
    ("een leverancier van de levering van de levering van de levering van de levering van de levering van de levering "
     "van de lever. Buddha Mar Gaya neemt je mee op een wilde rit.",
     "a procurer of dead bodies, the paparazzi and a couple of cops. Buddha Mar Gaya takes you on a rip-roaring ride."),
    ("de koningin hem wil doden om de: de: de: de: de: de: de: de: the queen to convince him otherwise",
     "the Queen is going to kill him to get the last two pieces, but the Queen convinces him otherwise"),
    ("hij bleef optreden. vervolg: vervolg: vervolg: vervolg: vervolg: vervolg: vervolg: vervatting: de: de:",
     "while continuing to perform on the sodina in musical ensembles."),
    # A sequence of 5 tokens 3 times; and a loop that varies its punctuation, counted over the words alone
    ("Vraag: Wat is de vraag: Wat is de vraag: Wat is de vraag: Wat is de vraag", "Question: What is the question?"),
    ("tijdunit for, for: for: for, for, for: for, unit", "The time unit of the trial."),
    # 5 copies of 4 tokens
    ("en wordt via de schoorsteen getrokken met de zak met de zak met de zak met de zak met de zak: eenmaal binnen",
     "is pulled up the chimney with the bag; once inside, he brings a few presents"),
])
def test_a_decode_loop_is_caught(nl, en):
    assert pub.decode_loop(nl, en)


def test_an_ellipsis_for_a_blank_is_no_decode_loop():
    # "... ." is 4 copies of ".", below the threshold; the English has a blank there, not a repeat
    assert not pub.decode_loop("Vraag: Volgens de tekst is het ... . Ja of nee?", "Question: According to the text, it is  _  . Yes or no?")


@pytest.mark.parametrize("nl, en", [
    (DIALOGUE_NL, DIALOGUE_EN),
    # The English has the same repeat, with its own punctuation, or inside one word (emoji written out)
    ("Geen emotie, geen emotie, geen emotie, geen emotie, geen emotie, geen emotie, geen emotie, geluk",
     "No emotion,No emotion,No emotion,No emotion,No emotion,No emotion,No emotion,happiness"),
    ("doelpunten: 0, 0, 0, 0, 0, 0, 1, 0", "goals: 0 0 0 0 0 0 1 0"),
    ("Hier, pak het. [huilend gezicht] [huilend gezicht] [huilend gezicht] [huilend gezicht] [huilend gezicht]",
     "here collect it" + "loudlycryingface" * 5),
    ("Hij zei het twee keer: kom op, kom op.", "He said it twice: come on, come on."),
    # Punctuation alone: emoji, exclamation marks, a table the translation drew
    ("Hier, pak het. 😭 😭 😭 😭 😭 😭", "here collect it &#128557;&#128557;&#128557;&#128557;&#128557;&#128557;"),
    ("CARLA MOET EENS ZIJN MOND HOUDEN!!!!!!!!", "CARLA NEEDS TO SHUT UP!!!!!!!!"),
    ("Artikel | Kleuren | Prijs\n------- | -------- | --------\nSokken | Zwart | $3",
     "Goods     Colors     Price Socks      Black $3"),
])
def test_a_repeat_of_the_english_is_no_decode_loop(nl, en):
    assert not pub.decode_loop(nl, en)


def test_english_left_in_a_dutch_text_is_caught():
    nl = ("Rahul heeft een unieke strategie om hogerop te komen: He provides his flat to his superiors at office, where "
          "they can safely bring their mistresses, girlfriends or prostitutes, in return for recommendation for his "
          "promotion. Ranjeet uses the flat to meet Neha. Tijdens een van hun avonturen in het appartement van Rahul "
          "vraagt Neha naar hun toekomst, wat leidt tot een ruzie.")
    assert pub.english_left(nl, PLOT_EN)
    assert pub.english_left("Vraag: What is the last name of the person who transcribed the music of an English "
                            "composer?\nAntwoord:",
                            "Question: What is the last name of the person who transcribed the music of an English "
                            "composer?\nAnswer:")


@pytest.mark.parametrize("nl, en", [
    (DIALOGUE_NL, DIALOGUE_EN),
    # Titles and names are copied, capitalized, with few lowercase words
    ("Het boek The Lord of the Rings van J.R.R. Tolkien verscheen in 1954 en werd een klassieker.",
     "The book The Lord of the Rings by J.R.R. Tolkien was published in 1954 and became a classic."),
    ("Ford speelde in de films Raiders of the Lost Ark, Kingdom of the Crystal Skull, The Sure Thing en Dead Man on "
     "Campus.",
     "Ford starred in the films Raiders of the Lost Ark, Kingdom of the Crystal Skull, The Sure Thing and Dead Man on "
     "Campus."),
    # A URL is copied as it is
    ("Het oorspronkelijke bericht: https://old.reddit.com/r/AmItheAsshole/comments/alhmmr/"
     "aita_for_breaking_up_out_of_fear_and_confusion",
     "Original post: https://old.reddit.com/r/AmItheAsshole/comments/alhmmr/"
     "aita_for_breaking_up_out_of_fear_and_confusion"),
    ("Rahul heeft een nieuwe strategie om de ladder van succes te beklimmen. Hij stelt zijn appartement ter "
     "beschikking aan zijn superieuren op kantoor, in ruil voor een aanbeveling voor zijn promotie.", PLOT_EN),
])
def test_names_titles_and_translations_are_no_english_left(nl, en):
    assert not pub.english_left(nl, en)


def test_a_row_fails_the_checks_of_either_field_in_order():
    loop = "de: de: de: de: de: de: de: de:"
    english = "Vraag: What is the last name of the person who transcribed the music of an English composer?"
    source = "Question: What is the last name of the person who transcribed the music of an English composer?"
    assert pub.publication_checks("Vraag: Hoeveel?", "Drie", "Q: How many?", "Three") == ()
    assert pub.publication_checks("Vraag: Hoeveel?", loop, "Q: How many?", "Three") == ("decode_loop",)
    assert pub.publication_checks(english, loop, source, "Three") == ("decode_loop", "english_left")


# A loop in the instruction of (0, 5), and English left in the instruction of (1, 4)
FAILING = DUTCH | {
    (0, 5): ("Vraag: Wat is de hoofdstad van de hoofdstad van de hoofdstad van de hoofdstad van de hoofdstad van de "
             "hoofdstad van Frankrijk?\nAntwoord:", "Parijs is de hoofdstad"),
    (1, 4): ("Vraag: How many children lived with their mother in the old house near the river?\nAntwoord:", "Drie")}
FAILING_ENGLISH = ENGLISH | {
    (1, 4): ("Q: How many children lived with their mother in the old house near the river?\nA:", "Three")}


def test_build_leaves_out_the_rows_that_fail_a_check_and_counts_them(tmp_path):
    out = tmp_path / "hf"
    pub.main(["build", *assembled(tmp_path, dutch=FAILING, english=FAILING_ENGLISH), "--out", str(out)])
    table = pq.read_table(next((out / "data").glob("*.parquet")))
    assert table.column("id").to_pylist() == ["w0-9", "w1-2"]
    report = json.loads((out / "build_report.json").read_text())
    assert (report["rows_of_the_experiment"], report["translated_checked"], report["rows"]) == (4, 3, 2)
    assert report["checks"] == {
        "decode_loop": {"rows": 1, "rows_failing": 1, "tasks": {"t0_task": 1}, "ids": ["w0-5"]},
        "english_left": {"rows": 1, "rows_failing": 1, "tasks": {"t0_task": 1}, "ids": ["w1-4"]}}
    card = (out / "README.md").read_text()
    for fact in ("6 English rows, of which 2 are in this dataset (1 translated from English, 1 of",
                 "translation failed a check (see \"Selection\" and \"Translation\"); 2 of them are Dutch",
                 "| translated from English (`translated = true`) | 1 |", "| total | 2 |",
                 "6. Checks for this publication: two more checks were run on each of the 3 translated rows, each on",
                 "out of this dataset only; the Dutch rows of the experiment were not changed. The 1 rows of Dutch tasks were",
                 "0 rows fail both and are counted under the first.",
                 "rows left out by these checks: `t0_task` (2).",
                 "Failed translations remain after the checks of step 6"):
        assert fact in card, fact
    lines = {line.split(" | ")[0]: line for line in card.splitlines() if line.startswith("| `decode_loop`")
             or line.startswith("| `english_left`")}
    assert lines["| `decode_loop`"].endswith(" | 1 |") and "5 or more times back to back" in lines["| `decode_loop`"]
    assert lines["| `english_left`"].endswith(" | 1 |") and "at least 3 English function words" in lines["| `english_left`"]
    assert "at least 5 other lowercase words of 3 letters or more" in lines["| `english_left`"]
    # The English of the checks stays out of the output
    written = b"".join(p.read_bytes() for p in out.rglob("*") if p.is_file()).decode("latin-1")
    assert FAILING_ENGLISH[(1, 4)][0].encode().decode("latin-1") not in written


def test_build_in_processes_gives_the_same_result(tmp_path):
    args = assembled(tmp_path, dutch=FAILING, english=FAILING_ENGLISH)
    pub.main(["build", *args[:-2], "--processes", "2", "--out", str(tmp_path / "hf")])
    assert pq.read_table(next((tmp_path / "hf" / "data").glob("*.parquet"))).column("id").to_pylist() == ["w0-9", "w1-2"]


def test_a_translated_row_without_its_english_fails(tmp_path):
    english = {k: v for k, v in ENGLISH.items() if k != (1, 4)}
    with pytest.raises(SystemExit, match=r"1 translated rows have no English row.*\(1, 4, 'flan/t0_zsopt_data/t0_task', None\)"):
        pub.main(["build", *assembled(tmp_path, english=english), "--out", str(tmp_path / "hf")])
    assert not (tmp_path / "hf").exists()


# ----------------------------------------------------------------------------------------------------------- upload

def fake_hub(monkeypatch):
    calls = []

    class HfApi:
        def create_repo(self, *args, **kwargs):
            calls.append(("create_repo", args, kwargs))

        def upload_folder(self, *args, **kwargs):
            calls.append(("upload_folder", args, kwargs))

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=HfApi))
    return calls


def test_upload_makes_a_public_dataset_repo_and_uploads_the_folder(tmp_path, monkeypatch, capsys):
    calls = fake_hub(monkeypatch)
    out = tmp_path / "hf"
    pub.main(["build", *assembled(tmp_path), "--out", str(out)])
    pub.main(["upload", "--folder", str(out), "--repo", "someone/flan-nl", "--public"])
    assert calls[0] == ("create_repo", ("someone/flan-nl",), {"repo_type": "dataset", "private": False, "exist_ok": True})
    name, _, kwargs = calls[1]
    assert name == "upload_folder" and kwargs["repo_id"] == "someone/flan-nl" and kwargs["repo_type"] == "dataset"
    assert kwargs["folder_path"] == str(out) and kwargs["allow_patterns"] == ["README.md", "data/train-*.parquet"]
    assert capsys.readouterr().out.strip().endswith("https://huggingface.co/datasets/someone/flan-nl")


def test_upload_needs_public_and_a_built_folder(tmp_path, monkeypatch):
    calls = fake_hub(monkeypatch)
    with pytest.raises(SystemExit, match="--public"):
        pub.main(["upload", "--folder", str(tmp_path), "--repo", "someone/flan-nl"])
    with pytest.raises(SystemExit, match="README.md and data"):
        pub.main(["upload", "--folder", str(tmp_path), "--repo", "someone/flan-nl", "--public"])
    assert calls == []
