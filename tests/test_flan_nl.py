"""flan_nl.py and translate.py on made-up FLAN rows and a made-up translator: the extract from made-up shards of
Open-Orca/FLAN, options, the substring rule, filters, the Dutch mix in place of F's FLAN rows, and shards that a restart
skips. No model, no network and no TPU."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import flan_nl as fn  # noqa: E402
import flan_select  # noqa: E402
import translate as tr  # noqa: E402


def words(text):
    return len(text.split())


def fake_dutch(text):
    """A translator that is easy to check: every segment becomes NL[<segment>]."""
    return f"NL[{text}]"


def translated(inputs, targets, translate=fake_dutch):
    instruction, answer, info = fn.plan_example(inputs, targets, words)
    table = {s: translate(s) for s in fn.segments_of(instruction) + fn.segments_of(answer)}
    return fn.render(instruction, table), fn.render(answer, table), info


@pytest.mark.parametrize("inputs, targets", [
    ("Premise: A man.\nHypothesis: A dog.\nOPTIONS:\n- yes\n- it is not possible to tell\n- no", "it is not possible to tell"),
    ("What is the best way to measure?\n\nOPTIONS: [+] Grab a teaspoon.; [+] Grab a tablespoon.;\nAnswer:", "Grab a teaspoon."),
    ("Tell me who it is.\nAvailable choices:\n +a Barbie.\n +the microwave.", "the microwave"),
    ("Available options: -Two.. -One.. -Four.. -Three..\nGiven those answer options, answer: Bob is six.", "Three."),
    ("pick from the following. -- Yes -- It's impossible to say -- No\nI think the answer is", "It's impossible to say"),
])
def test_an_answer_that_is_an_option_gets_the_options_translation(inputs, targets):
    instruction, answer, info = fn.plan_example(inputs, targets, words)
    assert info["answer"] == "option"
    nl_inputs, nl_targets, _ = translated(inputs, targets)
    option = fn._strip_end(targets)
    assert nl_targets == f"NL[{option}]" + targets[len(option):]
    # The Dutch answer is word for word one of the Dutch options
    assert f"NL[{option}]" in nl_inputs
    assert not fn.answer_lost(inputs, targets, nl_inputs, nl_targets)


@pytest.mark.parametrize("inputs, targets, labels", [
    ("What will happen to Lee?\n\nOPTIONS: (i). be frustrated (ii). be happy (iii). stand trial\nThe answer is:", "(ii).",
     ["(i).", "(ii).", "(iii)."]),
    ("Who is \"it\"?\nChoose your answer from:\n a). The cat\n b). the glass", "a).", ["a).", "b)."]),
    ("Select from the following. i. insulin; ii. lipids; iii. amino acids; iv. glucose;\n\nQ: What?\nAnswer:", "iv.",
     ["i.", "ii.", "iii.", "iv."]),
    ("Options: (A) How to change (B) Improve students (C) Only a little\nSolution:", "B", ["(A)", "(B)", "(C)"]),
    ("Question: Is it?\nChoices:\n1). yes.\n2). it is not possible to tell.\n3). no.", "1).", ["1).", "2).", "3)."]),
])
def test_labels_stay_as_they_are_and_every_option_is_a_segment_of_its_own(inputs, targets, labels):
    instruction, answer, info = fn.plan_example(inputs, targets, words)
    assert info["answer"] == "label" and info["options"] == len(labels)
    assert answer == [("L", targets)]
    literal = "".join(v for k, v in instruction if k == "L")
    segments = fn.segments_of(instruction)
    for label in labels:
        assert label in literal and not any(label in s for s in segments[1:])
    options = [inputs[start:stop] for _, _, start, stop in fn.find_options(inputs)]
    assert all(option in segments for option in options)
    nl_inputs, nl_targets, _ = translated(inputs, targets)
    assert nl_targets == targets and all(label in nl_inputs for label in labels)


def test_identity_translation_gives_back_every_example_unchanged():
    cases = [("Is this review positive?\nOPTIONS:\n- No\n- Yes", "Yes"), ("[73, 878, 639]\nOutput:", "[878, 11]"),
             ("  Leading and trailing  \n", "  answer with spaces  "), ("Answer with \"plausible\" or \"not plausible\".", "not plausible"),
             ("A. first. B. second. C. third.", "C."), ("Question?\n\nAvailable choices: [I] negative [II] positive", "[I]")]
    for inputs, targets in cases:
        nl_inputs, nl_targets, _ = translated(inputs, targets, lambda s: s)
        assert (nl_inputs, nl_targets) == (inputs, targets)


def test_text_without_words_is_not_translated():
    # The instruction has a word, so it is one segment, list and all; the answer has none and stays as it is
    instruction, answer, _ = fn.plan_example("[73, 878, 639, 977]\nOutput:", "[878, 11]", words)
    assert answer == [("L", "[878, 11]")] and instruction == [("S", "[73, 878, 639, 977]\nOutput:")]
    assert fn.text_pieces(" 4.5 ", words) == [("L", " 4.5 ")] and not fn.needs_translation("(B). 3 - 4")


def test_long_text_is_cut_at_paragraphs_then_sentences():
    paragraph = " ".join(["word"] * 300) + "."
    text = "\n\n".join([paragraph] * 3)
    pieces = fn.text_pieces(text, words, limit=512, splits=(splits := []))
    assert [k for k, _ in pieces] == ["S", "L", "S", "L", "S"] and splits == [2]
    assert fn.render(pieces, {s: s for s in fn.segments_of(pieces)}) == text
    sentences = " ".join(["Short sentence number one."] * 200)
    assert all(words(s) <= 512 for s in fn.segments_of(fn.text_pieces(sentences, words, limit=512)))
    with pytest.raises(fn.SegmentTooLong):
        fn.text_pieces(" ".join(["word"] * 600), words, limit=512)


@pytest.mark.parametrize("inputs_en, targets_en, inputs_nl, targets_nl, lost", [
    # Extractive: the answer is a span of the passage in both languages
    ("The papers were found by Sinister Foxx.", "Sinister Foxx", "De papieren zijn gevonden door Sinister Foxx.", "Sinister Foxx", False),
    ("Answer \"plausible\" or \"not plausible\".", "not plausible", "Antwoord \"aannemelijk\" of \"niet aannemelijk\".", "niet aannemelijk", False),
    # The span's own translation differs from its translation inside the passage
    ("Answer \"plausible\" or \"not plausible\".", "not plausible", "Antwoord \"plausibel\" of \"niet plausibel\".", "niet aannemelijk", True),
    # Whole words only, case and spacing aside: "no" is not in "know"
    ("I know nothing.", "no", "Ik weet niets.", "nee", False),
    ("Say Yes  or No.", "no", "Zeg Ja of Nee.", "NEE", False),
    # Not in the English instruction: no rule
    ("Write a review.", "A fine film.", "Schrijf een recensie.", "Een mooie film.", False),
])
def test_substring_rule(inputs_en, targets_en, inputs_nl, targets_nl, lost):
    assert fn.answer_lost(inputs_en, targets_en, inputs_nl, targets_nl) == lost


def test_task_filters():
    assert fn.task_drop_reason("glue/cola:2.0.0") == "english_form"
    assert fn.task_drop_reason("task107_splash_question_to_sql") == "code"
    assert fn.task_drop_reason("task183_rhyme_generation") == "word_game"
    assert fn.task_drop_reason("task583_udeps_eng_coarse_pos_tagging") == "token_level"
    assert fn.task_drop_reason("task560_alt_translation_en_entk") == "translation"
    assert fn.task_drop_reason("super_glue/wic:1.0.2") == "english_words"
    assert fn.task_drop_reason("task142_odd-man-out_classification_no_category") == "word_relations"
    assert fn.task_drop_reason("task382_hybridqa_answer_generation") == "token_level"
    assert fn.task_drop_reason("task344_hybridqa_answer_generation") is None
    assert fn.task_drop_reason("task1155_bard_analogical_reasoning_trash_or_treasure") == "word_relations"
    assert fn.task_drop_reason("glue/sst2:2.0.0") is None and fn.task_drop_reason("task1312_amazonreview_polarity_classification") is None
    assert "task1543_conll2002_parts_of_speech_tagging_answer_generation" in fn.DUTCH_TASKS
    assert not fn.DUTCH_TASKS & set(fn._REASON)
    assert fn.task_of("flan/niv2_zsopt_data/task391_causal_relationship") == "task391_causal_relationship"
    with pytest.raises(ValueError):
        fn.task_of("culturax")


def test_generic_checks_of_a_translation():
    dutch = lambda _: "nl"  # noqa: E731
    english = lambda _: "en"  # noqa: E731
    source = "The weather is lovely today and we will go to the beach."
    good = "Het weer is vandaag heerlijk en we gaan naar het strand."
    assert fn.check_translation(source, good, True, dutch) is None
    assert fn.check_translation(source, good, False, dutch) == "truncated"
    assert fn.check_translation(source, source, True, english) == "not_dutch"
    assert fn.check_translation(source, "Ja.", True, dutch) == "length_ratio"
    assert fn.check_translation(source, good + " " + good + " " + good, True, dutch) == "length_ratio"
    assert fn.check_translation(source, "Hier is de vertaling: " + good, True, dutch) == "commentary"
    assert fn.check_translation("not liking my parents", "niet overweg kunnen met / een hekel hebben aan", True, dutch) == "alternatives"
    assert fn.check_translation("km/h or mph", "km/u of mph / mijl", True, dutch) is None
    assert fn.check_translation("Note: bring a towel.", "Opmerking: neem een handdoek mee.", True, dutch) is None
    # Too short for a language ID: names and single words pass
    assert fn.check_translation("Sinister Foxx", "Sinister Foxx", True, english) is None
    # langid's English for names and for a Dutch sentence is not enough: the output must have English in it
    assert fn.check_translation("Jonathan Luke Ke Huy Quan", "Jonathan Luke Ke Huy Quan", True, english) is None
    assert fn.check_translation("What did Russell Simmons think of West's apology?",
                                "Wat vond Russell Simmons van Wests excuses?", True, english) is None
    # "Let op:" translates FLAN's own "Note that"; a translator's note does not
    assert fn.check_translation("Note that URLs in the text have been replaced with [Link].",
                                "Let op: de URL's in de tekst zijn vervangen door [Link].", True, dutch) is None
    assert fn.check_translation("The cat sat on the mat today.", "De kat zat vandaag op de mat. (Opmerking: letterlijk)",
                                True, dutch) == "commentary"
    assert fn.check_translation("cabin", "hut; cabine", True, dutch) == "alternatives"
    assert fn.check_translation("Counselor", "Begeleider/adviseur", True, dutch) == "alternatives"
    assert fn.check_translation("elderly", "ouder\nbejaard\nsenior", True, dutch) == "alternatives"
    assert fn.check_translation("Answer:\nyes", "Antwoord:\nja", True, dutch) is None
    assert fn.check_translation("Not similar", "Niet vergelijkbaarक", True, dutch) == "foreign_script"
    assert fn.check_translation("Dood Ari - Part 1", "Dood Ari – Deel 1", True, dutch) is None


def test_the_two_repairs_are_made_and_named():
    assert fn.repair_translation("cattle.", "rundvee\u0964") == ("rundvee.", ["danda"])
    assert fn.repair_translation("Not similar", "Niet vergelijkbaar\u0964") == ("Niet vergelijkbaar", ["danda", "final_period"])
    assert fn.repair_translation("It rains. It pours.", "Het regent\u0964 Het giet\u0964") == ("Het regent. Het giet.", ["danda"])
    assert fn.repair_translation("cattle.", "rundvee.") == ("rundvee.", [])
    assert fn.repair_translation("\u0964 stays", "\u0964 blijft") == ("\u0964 blijft", [])
    assert fn.repair_translation("Yes", "Ja.") == ("Ja", ["final_period"])
    assert fn.repair_translation("the snake", "de slang.") == ("de slang", ["final_period"])
    assert fn.repair_translation("Wait for it...", "Wacht maar...") == ("Wacht maar...", [])
    assert fn.repair_translation("Is it done?", "Is het klaar?") == ("Is het klaar?", [])
    assert fn.repair_translation("Answer:", "Antwoord.") == ("Antwoord.", [])


def test_typographic_quotes_and_dashes_count_as_plain_ones():
    assert not fn.answer_lost("Choose: Cars & Other Vehicles, Home", "Cars & Other Vehicles",
                              "Kies: Auto's en andere voertuigen, Huis", "Auto’s en andere voertuigen")
    assert not fn.answer_lost("Is it Kill Ari - Part 1?", "Kill Ari - Part 1", "Is het Dood Ari - Deel 1?", "Dood Ari – Deel 1")


def _mix_rows():
    """F's sequence of a host: web rows with odd bytes, FLAN rows in between, in position order."""
    texts = ["Eerste document.\n\nMet één alinea.", "  spaties  en ⁇ tekens ", "Derde\tdocument", "laatste"]
    rows, position = [], 0
    for k, text in enumerate(texts):
        rows.append((position, text, None, None, "culturax"))
        position += 1
        for j in range(2):
            rows.append((position, None, f"instruction {k}.{j}", f"answer {k}.{j}", "flan/t0_zsopt_data/some_task"))
            position += 1
    return rows


def test_dutch_mix_replaces_flan_rows_in_place_and_keeps_web_rows_byte_identical(tmp_path):
    rows = _mix_rows()
    flan_positions = [p for p, _, _, _, s in rows if s != "culturax"]
    dropped = flan_positions[3]
    dutch = {p: (f"instructie {p}", f"antwoord {p}") for p in flan_positions if p != dropped}
    sequence, web_texts, pool, counts = fn.dutch_sequence(rows, dutch)
    assert counts == {"web": 4, "flan_in": 8, "flan_dropped": 1, "flan_kept": 7}
    # The order of F with the dropped row taken out, every FLAN row at the place of its English one
    expected = [("web", p) if s == "culturax" else ("flan", p) for p, _, _, _, s in rows if p != dropped]
    got = [("web", None) if kind == "web" else ("flan", pool[i][3]) for kind, i in sequence]
    assert got == [(kind, None if kind == "web" else p) for kind, p in expected]
    assert all(pool[i][:2] == dutch[pool[i][3]] for kind, i in sequence if kind == "flan")

    # Written as F's builder writes a mix (here with the trainer's shuffle replaced by a fixed permutation)
    pool_rows = [(x, y, s, 0) for x, y, s, _ in pool]
    order = np.random.default_rng(0).permutation(len(sequence))
    path = tmp_path / "mix.parquet"
    flan_select.write_mix(path, sequence, web_texts, pool_rows, order, "test-build")
    table = pq.read_table(path).sort_by("position")
    web_in = [t for _, t, _, _, s in rows if s == "culturax"]
    web_out = [t for t, s in zip(table.column("text").to_pylist(), table.column("source").to_pylist()) if s == "culturax"]
    assert [t.encode() for t in web_out] == [t.encode() for t in web_in]
    assert pq.ParquetFile(path).schema_arrow.metadata[flan_select.METADATA_KEY] == b"test-build"


def test_dutch_mix_refuses_rows_out_of_order_and_dutch_rows_without_a_place():
    rows = _mix_rows()
    with pytest.raises(ValueError, match="position order"):
        fn.dutch_sequence(rows[1:] + rows[:1], {})
    with pytest.raises(ValueError, match="no FLAN row"):
        fn.dutch_sequence(rows, {0: ("instructie", "antwoord")})


def test_a_restart_translates_only_the_shards_that_are_not_stored():
    ids = list(range(10))
    texts = [f"segment {i}" for i in ids]
    shards = tr.shards_of(ids, texts, 4)
    assert [name for name, _, _ in shards] == ["shard-00000.parquet", "shard-00001.parquet", "shard-00002.parquet"]
    assert tr.shards_of(ids, texts, 4) == shards
    store, calls = {}, []

    def translate(shard_ids, shard_texts):
        calls.append(list(shard_ids))
        if len(calls) == 2 and not store.get("crashed"):
            store["crashed"] = True
            raise RuntimeError("preempted")
        return {"id": list(shard_ids), "text_nl": [fake_dutch(t) for t in shard_texts]}, 1.0

    def keep(name, rows, seconds):
        store[name] = rows

    with pytest.raises(RuntimeError, match="preempted"):
        tr.translate_pending(shards, set(), translate, keep)
    assert set(store) - {"crashed"} == {"shard-00000.parquet"}
    done = {name for name in store if name.endswith(".parquet")}
    calls.clear()
    assert tr.translate_pending(shards, done, translate, keep) == ["shard-00001.parquet", "shard-00002.parquet"]
    assert calls == [[4, 5, 6, 7], [8, 9]]
    assert tr.translate_pending(shards, set(store) - {"crashed"}, translate, keep) == []


def test_a_shard_with_translations_of_other_segments_is_refused():
    shards = tr.shards_of([1, 2], ["a", "b"], 2)
    with pytest.raises(RuntimeError, match="not those"):
        tr.translate_pending(shards, set(), lambda i, t: ({"id": [2, 1]}, 1.0), lambda *a: None)


class FakeDevice:
    """The device side of decode_all without a model: prompt [a, n] continues with n tokens 1000 a + k, then <end_of_turn>,
    then noise, as a real slot keeps generating past its end."""

    def __init__(self, slots):
        self.programs = [None] * slots
        self.positions = [0] * slots
        self.prefills = 0

    @staticmethod
    def completion(prompt):
        a, n = prompt[0], prompt[1]
        return [1000 * a + k for k in range(n)] + [tr.EOS_IDS[1]]

    def prefill(self, tokens):
        self.prefills += 1
        return list(tokens), np.int32(self.completion(tokens)[0])

    def insert(self, prefix, slot):
        self.programs[slot], self.positions[slot] = self.completion(prefix), 1

    def generate(self):
        out = []
        for s, program in enumerate(self.programs):
            position = self.positions[s]
            out.append(program[position] if program is not None and position < len(program) else 7)
            self.positions[s] += 1
        return np.asarray(out, np.int32)

    @staticmethod
    def fetch(arrays):
        return np.asarray(arrays)


@pytest.mark.parametrize("slots, sync_every", [(1, 1), (3, 4), (8, 16), (40, 8)])
def test_decode_all_gives_every_prompt_its_own_completion(slots, sync_every):
    rng = np.random.default_rng(slots)
    prompts = [[a, int(rng.integers(0, 30))] + [5] * int(rng.integers(0, 6)) for a in range(1, 26)]
    prompts.append([99, 60])  # longer than max_decode
    device = FakeDevice(slots)
    outputs = tr.decode_all(prompts, slots, device, max_decode=40, sync_every=sync_every)
    assert device.prefills == len(prompts)
    for prompt, (tokens, ended) in zip(prompts, outputs):
        expected = FakeDevice.completion(prompt)[:-1]
        if len(expected) + 1 <= 40:
            assert (tokens, ended) == (expected, True)
        else:
            assert (tokens, ended) == (expected[:40], False)


def test_a_completion_that_ends_at_once_is_empty():
    outputs = tr.decode_all([[1, 0], [2, 3]], 1, FakeDevice(1), max_decode=8, sync_every=2)
    assert outputs == [([], True), ([2000, 2001, 2002], True)]


TEMPLATES = tuple(sorted(["Q:", "A:", "Given the question:", "Answer the following question:", "Output:", "Input:",
                          "Given the task definition and input, reply with output.", "OPTIONS:"], key=lambda k: -len(k)))


def test_template_strings_at_line_starts_are_protected_pieces():
    inputs = ("Given the task definition and input, reply with output. In this task, classify.\n\nInput: A dog.\n"
              "Output:")
    instruction, answer, _ = fn.plan_example(inputs, "dog", words, templates=TEMPLATES)
    assert [v for k, v in instruction if k == "T"] == ["Given the task definition and input, reply with output.",
                                                       "Input:", "Output:"]
    assert all("Input:" not in v and "Output:" not in v for v in fn.segments_of(instruction))
    table = {"Given the task definition and input, reply with output.": "Gegeven de taak en de invoer, antwoord.",
             "Input:": "Invoer:", "Output:": "Uitvoer:"}
    nl = fn.render(instruction, {s: f"NL[{s}]" for s in fn.segments_of(instruction)}, table)
    assert nl == "Gegeven de taak en de invoer, antwoord. NL[In this task, classify.]\n\nInvoer: NL[A dog.]\nUitvoer:"
    # Identity translations give the English back, templates and all
    identity = {s: s for s in fn.segments_of(instruction)}
    assert fn.render(instruction, identity, {t: t for t in TEMPLATES}) == inputs


def test_template_strings_chain_and_need_a_line_start():
    text = "Q: Answer the following question: Who?\nA:"
    assert [text[a:b] for a, b in fn.template_spans(text, TEMPLATES)] == ["Q:", "Answer the following question:", "A:"]
    # Not at a line start: content, translated with its sentence
    assert fn.template_spans("He said Q: no.", TEMPLATES) == []
    assert fn.template_spans("Q:Suggest a title.", TEMPLATES) == [(0, 2)]
    # After a list of options the tail starts a line only after a newline
    instruction, _, _ = fn.plan_example("Is it?\nOPTIONS:\n- yes\n- no\nA:", "yes", words, templates=TEMPLATES)
    assert [v for k, v in instruction if k == "T"] == ["OPTIONS:", "A:"]


def test_a_variant_is_chosen_per_example_deterministically():
    variants = ["Gegeven de vraag:", "Bij de vraag:", "Vraag:"]
    chosen = [fn.choose_variant(0, p, "Given the question:", variants) for p in range(3000)]
    assert chosen == [fn.choose_variant(0, p, "Given the question:", variants) for p in range(3000)]
    assert set(chosen) == set(variants) and min(chosen.count(v) for v in variants) > 800
    assert fn.choose_variant(1, 5, "Q:", ["V:"]) == "V:"


def test_the_phrasing_table_is_checked(tmp_path):
    path = tmp_path / "templates_nl.json"
    path.write_text('{"Q:": ["Vraag:", "V:"], "A:": ["Antwoord:"]}')
    assert fn.read_phrasing_table(path, {"Q:", "A:", "Output:"}) == {"Q:": ["Vraag:", "V:"], "A:": ["Antwoord:"]}
    path.write_text('{"Qq:": ["Vraag:"]}')
    with pytest.raises(ValueError, match="not a template string"):
        fn.read_phrasing_table(path, {"Q:"})
    path.write_text('{"Q:": []}')
    with pytest.raises(ValueError, match="one or more"):
        fn.read_phrasing_table(path, {"Q:"})


def test_an_option_label_is_not_a_template_string():
    inputs = "Why did Alex do this?\nA: scare people\nB: become a dictator\nC: give people a future\nA:"
    instruction, answer, info = fn.plan_example(inputs, "A", words, templates=TEMPLATES)
    assert info["options"] == 3 and [v for k, v in instruction if k == "T"] == ["A:"]
    assert fn.render(instruction, {s: s for s in fn.segments_of(instruction)}, {"A:": "Antwoord:"}).endswith(
        "\nA: scare people\nB: become a dictator\nC: give people a future\nAntwoord:")


def test_a_rendering_is_read_where_the_lines_agree():
    assert fn._rendering("Now, understand the problem?", "Teacher: Now, understand the problem? Solve it.",
                         "Docent: Begrijpt u het probleem? Los het op.") == "Begrijpt u het probleem?"
    assert fn._rendering("Q:", "Q: " + "word " * 80, "Een heel lang verhaal zonder dubbele punt " * 5 + ": x") is None


def test_variants_are_classified_by_register():
    assert fn.register_of("Begrijp je het probleem nu?") == "je"
    assert fn.register_of("Begrijpt u het probleem nu?") == "u"
    assert fn.register_of("Beantwoord alstublieft de volgende vraag:") == "u"
    assert fn.register_of("Beantwoord de volgende vraag:") == "neutral"
    assert fn.register_of("Uitvoer:") == "neutral"


def test_the_register_of_an_example_comes_from_its_machine_translation_or_its_hash():
    assert fn.example_register(0, 1, "U krijgt een zin. Geef uw antwoord.") == ("u", "mt")
    assert fn.example_register(0, 1, "Je krijgt een zin, geef jouw antwoord. U?") == ("je", "mt")
    hashed = [fn.example_register(0, p, "Een zin zonder aanspreekvorm.") for p in range(400)]
    assert {how for _, how in hashed} == {"hash"} and 150 < sum(r == "u" for r, _ in hashed) < 250
    assert hashed == [fn.example_register(0, p, "Andere tekst.") for p in range(400)]


def test_only_variants_of_the_register_and_neutral_ones_are_chosen():
    variants = ["Begrijp je het probleem nu?", "Begrijpt u het probleem nu?", "Is het probleem duidelijk?"]
    assert fn.variants_for("u", variants) == (["Begrijpt u het probleem nu?", "Is het probleem duidelijk?"], True)
    assert fn.variants_for("je", ["Begrijpt u het?"]) == (["Begrijpt u het?"], False)
    for p in range(200):
        assert fn.choose_variant(0, p, "Now, understand the problem?", fn.variants_for("u", variants)[0]) != variants[0]


def test_tail_after_sep_is_protected_and_extra_keys_load(tmp_path):
    text = "Head: PersonX eats<sep>Tail: to be full"
    assert [text[a:b] for a, b in fn.template_spans(text, ("Head:", "Tail:"))] == ["Head:", "Tail:"]
    (tmp_path / "top.json").write_text('{"templates": [{"english": "Q:"}, {"english": "Head:"}]}')
    (tmp_path / "extra.json").write_text('["Tail:"]')
    (tmp_path / "table.json").write_text('{"Q:": ["Vraag:"], "Given the question:": ["Gegeven de vraag:"]}')
    keys = fn.load_templates([tmp_path / "top.json", tmp_path / "extra.json", tmp_path / "table.json"])
    assert keys == ("Given the question:", "Head:", "Tail:", "Q:")


def test_head_and_tail_labels_follow_the_terms_of_the_translation():
    def labels(text, position=0):
        return fn.head_tail_labels(0, position, text)

    assert labels("U krijgt een beginzin en een eindzin, gescheiden door <sep>.") == (("Beginzin:", "Eindzin:"), "beginzin/eindzin")
    assert labels("Een inleidende zin en een afsluitende zin.")[0] == ("Inleidende zin:", "Afsluitende zin:")
    assert labels("De kop en de staart van de gebeurtenis.")[0] == ("Kop:", "Staart:")
    assert labels("U krijgt 'Head' en 'Tail'.")[0] == ("Head:", "Tail:")
    assert labels("Het begin en het einde.")[0] == ("Begin:", "Einde:")
    # Beginzin does not count as begin
    assert labels("Een beginzin, en het einde.")[1] == "hash"
    fallback = {labels("Geen paar hier.", p)[0] for p in range(50)}
    assert fallback == {("Beginzin:", "Eindzin:"), ("Kop:", "Staart:")}
    assert "Head and Tail, separated with <sep>." not in fn.load_templates([])


def test_a_variant_named_by_the_translation_wins():
    assert fn.lexical_choice(["Context:", "Achtergrond:"], "Lees de achtergrond en beantwoord de vraag.") == "Achtergrond:"
    assert fn.lexical_choice(["Tekst:", "Tekstfragment:"], "Een tekstfragment uit een tekst.") is None
    assert fn.lexical_choice(["Opties:", "Antwoordmogelijkheden:"], "Kies een antwoord.") is None
    assert fn.lexical_choice(["Probleem:", "Opdracht:"], "In deze opdracht krijgt u een zin.") == "Opdracht:"


def test_the_table_may_hold_an_unprotected_key(tmp_path):
    path = tmp_path / "t.json"
    path.write_text('{"Head and Tail, separated with <sep>.": ["Beginzin en eindzin, gescheiden door <sep>."], "Q:": ["V:"]}')
    assert set(fn.read_phrasing_table(path, {"Q:"})) == {"Q:", "Head and Tail, separated with <sep>."}


# ------------------------------------------------------------------------------------- extract from Open-Orca/FLAN

ENGLISH_ONLY = {"input": ["English"], "output": ["English"], "instruction": ["English"]}
NIV2 = {"task001_en": ENGLISH_ONLY, "task002_fr": {"input": ["French"], "output": ["English"], "instruction": ["English"]},
        "task1612_sick_label_classification": ENGLISH_ONLY}
# (shards, tasks) per folder; cot has fewer shards than the hosts of the tests, so each host reads its own rows of it
FOLDER_TASKS = {"cot_zsopt_data": (1, ["cot_a"]), "dialog_zsopt_data": (3, ["dialog_a"]),
                "flan_zsopt_data": (5, ["rte:2.0.0", "wmt16_translate/de-en", "hellaswag:1.1.0", "anli_r1:0.1.0"]),
                "flan_zsnoopt_data": (2, ["rte:2.0.0"]),
                "niv2_zsopt_data": (4, ["task001_en", "task002_fr", "task1612_sick_label_classification"]),
                "t0_zsopt_data": (8, ["t0_a", "t0_b"]), "t0_zsnoopt_data": (2, ["t0_a"])}


def word_tokenizer(texts, add_special_tokens=True, split_special_tokens=False):
    """A tokenizer of one token per word, called as select_flan calls the real one."""
    assert add_special_tokens is False and split_special_tokens is True
    return {"input_ids": [text.split() for text in texts]}


def flan_shards(rows_per_shard=30):
    """{folder: [table]} like Open-Orca/FLAN: rows of every filter of select_flan, and rows of the same content."""
    folders = {}
    for folder, (n, tasks) in FOLDER_TASKS.items():
        tables = []
        for s in range(n):
            rows = []
            for r in range(rows_per_shard):
                if folder.startswith("t0_") and r == 2:
                    rows.append(rows[0])  # the same task, inputs and targets twice in one shard
                    continue
                task, inputs, targets = tasks[(s + r) % len(tasks)], f"Question {folder} {s} {r} about this?", f"Answer {s} {r}"
                if r % 11 == 3:
                    inputs += " ⁇"
                elif r % 11 == 5:
                    targets = " "
                elif r % 11 == 7:
                    inputs += " word" * 20  # over max_length
                rows.append((task, inputs, targets))
            tables.append(pa.table({"inputs": [x for _, x, _ in rows], "targets": [y for _, _, y in rows],
                                    "_task_name": [t for t, _, _ in rows], "_template_idx": list(range(len(rows))),
                                    "_task_source": ["Src"] * len(rows),
                                    "_template_type": ["zs_opt" if "_zsopt_" in folder else "zs_noopt"] * len(rows)}))
        folders[folder] = tables
    return folders


class FakeSource:
    """flan_select.HubShards without the network: {folder: [table]} as the shards of the revision."""

    def __init__(self, folders):
        self.folders = folders

    def shards(self, folder):
        return [f"{folder}/part.{k}.parquet" for k in range(len(self.folders[folder]))]

    def tables(self, names):
        for name in names:
            folder, part = name.split("/")
            yield name, self.folders[folder][int(part.split(".")[1])]


def test_extract_from_the_hub_takes_the_rows_of_fs_mix_in_their_order(tmp_path):
    folders = flan_shards()
    world, cap, max_length, seed = 2, 20, 20, 42
    for rank in range(world):
        # F's route, as its builder took it: the pool, shuffled; a mix of web documents and FLAN rows until FLAN is a
        # share of the tokens; the mix file in the order the trainer's shuffle undoes; its FLAN rows by extract --mix-list
        selection = argparse.Namespace(rank=rank, world=world, cap=cap, max_length=max_length)
        pool, report = flan_select.select_flan(selection, word_tokenizer, FakeSource(folders), NIV2)
        shuffled = np.random.default_rng([seed, rank]).permutation(len(pool))
        pool = [pool[i] for i in shuffled]
        sequence, used = flan_select.interleave([40] * 20, [p[3] for p in pool], 0.3)
        assert 0 < used < len(pool)
        mix = tmp_path / f"mix-{rank}.parquet"
        order = np.random.default_rng(1).permutation(len(sequence))
        flan_select.write_mix(mix, sequence, [f"web document {i}" for i in range(20)], pool, order, f"build-{rank}")
        (tmp_path / f"list-{rank}.txt").write_text(f"{mix}\n")
        fn.main(["extract", "--rank", str(rank), "--mix-list", str(tmp_path / f"list-{rank}.txt"),
                 "--out", str(tmp_path / f"f-{rank}.parquet")])
        from_mix = pq.read_table(tmp_path / f"f-{rank}.parquet")

        table, manifest = fn.hub_extract(rank, FakeSource(folders), word_tokenizer, NIV2, world, cap, max_length, seed,
                                         rows=used)
        assert table.column_names == from_mix.column_names
        assert table.schema.types == from_mix.schema.types
        for column in ("inputs", "targets", "source"):
            assert table.column(column).to_pylist() == from_mix.column(column).to_pylist()
        assert table.column("host").to_pylist() == [rank] * used
        # The place in the shuffled pool; F's positions count the web documents too
        assert table.column("position").to_pylist() == list(range(used))
        assert from_mix.column("position").to_pylist() != list(range(used))
        assert (manifest["flan"]["pool_examples"], manifest["flan"]["examples"]) == (len(pool), used)
        assert manifest["flan"]["folders"] == report["folders"]
        assert pq.read_metadata(tmp_path / f"f-{rank}.parquet").metadata[flan_select.METADATA_KEY] == f"build-{rank}".encode()
        assert table.schema.metadata[flan_select.METADATA_KEY] == manifest["build"].encode()
    # The made-up shards exercise every filter, the cap, and a folder left before its last shard once every task is at
    # its cap
    dropped = {why for f in report["folders"].values() for why in f["dropped"]}
    assert dropped == {"unknown_character", "empty", "too_long", "language", "gpt_generated", "dumb_leak"}
    assert any(f["tasks_at_cap"] for f in report["folders"].values())
    assert any(f["shards_read"] < f["units_of_this_host"] for f in report["folders"].values())


def test_extract_from_the_hub_defaults_to_the_published_selection(tmp_path, monkeypatch):
    folders = flan_shards()
    monkeypatch.setattr(flan_select, "HubShards", lambda download_dir, ahead=4: FakeSource(folders))
    monkeypatch.setattr(flan_select, "load_tokenizer", lambda name: word_tokenizer)
    niv2 = tmp_path / "niv2.json"
    niv2.write_text(json.dumps({"tasks": NIV2}))
    (tmp_path / "en").mkdir()
    out = tmp_path / "en" / "w1.parquet"
    args = ["extract", "--rank", "1", "--tokenizer", "some/tokenizer", "--download-dir", str(tmp_path / "shards"),
            "--niv2-languages", str(niv2), "--out", str(out)]
    # The published dataset's 1,357,022 English rows over its 4 hosts
    published = fn.PUBLISHED_SELECTION
    assert (published["world"], published["cap"], published["max_length"], published["seed"]) == (4, 2000, 1024, 42)
    assert sum(published["rows"]) == 1_357_022 and len(published["rows"]) == len(published["pool"]) == 4
    # By default host 1 takes the published count, and only from a pool of the published size
    with pytest.raises(SystemExit, match=r"host 1: a pool of \d+ rows, the published selection's has 607,927"):
        fn.main(args)
    # Another selection needs --rows
    with pytest.raises(SystemExit, match="pass --rows"):
        fn.main([*args, "--cap", "20"])
    with pytest.raises(SystemExit, match="extract from Open-Orca/FLAN needs --tokenizer"):
        fn.main(["extract", "--rank", "1", "--out", str(out)])
    assert not out.exists()

    fn.main([*args, "--world", "2", "--cap", "20", "--max-length", "20", "--rows", "all"])
    table = pq.read_table(out)
    manifest = json.loads(out.with_suffix(".json").read_text())
    config = manifest["config"]
    assert (config["rank"], config["world"], config["cap"], config["max_length"], config["seed"]) == (1, 2, 20, 20, 42)
    assert (config["rows"], config["flan_share"], config["tokenizer"]) == (None, None, "some/tokenizer")
    assert config["flan"] == f"{flan_select.FLAN_REPO}@{flan_select.FLAN_REVISION}"
    assert table.num_rows == manifest["flan"]["pool_examples"] == manifest["flan"]["examples"]
    assert pq.read_metadata(out).metadata[flan_select.METADATA_KEY] == manifest["build"].encode()

    fn.main([*args[:-2], "--out", str(tmp_path / "w1-5.parquet"), "--world", "2", "--cap", "20", "--max-length", "20",
             "--rows", "5"])
    assert pq.read_table(tmp_path / "w1-5.parquet").to_pylist() == table.slice(0, 5).to_pylist()
    with pytest.raises(SystemExit, match="rows asked, the pool has"):
        fn.main([*args, "--world", "2", "--cap", "20", "--max-length", "20", "--rows", "100000"])


# ------------------------------------------------------------------------------------------------------------ replan

REPLAN_TASK = "flan/t0_zsopt_data/t0_a"
# 750 words without a sentence boundary: too long for a segment of 512, not for one of 2,048
FACTS = "Facts: " + " - ".join(f"fact{k}: value{k}" for k in range(250))
WORDS_2100 = " ".join(f"term{k}" for k in range(2100))


def replan_rows():
    """(host, position, source, inputs, targets); (0, 2), (1, 3) and (1, 5) are too long for the plan's 512 tokens a
    segment"""
    return [(0, 0, REPLAN_TASK, "Q: Is item 1 red?\nOPTIONS:\n- yes\n- no\nA:", "yes"),
            (1, 1, REPLAN_TASK, "Write a word for 3 things.\nOutput:", "many 3 things"),
            # its answer is a segment of the plan already
            (0, 2, REPLAN_TASK, f"{FACTS}\nWrite a word for 3 things.\nOutput:", "many 3 things"),
            # too long for 2,048 as well
            (1, 3, REPLAN_TASK, f"Q: {WORDS_2100}\nA:", "yes"),
            (0, 4, REPLAN_TASK, "Q: Is item 4 blue?\nOPTIONS:\n- yes\n- no\nA:", "no"),
            # one option a segment of the plan already, the other a new short one
            (1, 5, REPLAN_TASK, f"Q: {FACTS}\nOPTIONS:\n- yes\n- maybe so\nA:", "maybe so")]


class WordTokenizer:
    """A token a word, as the plan's tokenizer is called (_plan_rows, _count_tokens)"""

    def __init__(self, count=str.split):
        self.count = count

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.count(text)}


@pytest.fixture
def word_plan(tmp_path, monkeypatch):
    """The English extracts of replan_rows and their plan from flan_nl.py plan (a word a token, in threads), plan2/;
    set_tokenizer(tokenizer) sets the tokenizer that _load_tokenizer loads from then on"""
    import concurrent.futures

    tokenizer = {"now": WordTokenizer()}

    def load(path, templates=()):
        monkeypatch.setattr(fn, "_TOKENIZER", tokenizer["now"])
        monkeypatch.setattr(fn, "_TEMPLATES", tuple(templates))

    monkeypatch.setattr(fn, "_load_tokenizer", load)
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor)
    en = tmp_path / "en"
    en.mkdir()
    for h in (0, 1):
        rows = [r for r in replan_rows() if r[0] == h]
        pq.write_table(pa.table({"host": [r[0] for r in rows], "position": [r[1] for r in rows], "source": [r[2] for r in rows],
                                 "inputs": [r[3] for r in rows], "targets": [r[4] for r in rows]}), en / f"w{h}.parquet")
    english = [str(en / "w0.parquet"), str(en / "w1.parquet")]
    (tmp_path / "plan2").mkdir()
    fn.main(["plan", "--english", *english, "--tokenizer", "words", "--out", str(tmp_path / "plan2"), "--world", "2",
             "--processes", "1", "--templates", str(ROOT / "templates_nl.json")])
    return tmp_path / "plan2", english, lambda t: tokenizer.update(now=t)


def plan_templates(plan):
    return json.loads((plan / "plan.json").read_text())["templates"]


def read_segments(directory, names):
    rows = []
    for name in names:
        rows += pq.read_table(directory / name).to_pylist()
    return rows


def test_replan_plans_the_too_long_examples_again_and_keeps_every_other_example_and_segment(tmp_path, word_plan):
    plan2, english, _ = word_plan
    before = json.loads((plan2 / "plan.json").read_text())
    assert before["status"] == {"translate": 3, "segment_too_long": 3} and before["max_segment_tokens"] == 512
    plan3 = tmp_path / "plan3"
    fn.main(["replan", "--plan", str(plan2), "--english", *english, "--tokenizer", "words", "--out", str(plan3),
             "--check", "3"])
    old, new = (pq.read_table(p / "examples.parquet") for p in (plan2, plan3))
    assert new.schema == old.schema
    old, new = old.to_pylist(), new.to_pylist()
    replanned = [k for k, r in enumerate(old) if r["status"] == "segment_too_long"]
    assert [(old[k]["host"], old[k]["position"]) for k in replanned] == [(0, 2), (1, 3), (1, 5)]
    assert [r for k, r in enumerate(new) if k not in replanned] == [r for k, r in enumerate(old) if k not in replanned]
    fits, still, options = (new[k] for k in replanned)
    assert still == old[replanned[1]] and (still["instruction"], still["answer"]) == ("[]", "[]")
    assert (fits["status"], fits["answer_kind"]) == ("translate", "text")
    assert (options["status"], options["answer_kind"], options["options"]) == ("translate", "option", 2)

    # The plan's segments keep their ids, files and token counts; the new ones follow, each in the file of its id
    n = before["segments"]
    for w in (0, 1):
        old_rows = pq.read_table(plan2 / f"segments-w{w}.parquet").to_pylist()
        new_rows = pq.read_table(plan3 / f"segments-w{w}.parquet").to_pylist()
        assert new_rows[:len(old_rows)] == old_rows
        assert all(r["id"] >= n and r["id"] % 2 == w for r in new_rows[len(old_rows):])
    every = {r["id"]: r for r in read_segments(plan3, ["segments-w0.parquet", "segments-w1.parquet"])}
    added = sorted(i for i in every if i >= n)
    short, long = (read_segments(plan3, [name]) for name in ("new-segments.parquet", "new-segments-long.parquet"))
    assert sorted(r["id"] for r in short + long) == added == list(range(n, len(every)))
    assert [(r["text"], r["tokens"]) for r in short] == [("maybe so", 2)]
    assert [(r["text"], r["tokens"]) for r in long] == [(f"{FACTS}\nOPTIONS:", 751), (f"{FACTS}\nWrite a word for 3 things.", 756)]
    # The answer of (0, 2) is the plan's segment of (1, 1), with its id; so is the option "yes" of (1, 5)
    segment_of = {r["text"]: r["id"] for r in every.values()}
    assert json.loads(fits["answer"]) == [["S", segment_of["many 3 things"]]] and segment_of["many 3 things"] < n
    assert segment_of["yes"] < n and ["S", segment_of["yes"]] in json.loads(options["instruction"])
    assert json.loads(options["answer"]) == [["S", segment_of["maybe so"]]]
    for example, english in ((fits, replan_rows()[2][3]), (options, replan_rows()[5][3])):
        instruction = [(k, every[v]["text"] if k == "S" else v) for k, v in json.loads(example["instruction"])]
        assert fn.render(instruction, {t: t for t in segment_of}, {t: t for t in plan_templates(plan2)}) == english

    after = json.loads((plan3 / "plan.json").read_text())
    assert after["status"] == {"translate": 5, "segment_too_long": 1} and after["segments"] == len(every)
    assert after["segment_tokens"] == before["segment_tokens"] + sum(every[i]["tokens"] for i in added)
    assert (after["max_segment_tokens"], after["templates"], after["template_segments"]) == \
        (512, before["templates"], before["template_segments"])
    assert after["replanned"] | {"plan": None} == {
        "plan": None, "status": "segment_too_long", "max_segment_tokens": 2048, "tokenizer": "words", "checked": 3,
        "examples": 3, "status_after": {"translate": 2, "segment_too_long": 1}, "new_segments": 3,
        "new_segment_tokens": 2 + 751 + 756, "new_segments_long": 2, "longest_new_segment": 756,
        "first_new_id": n, "ids": ["w0-2", "w1-3", "w1-5"]}

    # A plan made by replan is not planned again, a limit must be larger than the plan's, the output a new directory
    with pytest.raises(SystemExit, match="was made by replan"):
        fn.main(["replan", "--plan", str(plan3), "--english", *english, "--tokenizer", "words", "--out", str(tmp_path / "p4")])
    with pytest.raises(SystemExit, match="not above the plan's limit of 512"):
        fn.main(["replan", "--plan", str(plan2), "--english", *english, "--tokenizer", "words", "--out", str(tmp_path / "p4"),
                 "--limit", "512"])
    with pytest.raises(SystemExit, match="is not empty"):
        fn.main(["replan", "--plan", str(plan2), "--english", *english, "--tokenizer", "words", "--out", str(plan3)])


def test_replan_stops_when_the_tokenizer_is_not_the_plans(tmp_path, word_plan):
    plan2, english, set_tokenizer = word_plan
    # Two tokens a word: the examples to translate get other token counts, and so the plan's would not hold
    set_tokenizer(WordTokenizer(lambda text: text.split() * 2))
    with pytest.raises(SystemExit, match=r"planned again (is not the plan's|at 512 tokens is)"):
        fn.main(["replan", "--plan", str(plan2), "--english", *english, "--tokenizer", "other", "--out", str(tmp_path / "p3")])
    assert not (tmp_path / "p3").exists()


def test_assembled_over_the_new_plan_only_the_examples_planned_again_change(tmp_path, word_plan):
    plan2, english, _ = word_plan
    plan3 = tmp_path / "plan3"
    fn.main(["replan", "--plan", str(plan2), "--english", *english, "--tokenizer", "words", "--out", str(plan3)])

    def dutch(text):
        words = {"is": "is", "item": "voorwerp", "red": "rood", "blue": "blauw", "yes": "ja", "no": "nee",
                 "write": "schrijf", "a": "een", "word": "woord", "for": "voor", "things": "dingen", "many": "veel"}
        return " ".join(words.get(w.lower().strip("?.:"), w) for w in text.split())

    def shard(directory, rows):
        directory.mkdir(parents=True)
        pq.write_table(pa.table({"id": [r["id"] for r in rows], "text_nl": [dutch(r["text"]) for r in rows],
                                 "ended": [True] * len(rows), "tokens_in": [r["tokens"] for r in rows],
                                 "tokens_out": [r["tokens"] for r in rows]}), directory / "shard-00000.parquet")

    shard(tmp_path / "tr" / "g4", read_segments(plan2, ["segments-w0.parquet", "segments-w1.parquet"]))
    shard(tmp_path / "tr" / "g4-long", read_segments(plan3, ["new-segments.parquet", "new-segments-long.parquet"]))
    out = {}
    for name, plan, translations in (("nl", plan2, tmp_path / "tr" / "g4"), ("nlb", plan3, tmp_path / "tr")):
        out[name] = tmp_path / name
        out[name].mkdir()
        fn.main(["assemble", "--english", *english, "--plan", str(plan), "--translations", str(translations),
                 "--out", str(out[name]), "--world", "2", "--processes", "1",
                 "--templates-nl", str(ROOT / "templates_nl.json")])
    rows = {name: {h: pq.read_table(d / f"w{h}.parquet").to_pylist() for h in (0, 1)} for name, d in out.items()}
    for h, position in ((0, 2), (1, 5)):
        assert [r for r in rows["nlb"][h] if r["position"] != position] == rows["nl"][h]
    added = {(h, r["position"]): r for h in (0, 1) for r in rows["nlb"][h] if (h, r["position"]) in ((0, 2), (1, 5))}
    assert sorted(added) == [(0, 2), (1, 5)]
    assert added[(0, 2)]["inputs"].startswith("Facts: fact0: value0 - ") and added[(0, 2)]["targets"] == "veel 3 dingen"
    assert added[(1, 5)]["inputs"].endswith("\n- ja\n- maybe so\nA:") and added[(1, 5)]["targets"] == "maybe so"
    reports = {name: json.loads((d / "report.json").read_text())["hosts"] for name, d in out.items()}
    assert (reports["nl"]["0"]["dropped:segment_too_long"], reports["nl"]["1"]["dropped:segment_too_long"]) == (1, 2)
    assert "dropped:segment_too_long" not in reports["nlb"]["0"] and reports["nlb"]["1"]["dropped:segment_too_long"] == 1
    for h in ("0", "1"):
        assert reports["nlb"][h]["translated"] == reports["nl"][h]["translated"] + 1
