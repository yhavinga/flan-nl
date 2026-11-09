#!/usr/bin/env python3
"""The Dutch FLAN: which FLAN rows are translated, in which pieces, and how they come back.

Steps (README.md), each a subcommand here except the translation itself (translate.py) and the publication (publish.py):
    extract   per host: this host's FLAN rows, from Open-Orca/FLAN (flan_select.py) or from a training mix of the
              experiment (--mix-list) -> <root>/en/w<rank>.parquet
    plan      on one machine: task filters, the pieces of every example, the distinct segments -> <root>/plan/
    assemble  on one machine: the translations back into examples, the checks -> <root>/nl/w<rank>.parquet, report

"F" in the comments is the FLAN arm of the experiment the published dataset comes from: its training mixes held the
FLAN rows that were translated, and "F-nl" is its Dutch counterpart. A FLAN row is (instruction, answer); its id is
(host, position), position being its place in the host's sequence: in F's training mix (the mix's `position` column),
or in the host's shuffled pool when extract reads Open-Orca/FLAN. An example becomes a list of pieces per side:
literals, kept as they are (labels such as "(B).", separators, whitespace, text without words), and segments,
translated once per distinct string. Multiple-choice options are segments of their own, and an answer that is an
option's text gets that option's translation, so the example stays consistent; an answer that is an option's label
stays the label.

Python 3.10 and the standard library only at import, so the tests run anywhere; the steps import pyarrow, numpy,
transformers and langid when they run.
"""

import hashlib
import json
import os
import re
import unicodedata
from collections import Counter

# Most source tokens (TranslateGemma's tokenizer) of one segment: its training data were "individual sentences and
# text blobs of up to 512 tokens" (TranslateGemma technical report, arXiv 2601.09012v2); longer text is split at
# paragraph, then line, then sentence boundaries.
MAX_SEGMENT_TOKENS = 512

# Tasks whose examples do not survive translation, by reason. Names are FLAN's _task_name (NIv2 with its number).
DROPPED_TASKS = {
    # Code, SQL, logical forms and other formal languages: the text is not English to translate
    "code": {
        "task076_splash_correcting_sql_mistake", "task077_splash_explanation_to_sql", "task107_splash_question_to_sql",
        "task868_cfq_mcd1_explanation_to_sql", "task869_cfq_mcd1_sql_to_explanation",
        "task921_code_x_glue_information_retreival", "task956_leetcode_420_strong_password_check",
        "task126_scan_structured_text_generation_command_action_all", "task127_scan_long_text_generation_action_command_all",
        "task128_scan_structured_text_generation_command_action_short",
        "task129_scan_long_text_generation_action_command_short",
        "task130_scan_structured_text_generation_command_action_long",
        "task131_scan_long_text_generation_action_command_long",
        "task110_logic2text_sentence_generation", "task210_logic2text_structured_text_generation",
        "task211_logic2text_classification", "task212_logic2text_classification",
    },
    # The label depends on the English form: grammar, acceptability, spelling, punctuation, case, fluency, word order
    "english_form": {
        "glue/cola:2.0.0", "task616_cola_classification", "task1346_glue_cola_grammatical_correctness_classification",
        "task1559_blimp_binary_classification", "task1560_blimp_binary_classification",
        "task021_mctaco_grammatical_logical", "task088_identify_typo_verification", "task089_swap_words_verification",
        "task138_detoxifying-lms_classification_fluency", "task132_dais_text_modification",
        "task428_senteval_inversion", "task515_senteval_odd_word_out", "task516_senteval_conjoints_inversion",
        "task673_google_wellformed_query_classification", "task674_google_wellformed_query_sentence_generation",
        "task675_google_wellformed_query_sentence_generation",
        "task1415_youtube_caption_corrections_grammar_correction",
        "task1416_youtube_caption_corrections_incorrect_grammar_classification",
        "task1557_jfleg_answer_generation", "task1558_jfleg_incorrect_answer_generation",
        "task1195_disflqa_disfluent_to_fluent_conversion", "task1622_disfl_qa_text_modication",
        "task1623_disfl_qa_disfluent_question_classification",
        "task1703_ljspeech_textmodification", "task1704_ljspeech_textmodification", "task1705_ljspeech_classification",
        "task1706_ljspeech_classification",
        "fix_punct", "true_case", "word_segment",
    },
    # Word games: rhymes, repetition, letters and strings, lists of random letter strings
    "word_game": {
        "task183_rhyme_generation", "task062_bigbench_repeat_copy_logic",
        "task113_count_frequency_of_letter", "task114_is_the_given_word_longest", "task157_count_vowels_and_consonants",
        "task158_count_frequency_of_words", "task159_check_frequency_of_words_in_sentence_pair",
        "task160_replace_letter_in_a_sentence", "task161_count_words_containing_letter",
        "task162_count_words_starting_with_letter", "task163_count_words_ending_with_letter",
        "task365_synthetic_remove_vowels", "task376_reverse_order_of_words", "task377_remove_words_of_given_length",
        "task378_reverse_words_of_given_length", "task850_synthetic_longest_palindrome", "task1188_count_max_freq_char",
        "task1189_check_char_in_string", "task1316_remove_duplicates_string", "task1148_maximum_ascii_value",
        "task1443_string_to_number", "task079_conala_concat_strings", "task600_find_the_longest_common_substring_in_two_strings",
        "task622_replace_alphabets_in_a_list_by_their_position_in_english_alphabet",
        "task063_first_i_elements", "task064_all_elements_except_first_i", "task078_all_elements_except_last_i",
        "task091_all_elements_from_index_i_to_j", "task099_reverse_elements_between_index_i_and_j",
        "task100_concatenate_all_elements_from_index_i_to_j",
        "task101_reverse_and_concatenate_all_elements_from_index_i_to_j",
        "task267_concatenate_and_reverse_all_elements_from_index_i_to_j",
        "task488_extract_all_alphabetical_elements_from_list_in_order", "task504_count_all_alphabetical_elements_in_list",
        "task506_position_of_all_alphabetical_elements_in_list",
        "task509_collate_of_all_alphabetical_and_numerical_elements_in_list_separately",
        "task523_find_if_numbers_or_alphabets_are_more_in_list", "task605_find_the_longest_common_subsequence_in_two_lists",
        "task636_extract_and_sort_unique_alphabets_in_a_list",
        "task755_find_longest_substring_and_replace_its_sorted_lowercase_version_in_both_lists",
        "task756_find_longert_substring_and_return_all_unique_alphabets_in_it",
        "task1542_every_ith_element_from_starting", "task1551_every_ith_element_from_kth_element",
    },
    # Labels per token, or the next or missing English word
    "token_level": {
        "task583_udeps_eng_coarse_pos_tagging", "task584_udeps_eng_fine_pos_tagging",
        "task1167_penn_treebank_coarse_pos_tagging", "task1168_brown_coarse_pos_tagging", "task155_count_nouns_verbs",
        "task585_preposition_classification", "task388_torque_token_classification", "task610_conllpp_ner",
        "task963_librispeech_asr_next_word_prediction", "task964_librispeech_asr_text_auto_completion",
        "task965_librispeech_asr_missing_word_prediction", "task603_wikitext-103_fill_in_the_blank", "lambada:1.0.0",
        # POS tags of the words of an English question, under a QA name
        "task345_hybridqa_answer_generation", "task346_hybridqa_classification",
        "task347_hybridqa_incorrect_answer_generation", "task382_hybridqa_answer_generation",
    },
    # Translation, tokenization, or text in other languages
    "translation": {
        "task547_alt_translation_entk_en", "task560_alt_translation_en_entk", "task874_opus_xhosanavy_sr",
        "task681_hope_edi_malayalam_text_classification",
    },
    # About English words: their senses, the language itself, overlapping words
    "english_words": {
        "super_glue/wic:1.0.2", "task625_xlwic_true_or_false_answer_generation",
        "task626_xlwic_sentence_based_on_given_word_sentence_generation",
        "task627_xlwic_word_with_same_meaning_sentence_generation",
        "task628_xlwic_word_with_different_meaning_sentence_generation",
        "task225_english_language_answer_generation", "task226_english_language_answer_relevance_classification",
        "task039_qasc_find_overlapping_words",
    },
    # Relations between bare words: out of context a word is translated in one sense, and the relation the answer
    # rests on can be gone ("outing, bacon, blt, club, submarine" became "uitje, spek, BLT-sandwich, club,
    # onderzeeër", where the sandwich sense of club and sub is lost; the main session, 8 Nov)
    "word_relations": {
        "task141_odd-man-out_classification_category", "task142_odd-man-out_classification_no_category",
        "task143_odd-man-out_classification_generate_category",
        "task1152_bard_analogical_reasoning_causation", "task1153_bard_analogical_reasoning_affordance",
        "task1154_bard_analogical_reasoning_travel", "task1155_bard_analogical_reasoning_trash_or_treasure",
        "task1156_bard_analogical_reasoning_tools", "task1157_bard_analogical_reasoning_rooms_for_containers",
        "task1158_bard_analogical_reasoning_manipulating_items", "task1159_bard_analogical_reasoning_containers",
        "task1418_bless_semantic_relation_classification", "task1429_evalution_semantic_relation_classification",
        "task1505_root09_semantic_relation_classification", "task1508_wordnet_antonyms", "task1509_evalution_antonyms",
        "task1510_evalution_relation_extraction", "task1582_bless_hypernym_generation",
        "task1583_bless_meronym_classification", "task1584_evalution_meronym_classification",
        "task1585_root09_hypernym_generation",
    },
}
# The six Dutch tasks of F's pool (experiments/flan-decay/niv2_languages.json): their text is Dutch already and only the
# task definition is English; they stay as in F
DUTCH_TASKS = {
    "task1133_xcsr_nl_commonsense_mc_classification", "task409_mickey_nl_sentence_perturbation_generation",
    "task1543_conll2002_parts_of_speech_tagging_answer_generation",
    "task1544_conll2002_named_entity_recognition_answer_generation",
    "task1545_conll2002_person_name_extraction_answer_generation",
    "task1546_conll2002_location_name_extraction_answer_generation",
}
_REASON = {task: reason for reason, tasks in DROPPED_TASKS.items() for task in tasks}
assert len(_REASON) == sum(map(len, DROPPED_TASKS.values())), "a task is in two drop lists"


def task_of(source):
    """'flan/<folder>/<task>' -> task"""
    parts = source.split("/", 2)
    if len(parts) != 3 or parts[0] != "flan":
        raise ValueError(f"not a FLAN source: {source!r}")
    return parts[2]


def task_drop_reason(task):
    return _REASON.get(task)


# ---------------------------------------------------------------------------------------------------------- options

_ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]
_FAMILIES = {
    "upper": [chr(c) for c in range(ord("A"), ord("Z") + 1)],
    "lower": [chr(c) for c in range(ord("a"), ord("z") + 1)],
    "digit": [str(n) for n in range(1, 21)],
    "roman": _ROMAN,
    "ROMAN": [r.upper() for r in _ROMAN],
}
# How a label is written around its letter, number or numeral, as FLAN's templates write them
_STYLES = ["({}).", "({})", "[{}].", "[{}]", "{}).", "{})", "{}.", "{}:", "Option{}:", "Option {}:"]
# The text right after these introduces a list of bullets rather than labels (FLAN's "OPTIONS:\n- yes\n- no")
_LIST_HEADER = re.compile(r"(?:OPTIONS|Options|Options are|Choices|Possible answers|Pick from|Choose your answer from|"
                          r"Available options|Available choices|Pick your answer from|Choose from|Select from|"
                          r"Answer options|Possible choices|[Pp]ick from the following\.|"
                          r"Select from the following\.)[:]?[ \t]*\n?[ \t]*")
_BULLETS = ["[+]", "[-]", "--", "-", "+", "*", "•"]
_MAX_OPTION_CHARS = 400


def _label_pattern(style, token):
    return re.escape(style.format(token))


def _labelled_run(text, family, style):
    """[(label start, label end), ...] of the longest run of this family's labels in order, or []."""
    tokens = _FAMILIES[family]
    best = []
    # Most instructions have no list of this kind: a plain substring test spares the regular expressions
    if style.format(tokens[0]) not in text or style.format(tokens[1]) not in text:
        return best
    first = re.compile(r"(?:(?<=\s)|(?<=^)|(?<=;))" + _label_pattern(style, tokens[0]) + r"(?=\s)")
    for match in first.finditer(text):
        run = [(match.start(), match.end())]
        for token in tokens[1:]:
            pattern = re.compile(r"(?<=\s)" + _label_pattern(style, token) + r"(?=\s)")
            following = pattern.search(text, run[-1][1])
            if following is None:
                break
            between = text[run[-1][1]:following.start()]
            # Labels of one list are close together, and a paragraph break ends the list
            if len(between) > _MAX_OPTION_CHARS or "\n\n" in between.strip("\n"):
                break
            run.append((following.start(), following.end()))
        if len(run) >= 2 and len(run) >= len(best):
            best = run
    return best


def _bullet_run(text):
    """[(bullet start, bullet end), ...] of a bullet list right after a list header, or []."""
    best = []
    for header in _LIST_HEADER.finditer(text):
        start = header.end()
        for bullet in _BULLETS:
            if not text.startswith(bullet, start):
                continue
            newline_list = "\n" in header.group(0)
            spans = [(start, start + len(bullet))]
            pattern = re.compile(r"(?<=\s)" + re.escape(bullet) + r"(?=\S| )")
            position = spans[-1][1]
            while True:
                end_of_line = text.find("\n", position)
                following = pattern.search(text, position)
                if following is None:
                    break
                if newline_list:
                    # One bullet per line: the next must start the next line, after its indentation
                    if end_of_line == -1 or text[end_of_line + 1:following.start()].strip(" \t") != "":
                        break
                elif end_of_line != -1 and following.start() > end_of_line:
                    break
                if following.start() - position > _MAX_OPTION_CHARS:
                    break
                spans.append((following.start(), following.end()))
                position = following.end()
            if len(spans) >= 2 and len(spans) > len(best):
                best = spans
    return best


def find_options(text):
    """The multiple-choice options of an instruction: [(label, label start, option start, option end)] or [].

    The option's text runs from its label to the next label; the last one to the end of its line. Trailing separators
    (";", ",", "." and whitespace) are not part of it, so "(A). negative." has the option "negative".
    """
    candidates = []
    for family in _FAMILIES:
        for style in _STYLES:
            run = _labelled_run(text, family, style)
            if run:
                candidates.append(run)
    bullets = _bullet_run(text)
    if bullets:
        candidates.append(bullets)
    if not candidates:
        return []
    # The most labels win; among equals the latest list, which is where the templates put the options
    run = max(candidates, key=lambda r: (len(r), r[0][0]))
    options = []
    for k, (label_start, label_end) in enumerate(run):
        if k + 1 < len(run):
            end = run[k + 1][0]
        else:
            end = text.find("\n", label_end)
            end = len(text) if end == -1 else end
        start = label_end
        while start < end and text[start] in " \t":
            start += 1
        stop = end
        while stop > start and text[stop - 1] in " \t\n;,.":
            stop -= 1
        if stop <= start:
            return []
        options.append((text[label_start:label_end], label_start, start, stop))
    return options


# --------------------------------------------------------------------------------------------------------- segments

_WORD = re.compile(r"[^\W\d_]{2,}")
# An answer such as "C", "(ii).", "[b].", "2)": a label, whatever the options of its instruction parsed to
_LABEL_ANSWER = re.compile(r"^[(\[]?(?:[A-Za-z]|\d{1,2}|[ivxIVX]{1,5})[)\]]?[.):]*$")


def needs_translation(text):
    """A word of two letters or more; numbers, labels and lists of numbers stay as they are."""
    return _WORD.search(text) is not None


class SegmentTooLong(ValueError):
    pass


def _split_long(text, count_tokens, limit):
    """text as parts of at most `limit` tokens, cut at paragraphs, then lines, then sentences: [(kind, str)], the
    separators between the parts as literals. Raises SegmentTooLong for a sentence longer than limit."""
    if count_tokens(text) <= limit:
        return [("S", text)]
    for boundary in (r"\n\s*\n", r"\n", r"(?<=[.!?])\s+"):
        parts = re.split(f"({boundary})", text)  # text, separator, text, ..., text
        if len(parts) == 1:
            continue
        out, current = [], parts[0]
        for i in range(1, len(parts), 2):
            separator, part = parts[i], parts[i + 1]
            if count_tokens(current + separator + part) <= limit:
                current += separator + part
            else:
                out += [("S", current), ("L", separator)]
                current = part
        out.append(("S", current))
        result = []
        for kind, piece in out:
            result.extend(_split_long(piece, count_tokens, limit) if kind == "S" else [(kind, piece)])
        return result
    raise SegmentTooLong(f"{count_tokens(text)} tokens without a sentence boundary: {text[:80]!r}")


def _wrap(text):
    """One part as pieces: the whitespace around it is literal, and so is the part without a word."""
    stripped = text.strip()
    lead = text[:len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()):] if stripped else ""
    return [("L", lead), ("S" if needs_translation(stripped) else "L", stripped), ("L", trail)]


def text_pieces(text, count_tokens, limit=MAX_SEGMENT_TOKENS, splits=None):
    """One stretch of text as pieces. splits, a list, gets the number of extra segments a long text was cut into."""
    if not needs_translation(text):
        return [("L", text)] if text else []
    stripped = text.strip()
    parts = _split_long(stripped, count_tokens, limit)
    if splits is not None and len(parts) > 1:
        splits.append(sum(kind == "S" for kind, _ in parts) - 1)
    pieces = [("L", text[:len(text) - len(text.lstrip())])]
    for kind, part in parts:
        pieces.extend(_wrap(part) if kind == "S" else [("L", part)])
    pieces.append(("L", text[len(text.rstrip()):]))
    return merge_literals(pieces)


def merge_literals(pieces):
    out = []
    for kind, value in pieces:
        if not value:
            continue
        if kind == "L" and out and out[-1][0] == "L":
            out[-1] = ("L", out[-1][1] + value)
        else:
            out.append((kind, value))
    return out


def _strip_end(text):
    return text.rstrip(" \t\n;,.")


def _template_ends(text, end, key):
    return key[-1] in ":.?!" or end == len(text) or text[end].isspace()


def template_spans(text, keys, at_line_start=True):
    """[(start, end)] of the template strings keys (longest first) at the line starts of text, right after <sep>
    (ATOMIC's "Head: ...<sep>Tail: ..."), and right after another one ("Q: Answer the following question: ...").
    at_line_start: whether text itself starts a line."""
    spans = []
    if not keys:
        return spans
    starts = ([0] if at_line_start else []) + [m.end() for m in re.finditer(r"\n|<sep>", text)]
    for position in starts:
        while True:
            while position < len(text) and text[position] in " \t":
                position += 1
            key = next((k for k in keys if text.startswith(k, position) and _template_ends(text, position + len(k), k)),
                       None)
            if key is None:
                break
            spans.append((position, position + len(key)))
            position += len(key)
    return spans


def instruction_pieces(text, count_tokens, limit, splits, templates, at_line_start):
    """text_pieces, with every template string of templates as a piece ("T", string) of its own: the assembly gives it
    the Dutch of the phrasing table instead of a machine translation (templates_nl.json)."""
    pieces, cursor = [], 0
    for start, end in template_spans(text, templates, at_line_start):
        pieces.extend(text_pieces(text[cursor:start], count_tokens, limit, splits))
        pieces.append(("T", text[start:end]))
        cursor = end
    pieces.extend(text_pieces(text[cursor:], count_tokens, limit, splits))
    return pieces


def plan_example(inputs, targets, count_tokens, limit=MAX_SEGMENT_TOKENS, templates=()):
    """(instruction pieces, answer pieces, info) of one FLAN example. info: options found, how the answer was mapped.
    templates: the template strings to protect, longest first (template_spans)."""
    options = find_options(inputs)
    pieces, position, splits = [], 0, []
    for k, (label, label_start, start, stop) in enumerate(options):
        before = inputs[position:label_start]
        # Before the first label is the question; between two options only the separators the option left out
        pieces.extend(instruction_pieces(before, count_tokens, limit, splits, templates, True) if k == 0
                      else [("L", before)])
        pieces.append(("L", inputs[label_start:start]))
        pieces.extend(text_pieces(inputs[start:stop], count_tokens, limit, splits))
        position = stop
    tail = inputs[position:]
    if options:
        # What closed the last option (".", ";") stays as it is; the text after it is translated
        separators = len(tail) - len(tail.lstrip(" \t\n;,."))
        pieces.append(("L", tail[:separators]))
        position += separators
        tail = tail[separators:]
    pieces.extend(instruction_pieces(tail, count_tokens, limit, splits, templates,
                                     position == 0 or inputs[position - 1] == "\n"))
    instruction = merge_literals(pieces)

    answer = targets.strip()
    lead = targets[:len(targets) - len(targets.lstrip())]
    trail = targets[len(targets.rstrip()):]
    labels = {_strip_end(label).strip("()[]") for label, _, _, _ in options}
    option_texts = {inputs[start:stop] for _, _, start, stop in options}
    if _LABEL_ANSWER.match(answer) or _strip_end(answer).strip("()[]") in labels:
        core, how = [("L", answer)], "label"
    elif _strip_end(answer) in option_texts and _strip_end(answer):
        option = _strip_end(answer)
        core, how = text_pieces(option, count_tokens, limit) + [("L", answer[len(option):])], "option"
    else:
        core, how = text_pieces(answer, count_tokens, limit, splits), "text"
    answer_pieces = merge_literals([("L", lead)] + core + [("L", trail)])
    return instruction, answer_pieces, {"options": len(options), "answer": how, "splits": sum(splits)}


def segments_of(pieces):
    return [value for kind, value in pieces if kind == "S"]


def render(pieces, translation, template_text=None):
    """The text of pieces: segments replaced by translation[segment], template strings by template_text[string]."""
    out = []
    for kind, value in pieces:
        out.append(value if kind == "L" else translation[value] if kind == "S" else template_text[value])
    return "".join(out)


# In the phrasing table (the hand-chosen Dutch template variants) but translated in context: 27B renders Head and Tail
# differently per example (beginzin/eindzin, kop/staart, ...), consistently within one, and the labels follow what it
# chose (head_tail_labels; decided 8 Nov)
UNPROTECTED = {"Head and Tail, separated with <sep>."}
# (the terms in an example's translated text, its labels for "Head:" and "Tail:"), checked in this order
HEAD_TAIL_PAIRS = [
    (("inleidende zin", "afsluitende zin"), ("Inleidende zin:", "Afsluitende zin:")),
    (("beginzin", "eindzin"), ("Beginzin:", "Eindzin:")),
    (("kop", "staart"), ("Kop:", "Staart:")),
    (("head", "tail"), ("Head:", "Tail:")),
    (("begin", "einde"), ("Begin:", "Einde:")),
]
# Where the text names no pair: one of these, by sha256 of the example's id
HEAD_TAIL_FALLBACK = [("Beginzin:", "Eindzin:"), ("Kop:", "Staart:")]


def _has_phrase(text, phrase):
    """phrase as whole words in text, both already casefolded."""
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def head_tail_labels(host, position, machine_text):
    """((Head label, Tail label), how) for one example: the pair whose terms its translated text uses, else a seeded one."""
    text = machine_text.casefold()
    for terms, labels in HEAD_TAIL_PAIRS:
        if all(_has_phrase(text, term) for term in terms):
            return labels, "/".join(terms)
    digest = hashlib.sha256(f"{host}:{position}:head-tail".encode()).hexdigest()
    return HEAD_TAIL_FALLBACK[int(digest, 16) % len(HEAD_TAIL_FALLBACK)], "hash"


def lexical_choice(variants, machine_text):
    """The one variant whose words (without the final colon) occur in the example's translated text, or None when
    none or several do: a label then names what the translation itself calls it ("Achtergrond:" after "achtergrond")."""
    text = machine_text.casefold()
    hits = [v for v in variants if _has_phrase(text, v.rstrip(":").strip().casefold())]
    return hits[0] if len(hits) == 1 else None


_JE = {"je", "jij", "jouw", "jou", "jezelf"}
_U = {"u", "uw", "uzelf", "alstublieft"}


def register_of(text):
    """"je", "u" or "neutral": how a Dutch text addresses the reader, by its whole words."""
    words = set(re.findall(r"[^\W\d_]+", text.casefold()))
    if words & _JE:
        return "je"
    if words & _U:
        return "u"
    return "neutral"


def example_register(host, position, machine_text):
    """(register, how) of one example: the one its machine-translated text uses more (u/uw against je/jij/jouw as
    whole words; TranslateGemma mostly writes u), else one chosen by sha256 of the example's id, so that every
    template string of the example is in the same register."""
    words = re.findall(r"[^\W\d_]+", machine_text.casefold())
    u = sum(w in _U - {"alstublieft"} for w in words)
    je = sum(w in _JE for w in words)
    if u != je:
        return ("u" if u > je else "je"), "mt"
    digest = hashlib.sha256(f"{host}:{position}:register".encode()).hexdigest()
    return ("je", "u")[int(digest, 16) % 2], "hash"


def variants_for(register, variants):
    """The variants of that register and the neutral ones; all of them, and False, when there are none."""
    fitting = [v for v in variants if register_of(v) in (register, "neutral")]
    return (fitting, True) if fitting else (list(variants), False)


def choose_variant(host, position, english, variants):
    """One of the hand-chosen Dutch variants of a template string for one example: sha256 of the example's id (host, position)
    and the English string, so a rerun chooses the same, and every occurrence in one example gets the same one."""
    digest = hashlib.sha256(f"{host}:{position}:{english}".encode()).hexdigest()
    return variants[int(digest, 16) % len(variants)]


def read_phrasing_table(path, protected):
    """templates_nl.json: {English template string: [Dutch variant, ...]}. Refused when absent, malformed, or with a
    key the plan did not protect (a typo would otherwise change nothing)."""
    table = json.loads(open(path).read())
    if not isinstance(table, dict):
        raise ValueError(f"{path}: not an object of English strings")
    for english, variants in table.items():
        if english not in protected and english not in UNPROTECTED:
            raise ValueError(f"{path}: {english!r} is not a template string of the plan")
        if not isinstance(variants, list) or not variants or not all(isinstance(v, str) and v.strip() for v in variants):
            raise ValueError(f"{path}: {english!r} needs a list of one or more non-empty Dutch strings")
    return table


# ---------------------------------------------------------------------------------------------------------- checks

# The translator writes typographic quotes and dashes where FLAN has plain ones, and not always the same way twice
_TYPOGRAPHY = str.maketrans({"‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"',
                             "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-"})


def _normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold().translate(_TYPOGRAPHY)
    return re.sub(r"\s+", " ", text).strip()


def contains(haystack, needle):
    """needle as a whole-word stretch of haystack, case and runs of whitespace ignored; an empty needle is never in it."""
    needle = _strip_end(_normalize(needle))
    if not needle:
        return False
    haystack = _normalize(haystack)
    left = r"(?<!\w)" if needle[0].isalnum() else ""
    right = r"(?!\w)" if needle[-1].isalnum() else ""
    return re.search(left + re.escape(needle) + right, haystack) is not None


def answer_lost(inputs_en, targets_en, inputs_nl, targets_nl):
    """Extractive QA: the English answer is in the English instruction, the Dutch answer not in the Dutch one."""
    return contains(inputs_en, targets_en) and not contains(inputs_nl, targets_nl)


# Phrases with which a translator talks about the translation instead of giving it. Not "Let op:" or "Opmerking:":
# those translate "Note that ..." of FLAN's own instructions (pilot of 8 Nov). Only counted when the source has none.
_COMMENTARY = re.compile(
    r"(here is the translation|here's the translation|translation:|translated text|hier is de vertaling|"
    r"vertaling:|de vertaling is|vertaald:|noot van de vertaler|opmerking van de vertaler|vertalersnoot|"
    r"\((?:opmerking|noot|note):|alternatie(?:f|ve)(?: vertaling)?:|als ai|as an ai|"
    r"ik kan (?:dit|deze|de) .{0,30}niet vertalen|i cannot translate|i can't translate)", re.IGNORECASE)
DUTCH_LANGUAGES = {"nl", "af"}  # langid takes short Dutch for Afrikaans now and then; a translator does not write it
LANGID_LANGUAGES = ["nl", "en", "de", "fr", "es", "it", "pt", "af"]
MIN_LANGID_WORDS = 5
# English words that Dutch does not have: langid alone takes names and short Dutch for English (pilot of 8 Nov), so
# an output is only "not Dutch" when it also has English in it
_ENGLISH_WORDS = {"the", "and", "to", "that", "for", "with", "it", "are", "this", "on", "by", "from", "at", "not",
                  "have", "has", "you", "your", "which", "they", "their", "would", "can", "what", "who", "there",
                  "been", "were", "these", "those", "should", "could", "about", "into", "than", "then", "also", "its"}
# TranslateGemma now and then ends a sentence with the Devanagari danda instead of a period ("rundvee।" for
# "cattle.", pilot of 8 Nov); as a cached translation of a frequent answer it would take every example with it
DANDA = "।"
_SCRIPTS = ("DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "TAMIL", "TELUGU", "KANNADA", "MALAYALAM", "SINHALA",
            "THAI", "LAO", "TIBETAN", "MYANMAR", "KHMER", "CJK", "HIRAGANA", "KATAKANA", "HANGUL", "ARABIC", "HEBREW",
            "CYRILLIC", "GREEK", "ARMENIAN", "GEORGIAN", "ETHIOPIC")


def repair_translation(source, output):
    """(output, [repairs]): the two repairs, each counted; everything else that is wrong drops the example.

    "danda": the danda put back as the period it stands for. "final_period": a period at the end that the source does
    not have, which 12B adds to short answers ("Yes" -> "Ja.", "the snake" -> "de slang.", pilot of 8 Nov), so the
    answers keep F's form."""
    repairs = []
    if DANDA in output and DANDA not in source:
        output = output.replace(DANDA, ".")
        repairs.append("danda")
    if output.endswith(".") and not output.endswith("..") and not source.rstrip().endswith((".", "!", "?", ":", ";")):
        output = output[:-1].rstrip()
        repairs.append("final_period")
    return output, repairs


def _foreign_script(source, output):
    for ch in set(output) - set(source):
        name = unicodedata.name(ch, "")
        if ord(ch) > 0x24F and any(script in name for script in _SCRIPTS):
            return True
    return False


def check_translation(source, output, ended, identify_language):
    """Why a segment's translation is unusable, or None. identify_language(text) -> language code."""
    if not ended:
        return "truncated"
    if not output.strip():
        return "empty"
    s, o = len(source), len(output)
    if (s >= 30 and not 0.5 <= o / s <= 2.2) or o > 3 * s + 30:
        return "length_ratio"
    match = _COMMENTARY.search(output)
    if match and not _COMMENTARY.search(source) and "translat" not in source.lower():
        return "commentary"
    # Several translations offered instead of one ("niet overweg kunnen met / geen goede band met", "hut; cabine",
    # "Begeleider/adviseur", and for single words a list of one per line: "ouder\nbejaard\nsenior\n...", 8 Nov)
    short = len(source.split()) <= 4
    if (" / " in output and "/" not in source) or (short and any(c in output and c not in source for c in ";/")) or \
            (short and "\n" not in source and "\n" in output.strip()):
        return "alternatives"
    if _foreign_script(source, output):
        return "foreign_script"
    words = _WORD.findall(output)
    if len(words) >= MIN_LANGID_WORDS and identify_language(output) not in DUTCH_LANGUAGES:
        english = sum(w.lower() in _ENGLISH_WORDS for w in words)
        if english >= 2 or english >= 0.15 * len(words):
            return "not_dutch"
    return None


# --------------------------------------------------------------------------------------------------------- the mix

def dutch_sequence(rows, dutch):
    """F's sequence of one host with every FLAN row replaced in place by its Dutch version; a row without one is gone.

    rows: (position, text, inputs, targets, source) of F's mix in position order. dutch: position -> (inputs, targets).
    Returns (sequence of ("web", i) and ("flan", j), web texts, FLAN pool [(inputs, targets, source, F's position)],
    counts). Web rows keep their text and their order.
    """
    sequence, web_texts, pool = [], [], []
    counts = Counter()
    expected = 0
    for position, text, inputs, targets, source in rows:
        if position != expected:
            raise ValueError(f"row {expected} of the sequence has position {position}: rows must come in position order")
        expected += 1
        if source == "culturax":
            if text is None or inputs is not None or targets is not None:
                raise ValueError(f"web row at position {position} is not a text row")
            sequence.append(("web", len(web_texts)))
            web_texts.append(text)
            counts["web"] += 1
            continue
        counts["flan_in"] += 1
        if position not in dutch:
            counts["flan_dropped"] += 1
            continue
        inputs_nl, targets_nl = dutch[position]
        sequence.append(("flan", len(pool)))
        pool.append((inputs_nl, targets_nl, source, position))
        counts["flan_kept"] += 1
    unknown = set(dutch) - {p for _, _, _, p in pool}
    if unknown:
        raise ValueError(f"{len(unknown)} Dutch rows have no FLAN row of F at their position, e.g. {sorted(unknown)[:5]}")
    return sequence, web_texts, pool, dict(counts)


# ---------------------------------------------------------------------------------------------------------- CLI

def _read_parquet(path, columns=None):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=columns)


def extract(args):
    """This host's FLAN rows: from Open-Orca/FLAN, or with --mix-list from its training mix of F."""
    (extract_mix if args.mix_list else extract_hub)(args)


# The selection of the published dataset, and the defaults of extract, which reproduce it given the tokenizer of the
# experiment: flan_select.select_flan at flan_select.FLAN_REVISION over 4 hosts, at most 2,000 rows per (folder, task),
# rows of at most 1,024 tokens, each host's pool shuffled with seed 42. "rows": the FLAN rows of each host's training
# mix of F, which used its shuffled pool until FLAN was 10% of the mix's tokens; "pool": each host's pool size as the
# builder reported it. With the default rows extract refuses a pool of another size, as another tokenizer or source
# would select other rows.
PUBLISHED_SELECTION = {"world": 4, "cap": 2000, "max_length": 1024, "seed": 42, "flan_share": 0.1,
                       "rows": (340_016, 339_792, 338_986, 338_228), "pool": (608_571, 607_927, 593_688, 592_806)}
NIV2_LANGUAGES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "niv2_languages.json")


def hub_extract(rank, source, tokenizer, niv2_languages, world, cap, max_length, seed, rows=None, expected_pool=None,
                about=None):
    """This host's FLAN rows straight from source (flan_select.HubShards, or a stand-in with its shards() and
    tables()): the pool of flan_select.select_flan in the order F's builder shuffled it, its first `rows` rows (None:
    all). Returns (table, manifest): the table as extract writes it, position being the place in the shuffled pool,
    and the manifest with the selection's report in the form of F's mix manifests, which publish.py link reads.
    expected_pool: the pool size to insist on. about: more entries of the manifest's config."""
    import argparse

    import pyarrow as pa

    import flan_select

    selection = argparse.Namespace(rank=rank, world=world, cap=cap, max_length=max_length)
    pool, report = flan_select.select_flan(selection, tokenizer, source, niv2_languages)
    if expected_pool is not None and len(pool) != expected_pool:
        raise SystemExit(f"host {rank}: a pool of {len(pool):,} rows, the published selection's has {expected_pool:,}: "
                         "another tokenizer or source selects other rows (pass --rows to take them anyway)")
    if rows is not None and rows > len(pool):
        raise SystemExit(f"host {rank}: {rows:,} rows asked, the pool has {len(pool):,}")
    taken = flan_select.shuffled_pool(pool, seed, rank)
    if rows is not None:
        taken = taken[:rows]
    config = {"rank": rank, "world": world, "flan": f"{flan_select.FLAN_REPO}@{flan_select.FLAN_REVISION}",
              "folders": list(flan_select.FLAN_FOLDERS), "cap": cap, "max_length": max_length, "seed": seed, "rows": rows,
              "flan_share": None, **(about or {})}
    build = flan_select.build_id(config)
    table = pa.table({"inputs": pa.array([p[0] for p in taken], pa.string()),
                      "targets": pa.array([p[1] for p in taken], pa.string()),
                      "source": pa.array([p[2] for p in taken], pa.string()),
                      "position": pa.array(range(len(taken)), pa.int64()),
                      "host": pa.array([rank] * len(taken), pa.int32())})
    table = table.replace_schema_metadata({flan_select.METADATA_KEY: build.encode(), b"flan_nl_source": config["flan"].encode()})
    manifest = {"build": build, "config": config, "flan": {"pool_examples": len(pool), "examples": len(taken), **report}}
    return table, manifest


def _rows(value):
    return value if value == "all" else int(value)


def extract_hub(args):
    """This host's FLAN rows from Open-Orca/FLAN: --out and its manifest, the same path ending in .json."""
    import pathlib

    import pyarrow.parquet as pq

    import flan_select

    if not args.tokenizer or not args.download_dir:
        raise SystemExit("extract from Open-Orca/FLAN needs --tokenizer and --download-dir (or --mix-list)")
    if not 0 <= args.rank < args.world:
        raise SystemExit(f"--rank {args.rank} is not in 0..{args.world - 1}")
    about = {"tokenizer": args.tokenizer,
             "niv2_languages": hashlib.sha256(pathlib.Path(args.niv2_languages).read_bytes()).hexdigest()}
    rows, expected_pool = (None if args.rows == "all" else args.rows), None
    if args.rows is None:
        given = {k: getattr(args, k) for k in ("world", "cap", "max_length", "seed")}
        published = {k: PUBLISHED_SELECTION[k] for k in given}
        if given != published:
            raise SystemExit(f"pass --rows: its default is the published selection's count per host, made with "
                             f"{published}, not {given}")
        rows, expected_pool = PUBLISHED_SELECTION["rows"][args.rank], PUBLISHED_SELECTION["pool"][args.rank]
        about["flan_share"] = PUBLISHED_SELECTION["flan_share"]
    niv2_languages = json.loads(pathlib.Path(args.niv2_languages).read_text())["tasks"]
    table, manifest = hub_extract(args.rank, flan_select.HubShards(args.download_dir), flan_select.load_tokenizer(args.tokenizer),
                                  niv2_languages, args.world, args.cap, args.max_length, args.seed, rows, expected_pool, about)
    out = pathlib.Path(args.out)
    pq.write_table(table, out, compression="zstd")
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"extract: host {args.rank}, {manifest['config']['flan']} (build {manifest['build']}): a pool of "
          f"{manifest['flan']['pool_examples']:,} rows, {table.num_rows:,} taken -> {out}, manifest "
          f"{out.with_suffix('.json')}", flush=True)


def extract_mix(args):
    """The FLAN rows of this host's mix of F, with their positions, in position order: the route of the published
    dataset, whose ids are these positions."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    mix = open(args.mix_list).read().split()
    if len(mix) != 1:
        raise SystemExit(f"{args.mix_list} lists {len(mix)} files, F has one mix per host")
    table = _read_parquet(mix[0], ["inputs", "targets", "source", "position"])
    build = pq.ParquetFile(mix[0]).schema_arrow.metadata[b"flan_decay_build"].decode()
    flan = table.filter(pc.not_equal(table.column("source"), "culturax"))
    flan = flan.sort_by("position")
    flan = flan.append_column("host", pa.array([args.rank] * flan.num_rows, pa.int32()))
    meta = {b"flan_nl_source_mix": mix[0].encode(), b"flan_decay_build": build.encode()}
    pq.write_table(flan.replace_schema_metadata(meta), args.out, compression="zstd")
    print(f"extract: host {args.rank}, mix {mix[0]} (build {build}): {table.num_rows:,} rows, "
          f"{flan.num_rows:,} FLAN rows -> {args.out}", flush=True)


_TOKENIZER = None
_TEMPLATES = ()


def _load_tokenizer(path, templates=()):
    global _TOKENIZER, _TEMPLATES
    from transformers import AutoTokenizer
    _TOKENIZER = AutoTokenizer.from_pretrained(path)
    _TEMPLATES = tuple(templates)


def _count_tokens(texts):
    return {t: len(_TOKENIZER(t, add_special_tokens=False)["input_ids"]) for t in texts}


def load_templates(paths):
    """The template strings to protect, longest first, from any number of files: templates_top30.json (its "english"
    entries), templates_extra.json (a list), or a phrasing table (its keys), so that every key of the phrasing table is
    protected."""
    keys = set()
    for path in paths or ():
        data = json.loads(open(path).read())
        if isinstance(data, dict) and "templates" in data:
            keys |= {e["english"] for e in data["templates"]}
        elif isinstance(data, list):
            keys |= set(data)
        elif isinstance(data, dict):
            keys |= set(data)
        else:
            raise ValueError(f"{path}: neither a template list nor a phrasing table")
    return tuple(sorted(keys - UNPROTECTED, key=lambda k: (-len(k), k)))


def _plan_rows(rows):
    """In a worker process: (status, instruction, answer, info, {segment: tokens}) per (source, inputs, targets)."""
    counted = {}

    def count_tokens(text):
        if text not in counted:
            counted[text] = len(_TOKENIZER(text, add_special_tokens=False)["input_ids"])
        return counted[text]

    out = []
    for source, inputs, targets in rows:
        task = task_of(source)
        reason = task_drop_reason(task)
        instruction, answer, info = [], [], {"options": 0, "answer": "", "splits": 0}
        if reason is not None:
            status = f"task:{reason}"
        elif task in DUTCH_TASKS:
            status = "dutch_task"
        else:
            try:
                instruction, answer, info = plan_example(inputs, targets, count_tokens, templates=_TEMPLATES)
                status = "translate"
            except SegmentTooLong:
                status = "segment_too_long"
        tokens = {v: count_tokens(v) for v in segments_of(instruction) + segments_of(answer)}
        out.append((status, instruction, answer, info, tokens))
    return out


def plan(args):
    """Task filters, the pieces of every example and the distinct segments, split over the workers by length."""
    import concurrent.futures

    import pyarrow as pa
    import pyarrow.parquet as pq

    examples = {k: [] for k in ("host", "position", "source", "status", "instruction", "answer", "options", "answer_kind",
                                "splits")}
    counted = {}
    report = Counter()
    chunk = 2000
    templates = load_templates(args.templates)
    with concurrent.futures.ProcessPoolExecutor(args.processes, initializer=_load_tokenizer,
                                                initargs=(args.tokenizer, templates)) as pool:
        # Each template string is also a segment of its own: its machine translation stands in where the phrasing
        # table has no entry for it
        counted.update(pool.submit(_count_tokens, list(templates)).result())
        for path in args.english:
            table = _read_parquet(path)
            columns = {c: table.column(c).to_pylist() for c in ("host", "position", "source", "inputs", "targets")}
            rows = list(zip(columns["source"], columns["inputs"], columns["targets"]))
            planned = [r for part in pool.map(_plan_rows, [rows[k:k + chunk] for k in range(0, len(rows), chunk)])
                       for r in part]
            for h, p, s, (status, instruction, answer, info, tokens) in zip(columns["host"], columns["position"],
                                                                           columns["source"], planned):
                counted.update(tokens)
                for key, value in (("host", h), ("position", p), ("source", s), ("status", status),
                                   ("instruction", instruction), ("answer", answer), ("options", info["options"]),
                                   ("answer_kind", info["answer"]), ("splits", info["splits"])):
                    examples[key].append(value)
                report[status] += 1
            print(f"plan: {path}: {len(examples['host']):,} examples, {len(counted):,} distinct segments so far", flush=True)

    def count_tokens(text):
        return counted[text]

    # Shortest first, dealt round-robin: every worker gets the same mix of lengths, and a shard of consecutive
    # segments has similar lengths, which suits the engine's length buckets
    texts = sorted(counted, key=lambda t: (counted[t], t))
    segment_ids = {t: i for i, t in enumerate(texts)}
    encode = {"L": lambda v: v, "S": lambda v: segment_ids[v], "T": lambda v: v}
    examples["instruction"] = [json.dumps([[k, encode[k](v)] for k, v in p], ensure_ascii=False) for p in examples["instruction"]]
    examples["answer"] = [json.dumps([[k, encode[k](v)] for k, v in p], ensure_ascii=False) for p in examples["answer"]]
    pq.write_table(pa.table(examples), f"{args.out}/examples.parquet", compression="zstd")
    tokens = [count_tokens(t) for t in texts]
    for worker in range(args.world):
        pq.write_table(pa.table({"id": list(range(worker, len(texts), args.world)), "text": texts[worker::args.world],
                                 "tokens": tokens[worker::args.world]}),
                       f"{args.out}/segments-w{worker}.parquet", compression="zstd")
    summary = {"examples": len(examples["host"]), "status": dict(report), "segments": len(texts),
               "segment_tokens": sum(tokens), "answer_kind": dict(Counter(examples["answer_kind"])),
               "with_options": sum(1 for n in examples["options"] if n),
               "examples_split": sum(1 for n in examples["splits"] if n), "extra_segments_from_splits": sum(examples["splits"]),
               "max_segment_tokens": MAX_SEGMENT_TOKENS, "tokenizer": args.tokenizer,
               "templates": list(templates), "template_segments": {t: segment_ids[t] for t in templates},
               "examples_with_templates": sum(1 for p in examples["instruction"] if '["T",' in p)}
    json.dump(summary, open(f"{args.out}/plan.json", "w"), indent=1, ensure_ascii=False)
    print(f"PLAN {json.dumps(summary)}", flush=True)


def sample(args):
    """A plan of N random examples to translate, with all their segments, for the pilot on one worker."""
    import random

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    examples = _read_parquet(f"{args.plan}/examples.parquet")
    examples = examples.filter(pc.equal(examples.column("status"), "translate"))
    chosen = sorted(random.Random(args.seed).sample(range(examples.num_rows), args.n))
    examples = examples.take(chosen)
    needed = {v for column in ("instruction", "answer") for pieces in examples.column(column).to_pylist()
              for kind, v in json.loads(pieces) if kind == "S"}
    needed |= set(json.loads(open(f"{args.plan}/plan.json").read()).get("template_segments", {}).values())
    parts = []
    for worker in range(args.world):
        table = _read_parquet(f"{args.plan}/segments-w{worker}.parquet")
        parts.append(table.filter(pc.is_in(table.column("id"), pa.array(sorted(needed), pa.int64()))))
    segments = pa.concat_tables(parts).sort_by("id")
    pq.write_table(examples, f"{args.out}/examples.parquet", compression="zstd")
    pq.write_table(segments, f"{args.out}/segments-w0.parquet", compression="zstd")
    # The assembly reads the protected template strings from it
    open(f"{args.out}/plan.json", "w").write(open(f"{args.plan}/plan.json").read())
    print(f"sample: {examples.num_rows} examples, {segments.num_rows} segments, "
          f"{sum(segments.column('tokens').to_pylist()):,} tokens -> {args.out}", flush=True)


# ------------------------------------------------------------------------------------------------- template text

# A label at the start of a line ("Q:", "Definition:", "Given the question:", "Detailed Instructions:"), or the first
# sentence of a line ("Given the task definition and input, reply with output."): the fixed text of FLAN's templates
_LABEL = re.compile(r"[A-Z][^\n:.?!]{0,40}?:")
_SENTENCE = re.compile(r"[A-Z][^\n:]{0,160}?[.?!](?=\s|$)")
MAX_LABEL_WORDS = 6
MAX_SENTENCE_WORDS = 16


def template_candidates(text):
    """The fixed-looking strings of an instruction: labels and first sentences at line starts, and again right after
    such a label ("Q: Answer the following question: ..."). A set, so a row counts once per string."""
    found = set()
    starts = [0] + [m.end() for m in re.finditer(r"\n", text)]
    for start in starts:
        position = start
        for _ in range(3):
            while position < len(text) and text[position] in " \t":
                position += 1
            label = _LABEL.match(text, position)
            if label and len(label.group(0).split()) <= MAX_LABEL_WORDS:
                found.add(label.group(0))
            sentence = _SENTENCE.match(text, position)
            if sentence and len(sentence.group(0).split()) <= MAX_SENTENCE_WORDS:
                found.add(sentence.group(0))
            if not label:
                break
            position = label.end()
    return found


def _rendering(english, source, output):
    """How output renders the template string english, if source starts a line with it and the alignment is plain:
    at the start of the segment, or on the same line of an output with as many lines."""
    end = ":" if english.endswith(":") else None

    def head(line):
        if end:
            k = line.find(":")
            return line[:k + 1].strip() if k >= 0 else None
        match = re.match(r"[^\n]*?[.?!](?=\s|$)", line)
        return match.group(0).strip() if match else None

    def plausible(got):
        return got if got and "\n" not in got and len(got) <= 2.5 * len(english) + 20 else None

    if source.startswith(english):
        return plausible(head(output))
    source_lines, output_lines = source.split("\n"), output.split("\n")
    if len(source_lines) == len(output_lines):
        for s, o in zip(source_lines, output_lines):
            k = s.find(english)
            if k < 0:
                continue
            # After as many labels on the line as the source has before the string ("Teacher: Now, understand ...")
            before = s[:k]
            if before.strip() and not before.rstrip().endswith(":"):
                continue
            o = o.lstrip()
            for _ in range(before.count(":")):
                colon = o.find(":")
                if colon < 0:
                    break
                o = o[colon + 1:].lstrip()
            else:
                return plausible(head(o))
    return None


def templates(args):
    """The most frequent fixed template strings over F's FLAN rows, with the 27B and 12B renderings of them."""
    import glob

    rows, tasks, total = Counter(), {}, 0
    for path in args.english:
        table = _read_parquet(path, ["source", "inputs"])
        for source, inputs in zip(table.column("source").to_pylist(), table.column("inputs").to_pylist()):
            total += 1
            for candidate in template_candidates(inputs):
                rows[candidate] += 1
                tasks.setdefault(candidate, set()).add(task_of(source))
    ranked = [s for s, _ in rows.most_common() if len(tasks[s]) >= args.min_tasks][:args.top]
    segments = {}
    for path in sorted(glob.glob(f"{args.plan}/segments-w*.parquet")):
        table = _read_parquet(path, ["id", "text"])
        segments.update(zip(table.column("id").to_pylist(), table.column("text").to_pylist()))
    renderings = {}
    for model, root in (("27b", args.out27b), ("12b", args.out12b)):
        seen = {s: Counter() for s in ranked}
        for path in sorted(glob.glob(f"{root}/**/*.parquet", recursive=True)):
            table = _read_parquet(path, ["id", "text_nl", "ended"])
            for i, out, ended in zip(*(table.column(c).to_pylist() for c in ("id", "text_nl", "ended"))):
                source = segments.get(i)
                if not ended or source is None:
                    continue
                for english in ranked:
                    if english in source:
                        got = _rendering(english, source, out)
                        if got:
                            seen[english][got] += 1
        renderings[model] = seen
    table = [{"english": s, "rows": rows[s], "share_of_flan_rows": rows[s] / total, "tasks": len(tasks[s]),
              "nl_27b": renderings["27b"][s].most_common(3), "nl_12b": renderings["12b"][s].most_common(3)}
             for s in ranked]
    json.dump({"flan_rows": total, "min_tasks": args.min_tasks, "templates": table},
              open(f"{args.out}/top{args.top}.json", "w"), indent=1, ensure_ascii=False)

    def cell(pairs):
        if not pairs:
            return "(not seen)"
        return "; ".join(f"`{t}` ({n})".replace("|", "\\|") for t, n in pairs)

    lines = [f"# The {args.top} most frequent template strings of F's FLAN rows", "",
             f"Over the {total:,} FLAN rows of F's 4 host mixes: labels and first sentences at the start of a line (and",
             "right after such a label), found in at least 3 tasks; rows = FLAN rows that have the string there. The",
             "renderings are how 27B and 12B translated the string inside a segment that starts a line with it (the",
             "segment starts with it, or its line has the same index in source and output); most frequent first, with",
             "counts. 27B: the pilot and its measurement shards; 12B: the pilot and the run so far.", "",
             "| English | rows | share | 27B | 12B |", "|---|---|---|---|---|"]
    for t in table:
        english = t["english"].replace("|", "\\|")
        lines.append(f"| `{english}` | {t['rows']:,} | {100 * t['share_of_flan_rows']:.1f}% | {cell(t['nl_27b'])} | {cell(t['nl_12b'])} |")
    open(f"{args.out}/top{args.top}.md", "w").write("\n".join(lines) + "\n")
    print(f"TEMPLATES {len(table)} of {len(rows):,} candidates -> {args.out}/top{args.top}.md", flush=True)


def _identify(texts):
    import langid
    langid.set_languages(LANGID_LANGUAGES)
    return [langid.classify(t)[0] for t in texts]


def assemble(args):
    """Translations back into examples; per host the Dutch rows and what was dropped why."""
    import concurrent.futures
    import glob

    import pyarrow as pa
    import pyarrow.parquet as pq

    segments = {}
    for path in sorted(glob.glob(f"{args.plan}/segments-w*.parquet")):
        table = _read_parquet(path, ["id", "text"])
        segments.update(zip(table.column("id").to_pylist(), table.column("text").to_pylist()))
    translated = {}
    for path in sorted(glob.glob(f"{args.translations}/**/*.parquet", recursive=True)):
        table = _read_parquet(path, ["id", "text_nl", "ended"])
        for i, out, ended in zip(*(table.column(c).to_pylist() for c in ("id", "text_nl", "ended"))):
            if i in translated:
                raise ValueError(f"segment {i} translated twice ({path})")
            translated[i] = (out, ended)
    missing = set(segments) - set(translated)
    if missing:
        raise SystemExit(f"{len(missing):,} of {len(segments):,} segments have no translation, e.g. {sorted(missing)[:5]}")
    ids = sorted(segments)
    repaired = Counter()
    for i in ids:
        out, fixed = repair_translation(segments[i], translated[i][0])
        translated[i] = (out, translated[i][1])
        repaired.update(fixed)
    # Language ID is the slow part: in processes, in chunks
    with concurrent.futures.ProcessPoolExecutor(args.processes) as pool:
        chunk = 2000
        outputs = [translated[i][0] for i in ids]
        languages = [lang for part in pool.map(_identify, [outputs[k:k + chunk] for k in range(0, len(ids), chunk)])
                     for lang in part]
    language = dict(zip(ids, languages))
    failed = {}
    for i in ids:
        out, ended = translated[i]
        why = check_translation(segments[i], out, ended, lambda _: language[i])
        if why:
            failed[i] = why
    print(f"assemble: {len(ids):,} segments, repaired {dict(repaired)}, {len(failed):,} unusable: "
          f"{dict(Counter(failed.values()))}", flush=True)

    examples = _read_parquet(f"{args.plan}/examples.parquet")
    columns = {c: examples.column(c).to_pylist() for c in examples.column_names}
    english = {}
    for path in args.english:
        table = _read_parquet(path)
        for h, p, x, y in zip(*(table.column(c).to_pylist() for c in ("host", "position", "inputs", "targets"))):
            english[(h, p)] = (x, y)
    translation = {i: translated[i][0] for i in ids}
    plan_summary = json.loads(open(f"{args.plan}/plan.json").read())
    template_segments = plan_summary.get("template_segments", {})
    phrasing = {}
    if template_segments:
        # A plan with protected template strings is assembled with the phrasing table or not at all
        if not args.templates_nl:
            raise SystemExit("the plan protects template strings: --templates-nl (templates_nl.json) is required")
        phrasing = read_phrasing_table(args.templates_nl, set(template_segments))
    variant_rows = {t: Counter() for t in template_segments}
    register_rows, head_tail_rows, decided = Counter(), Counter(), Counter()
    rows = {h: {"position": [], "source": [], "inputs": [], "targets": []} for h in range(args.world)}
    report = {h: Counter() for h in range(args.world)}
    for k in range(len(columns["host"])):
        h, p, status = columns["host"][k], columns["position"][k], columns["status"][k]
        report[h]["flan_rows"] += 1
        # The lists in this file decide, also for a task added after the plan was made (its segments were
        # translated for nothing)
        reason = task_drop_reason(task_of(columns["source"][k]))
        if reason is not None and status == "translate":
            status = f"task:{reason}"
        x, y = english[(h, p)]
        if status == "dutch_task":
            report[h]["kept_dutch_task"] += 1
            out = (x, y)
        elif status != "translate":
            report[h][f"dropped:{status}"] += 1
            continue
        else:
            instruction = [(kind, value) for kind, value in json.loads(columns["instruction"][k])]
            answer = [(kind, value) for kind, value in json.loads(columns["answer"][k])]
            used = {v for kind, v in instruction if kind == "T"}
            # Each template string once per example: a variant of the table in the register of the example, or its
            # machine translation
            machine = " ".join(translation[v] for kind, v in instruction + answer if kind == "S")
            register, how = example_register(h, p, machine)
            template_text, mismatched, how_chosen = {}, 0, Counter()
            if used & {"Head:", "Tail:"}:
                (head, tail), head_tail = head_tail_labels(h, p, machine)
                template_text.update({k: v for k, v in (("Head:", head), ("Tail:", tail)) if k in used})
            for t in used - {"Head:", "Tail:"}:
                if t in phrasing:
                    fitting, ok = variants_for(register, phrasing[t])
                    mismatched += not ok
                    lexical = lexical_choice(fitting, machine) if len(fitting) > 1 else None
                    template_text[t] = lexical or choose_variant(h, p, t, fitting)
                    if len(fitting) > 1:
                        how_chosen["lexical" if lexical else "seeded"] += 1
                else:
                    template_text[t] = translation[template_segments[t]]
            bad = [failed[v] for kind, v in instruction + answer if kind == "S" and v in failed] + \
                  [failed[template_segments[t]] for t in used if t not in phrasing and template_segments[t] in failed]
            if bad:
                report[h][f"dropped:segment_{bad[0]}"] += 1
                continue
            out = (render(instruction, translation, template_text), render(answer, translation, template_text))
            if answer_lost(x, y, *out):
                report[h]["dropped:answer_not_in_instruction"] += 1
                continue
            if not out[0].strip() or not out[1].strip():
                report[h]["dropped:empty"] += 1
                continue
            report[h]["translated"] += 1
            for t in used:
                variant_rows[t][template_text[t] if t in phrasing else "(machine translation)"] += 1
            if used:
                register_rows[f"{register} ({how})"] += 1
                register_rows["rows with a variant outside the register"] += mismatched > 0
                decided.update(how_chosen)
            if used & {"Head:", "Tail:"}:
                head_tail_rows[f"{template_text.get('Head:', '-')} {template_text.get('Tail:', '-')} ({head_tail})"] += 1
        for key, value in (("position", p), ("source", columns["source"][k]), ("inputs", out[0]), ("targets", out[1])):
            rows[h][key].append(value)
    for h in range(args.world):
        pq.write_table(pa.table(rows[h]), f"{args.out}/w{h}.parquet", compression="zstd")
    summary = {"segments": len(ids), "segments_repaired": dict(repaired), "segments_unusable": dict(Counter(failed.values())),
               "hosts": {h: dict(sorted(report[h].items())) for h in range(args.world)},
               "template_variants": {t: dict(c.most_common()) for t, c in variant_rows.items()},
               "template_registers": dict(register_rows), "head_tail_pairs": dict(head_tail_rows.most_common()),
               "variant_decisions": dict(decided)}
    json.dump(summary, open(f"{args.out}/report.json", "w"), indent=1, ensure_ascii=False)
    print(f"ASSEMBLE {json.dumps(summary)}", flush=True)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="The Dutch FLAN of arm F-nl")
    sub = parser.add_subparsers(dest="step", required=True)
    p = sub.add_parser("extract", help="this host's FLAN rows: from Open-Orca/FLAN, or from its training mix of F")
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--out", required=True,
                   help="<root>/en/w<rank>.parquet; from Open-Orca/FLAN also its manifest, <root>/en/w<rank>.json")
    p.add_argument("--tokenizer", help="from Open-Orca/FLAN: the Hugging Face tokenizer whose counts decide --max-length "
                                       "(the experiment's tokenizer reproduces the published selection)")
    p.add_argument("--download-dir", help="from Open-Orca/FLAN: the shards while they are read; each is deleted after")
    p.add_argument("--niv2-languages", default=NIV2_LANGUAGES,
                   help="the languages of the NIv2 tasks (default: niv2_languages.json beside this file)")
    # The defaults are the published dataset's selection (PUBLISHED_SELECTION)
    p.add_argument("--world", type=int, default=PUBLISHED_SELECTION["world"], help="hosts the selection is split over (default: 4)")
    p.add_argument("--cap", type=int, default=PUBLISHED_SELECTION["cap"],
                   help="most rows per (folder, task) over all hosts (default: 2000)")
    p.add_argument("--max-length", type=int, default=PUBLISHED_SELECTION["max_length"],
                   help="most tokens of instruction and answer, plus 3 (default: 1024)")
    p.add_argument("--seed", type=int, default=PUBLISHED_SELECTION["seed"], help="seed of the order of each host's pool (default: 42)")
    p.add_argument("--rows", type=_rows, default=None,
                   help="this host's rows: the first N of its shuffled pool, or 'all' (default: the published dataset's "
                        "count for this host, with a check of the pool's size)")
    p.add_argument("--mix-list", help="instead of Open-Orca/FLAN: a file naming this host's training mix of F (the "
                                      "parquet file of the experiment's builder); its FLAN rows keep their positions")
    p = sub.add_parser("plan", help="filters, pieces and distinct segments")
    p.add_argument("--english", nargs="+", required=True, help="the extract files of all hosts")
    p.add_argument("--tokenizer", required=True, help="TranslateGemma's tokenizer directory")
    p.add_argument("--out", required=True)
    p.add_argument("--world", type=int, default=4)
    p.add_argument("--processes", type=int, default=64)
    p.add_argument("--templates", nargs="*", default=None,
                   help="templates_top30.json, templates_extra.json and/or a phrasing table: template strings to protect")
    p = sub.add_parser("sample", help="a plan of N random examples, for the pilot")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--world", type=int, default=4)
    p = sub.add_parser("templates", help="the most frequent fixed template strings and their renderings")
    p.add_argument("--english", nargs="+", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--out27b", required=True, help="Directory of 27B shards (searched recursively)")
    p.add_argument("--out12b", required=True, help="Directory of 12B shards (searched recursively)")
    p.add_argument("--out", required=True)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--min-tasks", type=int, default=3)
    p = sub.add_parser("assemble", help="the translations back into examples")
    p.add_argument("--english", nargs="+", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--translations", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--world", type=int, default=4)
    p.add_argument("--processes", type=int, default=64)
    p.add_argument("--templates-nl", default=None, help="templates_nl.json, the hand-chosen Dutch variants of the template strings")
    args = parser.parse_args(argv)
    {"extract": extract, "plan": plan, "sample": sample, "templates": templates, "assemble": assemble}[args.step](args)


if __name__ == "__main__":
    main()
