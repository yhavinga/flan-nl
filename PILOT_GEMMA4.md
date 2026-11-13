# Pilot: FLAN-nl with Gemma 4 31B-it instead of TranslateGemma 27B

The question: should the FLAN-nl segments be translated again with Gemma 4 31B-it (`google/gemma-4-31B-it` at
revision `842da3794eaa0b77d5f08bae87a17459d91ff475`) instead of TranslateGemma 27B, the model of the published dataset?
The pilot translates a sample of the plan's segments with Gemma 4 and compares the result with TranslateGemma's
published translation of the same segments. `g4_pilot.py` draws the sample, measures and reports;
`translate_gemma4.py` is the translation step, the one a full run would use.

**The decision rule below was fixed on 14 Nov 2026 at 00:17 CET, before any Gemma 4 translation of the pilot
existed.** It is applied as it stands; the report says for each part whether it holds.

## Sample (seed `flan-g4-pilot/0`)

All from `plan2/` (the published plan), segments of examples with status `translate` only.

- **S, the segment sample (~2,000 segments).** Cells: length class (`tiny`, `small`, `medium`, `long`, `xlong`, by the
  plan's token count) x instruction-like or not (the segment looks like an instruction, a question, a request to
  translate, or code: `INSTRUCTION_LIKE`, `CODE_LIKE` in `g4_pilot.py`) x answer kind of the segment's home example
  (`label`, `option`, `text`; the home example is the first example that uses the segment in a seeded order). 67
  segments a cell (all of a smaller cell), at most 2 per task in a cell. The 30 protected template strings are not
  in S. Estimates over S are weighted by cell (weight = segments of the cell in the plan / segments drawn from it), so
  they stand for the plan's segments.
- **E, whole examples (402).** 134 per answer kind, at most one per task, with all their segments and the template
  strings, so that `flan_nl.py assemble` can be run on them with either translation.
- (Corrected at 00:31 CET, before any Gemma 4 translation, the rule unchanged: the plan's empty answer kind is that of
  the examples it does not translate, so the kinds of S and E are `label`, `option` and `text`; S had 4 kinds of 50 a
  cell and E 4 kinds of 100.)
- **T, throughput segments.** Per worker of the v4-32 and per class, random segments of the class (tiny 2,048, small
  1,024, medium 512, long 384, xlong 384), so that every class fills its decode slots for several shards; their
  seconds give the cost of a full run. They are measured like S, unweighted, as a uniform sample within each class.

## Translation

Gemma 4 31B-it in MaxText (fork `yhavinga/maxtext` at `2249474`; its decode is correct while the decode limit is at
most 1,024 tokens, which every class here keeps), bf16, greedy, the model's chat template with an empty thought
channel, the prompt `flan-v1` of `models.yml` (from English into Dutch), the classes and slots of `gemma4-31b-flan`
in `models.yml`. Each worker of a v4-32 is a TPU of its own 4 chips.

## Measures (Gemma 4 and TranslateGemma on the same segments)

- **af** (finished): the translation ended with a stop token within the decode limit, and `publish.decode_loop` does
  not flag it. Also reported apart: not ended (truncated), and loops.
- **chatter**: an introduction or a note (`g4_pilot.chatter`), or `flan_nl.py`'s commentary check.
- **executed** (the segment carried out, answered, completed or continued instead of translated): any of
  `question_lost` (the source ends in a question mark, the output does not), `too_long` (output characters > 2.2 x
  the source's + 20), `too_short` (a source of at least 40 characters, output < 0.4 x its characters),
  `added_lines` (at least 2 more non-empty lines than the source), `answer_marker` (a line of the output starts with
  an answer label such as `Antwoord:` or "Het antwoord is" that the source does not have), `assistant_opening` (the
  output opens as an assistant: "Zeker", "Hier is", "Sure", ... where the source does not). The detector is checked
  by hand on a sample of flagged and unflagged segments of both models; its precision on each model's flagged
  segments (when 20 or more were read) corrects that model's rate.
- **labels and options**: `flan_nl.py assemble` (its checks and repairs) on E with each translation; an example with
  answer kind `label` or `option` fails when assemble leaves it out, or when two different English options of it
  have the same Dutch translation.
- **rows kept**: the share of E that assemble keeps.
- Reported, not part of the rule: length ratio Dutch/English (characters, words), MetricX-24-QE XXL from English
  into Dutch (it favours TranslateGemma: human Dutch references score worse than TranslateGemma on 73-77% of
  segments), the repairs, the template variants and forms of address, the checks per segment.

## Blind material

100 segments of S, 20 per class, half of them instruction-like where the class has them, leaving out segments whose
two translations are equal (counted). Per segment the English source and the two translations as A and B in a
seeded random order, in 4 portions of 25; the key is kept apart. A fresh reviewer gives per segment: the number of
errors of meaning in A and in B, whether A and whether B carries out the task instead of translating it (yes/no),
which is more natural Dutch (A, B, equal), and which is preferred (A, B, equal).

## The decision rule

Gemma 4 replaces TranslateGemma for FLAN-nl when all five hold:

1. **Blind**: Gemma 4's errors of meaning, summed over the blind segments, are at most TranslateGemma's, and Gemma 4
   is preferred in at least 50% of the judgements that are not "equal".
2. **Executed**: over S (weighted, corrected by the hand check), Gemma 4's rate is at most TranslateGemma's + 1.0
   percentage point; and in the blind set the reviewer marks Gemma 4 as executed at most 2 times more than
   TranslateGemma.
3. **Labels and options**: over the examples of E with answer kind `label` or `option`, Gemma 4's share of failed
   examples is at most TranslateGemma's + 0.5 percentage point.
4. **Finished**: over S (weighted), Gemma 4's share of af is at least TranslateGemma's - 0.5 percentage point.
5. **Rows kept**: over E, Gemma 4's share of rows kept by assemble is at least TranslateGemma's - 1.0 percentage
   point.

Why these: every part is relative to TranslateGemma, the translation in use, not an absolute threshold it might not
meet itself. The blind review is the measure of quality (MetricX favours TranslateGemma, so it is reported but not
decisive). Carrying out the task instead of translating it and breaking labels or options are the risks of a chat
model on FLAN, where every segment is part of an instruction; finished translations and kept rows guard against
losing data. The margins allow the noise of samples of this size (1 pp of S is 20 segments, 0.5 pp of ~200
multiple-choice examples is one example) but no systematic loss.
