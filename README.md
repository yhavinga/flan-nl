# FLAN-nl

This repository holds the pipeline that translated a sample of the zero-shot FLAN Collection, as published in
[Open-Orca/FLAN](https://huggingface.co/datasets/Open-Orca/FLAN), into Dutch with TranslateGemma 27B, and published the
result as a Hugging Face dataset (called "the published dataset" below). It selects FLAN rows, splits them into
segments, translates the segments on TPU with MaxText, puts the translations back together, checks them, links every
row to its source row in Open-Orca/FLAN, and builds and uploads the dataset with its dataset card.

## What it does and why

- **Selection** (`flan_select.py`, `flan_nl.py extract`). Rows come from the seven zero-shot folders of Open-Orca/FLAN
  at revision `6845b1b3b53c6d5c5b1e49767ed759df3fc246cc`. The selection is split over 4 hosts. It leaves out tasks that
  overlap with Dutch evaluation data (SICK, DBRD, Belebele, FLORES), tasks with GPT-generated answer options (HellaSwag,
  MOCHA), tasks with text in languages other than English and Dutch (by `niv2_languages.json` for NIv2), rows with
  `⁇` (U+2047), rows with an empty instruction or answer, and rows of more than 1,024 tokens (instruction and answer).
  It keeps at most 2,000 rows per (folder, task). Each host shuffles its rows with a fixed seed and takes the first ones.
- **Tasks that do not survive translation are left out** (`DROPPED_TASKS` in `flan_nl.py`): code, SQL and other
  formal languages; tasks whose label depends on the English form (grammar, spelling, punctuation, word order); word
  and letter games; labels per token (POS tagging, NER) and next-word prediction; translation tasks; tasks about
  English words; relations between bare words, where a word translated without context can lose the sense the answer
  depends on. Six NIv2 tasks whose text is already Dutch are kept as they are, not translated.
- **Segments.** Each instruction and answer is cut into segments, and each distinct segment is translated once. A
  segment of more than 512 tokens (TranslateGemma's tokenizer) is cut at paragraphs, then lines, then sentences.
- **Consistent multiple-choice options.** Options (labelled A/B/C, a/b/c, 1/2/3, i/ii/iii, `Option1:`, or bullet lists
  after a header such as `OPTIONS:`) are segments of their own; labels and separators are kept as they are. An answer
  that is the text of an option gets the same translation as that option, so the Dutch answer is still one of the
  Dutch options. An answer that is a label stays the label.
- **Protected template strings.** 30 fixed strings of FLAN's templates (labels such as `Q:`, `Answer:`,
  `Definition:`, and opening sentences such as `Given the task definition and input, reply with output.`) are not
  machine translated. They are replaced by hand-chosen Dutch variants from `templates_nl.json`; the table's one other
  entry, the sentence `Head and Tail, separated with <sep>.`, is translated in context. A row uses one form of
  address throughout (`je` or `u`: the one its machine translation uses, else one chosen by a hash of the row's id),
  and among the variants of that form the one whose words the translation itself uses, else one chosen by hash. The
  labels `Head:` and `Tail:` follow the terms the translated task definition uses for them.
  `templates_top30.md` lists the 30 most frequent template strings with their frequency and how the models translated
  them; `templates_top30.json` and `templates_extra.json` are the lists the plan protects.
- **Translation** (`translate.py`). TranslateGemma 27B (`google/translategemma-27b-it` at revision
  `7d10f0b72f89a2d0f268cea30727d8b77c0d25c2`), its language model converted to a MaxText checkpoint (`convert.py`),
  runs on TPU v4 in bf16 with greedy decoding and the model's chat template (source language `en`, target `nl`).
  Segments are grouped in five length classes and translated in shards; a stored shard is skipped when a run is
  restarted.
- **Checks at assembly** (`flan_nl.py assemble`). A segment is unusable, and every row with it is left out, when its
  translation did not end within the decode limit, is empty, has an implausible length, contains translator
  commentary, offers alternatives, has letters of a script the source does not have, or is not Dutch (`langid`). Two
  repairs are made: a Devanagari danda (`।`) becomes a period, and a final period the source does not have is removed.
  For extractive answers (the English answer is a stretch of the English instruction), a row is left out when the
  Dutch answer is not a stretch of the Dutch instruction.
- **Publication** (`publish.py`). Every row is linked to its file and row in Open-Orca/FLAN. Two further checks,
  `decode_loop` and `english_left`, leave out translations that repeat themselves or keep a stretch of the English
  (see "Known limitations"). The English text is read for these checks only; it is not written to the dataset.

## Files

| file | purpose |
|---|---|
| `flan_select.py` | the selection of FLAN rows (vendored, unchanged in its logic, from the data builder of the experiment the published dataset comes from) |
| `flan_nl.py` | `extract`, `plan`, `assemble`; also `sample` (a pilot of N examples) and `templates` (how `templates_top30.*` were made) |
| `setup.sh`, `convert.py` | the environment on the TPU VM, and TranslateGemma converted to a MaxText checkpoint |
| `translate.sh`, `translate.py` | the translation on each TPU worker |
| `publish.py` | `link`, `build`, `upload` |
| `niv2_languages.json` | the languages of each NIv2 task, from the task metadata of `allenai/natural-instructions` at the commit it names |
| `templates_top30.json`, `templates_top30.md`, `templates_extra.json`, `templates_nl.json` | the protected template strings and their Dutch variants |
| `tests/` | tests on made-up data; no network, model or TPU |

The comments in the code refer to the experiment the published dataset comes from: "F" is its FLAN arm, whose
training mixes held the FLAN rows that were translated, and "F-nl" its Dutch counterpart. "Host" and "worker" are the 4
hosts of a TPU v4-32.

## Requirements

- **CPU steps** (`extract`, `plan`, `assemble`, `publish.py`): Python 3.10 or later with `pyarrow`, `numpy`,
  `transformers`, `huggingface_hub` and `langid==1.1.6`:
  `pip install pyarrow numpy transformers huggingface_hub langid==1.1.6`. The plan and the assembly run in many
  processes (`--processes`); the published run used 96 processes on a TPU v4 host with about 400 GB of RAM.
- **TPU steps** (`setup.sh`, `convert.py`, `translate.sh`, `translate.py`): a Cloud TPU VM of v4 hosts with 4 chips
  each, `gsutil` and a `gs://` directory for the checkpoint, the plan and the translations. `setup.sh` installs `uv`,
  Python 3.12, MaxText 0.2.5 with JAX for TPU, `langid`, and (for the conversion only) `torch` for CPU, all under
  `/dev/shm/flan_nl`. Each worker translates its own share as a TPU of its own 4 chips. The scripts read the worker
  number from the hostname (`...-w-<N>`), as on Cloud TPU VMs.
- **Hugging Face token.** TranslateGemma is a gated model: accept its terms on Hugging Face, and put a token in
  `~/.cache/huggingface/token` on each machine, for example with `hf auth login`, which reads it from the terminal.
  The scripts never take the token as an argument and never print it; `huggingface_hub` reads it from that file. The
  upload needs a token with write access.

## Steps

Clone this repository on each machine and run the commands from its directory. `en/`, `plan2/`, `nl/`, `links/` and
`hf/` are local directories; `ROOT` is a `gs://` directory of your own.

1. **Extract** (CPU and network, once per host, `--rank` 0 to 3):

   ```
   mkdir -p en
   python flan_nl.py extract --rank 0 --tokenizer <hub id of the tokenizer> --download-dir /tmp/oo-flan --out en/w0.parquet
   ```

   This reads the shards of Open-Orca/FLAN of this host (each is downloaded to `--download-dir` and deleted after it is
   read) and writes `en/w0.parquet` (columns `inputs`, `targets`, `source` = `flan/<folder>/<task>`, `position`,
   `host`) and the manifest of the selection, `en/w0.json`, which `publish.py link` reads.

   The defaults are the parameters of the published selection: `--world 4`, `--cap 2000`, `--max-length 1024`,
   `--seed 42`, and the number of rows each host took (340,016, 339,792, 338,986 and 338,228 for hosts 0 to 3). With
   these defaults `extract` also checks that the host's pool has the size of the published selection's pool (608,571,
   607,927, 593,688 and 592,806 rows) and stops otherwise. `--max-length` counts tokens with `--tokenizer`; the
   published selection used the tokenizer of the experiment (a Llama tokenizer for Dutch with 32,000 tokens), and
   another tokenizer selects other rows near the length limit. For another selection pass `--rows N` (the first N rows
   of the host's shuffled pool) or `--rows all`.

   `position` is the place of the row in the host's shuffled pool (0, 1, 2, ...). In the published dataset the id
   `w<host>-<position>` is the row's position in a training mix of the experiment, which also held web documents;
   those positions cannot be rebuilt from Open-Orca/FLAN. So the ids differ from the published ones, and so do the
   template variants, the form of address and the `Head:`/`Tail:` labels of rows where these were chosen by a hash of
   the id. `--mix-list <file>` takes the FLAN rows of a host's training mix of the experiment instead, with those
   positions; it needs the mix files, which are not published.

2. **Plan** (CPU, one machine; needs TranslateGemma's tokenizer, which `setup.sh` downloads to
   `/dev/shm/flan_nl/tokenizer-27b`):

   ```
   mkdir -p plan2
   python flan_nl.py plan --english en/w0.parquet en/w1.parquet en/w2.parquet en/w3.parquet \
       --tokenizer /dev/shm/flan_nl/tokenizer-27b \
       --templates templates_top30.json templates_extra.json templates_nl.json --out plan2 --processes 64
   ```

   Output: `plan2/examples.parquet` (the pieces of every example and its status: `translate`, `dutch_task`,
   `task:<reason>` or `segment_too_long`), `plan2/segments-w<N>.parquet` (the distinct segments, dealt over the 4
   workers by length) and `plan2/plan.json` (counts). Copy the segment files to `$ROOT/plan2/`.

3. **Environment and checkpoint** (TPU VM, every worker; the conversion on one worker):

   ```
   ROOT=gs://<bucket>/<dir> bash setup.sh            # on every worker
   ROOT=gs://<bucket>/<dir> bash setup.sh convert    # on one worker: $ROOT/translategemma-27b-maxtext
   ```

4. **Translate** (TPU, every worker):

   ```
   ROOT=gs://<bucket>/<dir> bash translate.sh
   ```

   Each worker reads `$ROOT/plan2/segments-w<N>.parquet` and translates class after class (`tiny`, `small`,
   `medium`, `long`, `xlong`) into `$ROOT/translations/27b/w<N>/<class>/shard-NNNNN.parquet` (columns `id`,
   `text_nl`, `ended`, `tokens_in`, `tokens_out`). The log is `~/logs/flan_nl_translate.log`; it ends in `EXIT <code>`.
   The same command resumes after a crash: stored shards are skipped. The published run translated 1,811,525 distinct
   segments (158.9 million source tokens) in about 12 hours on the 4 hosts of a TPU v4-32.

5. **Assemble** (CPU, one machine):

   ```
   mkdir -p out27b nl
   gsutil -m rsync -r $ROOT/translations/27b out27b
   python flan_nl.py assemble --english en/w0.parquet en/w1.parquet en/w2.parquet en/w3.parquet --plan plan2 \
       --translations out27b --templates-nl templates_nl.json --out nl --processes 64
   ```

   Output: `nl/w<N>.parquet` (columns `position`, `source`, `inputs`, `targets`) and `nl/report.json` (per host the
   rows translated and left out by reason, the repairs, and the template variants used). It stops when a segment has
   no translation.

6. **Publish** (CPU and network):

   ```
   python publish.py link --english en --manifests en --out links --download-dir /tmp/oo-flan
   python publish.py build --nl nl --links links --plan plan2 --english en --out hf \
       --repo <owner>/<dataset> --code-url <URL of this repository>
   python publish.py upload --folder hf --repo <owner>/<dataset> --public
   ```

   `link` reads the same shards as `extract` again and gives every row its `flan_file` and `flan_row`. `build` writes
   `hf/data/train-*.parquet`, `hf/README.md` (the dataset card, every number in it taken from the data and the
   reports) and `hf/build_report.json` (the rows each publication check left out, by task and id; not uploaded). It
   stops, before writing anything, when a row has no link, a link is of another task, or the reports do not add up to
   the data, and it refuses an `--out` that is not empty. Read the card before the upload. `upload` creates the
   dataset repo as a public one if it does not exist, and uploads `README.md` and `data/`.

## Outputs

The dataset has one split, `train`, in parquet files under 400 MB, sorted by (`flan_file`, `flan_row`):

| field | type | description |
|---|---|---|
| `id` | string | `w<host>-<position>`; unique |
| `inputs` | string | the instruction, in Dutch |
| `targets` | string | the answer, in Dutch |
| `translated` | bool | false for the rows of the six Dutch NIv2 tasks, kept as they were |
| `task_name` | string | `_task_name` of the source row |
| `task_source` | string | `_task_source` of the source row |
| `folder` | string | the folder of the source file in Open-Orca/FLAN |
| `template_type` | string | `_template_type` of the source row (`zs_opt` or `zs_noopt`) |
| `template_idx` | int64 | `_template_idx` of the source row |
| `flan_revision` | string | the revision of Open-Orca/FLAN |
| `flan_file` | string | the parquet file of the source row in Open-Orca/FLAN |
| `flan_row` | int64 | the index of the source row in that file, from 0 |

Open-Orca/FLAN has no id column; (`flan_revision`, `flan_file`, `flan_row`) identifies the source row.

## Tests

The tests use made-up data and need `pytest`, `numpy` and `pyarrow`; they use no network, model or TPU. The code is
checked with ruff's pycodestyle, pyflakes and isort rules:

```
python -m pytest -q
ruff check --line-length 100 --target-version py310 --select E,F,I,W --ignore E501 .
```

## Licences

- **Code**: Apache License 2.0 (`LICENSE`).
- **Data**: Open-Orca/FLAN is released under CC BY 4.0, and its card says: "The data is subject to all the same
  licensing of the component datasets." The published dataset is released under CC BY 4.0 with the same condition.
- **TranslateGemma**: provided under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Section 3.3 says:
  "Google claims no rights in Outputs you generate using Gemma." Using the model requires accepting these terms. If
  you train a model on the translations, read the definition of "Model Derivatives" in Section 1.1(e) of those terms.

## Known limitations

- **Decode loops.** The model sometimes repeats a word or phrase many times, mostly in long passages (film plots, news
  articles, Wikipedia paragraphs). A segment whose loop runs to the decode limit is left out by the assembly
  (`truncated`); a loop that stops before the limit and keeps the translation within the assembly's length limits
  passes its checks.
- **Long passages left half in English.** The model sometimes keeps a stretch of a long passage in English. `langid`
  takes a text that is mostly Dutch for Dutch, so the assembly's `not_dutch` check does not catch it.
- Both are filtered at publication, by `decode_loop` and `english_left` in `publish.py` (the rules are in the dataset
  card). In the published dataset they left out about 0.6% of the translated rows. Their thresholds were chosen by
  reading random samples of the rows they flag and of the rows just below the thresholds: in those samples nearly all
  rows flagged by `decode_loop` and about 95% of those flagged by `english_left` were failed translations. Rows
  wrongly left out are mostly song lyrics, quotations and titles that the translation kept in English. Failed
  translations below the thresholds remain in the dataset.
- The translations are machine translations and were not reviewed by people; errors of meaning, wrong terms and
  literal renderings remain.
- The dataset card that `publish.py build` writes describes the published dataset: its selection, the experiment it
  comes from, and the template phrasings. Edit it when you change the selection or the pipeline.
- The TPU scripts are written for Cloud TPU VMs of v4 hosts (4 chips and about 400 GB of RAM per host, `/dev/shm` as
  working space, `gsutil`). `MODEL=12b` selects TranslateGemma 12B, which was used in a pilot; the published dataset
  was translated with 27B.
