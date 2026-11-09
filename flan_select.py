"""The selection of FLAN rows behind the published dataset: Open-Orca/FLAN at a fixed revision, split over hosts,
filtered, and capped per (folder, task).

Vendored from the data builder of the experiment the published dataset comes from (its `scripts/build_decay_mix.py`,
the version whose sha256 the builder's manifests record, 2f5adb05...). The functions and constants below are copied
from it unchanged, so the selection is the same; what this repository does not use (the web text, the trainer's
stream, the build of a training mix) is left out. Three small pieces come from elsewhere in that builder:
`shuffled_pool` is the shuffle line of its `build()`, `load_tokenizer` the tokenizer setup of its `main()`, and
`add_answer_token` the function of the trainer it imported.

FLAN: Open-Orca/FLAN at a fixed revision, only the zero-shot folders. Host r takes shards r, r + world, ... of a folder
(every shard is a random sample of its folder, measured); a folder of fewer shards than hosts is split by row index.
Filters, in this order: tasks that leak DUMB or the zero-shot eval (sick, dbrd, belebele, flores), tasks with
GPT-generated parts (HellaSwag, MOCHA), tasks with text in other languages than English and Dutch (WMT, para_crawl,
NIv2 by the languages in niv2_languages.json), rows with U+2047, rows with an empty instruction or answer, and examples
of more than max_length tokens. Then at most ceil(cap / world) rows per (folder, task) on this host, in shard order.

The builder shuffled each host's pool with a fixed seed and used its first rows for the host's training mix
(`interleave`, `write_mix`); `flan_nl.py extract` takes the same first rows straight from the shuffled pool.

Reads Hugging Face with the token in ~/.cache/huggingface/token if there is one (huggingface_hub finds it); the token is
never printed.
"""

import concurrent.futures
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

FLAN_REPO = "Open-Orca/FLAN"
# The revision whose shards were measured for the design (2026-11-03) and for this script
FLAN_REVISION = "6845b1b3b53c6d5c5b1e49767ed759df3fc246cc"
# All zero-shot folders of the revision; niv2, cot and dialog have no zsnoopt variant there
FLAN_FOLDERS = ("cot_zsopt_data", "dialog_zsopt_data", "flan_zsopt_data", "flan_zsnoopt_data", "niv2_zsopt_data",
                "t0_zsopt_data", "t0_zsnoopt_data")
FLAN_COLUMNS = ["inputs", "targets", "_task_name", "_template_type"]
# What the tokenizer of T5 put for characters it lacked; FLAN rows with it lost text
UNKNOWN_CHARACTER = "⁇"
LANGUAGES_KEPT = {"English", "Dutch"}
METADATA_KEY = b"flan_decay_build"
WEB_SOURCE = "culturax"
ANSWER_TOKEN = "<answer>"


def task_exclusion(folder, task, niv2_languages):
    """Why a task of FLAN is left out, or None."""
    name = task.lower()
    if any(word in name for word in ("sick", "dbrd")):
        return "dumb_leak"  # task1612-task1615 are SICK: SICK-NL is its translation
    if any(word in name for word in ("belebele", "flores")):
        return "eval_leak"  # Belebele's passages are FLORES-200
    if "hellaswag" in name or "mocha" in name:
        return "gpt_generated"  # HellaSwag's wrong endings (Zellers et al. 2019, App. B), MOCHA's candidates (DPI)
    if folder.startswith("flan_") and (name.startswith("wmt") or name.startswith("para_crawl")):
        return "language"
    if folder.startswith("niv2_"):
        if task not in niv2_languages:
            raise KeyError(f"{folder}/{task} is not in --niv2-languages: that file is not of this revision of {FLAN_REPO}")
        languages = niv2_languages[task]
        if not set(languages["input"]) | set(languages["output"]) | set(languages["instruction"]) <= LANGUAGES_KEPT:
            return "language"
    return None


def host_units(n_shards, rank, world):
    """(shard index, row residue or None) of this host: whole shards, or every world-th row of all of them."""
    if n_shards >= world:
        return [(index, None) for index in range(rank, n_shards, world)]
    return [(index, rank) for index in range(n_shards)]


def interleave(web_tokens, flan_tokens, flan_share):
    """The sequence of ("web", i) and ("flan", j): after each web document, FLAN until it is flan_share of the tokens.

    Returns (sequence, FLAN examples used). Raises when the FLAN pool runs out before the web documents.
    """
    sequence, used = [], 0
    web_total = flan_total = 0
    for i, tokens in enumerate(web_tokens):
        sequence.append(("web", i))
        web_total += tokens
        while flan_total < flan_share * (web_total + flan_total):
            if used == len(flan_tokens):
                raise RuntimeError(f"the FLAN pool of {len(flan_tokens)} examples ran out after web document {i + 1} of "
                                   f"{len(web_tokens)} ({web_total} web tokens): raise --cap")
            sequence.append(("flan", used))
            flan_total += flan_tokens[used]
            used += 1
    return sequence, used


class HubShards:
    """The shards of FLAN_REPO at FLAN_REVISION, downloaded a few ahead into download_dir and deleted once read."""

    def __init__(self, download_dir, ahead=4):
        from huggingface_hub import HfApi

        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.ahead = ahead
        self.files = HfApi().list_repo_files(FLAN_REPO, repo_type="dataset", revision=FLAN_REVISION)

    def shards(self, folder):
        names = [f for f in self.files if f.startswith(folder + "/") and f.endswith(".parquet")]
        return sorted(names, key=lambda name: int(re.search(r"part\.(\d+)\.parquet$", name).group(1)))

    def _download(self, name):
        from huggingface_hub import hf_hub_download

        return hf_hub_download(FLAN_REPO, name, repo_type="dataset", revision=FLAN_REVISION, local_dir=self.download_dir)

    def tables(self, names):
        """(name, table) in the order of names; the caller may stop early."""
        import pyarrow.parquet as pq

        executor = concurrent.futures.ThreadPoolExecutor(self.ahead)
        waiting, pending = iter(names), []

        def refill():
            # At most `ahead` shards on their way or in download_dir, so an early stop wastes little
            while len(pending) < self.ahead:
                name = next(waiting, None)
                if name is None:
                    return
                pending.append((name, executor.submit(self._download, name)))

        try:
            refill()
            while pending:
                name, future = pending.pop(0)
                path = future.result()
                table = pq.read_table(path, columns=FLAN_COLUMNS)
                os.remove(path)
                refill()
                yield name, table
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            for _, future in pending:
                if future.done() and not future.cancelled() and future.exception() is None and os.path.exists(future.result()):
                    os.remove(future.result())


def select_flan(args, tokenizer, source, niv2_languages):
    """This host's FLAN pool: [(inputs, targets, source, tokens)] in shard order, and what was left out why."""
    per_host_cap = math.ceil(args.cap / args.world)
    pool, report = [], {"per_host_cap": per_host_cap, "folders": {}}
    for folder in FLAN_FOLDERS:
        names = source.shards(folder)
        if not names:
            raise RuntimeError(f"{FLAN_REPO}@{FLAN_REVISION} has no shards in {folder}")
        units = host_units(len(names), args.rank, args.world)
        counts, dropped, excluded = Counter(), Counter(), {}
        shards_read = 0
        for (index, residue), (name, table) in zip(units, source.tables([names[index] for index, _ in units])):
            shards_read += 1
            # The folder is the template type: the cap is per (folder, task)
            template_types = set(table.column("_template_type").unique().to_pylist())
            if template_types != {"zs_opt" if "_zsopt_" in folder else "zs_noopt"}:
                raise RuntimeError(f"{name} has template types {template_types}, not those of {folder}")
            tasks = table.column("_task_name").to_pylist()
            rows = range(len(tasks)) if residue is None else range(residue, len(tasks), args.world)
            taken_here = Counter()
            candidates = []
            for row in rows:
                task = tasks[row]
                if task not in excluded:
                    excluded[task] = task_exclusion(folder, task, niv2_languages)
                if excluded[task] is not None:
                    dropped[excluded[task]] += 1
                    continue
                # A few more than the task still needs leaves room, as some fail the filters below
                need = per_host_cap - counts[task]
                if need <= 0 or taken_here[task] >= need * 1.2 + 5:
                    continue
                taken_here[task] += 1
                candidates.append(row)
            # Only these become Python strings: a shard holds ~300,000 rows, of which a few thousand are wanted
            inputs = table.column("inputs").take(candidates).to_pylist()
            targets = table.column("targets").take(candidates).to_pylist()
            lengths_in = [len(ids) for ids in tokenizer(inputs, add_special_tokens=False, split_special_tokens=True)["input_ids"]]
            lengths_out = [len(ids) for ids in tokenizer(targets, add_special_tokens=False, split_special_tokens=True)["input_ids"]]
            for row, text_in, text_out, n_in, n_out in zip(candidates, inputs, targets, lengths_in, lengths_out):
                task = tasks[row]
                if counts[task] >= per_host_cap:
                    continue
                if UNKNOWN_CHARACTER in text_in or UNKNOWN_CHARACTER in text_out:
                    dropped["unknown_character"] += 1
                elif not text_in.strip() or not text_out.strip():
                    dropped["empty"] += 1
                elif n_in + n_out + 3 > args.max_length:
                    dropped["too_long"] += 1
                else:
                    counts[task] += 1
                    pool.append((text_in, text_out, f"flan/{folder}/{task}", n_in + n_out + 3))
            print(f"flan: {name}: {sum(counts.values()):,} rows of {len(counts)} tasks so far in {folder}, pool {len(pool):,}", flush=True)
            if counts and all(counts[task] >= per_host_cap for task, why in excluded.items() if why is None):
                break
        report["folders"][folder] = {
            "shards": len(names), "units_of_this_host": len(units), "shards_read": shards_read,
            "rows": sum(counts.values()), "tasks": len(counts), "tasks_at_cap": sum(n >= per_host_cap for n in counts.values()),
            "dropped": dict(dropped), "excluded_tasks": {task: why for task, why in sorted(excluded.items()) if why is not None},
            "rows_per_task": dict(sorted(counts.items())),
        }
    return pool, report


def shuffled_pool(pool, seed, rank):
    """The pool in the order the builder used it (the shuffle line of its build())."""
    # The pool is in shard order, which groups rows by folder: shuffled, every stretch of the mix has every folder
    shuffled = np.random.default_rng([seed, rank]).permutation(len(pool))
    return [pool[i] for i in shuffled]


def build_id(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def write_mix(path, sequence, web_texts, pool, order, build):
    """Row order[t] of the file holds sequence[t], so that the trainer's shuffle yields the sequence."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    schema = pa.schema([("text", pa.string()), ("inputs", pa.string()), ("targets", pa.string()), ("source", pa.string()),
                        ("position", pa.int64())], metadata={METADATA_KEY: build.encode()})
    with pq.ParquetWriter(path, schema, compression="zstd") as writer:
        for start in range(0, len(order), 10_000):
            positions = inverse[start:start + 10_000]
            rows = [sequence[t] for t in positions]
            writer.write_table(pa.table({
                "text": [web_texts[i] if kind == "web" else None for kind, i in rows],
                "inputs": [pool[i][0] if kind == "flan" else None for kind, i in rows],
                "targets": [pool[i][1] if kind == "flan" else None for kind, i in rows],
                "source": [WEB_SOURCE if kind == "web" else pool[i][2] for kind, i in rows],
                "position": positions,
            }, schema=schema))


def add_answer_token(tokenizer) -> int:
    """Add <answer> as a special token and return its id; call after <mask> is in the tokenizer.

    It goes at the end of the vocabulary, so every other id stays what the start checkpoint was trained with.
    """
    if tokenizer.mask_token_id is None:
        raise ValueError("add <mask> before <answer>: the order of the added tokens fixes their ids")
    mask_token_id, size = tokenizer.mask_token_id, len(tokenizer)
    if ANSWER_TOKEN in tokenizer.get_vocab():
        raise ValueError(f"{ANSWER_TOKEN} is in the tokenizer already, at id {tokenizer.convert_tokens_to_ids(ANSWER_TOKEN)}")
    tokenizer.add_special_tokens({"additional_special_tokens": [ANSWER_TOKEN]}, replace_additional_special_tokens=False)
    answer_token_id = tokenizer.convert_tokens_to_ids(ANSWER_TOKEN)
    if tokenizer.mask_token_id != mask_token_id or answer_token_id != size or len(tokenizer) != size + 1:
        raise ValueError(f"{ANSWER_TOKEN} got id {answer_token_id} and <mask> {tokenizer.mask_token_id}; expected {size} "
                         f"and {mask_token_id}, the end of a vocabulary of {size}")
    return answer_token_id


def load_tokenizer(name):
    """The tokenizer as the builder's main() set it up; its counts decide the max_length filter of select_flan."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, split_special_tokens=True)
    if tokenizer.mask_token is None:
        # As the trainer does: the text is tokenized the same with or without these, but the ids must be the trainer's
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
    add_answer_token(tokenizer)
    return tokenizer
