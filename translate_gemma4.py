#!/usr/bin/env python3
"""Gemma 4 31B-it on the 4 chips of one host: this worker's segments from English to Dutch, greedy, in resumable
shards; the Gemma 4 counterpart of translate.py, with the same input and output.

Usage (translate_gemma4.sh, on each worker, as a TPU of its own 4 chips):
    python translate_gemma4.py --segments <plan>/segments-w0.parquet --out gs://.../translations/g4/w0/small \\
        --ckpt gs://.../gemma-4-31b-it-maxtext/0/items --tokenizer /dev/shm/flan_nl/tokenizer-gemma4 \\
        --length-class small

The model, its prompt (flan-v1), its length classes and its slots are those of models.yml (--model, default
gemma4-31b-flan); the engine is gemma4_engine.py (MaxText's MaxEngine, continuous batching over the slots, greedy).
A segment's class is decided by the plan's token count (column tokens), as in translate.py. Every prompt of the class
is built and measured before the engine starts: one longer than the class's largest prefill length stops the run.
Shards, their names and their order are translate.py's (shards_of, a fixed random order, a stored shard skipped at a
restart; the bucket through the Python client), and so are the shard's columns (id, text_nl, ended, tokens_in, tokens_out): flan_nl.py assemble reads
them as it reads TranslateGemma's.
"""

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import gemma4_engine as ge
import translate as tg

DEFAULT_MODEL = "gemma4-31b-flan"


def class_bounds(model, length_class):
    """(the most tokens of the class before, the most of this class): a segment of n tokens is in the class when
    low < n <= high"""
    classes = ge.MODELS[model]["length_classes"]
    names = list(classes)
    k = names.index(length_class)
    return (classes[names[k - 1]]["max_source"] if k else 0), classes[length_class]["max_source"]


def select(ids, texts, tokens, low, high):
    chosen = [k for k, n in enumerate(tokens) if low < n <= high]
    return [ids[k] for k in chosen], [texts[k] for k in chosen]


# The bucket through the Python client, not gsutil: on these images gsutil is a snap, and a snap refuses to start under
# a process with no_new_privs, which this one has once libtpu is up (gemma4_engine.py; smoke test of 14 Nov on a v4-32)
def stored_shards(client, out):
    """The names of the shards stored under out (gs://bucket/dir)"""
    bucket, prefix = ge.split_gs(out.rstrip("/") + "/")
    return {Path(b.name).name for b in client.list_blobs(bucket, prefix=prefix) if b.name.endswith(".parquet")}


def upload(client, local, url):
    bucket, name = ge.split_gs(url)
    client.bucket(bucket).blob(name).upload_from_filename(str(local))


def check_prompts(codec, texts, shapes):
    """The prompt ids of every text; a prompt longer than the largest prefill length is an error before any work"""
    prompts = [codec.encode(t) for t in texts]
    longest = max((len(p) for p in prompts), default=0)
    if longest > shapes["prefill_lengths"][-1]:
        k = max(range(len(prompts)), key=lambda i: len(prompts[i]))
        raise SystemExit(f"a prompt of {longest} tokens is longer than the class's largest prefill length "
                         f"{shapes['prefill_lengths'][-1]}: {texts[k][:80]!r}")
    return longest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gemma 4 on one host: segments from English to Dutch")
    parser.add_argument("--segments", required=True, help="segments-w<N>.parquet of the plan (local; id, text, tokens)")
    parser.add_argument("--out", required=True, help="Bucket directory of this worker's shards of this length class")
    parser.add_argument("--ckpt", required=True, help="The MaxText parameters, <checkpoint>/0/items")
    parser.add_argument("--tokenizer", required=True, help="Gemma 4's tokenizer directory (with its chat template)")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(ge.MODELS))
    parser.add_argument("--prompt", default=None, help="A prompt of models.yml (default: the model's)")
    parser.add_argument("--length-class", required=True)
    parser.add_argument("--slots", type=int, default=None, help="Override the class's number of decode slots")
    parser.add_argument("--shard-size", type=int, default=1024, help="Segments per shard (default: 1024)")
    parser.add_argument("--max-shards", type=int, default=None, help="Stop after this many new shards (smoke test)")
    parser.add_argument("--sync-every", type=int, default=8, help="Generate steps between transfers (default: 8)")
    parser.add_argument("--work-dir", default="/dev/shm/flan_nl/work-g4", help="Local shards, MaxText's output and "
                        "JAX's compilation cache")
    args = parser.parse_args(argv)
    model = ge.MODELS[args.model]
    if args.length_class not in model["length_classes"]:
        raise SystemExit(f"--length-class {args.length_class} is not a class of {args.model}: {list(model['length_classes'])}")

    import pyarrow as pa
    import pyarrow.parquet as pq

    shapes = dict(model["length_classes"][args.length_class], slots=args.slots or model["slots"][args.length_class])
    low, high = class_bounds(args.model, args.length_class)
    table = pq.read_table(args.segments, columns=["id", "text", "tokens"])
    ids, texts, tokens = (table.column(c).to_pylist() for c in ("id", "text", "tokens"))
    largest = max(c["max_source"] for c in model["length_classes"].values())
    if max(tokens, default=0) > largest:
        raise SystemExit(f"{args.segments} has a segment of {max(tokens)} tokens, more than the classes allow ({largest})")
    ids, texts = select(ids, texts, tokens, low, high)
    shards = tg.shards_of(ids, texts, args.shard_size)
    shards = random.Random(0).sample(shards, len(shards))
    client = ge.gcs_client()
    done = stored_shards(client, args.out)
    pending = sum(name not in done for name, _, _ in shards)
    print(f"{args.length_class}: {len(ids):,} segments in {len(shards)} shards, {len(shards) - pending} done before; "
          f"shapes {json.dumps(shapes)}", flush=True)
    if not pending:
        return

    import jax
    from transformers import AutoTokenizer

    if (jax.device_count(), jax.process_count()) != (model["chips"], 1):
        raise SystemExit(f"needs one process with {model['chips']} chips, has {jax.process_count()} process(es) with "
                         f"{jax.device_count()} devices: set the TPU_* variables of translate_gemma4.sh")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    codec = ge.TurnCodec(tokenizer, model, args.prompt or model["prompt"], "en", "nl")
    codec.check(texts[:3] + texts[-3:])
    longest = check_prompts(codec, texts, shapes)
    print(f"MAXTEXT {json.dumps({'model': args.model, 'prompt': args.prompt or model['prompt'], 'class': args.length_class, 'shapes': shapes, 'longest_prompt': longest, 'ckpt': args.ckpt, **ge.maxtext_source(), 'patches': ge.patch_maxtext()})}", flush=True)
    started = time.time()
    engine, params = ge.make_engine(args.model, args.ckpt, args.tokenizer, args.work_dir, shapes)
    device = ge.EngineSlots(engine, params, shapes["prefill_lengths"])
    print(f"engine ready in {time.time() - started:.0f} s", flush=True)
    local = Path(args.work_dir) / "shards" / args.length_class
    local.mkdir(parents=True, exist_ok=True)
    totals = {"segments": 0, "tokens_in": 0, "tokens_out": 0, "seconds": 0.0}

    def translate(shard_ids, shard_texts):
        return ge.translate_shard(device, codec, tokenizer, shard_ids, shard_texts, shapes, args.sync_every,
                                  model["eos_ids"])

    def store(name, rows, seconds):
        path = local / name
        pq.write_table(pa.table(rows), path, compression="zstd")
        upload(client, path, f"{args.out}/{name}")
        path.unlink()
        n_in, n_out = sum(rows["tokens_in"]), sum(rows["tokens_out"])
        for key, value in (("segments", len(rows["id"])), ("tokens_in", n_in), ("tokens_out", n_out), ("seconds", seconds)):
            totals[key] += value
        print(f"SHARD {json.dumps({'class': args.length_class, 'shard': name, 'segments': len(rows['id']), 'tokens_in': n_in, 'tokens_out': n_out, 'seconds': round(seconds, 1), 'out_per_s': round(n_out / seconds), 'segments_per_s': round(len(rows['id']) / seconds, 2), 'not_ended': rows['ended'].count(False)})}", flush=True)

    tg.translate_pending(shards, done, translate, store, args.max_shards)
    print(f"DONE {json.dumps({'class': args.length_class, **totals, 'out_per_s': round(totals['tokens_out'] / max(totals['seconds'], 1e-9))})}", flush=True)


if __name__ == "__main__":
    # As translate.py: a thread that is no daemon would keep a failed process alive and holding the TPU, so the
    # process ends here with the error's traceback and code
    code = 0
    try:
        main()
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
        if not isinstance(stop.code, int) and stop.code is not None:
            print(stop.code, file=sys.stderr)
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
