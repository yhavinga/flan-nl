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

With --units (g4_full.py units) the worker does not translate a segment file of its own: it goes through the units of
the class in units.json, claims each free one in --claims (g4_full.claim; owner <vm>/w<k>, run), translates it and
stores it, so that every worker of every VM takes the next free unit and a restart goes on where the work is.
--segments is then the local directory of the plan's segment files and --out the class's directory (<unit>.parquet).
The engine starts at the first unit claimed; a class without a free unit costs no engine. A unit's prompts are
measured before it is translated.
"""

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import g4_full as g4f
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
    parser.add_argument("--out", required=True, help="Bucket directory of this worker's shards of this length class "
                        "(with --units: of the class's units, <unit>.parquet)")
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
    parser.add_argument("--units", default=None, help="units.json of g4_full.py units: claim units of the class instead "
                        "of translating --segments (then the local directory of the plan's segment files)")
    parser.add_argument("--claims", default=None, help="With --units: the gs:// directory of the claims")
    parser.add_argument("--owner", default=None, help="With --units: <vm>/w<k>, the claims' owner")
    parser.add_argument("--run", default=None, help="With --units: the run (a new one takes over its VM's claims)")
    parser.add_argument("--lease", type=int, default=g4f.LEASE_SECONDS, help="With --units: seconds a claim holds")
    parser.add_argument("--max-units", type=int, default=None, help="With --units: stop after this many (smoke test)")
    args = parser.parse_args(argv)
    model = ge.MODELS[args.model]
    if args.length_class not in model["length_classes"]:
        raise SystemExit(f"--length-class {args.length_class} is not a class of {args.model}: {list(model['length_classes'])}")
    if args.units:
        if not (args.claims and args.owner and args.run):
            raise SystemExit("--units needs --claims, --owner and --run")
        return work(args, model)

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


def work(args, model):
    """--units: the free units of the class, claimed one by one (g4_full.work_units)"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    info = json.loads(Path(args.units).read_text())
    if info["model"] != args.model:
        raise SystemExit(f"{args.units} is a plan of {info['model']}, not {args.model}")
    units = [u for u in info["units"] if u["class"] == args.length_class]
    size = info["classes"][args.length_class]["size"]
    low, high = class_bounds(args.model, args.length_class)
    shapes = dict(model["length_classes"][args.length_class], slots=args.slots or model["slots"][args.length_class])
    client = ge.gcs_client()
    claims = g4f.GcsClaims(args.claims, client)
    me = {"owner": args.owner, "run": args.run}
    out = args.out.rstrip("/")
    listing = {"at": None, "names": set()}

    def stored():
        if listing["at"] is None or time.time() - listing["at"] > 60:
            listing["names"] = {name[:-len(".parquet")] for name in stored_shards(client, out)}
            listing["at"] = time.time()
        return listing["names"]

    pending = sum(u["name"] not in stored() for u in units)
    print(f"{args.length_class}: {len(units)} units, {len(units) - pending} stored before; owner {args.owner}, run "
          f"{args.run}; shapes {json.dumps(shapes)}", flush=True)
    if not pending:
        return
    state = {}
    tables = {}

    def start():
        import jax
        from transformers import AutoTokenizer

        if (jax.device_count(), jax.process_count()) != (model["chips"], 1):
            raise SystemExit(f"needs one process with {model['chips']} chips, has {jax.process_count()} process(es) "
                             f"with {jax.device_count()} devices: set the TPU_* variables of translate_gemma4.sh")
        state["tokenizer"] = AutoTokenizer.from_pretrained(args.tokenizer)
        state["codec"] = ge.TurnCodec(state["tokenizer"], model, args.prompt or model["prompt"], "en", "nl")
        print(f"MAXTEXT {json.dumps({'model': args.model, 'prompt': args.prompt or model['prompt'], 'class': args.length_class, 'shapes': shapes, 'ckpt': args.ckpt, **ge.maxtext_source(), 'patches': ge.patch_maxtext()})}", flush=True)
        started = time.time()
        engine, params = ge.make_engine(args.model, args.ckpt, args.tokenizer, args.work_dir, shapes)
        state["device"] = ge.EngineSlots(engine, params, shapes["prefill_lengths"])
        print(f"engine ready in {time.time() - started:.0f} s", flush=True)

    def translate(u):
        if u["segments"] not in tables:
            t = pq.read_table(Path(args.segments) / u["segments"], columns=["id", "text", "tokens"])
            tables[u["segments"]] = [t.column(c).to_pylist() for c in ("id", "text", "tokens")]
        ids, texts = g4f.unit_rows(u, size, *tables[u["segments"]], low, high)
        if not state:
            start()
        state["codec"].check(texts[:2] + texts[-1:])
        check_prompts(state["codec"], texts, shapes)
        rows, seconds = ge.translate_shard(state["device"], state["codec"], state["tokenizer"], ids, texts, shapes,
                                           args.sync_every, model["eos_ids"])
        if rows["id"] != ids:
            raise RuntimeError(f"{u['name']}: the translations are not those of the unit's segments")
        return rows, seconds

    local = Path(args.work_dir) / "units"
    local.mkdir(parents=True, exist_ok=True)

    def store(u, rows, seconds):
        path = local / f"{u['name']}.parquet"
        pq.write_table(pa.table(rows), path, compression="zstd")
        upload(client, path, f"{out}/{u['name']}.parquet")
        path.unlink()
        n_out = sum(rows["tokens_out"])
        print(f"SHARD {json.dumps({'class': args.length_class, 'shard': u['name'], 'owner': args.owner, 'segments': len(rows['id']), 'tokens_in': sum(rows['tokens_in']), 'tokens_out': n_out, 'seconds': round(seconds, 1), 'out_per_s': round(n_out / seconds), 'segments_per_s': round(len(rows['id']) / seconds, 2), 'not_ended': rows['ended'].count(False)})}", flush=True)

    bucket, prefix = ge.split_gs(out)

    def exists(name):
        return client.bucket(bucket).blob(f"{prefix}/{name}.parquet").exists()

    started = time.time()
    done = g4f.work_units(units, stored, lambda name: g4f.claim(claims, name, me, args.lease), exists, translate,
                          store, args.max_units)
    print(f"CLASS {json.dumps({'class': args.length_class, 'units': len(units), 'done_here': len(done), 'engine': bool(state), 'seconds': round(time.time() - started)})}", flush=True)


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
