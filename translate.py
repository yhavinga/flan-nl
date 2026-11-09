#!/usr/bin/env python3
"""TranslateGemma on the 4 chips of one host: this worker's segments from English to Dutch, greedy, in resumable shards.

Usage (translate.sh, on each worker, as a TPU of its own 4 chips):
    python translate.py --segments <plan>/segments-w0.parquet --out gs://.../out/w0 --ckpt gs://.../0/items \\
        --tokenizer /dev/shm/flan_nl/tokenizer-27b --model gemma3-27b --length-class small

The model is MaxText's gemma3 with the converted checkpoint in MaxText's MaxEngine: a KV cache of fixed shape
(slots x max_target_length), prompts padded to a few prefill lengths, continuous batching over the slots (decode_all),
greedy decoding, a stop at <end_of_turn> or <eos>. The tensor axis of the mesh is the host's 4 chips. Segments are
taken in shards of consecutive ids (similar lengths, as the plan sorts them); a shard is written to the bucket at once
when it is done, so a restart skips every shard that is there. A segment that hits the decode limit is written with
ended = false; the assemble step drops its examples.

The prompt is the model's chat template (chat_template.jinja of the Hugging Face repo): a user turn with
{type: text, source_lang_code: en, target_lang_code: nl, text}, which the template turns into "You are a professional
English (en) to Dutch (nl) translator. ... Please translate the following English text into Dutch:\n\n\n<text>".
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# The engine's shapes per length class: the longest source segment (tokens; the class takes those above the class
# before it), the prefill lengths a prompt is padded to (the template adds ~70 tokens), and the decode limit. A step
# attends over the whole cache of every slot, prefill and decode part, so its cost grows with prefill + decode (8 Nov
# on a v4 host, 12B: 38 ms at 256 slots x 512): the classes keep that length close to what their segments need.
LENGTH_CLASSES = {
    "tiny": {"max_source": 32, "prefill_lengths": [128], "max_decode": 96},
    "small": {"max_source": 128, "prefill_lengths": [128, 256], "max_decode": 256},
    "medium": {"max_source": 256, "prefill_lengths": [256, 384], "max_decode": 448},
    "long": {"max_source": 384, "prefill_lengths": [384, 512], "max_decode": 640},
    "xlong": {"max_source": 512, "prefill_lengths": [512, 640], "max_decode": 832},
}
CLASS_ORDER = ["tiny", "small", "medium", "long", "xlong"]
# Slots per class: slots x (prefill + decode) x the KV of a token in bf16 (12B: 48 layers x 8 heads x 256 x 2 x 2 =
# 0.39 MB; 27B: 62 x 16 x 128 x 2 x 2 = 0.51 MB) next to the weights (24 and 54 GB) in the host's 4 x 30.7 GiB.
# 27B measured on a v4 host on 8 Nov: tiny 384 x 224 peaks at 27.0 GiB per chip, small 224 x 512 at 28.9 GiB; the
# others hold about as many slot-tokens. (The 12B run of 8 Nov had four classes, long being 256-512 with 128 slots.)
SLOTS = {
    "gemma3-12b": {"tiny": 512, "small": 320, "medium": 224, "long": 160, "xlong": 128},
    "gemma3-27b": {"tiny": 384, "small": 224, "medium": 136, "long": 104, "xlong": 80},
}
# generation_config.json of TranslateGemma: <eos> and <end_of_turn>
EOS_IDS = [1, 106]
_MARK = ""  # a private-use character, never in FLAN: where the text goes in the rendered template


def messages(text):
    return [{"role": "user", "content": [{"type": "text", "source_lang_code": "en", "target_lang_code": "nl",
                                          "text": text}]}]


class PromptCodec:
    """Prompt ids of a segment: the template rendered once around a mark, checked against the template itself."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        rendered = tokenizer.apply_chat_template(messages(_MARK), tokenize=False, add_generation_prompt=True)
        self.prefix, self.suffix = rendered.split(_MARK)
        if "English (en) to Dutch (nl)" not in self.prefix or not self.suffix.endswith("<start_of_turn>model\n"):
            raise RuntimeError(f"unexpected chat template: {rendered!r}")

    def encode(self, text):
        return self.tokenizer(self.prefix + text + self.suffix, add_special_tokens=False)["input_ids"]

    def check(self, texts):
        for text in texts:
            expected = self.tokenizer.apply_chat_template(messages(text), tokenize=True, add_generation_prompt=True)
            if isinstance(expected, dict) or hasattr(expected, "keys"):
                expected = expected["input_ids"]
            if list(expected) != self.encode(text):
                raise RuntimeError(f"prompt ids differ from the chat template's for {text[:60]!r}")


def make_result_tokens_a_pytree():
    """Without JetStream, MaxText's engine returns a plain stand-in for JetStream's ResultTokens (a flax struct
    dataclass, so a pytree), and jit cannot return a plain object. Registered here as JetStream's is: arrays as children,
    the index tuples as static data."""
    import jax
    from maxtext.inference.maxengine import maxengine

    api = maxengine.engine_api
    if not getattr(api, "_IS_STUB", False):
        return
    cls = api.ResultTokens

    def flatten(r):
        return (r.data, r.log_prob), (r.tokens_idx, r.valid_idx, r.length_idx, r.samples_per_slot)

    def unflatten(aux, children):
        return cls(data=children[0], log_prob=children[1], tokens_idx=aux[0], valid_idx=aux[1], length_idx=aux[2],
                   samples_per_slot=aux[3])

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)


def make_decode_cache_sharded():
    """MaxText 0.2.5 makes the zeros of the whole decode cache eagerly, unsharded, on one chip, before its jit makes them
    again sharded (MaxEngine._init_decode_state_nnx): a cache larger than one chip's free memory fails there with
    RESOURCE_EXHAUSTED (pilot of 8 Nov, 47 GB of cache). That is the one call for the autoregressive cache outside a
    jit; given shapes instead of arrays, the jit alone makes the zeros, sharded. The prefill calls are traced and keep
    MaxText's own code."""
    import jax
    from flax import nnx
    from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE
    from maxtext.inference.maxengine import maxengine

    original = maxengine.MaxEngine._nnx_init_cache_dict

    def init_cache_dict(self, mode=maxengine.MODEL_MODE_PREFILL):
        if mode != MODEL_MODE_AUTOREGRESSIVE:
            return original(self, mode)
        _, cache_state, _ = nnx.split(self._abstract_model_for_mode(mode), nnx.Cache, ...)
        return jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), nnx.to_pure_dict(cache_state))

    maxengine.MaxEngine._nnx_init_cache_dict = init_cache_dict


def make_engine(args, shapes):
    """MaxText's MaxEngine with the converted checkpoint: (engine, params)."""
    from maxtext.configs import pyconfig
    from maxtext.inference.maxengine import maxengine

    max_prefill = shapes["prefill_lengths"][-1]
    config = pyconfig.initialize(
        [""], model_name=args.model, load_parameters_path=args.ckpt, run_name="flan_nl", log_config=False,
        # Not the home directory: the boot disk is nearly full
        base_output_directory="/dev/shm/flan_nl/maxtext_output",
        enable_checkpointing=False, async_checkpointing=False, skip_jax_distributed_system=True, scan_layers=False,
        use_multimodal=False, attention="dot_product", dtype="bfloat16", weight_dtype="bfloat16",
        ici_fsdp_parallelism=1, ici_tensor_parallelism=4, per_device_batch_size=shapes["slots"] / 4,
        max_prefill_predict_length=max_prefill, max_target_length=max_prefill + shapes["max_decode"],
        decode_sampling_strategy="greedy", return_log_prob=False, tokenizer_type="huggingface",
        tokenizer_path=args.tokenizer)
    engine = maxengine.MaxEngine(config)
    return engine, engine.load_params()


class EngineSlots:
    """The device side of decode_all: MaxEngine's prefill, insert and generate on one decode state."""

    def __init__(self, engine, params, prefill_lengths):
        self.engine, self.params, self.prefill_lengths = engine, params, prefill_lengths
        self.state = engine.init_decode_state()

    def prefill(self, tokens):
        import jax.numpy as jnp

        length = next(n for n in self.prefill_lengths if n >= len(tokens))
        padded = np.zeros(length, np.int32)
        padded[:len(tokens)] = tokens
        prefix, first = self.engine.prefill(params=self.params, padded_tokens=jnp.asarray(padded), true_length=len(tokens))
        return prefix, first.data[0, 0]

    def insert(self, prefix, slot):
        self.state = self.engine.insert(prefix, self.state, slot=slot)

    def generate(self):
        self.state, result = self.engine.generate(self.params, self.state)
        return result.data[:, 0]

    @staticmethod
    def fetch(arrays):
        import jax

        return np.asarray(jax.device_get(arrays))


def decode_all(prompts, slots, device, max_decode, sync_every):
    """Greedy completions of prompts over a fixed number of slots: [(tokens without the stop token, ended)].

    Continuous batching with the engine's own pieces: every free slot gets the next prompt (prefill, insert; the
    prefill's token is the first of the completion), then sync_every generate steps run for all slots at once and come
    back to the host in one transfer. A slot whose completion ends (a stop token, or max_decode tokens) is free for
    the next prompt; the steps it ran past its end are dropped. MaxText's OfflineEngine does the same with a thread
    per token and reached 344 tokens/s on a v4 host where these pieces allow several times that (8 Nov).
    Prompts go longest first, so the last steps of a shard are spent on short ones.
    """
    queue = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    queue.reverse()  # pop() from the end gives the longest first
    outputs = [None] * len(prompts)
    owner = [-1] * slots
    tokens = [[] for _ in range(slots)]

    def finished(s):
        return bool(tokens[s]) and (tokens[s][-1] in EOS_IDS or len(tokens[s]) >= max_decode)

    while queue or any(i >= 0 for i in owner):
        firsts = []
        for s in range(slots):
            if owner[s] < 0 and queue:
                i = queue.pop()
                prefix, first = device.prefill(prompts[i])
                device.insert(prefix, s)
                owner[s], tokens[s] = i, []
                firsts.append((s, first))
        steps = [device.generate() for _ in range(sync_every)]
        if firsts:
            for (s, _), token in zip(firsts, device.fetch([f for _, f in firsts])):
                tokens[s].append(int(token))
        for step in device.fetch(steps):
            for s in range(slots):
                if owner[s] >= 0 and not finished(s):
                    tokens[s].append(int(step[s]))
        for s in range(slots):
            if owner[s] >= 0 and finished(s):
                ended = tokens[s][-1] in EOS_IDS
                outputs[owner[s]] = (tokens[s][:-1] if ended else tokens[s], ended)
                owner[s] = -1
    return outputs


def shards_of(ids, texts, shard_size):
    """[(name, ids, texts)]: the same names for the same segments at every start, so a restart finds its shards."""
    return [(f"shard-{k // shard_size:05d}.parquet", ids[k:k + shard_size], texts[k:k + shard_size])
            for k in range(0, len(ids), shard_size)]


def translate_pending(shards, done, translate, store, max_shards=None):
    """translate(ids, texts) -> (rows, seconds), then store(name, rows, seconds), for every shard not in done; a shard
    is stored whole or not at all, so a crash in the middle of one costs only that one. Returns the names stored."""
    stored = []
    for name, shard_ids, shard_texts in shards:
        if name in done:
            continue
        if max_shards is not None and len(stored) == max_shards:
            break
        rows, seconds = translate(shard_ids, shard_texts)
        if rows["id"] != list(shard_ids):
            raise RuntimeError(f"{name}: the translations are not those of the shard's segments")
        store(name, rows, seconds)
        stored.append(name)
    return stored


def done_shards(out):
    listed = subprocess.run(["gsutil", "ls", f"{out}/"], capture_output=True, text=True)
    if listed.returncode != 0:
        if "matched no objects" in listed.stderr or "One or more URLs matched no objects" in listed.stderr:
            return set()
        raise RuntimeError(f"gsutil ls {out}/ failed: {listed.stderr.strip()}")
    return {Path(line).name for line in listed.stdout.split() if line.endswith(".parquet")}


def translate_shard(device, codec, tokenizer, ids, texts, shapes, sync_every):
    prompts = [codec.encode(text) for text in texts]
    started = time.time()
    outputs = decode_all(prompts, shapes["slots"], device, shapes["max_decode"], sync_every)
    seconds = time.time() - started
    rows = {"id": [], "text_nl": [], "ended": [], "tokens_in": [], "tokens_out": []}
    for i, prompt, (completion, ended) in zip(ids, prompts, outputs):
        rows["id"].append(i)
        rows["text_nl"].append(tokenizer.decode(completion, skip_special_tokens=True).strip())
        rows["ended"].append(ended)
        rows["tokens_in"].append(len(prompt))
        rows["tokens_out"].append(len(completion) + int(ended))
    return rows, seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description="TranslateGemma on one host: segments from English to Dutch")
    parser.add_argument("--segments", required=True, help="segments-w<N>.parquet of the plan (local)")
    parser.add_argument("--out", required=True, help="Bucket directory of this worker's shards of this length class")
    parser.add_argument("--ckpt", required=True, help="The MaxText parameters, <checkpoint>/0/items")
    parser.add_argument("--tokenizer", required=True, help="TranslateGemma's tokenizer directory")
    parser.add_argument("--model", default="gemma3-27b", choices=sorted(SLOTS))
    parser.add_argument("--length-class", required=True, choices=sorted(LENGTH_CLASSES))
    parser.add_argument("--slots", type=int, default=None, help="Override the class's number of decode slots")
    parser.add_argument("--shard-size", type=int, default=4096, help="Segments per shard (default: 4096)")
    parser.add_argument("--max-shards", type=int, default=None, help="Stop after this many new shards (pilot)")
    parser.add_argument("--sync-every", type=int, default=8, help="Generate steps between transfers (default: 8)")
    args = parser.parse_args(argv)

    import jax
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    if (jax.device_count(), jax.process_count()) != (4, 1):
        raise SystemExit(f"needs one process with 4 chips, has {jax.process_count()} process(es) with "
                         f"{jax.device_count()} devices: set the TPU_* variables of translate.sh")
    shapes = dict(LENGTH_CLASSES[args.length_class], slots=args.slots or SLOTS[args.model][args.length_class])
    k = CLASS_ORDER.index(args.length_class)
    low = LENGTH_CLASSES[CLASS_ORDER[k - 1]]["max_source"] if k else 0
    table = pq.read_table(args.segments)
    ids, texts, tokens = (table.column(c).to_pylist() for c in ("id", "text", "tokens"))
    if max(tokens, default=0) > LENGTH_CLASSES[CLASS_ORDER[-1]]["max_source"]:
        raise SystemExit(f"{args.segments} has a segment of {max(tokens)} tokens, more than the plan allows")
    chosen = [k for k, n in enumerate(tokens) if low < n <= shapes["max_source"]]
    ids, texts = [ids[k] for k in chosen], [texts[k] for k in chosen]
    # Shards hold consecutive ids, so similar lengths; they are done in a fixed random order, so that the time of the
    # first few is a fair sample of the whole class (the projection of the run's length)
    shards = shards_of(ids, texts, args.shard_size)
    shards = random.Random(0).sample(shards, len(shards))
    done = done_shards(args.out)
    pending = sum(name not in done for name, _, _ in shards)
    print(f"{args.length_class}: {len(ids):,} segments in {len(shards)} shards, {len(shards) - pending} done before; "
          f"shapes {json.dumps(shapes)}", flush=True)
    if not pending:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    codec = PromptCodec(tokenizer)
    codec.check(texts[:3] + texts[-3:])
    make_result_tokens_a_pytree()
    make_decode_cache_sharded()
    started = time.time()
    engine, params = make_engine(args, shapes)
    device = EngineSlots(engine, params, shapes["prefill_lengths"])
    print(f"engine ready in {time.time() - started:.0f} s", flush=True)
    local = Path("/dev/shm/flan_nl/shards") / args.length_class
    local.mkdir(parents=True, exist_ok=True)
    totals = {"segments": 0, "tokens_in": 0, "tokens_out": 0, "seconds": 0.0}

    def store(name, rows, seconds):
        path = local / name
        pq.write_table(pa.table(rows), path, compression="zstd")
        subprocess.run(["gsutil", "-q", "cp", str(path), f"{args.out}/{name}"], check=True)
        path.unlink()
        n_in, n_out = sum(rows["tokens_in"]), sum(rows["tokens_out"])
        for key, value in (("segments", len(rows["id"])), ("tokens_in", n_in), ("tokens_out", n_out), ("seconds", seconds)):
            totals[key] += value
        print(f"SHARD {json.dumps({'class': args.length_class, 'shard': name, 'segments': len(rows['id']), 'tokens_in': n_in, 'tokens_out': n_out, 'seconds': round(seconds, 1), 'out_per_s': round(n_out / seconds), 'all_per_s': round((n_in + n_out) / seconds), 'not_ended': rows['ended'].count(False)})}", flush=True)

    translate_pending(shards, done, lambda i, t: translate_shard(device, codec, tokenizer, i, t, shapes, args.sync_every), store, args.max_shards)
    print(f"DONE {json.dumps({'class': args.length_class, **totals, 'out_per_s': round(totals['tokens_out'] / max(totals['seconds'], 1e-9))})}", flush=True)


if __name__ == "__main__":
    # A thread that is no daemon (MaxText's OfflineEngine had one, fsspec's IO loop is another) would keep a failed
    # process alive, holding the TPU and never reaching translate.sh's EXIT line (pilot of 8 Nov). So the process ends
    # here, with the error's traceback and code, whatever the threads do.
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
