# A copy of github.com/yhavinga/nl-backtranslation gemma4_bt/gemma4_engine.py at commit 2057072 (13 Nov 2026), unchanged
# but for this comment: models.yml is FLAN-nl's own (beside this file), and the environment is setup_gemma4.sh rather
# than gemma4_bt/setup.sh. A fix to the engine belongs in both repositories.
"""Gemma 4 31B-it in MaxText on the 4 chips of one process: the model and its prompts (models.yml beside this file),
greedy decoding with continuous batching over a fixed number of slots, and the bucket through the Python client.

The model runs in MaxText's MaxEngine: a KV cache of fixed shape (slots x max_target_length), prompts padded to a few
prefill lengths, continuous batching over the slots (decode_all), greedy decoding, a stop at the model's stop tokens.
The tensor axis of the mesh is the process's chips. The engine part is that of translate.py (TranslateGemma), with
the model's settings from models.yml and two workarounds that apply only where the installed MaxText still needs them
(patch_maxtext). Gemma 4 runs in MaxText from git at a fixed commit (gemma4_bt/setup.sh).

A prompt is one user turn in the model's own chat format: prefix + user_text + suffix of the model's turn, tokenized
without special tokens (the prefix holds <bos>); TurnCodec.check compares it with the tokenizer's chat template.
"""

import json
import time
from pathlib import Path

import numpy as np
import yaml

MODELS_FILE = Path(__file__).with_name("models.yml")
CONFIG = yaml.safe_load(MODELS_FILE.read_text(encoding="utf-8"))
MODELS = CONFIG["models"]
PROMPTS = CONFIG["prompts"]
# The language codes this module takes, with the names the prompts give them
LANGUAGES = {"en": "English", "nl": "Dutch"}


def user_text(prompt, text, source="nl", target="en"):
    """The user turn of a prompt of models.yml: its plain template with the language names and the stripped text"""
    names = {"source": LANGUAGES[source], "target": LANGUAGES[target], "source_code": source, "target_code": target}
    return prompt["plain"].format(text=text.strip(), **names)


class TurnCodec:
    """Prompt ids of a segment: user_text between the strings of the model's user turn. check compares them with the
    tokenizer's chat template (a plain-text user turn)."""

    def __init__(self, tokenizer, model, prompt_name, source="nl", target="en"):
        if prompt_name not in PROMPTS:
            raise ValueError(f"no prompt {prompt_name!r} in {MODELS_FILE.name}")
        self.tokenizer, self.turn, self.source, self.target = tokenizer, model["turn"], source, target
        self.prompt = PROMPTS[prompt_name]

    def text(self, text):
        return self.turn["prefix"] + user_text(self.prompt, text, self.source, self.target) + self.turn["suffix"]

    def encode(self, text):
        return self.tokenizer(self.text(text), add_special_tokens=False)["input_ids"]

    def check(self, texts):
        if not self.turn.get("check"):
            return
        for text in texts:
            user = user_text(self.prompt, text, self.source, self.target)
            expected = self.tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=True,
                                                          add_generation_prompt=True)
            if isinstance(expected, dict) or hasattr(expected, "keys"):
                expected = expected["input_ids"]
            if list(expected) != self.encode(text):
                raise RuntimeError(f"prompt ids differ from the chat template's for {text[:60]!r}")


def make_result_tokens_a_pytree():
    """Without JetStream, MaxText's engine returns a plain stand-in for JetStream's ResultTokens (a flax struct
    dataclass, so a pytree), and jit cannot return a plain object. Registered here as JetStream's is: arrays as children,
    the index tuples as static data. A MaxText whose stub is a pytree already refuses a second registration with a
    ValueError, which is then the sign that nothing is needed. Returns whether it registered the stub."""
    import jax
    from maxtext.inference.maxengine import maxengine

    api = maxengine.engine_api
    if not getattr(api, "_IS_STUB", False):
        return False
    cls = api.ResultTokens

    def flatten(r):
        return (r.data, r.log_prob), (r.tokens_idx, r.valid_idx, r.length_idx, r.samples_per_slot)

    def unflatten(aux, children):
        return cls(data=children[0], log_prob=children[1], tokens_idx=aux[0], valid_idx=aux[1], length_idx=aux[2],
                   samples_per_slot=aux[3])

    try:
        jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    except ValueError as error:
        if "Duplicate" not in str(error):
            raise
        return False
    return True


def decode_cache_fix_present():
    """Whether MaxText makes the NNX decode cache from shapes inside its init jit (jax.eval_shape over
    _nnx_init_cache_dict in _init_decode_state_nnx)"""
    import inspect

    from maxtext.inference.maxengine import maxengine

    return "eval_shape" in inspect.getsource(maxengine.MaxEngine._init_decode_state_nnx)


def make_decode_cache_sharded():
    """MaxText 0.2.5 makes the zeros of the whole decode cache eagerly, unsharded, on one chip, before its jit makes them
    again sharded (MaxEngine._init_decode_state_nnx): a cache larger than one chip's free memory fails there with
    RESOURCE_EXHAUSTED. That is the one call for the autoregressive cache outside a jit; given shapes instead of arrays,
    the jit alone makes the zeros, sharded."""
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


def maxtext_source():
    """The installed MaxText: its version and, for an install from git, the URL and commit"""
    import importlib.metadata

    dist = importlib.metadata.distribution("maxtext")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    return {"version": dist.version, "url": direct.get("url"), "commit": direct.get("vcs_info", {}).get("commit_id")}


def patch_maxtext():
    """The two workarounds, each only where this MaxText still needs it: {name: applied}"""
    applied = {"result_tokens_pytree": make_result_tokens_a_pytree(), "decode_cache_sharded": False}
    if not decode_cache_fix_present():
        make_decode_cache_sharded()
        applied["decode_cache_sharded"] = True
    return applied


def make_engine(model, ckpt, tokenizer, work_dir, shapes):
    """MaxText's MaxEngine with the converted checkpoint (ckpt: <checkpoint>/0/items): (engine, params)"""
    from maxtext.configs import pyconfig
    from maxtext.inference.maxengine import maxengine

    chips = MODELS[model]["chips"]
    max_prefill = shapes["prefill_lengths"][-1]
    config = pyconfig.initialize(
        [""], model_name=MODELS[model]["maxtext_model"], load_parameters_path=ckpt, run_name="bt_gemma4",
        log_config=False,
        # Not the home directory: the boot disk is small
        base_output_directory=f"{work_dir}/maxtext_output",
        # pyconfig points JAX's compilation cache here; its default is ~/jax_cache on the boot disk
        jax_cache_dir=f"{work_dir}/jax_cache",
        enable_checkpointing=False, async_checkpointing=False, skip_jax_distributed_system=True, scan_layers=False,
        use_multimodal=False, attention="dot_product", dtype="bfloat16", weight_dtype="bfloat16",
        ici_fsdp_parallelism=1, ici_tensor_parallelism=chips, per_device_batch_size=shapes["slots"] / chips,
        max_prefill_predict_length=max_prefill, max_target_length=max_prefill + shapes["max_decode"],
        decode_sampling_strategy="greedy", return_log_prob=False, tokenizer_type="huggingface",
        tokenizer_path=tokenizer)
    engine = maxengine.MaxEngine(config)
    return engine, engine.load_params()


class EngineSlots:
    """The device side of decode_all: MaxEngine's prefill, insert and generate on one decode state"""

    def __init__(self, engine, params, prefill_lengths):
        self.engine, self.params, self.prefill_lengths = engine, params, prefill_lengths
        self.state = engine.init_decode_state()

    def prefill(self, tokens):
        import jax.numpy as jnp

        length = next((n for n in self.prefill_lengths if n >= len(tokens)), None)
        if length is None:
            raise RuntimeError(f"a prompt of {len(tokens)} tokens is longer than the largest prefill length "
                               f"{self.prefill_lengths[-1]}")
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


def decode_all(prompts, slots, device, max_decode, sync_every, eos_ids):
    """Greedy completions of prompts over a fixed number of slots: [(tokens without the stop token, ended)].

    Continuous batching with the engine's own pieces: every free slot gets the next prompt (prefill, insert; the
    prefill's token is the first of the completion), then sync_every generate steps run for all slots at once and come
    back to the host in one transfer. A slot whose completion ends (a stop token, or max_decode tokens) is free for
    the next prompt; the steps it ran past its end are dropped. Prompts go longest first, so the last steps of a shard
    are spent on short ones.
    """
    queue = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    queue.reverse()  # pop() from the end gives the longest first
    outputs = [None] * len(prompts)
    owner = [-1] * slots
    tokens = [[] for _ in range(slots)]

    def finished(s):
        return bool(tokens[s]) and (tokens[s][-1] in eos_ids or len(tokens[s]) >= max_decode)

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
                ended = tokens[s][-1] in eos_ids
                outputs[owner[s]] = (tokens[s][:-1] if ended else tokens[s], ended)
                owner[s] = -1
    return outputs


def translate_shard(device, codec, tokenizer, ids, texts, shapes, sync_every, eos_ids):
    """The translations of one shard as columns: id, text_<target>, ended, tokens_in, tokens_out (with the stop
    token); and the seconds of decoding"""
    prompts = [codec.encode(text) for text in texts]
    started = time.time()
    outputs = decode_all(prompts, shapes["slots"], device, shapes["max_decode"], sync_every, eos_ids)
    seconds = time.time() - started
    column = f"text_{codec.target}"
    rows = {"id": [], column: [], "ended": [], "tokens_in": [], "tokens_out": []}
    for i, prompt, (completion, ended) in zip(ids, prompts, outputs):
        rows["id"].append(i)
        rows[column].append(tokenizer.decode(completion, skip_special_tokens=True).strip())
        rows["ended"].append(ended)
        rows["tokens_in"].append(len(prompt))
        rows["tokens_out"].append(len(completion) + int(ended))
    return rows, seconds


# The bucket through the Python client, not gsutil: on the v6e image gsutil is a snap, and a snap refuses to start
# ("need to run as root or suid") under a process with no_new_privs, which this one has once libtpu is up.
# google-cloud-storage comes with maxtext[tpu].
def gcs_client():
    from google.cloud import storage

    return storage.Client()


def split_gs(url):
    if not url.startswith("gs://"):
        raise ValueError(f"not a gs:// URL: {url}")
    bucket, _, name = url[len("gs://"):].partition("/")
    if not bucket:
        raise ValueError(f"no bucket in {url}")
    return bucket, name
