#!/usr/bin/env python3
"""TranslateGemma (a Gemma3ForConditionalGeneration checkpoint) to a MaxText checkpoint of its language model only.

Usage (setup.sh convert, on the CPU of one worker):
    JAX_PLATFORMS=cpu python convert.py --hf-dir /dev/shm/flan_nl/hf-27b --model gemma3-27b --out gs://.../translategemma-27b-maxtext

This is MaxText's own converter (maxtext.checkpoint_conversion.to_maxtext, lazy loading, unscanned layers, no vision
tower) with one difference: the names of the tensors. MaxText 0.2.5 maps Gemma 3 to the names that transformers >= 4.52
gives the multimodal model (`model.language_model.layers.N...`), while TranslateGemma's safetensors still carry the
older layout (`language_model.model.layers.N...`). Its eager path renames through transformers; the lazy path reads the
files and does not, so the loader here asks for each tensor under the name the files use.
"""

import argparse
import os
import sys

# Before jax starts: the checkpoint is sharded over this many simulated CPU devices, as to_maxtext does from its CLI
SIMULATED_DEVICES = 16
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={SIMULATED_DEVICES}"

import jax  # noqa: E402

MAXTEXT_PREFIX = "model.language_model."
FILE_PREFIX = "language_model.model."


def file_key(key):
    """The name in TranslateGemma's safetensors of the tensor MaxText asks for."""
    return FILE_PREFIX + key[len(MAXTEXT_PREFIX):] if key.startswith(MAXTEXT_PREFIX) else key


def main(argv=None):
    parser = argparse.ArgumentParser(description="TranslateGemma's language model to a MaxText checkpoint")
    parser.add_argument("--hf-dir", required=True, help="Local snapshot of the Hugging Face repo")
    parser.add_argument("--model", required=True, choices=["gemma3-27b", "gemma3-12b"])
    parser.add_argument("--out", required=True, help="base_output_directory; the parameters land in <out>/0/items")
    args = parser.parse_args(argv)

    jax.config.update("jax_platforms", "cpu")
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    from maxtext.checkpoint_conversion import to_maxtext

    original = to_maxtext.LazyHFLoader.get_tensor

    def get_tensor(self, key):
        return original(self, file_key(key))

    to_maxtext.LazyHFLoader.get_tensor = get_tensor
    to_maxtext.main(
        args=[sys.argv[0], f"model_name={args.model}", "scan_layers=false", "use_multimodal=false",
              "skip_jax_distributed_system=True", "hardware=cpu", f"base_output_directory={args.out}",
              "run_name=translategemma_conversion"],
        lazy_load_tensors=True, hf_model_path=args.hf_dir, save_dtype="bfloat16",
        simulated_cpu_devices_count=SIMULATED_DEVICES)


if __name__ == "__main__":
    main()
