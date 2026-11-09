#!/bin/bash
# On every worker of a TPU VM of v4 hosts (the published run: a v4-32, 4 workers): the environment of the translation,
# and on one worker the conversion of TranslateGemma into a MaxText checkpoint in the bucket.
#   ROOT=gs://<bucket>/<dir> bash setup.sh            the venv (MaxText 0.2.5, JAX for TPU) and a check of it
#   ROOT=gs://<bucket>/<dir> bash setup.sh convert    also: TranslateGemma from Hugging Face through /dev/shm into
#                                                     $CKPT, marked done last
# Everything lives in /dev/shm/flan_nl: the boot disks of the published run had ~9 GB free, while the hosts have ~400 GB
# of RAM. A reboot empties /dev/shm; a second start rebuilds what is gone and keeps what is there.
# Needs ~/.cache/huggingface/token (the model is gated); the token is never printed.
# Environment: ROOT (required: a gs:// directory for the checkpoint, the plan and the translations).
# Overrides: MODEL (27b or 12b), CKPT (the checkpoint directory in the bucket).
set -euo pipefail
MODEL=${MODEL:-27b}
case $MODEL in
  27b) REVISION=7d10f0b72f89a2d0f268cea30727d8b77c0d25c2 ;;
  12b) REVISION=d1b225e1caa17f1ddc7e62065d8637d0923f34e2 ;;
  *) echo "MODEL is 27b or 12b, not $MODEL"; exit 1 ;;
esac
REPO=google/translategemma-$MODEL-it
ROOT=${ROOT:?set ROOT to a gs:// directory for the checkpoint, the plan and the translations}
CKPT=${CKPT:-$ROOT/translategemma-$MODEL-maxtext}
SHM=/dev/shm/flan_nl
VENV=$SHM/venv
PY=$VENV/bin/python
MAXTEXT_PIN=0.2.5
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$SHM/uv-cache UV_PYTHON_INSTALL_DIR=$SHM/python
[ -s ~/.cache/huggingface/token ] || { echo "$(hostname): no ~/.cache/huggingface/token"; exit 1; }
mkdir -p "$SHM" ~/logs
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v uv > /dev/null || { echo "$(hostname): uv is not on PATH after installing it"; exit 1; }

# MaxText 0.2.5 needs Python 3.12; the image has 3.10. --resolution=lowest is how MaxText pins its extras: the floors
# are its tested lock. No JetStream: its token_utils pulls in seqio and
# TensorFlow, and translate.py needs only the engine (it makes MaxText's stand-in for ResultTokens a pytree).
[ -x "$PY" ] || uv venv -q --python 3.12 "$VENV"
if ! VIRTUAL_ENV=$VENV uv pip show maxtext 2> /dev/null | grep -q "^Version: $MAXTEXT_PIN$"; then
  VIRTUAL_ENV=$VENV uv pip install -q "maxtext[tpu]==$MAXTEXT_PIN" --resolution=lowest
fi
# Language ID of the translations (translate.py)
VIRTUAL_ENV=$VENV uv pip install -q langid==1.1.6
"$PY" -c "import importlib.metadata as m; print('$(hostname):', {p: m.version(p) for p in ('maxtext', 'jax', 'jaxlib', 'libtpu', 'flax', 'orbax-checkpoint', 'transformers', 'tokenizers', 'pyarrow', 'langid')})"

# The tokenizer and the template of the model, small; every worker needs them to build its prompts
TOK=$SHM/tokenizer-$MODEL
if [ ! -s "$TOK/tokenizer.json" ]; then
  "$PY" -c "from huggingface_hub import snapshot_download as s; s('$REPO', revision='$REVISION', local_dir='$TOK', allow_patterns=['tokenizer*', 'special_tokens_map.json', 'added_tokens.json', 'chat_template.jinja', 'config.json', 'generation_config.json'])"
fi

[ "${1:-}" = convert ] || { echo "$(hostname): environment ready in $VENV, $(df -h /dev/shm | tail -1 | awk '{print $4}') free in /dev/shm"; exit 0; }

if gsutil -q stat "$CKPT/done.txt"; then
  echo "$(hostname): $CKPT is done before"
  exit 0
fi
# Only the conversion reads bf16 safetensors, which numpy cannot; the translation itself has no torch
VIRTUAL_ENV=$VENV UV_TORCH_BACKEND=cpu uv pip install -q torch==2.9.1
HF=$SHM/hf-$MODEL
echo "$(hostname): downloading $REPO@$REVISION to $HF at $(date +%H:%M:%S)"
HF_HUB_DISABLE_PROGRESS_BARS=1 "$PY" -c "from huggingface_hub import snapshot_download as s; s('$REPO', revision='$REVISION', local_dir='$HF')"
echo "$(hostname): converting at $(date +%H:%M:%S), $(du -sh "$HF" | cut -f1) of weights"
# A half-written checkpoint from an earlier attempt would be mixed with this one: orbax refuses an existing step
gsutil -q ls "$CKPT/" > /dev/null 2>&1 && { echo "$(hostname): $CKPT exists without done.txt; remove it first (gsutil -m rm -r $CKPT)"; exit 1; }
JAX_PLATFORMS=cpu "$PY" -u "$(dirname "$0")/convert.py" --hf-dir "$HF" --model "gemma3-$MODEL" --out "$CKPT"
# orbax writes commit_success.txt last, so its absence means an unfinished save
gsutil -q stat "$CKPT/0/commit_success.txt" || { echo "$(hostname): no $CKPT/0/commit_success.txt after the conversion"; exit 1; }
echo "$REPO@$REVISION converted by MaxText $MAXTEXT_PIN on $(hostname) at $(date -u +%FT%TZ)" | gsutil -q cp - "$CKPT/done.txt"
# The safetensors stay in $HF until they are removed by hand (rm -f $HF/*.safetensors): this script deletes nothing
echo "$(hostname): $CKPT done at $(date +%H:%M:%S)"
