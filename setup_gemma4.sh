#!/bin/bash
# On every worker of a TPU VM of v4 hosts: the environment of the Gemma 4 translation (translate_gemma4.py), in
# /dev/shm/flan_nl. Each step is skipped when its result is there.
#   bash setup_gemma4.sh
# venv-g4: MaxText from git at a fixed commit of the fork yhavinga/maxtext (branch fix-nnx-decode-cache-abstract-init:
# MaxText's main of 9 Nov 2026 with the NNX decode cache made from shapes inside the init jit; main has the
# sliding-window decode mask measured in true positions, which MaxText 0.2.5 of setup.sh lacks), installed with the
# fork's tpu-requirements.txt, plus langid, PyYAML and CPU torch (for MetricX's weights). Without JetStream:
# gemma4_engine.py registers MaxText's stub ResultTokens as a pytree. VENV=<dir> takes an existing venv instead, when
# it was installed from the same fork commit (checked). tokenizer-gemma4: Gemma 4 31B-it's tokenizer and chat
# template at the revision of models.yml.
# The checkpoint is not made here: Gemma 4 31B-it converted by MaxText's to_maxtext (CKPT of translate_gemma4.sh).
# Needs ~/.cache/huggingface/token (Gemma 4 is gated); the token is never printed.
set -euo pipefail
SHM=/dev/shm/flan_nl
FORK=https://github.com/yhavinga/maxtext.git
FORK_COMMIT=2249474b93b971d8248520e3a09b94b4ce626ce4
GEMMA4=google/gemma-4-31B-it
GEMMA4_REV=842da3794eaa0b77d5f08bae87a17459d91ff475
VENV=${VENV:-$SHM/venv-g4}
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$SHM/uv-cache UV_PYTHON_INSTALL_DIR=$SHM/python HF_HUB_DISABLE_PROGRESS_BARS=1
[ -s ~/.cache/huggingface/token ] || { echo "$(hostname): no ~/.cache/huggingface/token"; exit 1; }
mkdir -p "$SHM" ~/logs
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v uv > /dev/null || { echo "$(hostname): uv is not on PATH after installing it"; exit 1; }

maxtext_commit() {
  "$1/bin/python" -c "import importlib.metadata as m, json; d = m.distribution('maxtext'); print(json.loads(d.read_text('direct_url.json') or '{}').get('vcs_info', {}).get('commit_id'))"
}
if [ -x "$VENV/bin/python" ] && [ "$VENV" != "$SHM/venv-g4" ]; then
  [ "$(maxtext_commit "$VENV")" = "$FORK_COMMIT" ] || { echo "$(hostname): $VENV has not MaxText $FORK_COMMIT"; exit 1; }
elif [ ! -s "$VENV/done.txt" ]; then
  [ -x "$VENV/bin/python" ] || uv venv -q --python 3.12 "$VENV"
  SRC=$SHM/maxtext-src
  [ -d "$SRC/.git" ] || git clone -q "$FORK" "$SRC"
  git -C "$SRC" fetch -q origin "$FORK_COMMIT"
  git -C "$SRC" checkout -q "$FORK_COMMIT"
  VIRTUAL_ENV=$VENV uv pip install -q -r "$SRC/src/dependencies/requirements/generated_requirements/tpu-requirements.txt"
  VIRTUAL_ENV=$VENV uv pip install -q --no-deps "maxtext @ git+$FORK@$FORK_COMMIT"
  VIRTUAL_ENV=$VENV uv pip install -q langid==1.1.6 pyyaml
  VIRTUAL_ENV=$VENV UV_TORCH_BACKEND=cpu uv pip install -q torch==2.9.1
  "$VENV/bin/python" -c "import importlib.metadata as m; print({p: m.version(p) for p in ('maxtext', 'jax', 'jaxlib', 'libtpu', 'flax', 'transformers', 'tokenizers', 'pyarrow', 'torch')})" \
    > "$VENV/done.txt"
fi
"$VENV/bin/python" -c "import yaml, langid, pyarrow, transformers, jax, google.cloud.storage" || { echo "$(hostname): $VENV lacks a package"; exit 1; }
echo "$(hostname): venv $VENV, MaxText $(maxtext_commit "$VENV")"
TOK=$SHM/tokenizer-gemma4
[ -s "$TOK/tokenizer.json" ] || "$VENV/bin/python" -c "from huggingface_hub import snapshot_download as s; s('$GEMMA4', revision='$GEMMA4_REV', local_dir='$TOK', allow_patterns=['tokenizer*', 'chat_template.jinja', 'config.json', 'generation_config.json'])"
echo "$(hostname): ready in $SHM, $(df -h /dev/shm | tail -1 | awk '{print $4}') free in /dev/shm"
