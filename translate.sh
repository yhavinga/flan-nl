#!/bin/bash
# On every worker of a TPU VM of v4 hosts (the published run: a v4-32, 4 workers), after setup.sh and the plan
# (README.md): this worker's segments through TranslateGemma, length class after length class (translate.py), detached
# from ssh, each worker a TPU of its own 4 chips.
#   ROOT=gs://<bucket>/<dir> bash translate.sh
# Log: ~/logs/flan_nl_translate.log, a fresh file per start that ends in "EXIT <code>"; a copy goes to $ROOT/logs/.
# Shards land in $OUT/<class>/; a second start skips every shard that is there.
# Environment: ROOT (required: the gs:// directory of setup.sh, with the checkpoint, the plan and the translations).
# Overrides: MODEL (27b, the model of the run, or 12b), SEGMENTS, OUT, CLASSES, EXTRA (more flags, e.g.
# "--max-shards 1"), NAME (of the log, for the pilot and the measurements).
set -euo pipefail
ROOT=${ROOT:?set ROOT to the gs:// directory of the checkpoint, the plan and the translations}
MODEL=${MODEL:-27b}
CKPT=$ROOT/translategemma-$MODEL-maxtext
SHM=/dev/shm/flan_nl
PY=$SHM/venv/bin/python
CODE=$(cd "$(dirname "$0")" && pwd)
WORKER=$(hostname | sed 's/.*-w-//')
SEGMENTS=${SEGMENTS:-$ROOT/plan2/segments-w$WORKER.parquet}
OUT=${OUT:-$ROOT/translations/$MODEL/w$WORKER}
CLASSES=${CLASSES:-tiny small medium long xlong}
EXTRA=${EXTRA:-}
NAME=${NAME:-flan_nl_translate}

[ -x "$PY" ] || { echo "$(hostname): no $PY, run setup.sh first"; exit 1; }
[ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "$(hostname): the TPU is in use"; exit 1; }
gsutil -q stat "$CKPT/done.txt" || { echo "$(hostname): $CKPT has no done.txt (setup.sh convert)"; exit 1; }
[ -s "$SHM/tokenizer-$MODEL/tokenizer.json" ] || { echo "$(hostname): no tokenizer in $SHM/tokenizer-$MODEL (setup.sh)"; exit 1; }
mkdir -p ~/logs "$SHM/segments"
local_segments=$SHM/segments/$(basename "$SEGMENTS")
gsutil -q cp "$SEGMENTS" "$local_segments"
case $MODEL in 27b) MAXTEXT_MODEL=gemma3-27b ;; 12b) MAXTEXT_MODEL=gemma3-12b ;; esac

# A host of the v4-32 as a TPU of its own: without these variables JAX takes the slice from the TPU metadata and waits
# for the other hosts (exit 134 on 7 Nov). MaxText runs with skip_jax_distributed_system, so nothing initializes a
# runtime of several processes.
cat > ~/run_$NAME.sh <<EOF
set -uo pipefail
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
# The engine compiles a program per prefill length and number of packed prompts, minutes in all; a restart reuses them
export JAX_COMPILATION_CACHE_DIR=$SHM/jax_cache
timeout 600 $PY -c "import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())" || { echo "preflight failed: not 4 chips of one process"; exit 2; }
echo "\$(date +%H:%M:%S) $(hostname): $SEGMENTS -> $OUT with $CKPT"
for class in $CLASSES; do
  $PY -u $CODE/translate.py --segments $local_segments --out $OUT/\$class --ckpt $CKPT/0/items \
    --tokenizer $SHM/tokenizer-$MODEL --model $MAXTEXT_MODEL --length-class \$class $EXTRA || exit \$?
done
EOF
setsid nohup bash -c "bash ~/run_$NAME.sh > ~/logs/$NAME.log 2>&1; echo EXIT \$? >> ~/logs/$NAME.log; gsutil -q cp ~/logs/$NAME.log $ROOT/logs/$NAME-w$WORKER.log" > /dev/null 2>&1 < /dev/null &
echo "started $NAME on $(hostname): ~/logs/$NAME.log, shards in $OUT"
