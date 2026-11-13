#!/bin/bash
# On every worker of a TPU VM of v4 hosts, after setup_gemma4.sh and the plan (README.md): this worker's segments
# through Gemma 4 31B-it, length class after length class (translate_gemma4.py), detached from ssh, each worker a TPU
# of its own 4 chips. The Gemma 4 counterpart of translate.sh.
#   ROOT=gs://<bucket>/<dir> CKPT=gs://<bucket>/<dir>/gemma-4-31b-it-maxtext bash translate_gemma4.sh
# Log: ~/logs/$NAME.log, a fresh file per start that ends in "EXIT <code>"; a copy goes to $ROOT/logs/. Shards land in
# $OUT/<class>/; a second start skips every shard that is there.
# Environment: ROOT (required: the gs:// directory of the plan and the translations), CKPT (required: Gemma 4 31B-it as
# a MaxText checkpoint, the directory that holds 0/items).
# Overrides: VENV (setup_gemma4.sh's), SEGMENTS, OUT, CLASSES, EXTRA (more flags, e.g. "--max-shards 1"), NAME (of the
# log and the job).
set -euo pipefail
ROOT=${ROOT:?set ROOT to the gs:// directory of the plan and the translations}
CKPT=${CKPT:?set CKPT to the MaxText checkpoint of Gemma 4 31B-it, the directory that holds 0/items}
SHM=/dev/shm/flan_nl
VENV=${VENV:-$SHM/venv-g4-f4d8615}  # setup_gemma4.sh's default; the pilot ran on the venv of 2249474
PY=$VENV/bin/python
CODE=$(cd "$(dirname "$0")" && pwd)
WORKER=$(hostname | sed 's/.*-w-//')
SEGMENTS=${SEGMENTS:-$ROOT/plan2/segments-w$WORKER.parquet}
OUT=${OUT:-$ROOT/translations/g4/w$WORKER}
CLASSES=${CLASSES:-tiny small medium long xlong}
EXTRA=${EXTRA:-}
NAME=${NAME:-flan_nl_translate_g4}

[ -x "$PY" ] || { echo "$(hostname): no $PY, run setup_gemma4.sh first"; exit 1; }
[ -e ~/logs/$NAME.log ] && { echo "$(hostname): ~/logs/$NAME.log exists; a new start takes a new NAME"; exit 1; }
[ -z "$(sudo lsof -t /dev/accel* /dev/vfio/* 2> /dev/null)" ] || { echo "$(hostname): the TPU is in use"; exit 1; }
gsutil -q stat "$CKPT/0/commit_success.txt" || { echo "$(hostname): $CKPT has no finished checkpoint (0/commit_success.txt)"; exit 1; }
[ -s "$SHM/tokenizer-gemma4/tokenizer.json" ] || { echo "$(hostname): no tokenizer in $SHM/tokenizer-gemma4 (setup_gemma4.sh)"; exit 1; }
mkdir -p ~/logs "$SHM/segments"
local_segments=$SHM/segments/$NAME-$(basename "$SEGMENTS")
gsutil -q cp "$SEGMENTS" "$local_segments"

# A host of the v4-32 as a TPU of its own (translate.sh); MaxText dumps HLO to /tmp/xla_dump whenever XLA_FLAGS is empty
cat > ~/run_$NAME.sh <<EOF
set -uo pipefail
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3 TPU_PROCESS_PORT=8476
export TPU_PROCESS_ADDRESSES=localhost:8476 CLOUD_TPU_TASK_ID=0 ALLOW_MULTIPLE_LIBTPU_LOAD=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 XLA_FLAGS=--xla_dump_hlo_as_text=false TPU_LOG_DIR=$SHM/tpu_logs
timeout 600 $PY -c "import jax; assert (jax.device_count(), jax.process_count()) == (4, 1), (jax.device_count(), jax.process_count()); print('devices', jax.devices())" || { echo "preflight failed: not 4 chips of one process"; exit 2; }
echo "\$(date +%H:%M:%S) $(hostname): $SEGMENTS -> $OUT with $CKPT"
for class in $CLASSES; do
  $PY -u $CODE/translate_gemma4.py --segments $local_segments --out $OUT/\$class --ckpt $CKPT/0/items \
    --tokenizer $SHM/tokenizer-gemma4 --length-class \$class $EXTRA || exit \$?
done
EOF
setsid nohup bash -c "bash ~/run_$NAME.sh > ~/logs/$NAME.log 2>&1; echo EXIT \$? >> ~/logs/$NAME.log; gsutil -q cp ~/logs/$NAME.log $ROOT/logs/$NAME-w$WORKER.log" > /dev/null 2>&1 < /dev/null &
echo "started $NAME on w$WORKER: ~/logs/$NAME.log, shards in $OUT"
