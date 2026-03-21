#!/bin/bash
# Auto-launched training script: wait for data → kill old → train best model
# Runs independently via nohup; output → /home/xlisp/PyPro/CodeGPT/train_best.log

LOG=/home/xlisp/PyPro/CodeGPT/train_best.log
CODEGPT=/home/xlisp/PyPro/CodeGPT
PYTHON=~/miniconda3/envs/codegpt/bin/python

exec >> "$LOG" 2>&1

echo "=============================================="
echo "$(date) — run_best_training.sh started"
echo "=============================================="

# ── 1. wait for 101 shards ────────────────────────────────────────────────────
echo "$(date) — Waiting for 101 shards to finish downloading..."
while [ $(ls ~/.cache/autoresearch/data/*.parquet 2>/dev/null | wc -l) -lt 101 ]; do
  n=$(ls ~/.cache/autoresearch/data/*.parquet 2>/dev/null | wc -l)
  echo "$(date) — $n/101 shards, $(du -sh ~/.cache/autoresearch/data/ | cut -f1)"
  sleep 60
done
echo "$(date) — All 101 shards ready: $(du -sh ~/.cache/autoresearch/data/ | cut -f1)"

# ── 2. kill old training processes ───────────────────────────────────────────
OLD=$(pgrep -f "train_autoresearch.py")
if [ -n "$OLD" ]; then
  echo "$(date) — Killing old training PIDs: $OLD"
  kill $OLD
  sleep 5
fi
pkill -f "prepare.py" 2>/dev/null
sleep 3

# ── 3. verify GPU free ────────────────────────────────────────────────────────
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
sleep 5

# ── 4. launch best-effect training ───────────────────────────────────────────
echo "$(date) — Starting BEST EFFECT training:"
echo "  DEPTH=12  (~125M params)"
echo "  TIME_BUDGET=86400s  (24 hours)"
echo "  TOTAL_BATCH_SIZE=131072"
echo "  DEVICE_BATCH_SIZE=2"
echo "  100 training shards from climbmix-400b-shuffle"
echo "----------------------------------------------"

cd "$CODEGPT"
$PYTHON -W ignore train_autoresearch.py

echo "=============================================="
echo "$(date) — Training finished!"
echo "Checkpoint: $CODEGPT/out-autoresearch/ckpt.pt"
echo "Run REPL:   $PYTHON $CODEGPT/repl_autoresearch.py"
echo "=============================================="
