#!/usr/bin/env bash
# Watchdog (instrument): keeps the pretrain alive and backs up checkpoints.
# - if `gpt train` is not running and the run is not DONE, restart it (which
#   resumes from the last checkpoint);
# - every loop, copy the checkpoint to an off-WSL backup (survives WSL issues);
# - exit when runs/latest/DONE appears.
# Usage: bash scripts/watchdog.sh [STEPS]
cd "$(dirname "$0")/.." || exit 1
STEPS=${1:-20000}
BK=/mnt/c/Users/Richard/Downloads/loom-ckpt-backup; mkdir -p "$BK"
mkdir -p runs/latest
while [ ! -f runs/latest/DONE ]; do
  if ! pgrep -f 'gpt train' >/dev/null 2>&1; then
    echo "$(date) watchdog: (re)starting training" >> runs/latest/watchdog.log
    bash scripts/train_run.sh "$STEPS" &
    sleep 10
  fi
  cp -f runs/latest/ckpt.bin "$BK/ckpt.bin" 2>/dev/null
  sleep 60
done
cp -f runs/latest/ckpt.bin "$BK/ckpt.bin" 2>/dev/null
echo "$(date) watchdog: DONE" >> runs/latest/watchdog.log
