#!/usr/bin/env bash
# Milestone pretrain launcher (instrument). Always uses `resume`, so a fresh
# start inits from scratch and any restart continues from the last checkpoint.
# Logs to runs/latest/train.log; touches runs/latest/DONE on clean completion.
cd "$(dirname "$0")/.." || exit 1
STEPS=${1:-20000}
mkdir -p runs/latest
echo "=== $(date) train_run start (target $STEPS steps) ===" >> runs/latest/train.log
zero/gpt train data/corpus.u16 runs/latest/ckpt.bin "$STEPS" resume >> runs/latest/train.log 2>&1
rc=$?
echo "=== $(date) gpt exited rc=$rc ===" >> runs/latest/train.log
[ $rc -eq 0 ] && touch runs/latest/DONE
exit $rc
