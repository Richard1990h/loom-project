# STATUS — live task queue (a fresh session resumes from here in <1 minute)

**Start every session:** run `make verify` (must be GREEN) and read this file +
CLAUDE.md + ZERO.md. Never build on red. State lives in the repo, not in memory.

Last updated: 2026-07-07. Repo: https://github.com/Richard1990h/loom-project

## Task queue (Days 4-5 as specified)
| # | rung | state | commit |
|---|------|-------|--------|
| 4a | fp32 engine port + dual-precision gradcheck | **done** | 74dc114 |
| gov | governance: reference.json, verify.sh, branch protection, STATUS, prove-it (verify RED demonstrated) | **done*** | a5511a9 |
| gov-ci | push CI workflow + demonstrate CI RED — BLOCKED: needs a `workflow`-scoped GitHub token (current token is `repo` only). verify.yml authored + staged locally | **blocked** | — |
| 4b | hand-written CUDA kernels (matmul ≥3× tiled, fused attn fwd/bwd, optimizer, gather/scatter) + hw.json | todo | — |
| 4c | CPU-fp64 vs GPU-fp32 training parity (gate before GPU training) | todo | — |
| 5d | BPE tokenizer in C (lossless round-trip proof, chars/token) | todo | — |
| 5e | full transformer LM in ZERO engine (causality + gradcheck on assembled stack) | todo | — |
| 5f | sampling + top-k + chat REPL binary + checkpoint format | todo | — |
| 5g | milestone pretrain (2–8M) + finetune on self-authored dialogues | todo | — |
| 5h | DAY5-FIRST-CONVERSATION report (raw transcript, verbatim) | todo | — |

States: todo / doing / done. Update the state + commit sha as each rung lands.

## Machine continuity — set ONCE on the host for multi-day runs (non-expert steps)
1. **Stop the PC sleeping mid-train:** Windows Settings → System → Power &
   battery → Screen and sleep → set "When plugged in, put my device to sleep"
   to **Never** (also set hibernate off: run `powercfg /hibernate off` in an
   admin PowerShell). WSL keeps running only while Windows is awake.
2. **Check the training is alive:** open Ubuntu and run
   `tmux attach -t train` (the run lives in a tmux session); detach with
   Ctrl-b then d. The watchdog is `scripts/watchdog.sh` — `pgrep -f watchdog`
   should show it running.
3. **Watch progress / find problems:** `tail -f ~/loom-project/runs/latest/train.log`
   shows live loss + checkpoints; the newest checkpoint is
   `~/loom-project/runs/latest/ckpt-*.bin`. A crashed run auto-resumes from the
   last checkpoint via the watchdog; if the log stops growing for >10 min and
   no new checkpoint appears, tell the human.

(Long-run infrastructure — tmux launcher, watchdog restart-from-checkpoint,
run dir layout — is created with rung 5g when the first real training starts.)
