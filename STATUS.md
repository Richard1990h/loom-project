# STATUS — live task queue (a fresh session resumes from here in <1 minute)

**Start every session:** run `make verify` (must be GREEN) and read this file +
CLAUDE.md + ZERO.md. Never build on red. State lives in the repo, not in memory.

**ACTIVE BRANCH: `work`** — resume here, not on `main`. Landing workflow (see
ZERO.md AMENDMENT 2026-07-07): develop on `work`, push it to origin after every
task; `main` is updated ONLY via a CI-gated PR at each rung boundary
(enforce_admins ON — direct pushes to main are rejected by design). Sandbox
review cites the `main` merge-commit sha.

Last updated: 2026-07-07. Repo: https://github.com/Richard1990h/loom-project

## Task queue (Days 4-5 as specified)
| # | rung | state | commit |
|---|------|-------|--------|
| 4a | fp32 engine port + dual-precision gradcheck | **done** | 74dc114 |
| gov | governance: reference.json, verify.sh, branch protection, STATUS, prove-it (verify RED demonstrated) | **done*** | a5511a9 |
| gov-ci | CI workflow live; GREEN on main; CI-RED demoed; protection binds (red PR merge refused) | **done** | b52ef89 |
| 4b | CUDA kernels: SGEMM 14303 GFLOP/s (bar 8190 HIT), fused attn fwd/bwd, RMSProp, gather/scatter — all fp64-verified; hw.json updated | **done** | c4544ce |
| data | Data Refinery stage-1: 14 docs / 5.68 MiB Gutenberg, manifest+provenance; DATA.md; human checkpoints | **done** | b4b1cd2 |
| 4c | CPU-fp64 vs GPU-fp32 parity PASS (worst200 3.98e-4, final rel 0.94%); GPU training unlocked | **done** | (landing) |
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

## Human checkpoints — moments the human personally tests
These belong to the human; STOP and alert loudly when each is reached. The
human's verdict is logged VERBATIM in ZERO.md (their words, not a summary).
1. **Day 5 — `make talk`.** After the chat fine-tune, the human runs the REPL
   and holds a live conversation. Machine cannot self-certify this.
2. **Blind A/B (base vs chat-tuned).** Same 10 prompts through both models,
   outputs shown blind/unlabeled; the human picks which is better per prompt.
3. **Tier-2 — 20 everyday questions.** After tier-2 training, the human asks
   20 ordinary questions and judges coherence.
Each verdict → an entry in ZERO.md quoting the human exactly.
