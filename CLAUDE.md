# CLAUDE.md — self-brief for every session (read this first)

This project is a from-scratch language model built at the lowest practical
layer. If your instructions reference checks or tasks you cannot find, read
CLAUDE.md, ZERO.md, HANDOFF.md in that order before asking.

Canonical tree lives in the **Ubuntu (WSL2) Linux filesystem** at
`~/loom-project` (`/root/loom-project`). The session host is native Windows;
drive ALL build/run steps through `wsl.exe -d Ubuntu` (the default WSL distro
is `docker-desktop` — always name Ubuntu explicitly). The Windows copy under
`C:\Users\Richard\Downloads\loom-project` is disposable.

Machine (gated, verified this session): RTX 4070 12 GB (12282 MiB, driver
610.47), Ryzen 9 7950X, 64 GB DDR5.

## PURITY DECISION (permanent)
- The `loom/` PyTorch prototype is **RETIRED** and never executes on this
  machine. It is archived as `archive-track1-torch/`.
- **No PyTorch, no ML libraries, no numerics libraries, ever.**
- Permitted foundations: hardware, OS, C compilers, mathematics.
- `nvcc` + the CUDA runtime are admitted as "the C compiler and driver layer
  for the GPU." GPU libraries (**cuBLAS, cuDNN, thrust are forbidden**). Every
  kernel is written by hand.
- Measurement instruments (probes, checkers, plotting) are instruments, not
  the organism — exempt from the no-library rule, but mark them as such.

## STANDING ORDERS
- No claim without an execution.
- Pre-register predictions and kill criteria BEFORE measuring.
- Publish misses next to hits.
- Ask the human before any run > 30 minutes or any download > 5 GB (the CUDA
  toolkit download is pre-approved).
- Never improvise missing context — consult CLAUDE.md, ZERO.md, HANDOFF.md in
  that order, then ask.
- Commit after every completed task with the measured numbers in the message.

## PHASE 0 (PURE) plan — in order, stop on any mismatch
0. **FINGERPRINT** — confirm the tree and sentinels before anything runs.
1. **Replicate the engine.** `cd zero && gcc -O2 -std=c11 -Wall -o day1 day1.c
   -lm && ./day1` → gradcheck max rel err < 1e-6 and final loss ~0.179
   (sandbox reference: 7.91e-08 / 0.1790). Mismatch → STOP.
2. **GPU toolchain.** `nvidia-smi` works inside Ubuntu; install CUDA toolkit
   for WSL if `nvcc` absent; hand-written CUDA probe prints device name + SM
   count (must be RTX 4070). No GPU libraries.
3. **Hardware truth → `hw.json`.** Every probe hand-written C/CUDA: host RAM
   triad GB/s; CPU fp32/fp64 GFLOPs; VRAM streaming-copy GB/s; PCIe H2D/D2H
   GB/s (pinned + pageable, timed cudaMemcpy); GPU fp32 matmul GFLOPs with
   BOTH a naive and a shared-memory tiled kernel. bf16/tensor-core: record
   null with a note (deferred, never faked). Spec sheets are not data.
4. **Day 2 — the optimizer.** Pre-register GD-degradation predictions
   (kappa = 1, 100, 10000) in ZERO.md; build `zero/day2.c` on the day1 engine
   with a quadratic-bowl benchmark + the day1 char-LM as a second testbed;
   derive per-parameter step adaptation from observed geometry; implement
   ours; benchmark vs plain GD at equal budgets; publish hits AND misses;
   day1 gradcheck must still pass after engine changes.
5. **Report.** Summarize every measured number, append to ZERO.md, commit
   `PHASE0-PURE-COMPLETE`, push, print `hw.json` + the optimizer table.

## Two tracks (historical)
- Track 1 — `archive-track1-torch/` (was `loom/`): PyTorch prototype. RETIRED
  before ever executing on the desktop. Sandbox-era measurements in its
  RESULTS.md and *.json remain valid historical records only.
- Track 2 — `zero/`: the real artifact. Pure C today (day1.c), CUDA next.

## Provenance note
The GitHub repo `Richard1990h/loom-project` (branch `main`) is authoritative
for this pure track. A prior improvised wrong-environment attempt is archived
on the RL branch `claude/loom-sandbox-replication-tyfmkz` — leave it alone.
