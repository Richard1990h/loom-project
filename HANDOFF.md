# HANDOFF — brief for the on-site Claude (Claude Code, user's desktop)

You are joining a from-scratch language-model project mid-flight. Read this
file, then ZERO.md, then loom/RESULTS.md, before executing anything.

## Mission
Build a language model designed for THIS machine (not datacenters):
small looping reasoning core in VRAM + large editable knowledge store in
system RAM, minimizing bytes touched per token. Long-term: coherent chat
and coding, private, running locally.

## Non-negotiable rules (full text in ZERO.md)
1. Permitted foundations: hardware, OS, C compiler, mathematics.
2. Track-2 artifact code: no ML frameworks, no numerics libraries, no
   copied architectures, no pretrained weights. Every choice traces to a
   stated requirement.
3. Declared convergence: if a derivation lands on prior art, keep it and
   say so in ZERO.md.
4. No claim without an execution. Pre-register predictions and kill
   criteria BEFORE running. Publish misses next to hits.
5. Measurement instruments (probes, checkers, plotting) are exempt from
   rule 2 — they are instruments, not the organism. Mark them as such.
6. Ask the human before: any run > 30 minutes, any download > 5 GB, any
   deviation from this brief.

## Two tracks
- **Track 1 — torch prototype** (`loom/`): PyTorch implementation used
  for fast hypothesis testing only. Will be retired.
- **Track 2 — ZERO** (`zero/`): the real artifact. Pure C today
  (day1.c: own PRNG, own reverse-mode autodiff, proofs), CUDA next.

## Verified state (all executed, 1 CPU core, sandbox)
- `loom/tests.py`: 11/11 proofs pass (exact product-key retrieval vs
  brute force; bit-exact causality; ACT weights sum to 1; hypernet
  zero-init identity; recursion adds zero params: 609,217 at k=1 and
  k=8; overfit CE 0.089).
- Controlled run, 533 steps, param-matched (609,217 vs 615,680):
  LOOM val CE 2.090 vs vanilla 2.108 at 3.9× wall-clock. Ablations:
  recursion +0.040 if removed, hypernet +0.038, memory +0.0004 (i.e.
  memory did NOTHING on Shakespeare).
- `zero/day1`: gradcheck vs finite differences max rel err 7.91e-08;
  micro-LM reached 0.1790 nats vs exact entropy floor 0.1777.
- `loom/kstarve.py` (knowledge-starving, pre-registered): 6,000 facts,
  22,561-param core, 9,000 steps. A(with memory) 23.2% recall;
  A(memory ablated) 3.1%; B(never memory) 9.6%.
  **P1 (A ≥ 0.90): FAILED (undertrained — curves still rising at
  cutoff). P2 (ablated < 0.05): passed. P3 (A − B > 0.20): FAILED
  (gap 13.7pp). Kill criterion (A ≤ B): not triggered.**
  Verdict: MIXED. Compute-censored, not settled. Your job #4.

## Target machine
RTX 4070 12 GB (~504 GB/s VRAM, PCIe 4.0 ≈ 25 GB/s effective),
64 GB DDR5, Ryzen 9 16c/32t. Replace all spec-sheet numbers with
measured ones (task 3) before any sizing decision.

## PHASE 0 — execute in order, stop on any mismatch
1. **Environment.** Verify gcc, nvcc, python3, torch with CUDA visible
   on the 4070. Record versions.
2. **Replicate the sandbox.** `cd loom && python3 tests.py` → expect
   11/11 PASS. `cd zero && gcc -O2 -std=c11 -Wall -o day1 day1.c -lm
   && ./day1` → expect gradcheck < 1e-6 and loss → ~0.179.
   Any mismatch: STOP, diagnose, report. Do not train on an engine
   that does not replicate.
3. **Hardware truth probe.** Write and run a probe measuring: VRAM
   bandwidth (GB/s), host→device and device→host PCIe (GB/s), fp32 and
   bf16 matmul throughput (TFLOPs). Raw CUDA preferred; torch
   acceptable if marked. Write `hw.json`. All future design math cites
   this file.
4. **Settle the starving experiment.** Port `loom/kstarve.py` to CUDA
   (move model and FACTS to device; this is Track 1, torch is fine).
   Delete `loom/kck/`, set STEPS_MAX = 60000, remove the time guard,
   rerun fresh. Pre-registered bars UNCHANGED: P1 A ≥ 0.90,
   P2 ablated < 0.05, P3 A − B > 0.20, kill if A ≤ B. Write
   `kstarve_gpu.json`. Report the verdict verbatim, including
   failures. Estimated minutes, not hours.
5. **Report.** Summarize 1–4 to the human with measured numbers, then
   await the next rung (Day 2: derive the optimizer; then scaling
   curve at 2M → 20M → 100M params to forecast the big run before
   committing weeks of GPU).

## File map
- HANDOFF.md — this brief
- ZERO.md — rules, derivations, audit log (append, never rewrite)
- zero/day1.c — Track 2 engine + proofs
- loom/loom.py, tests.py, train.py — Track 1 model, proofs, experiment
- loom/kstarve.py, kstarve.json — starving experiment + sandbox result
- loom/RESULTS.md, results.json — measured Shakespeare run
- loom/shakespeare.txt — corpus for replication
- loom-architecture.html — design-targets page. Numbers on it are
  TARGETS, not results. RESULTS.md and *.json are the only measured
  sources.
