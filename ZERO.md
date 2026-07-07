# ZERO — design log

A language model built from the lowest practical layer, with every choice
derived, every derivative proven, and every convergence with prior art
declared. This document is the audit trail.

## Rules of engagement
1. **Permitted foundations:** hardware, operating system, C compiler,
   mathematics. (The only thing beneath this layer is fabricating
   silicon.)
2. **Forbidden:** ML frameworks, numerics libraries, copied
   architectures, pretrained weights, and any choice justified by
   "that's how it's done." Every choice must trace to a stated
   requirement.
3. **Convergence is declared, not hidden.** If our derivation lands
   where prior work landed, we keep the result because *we* derived it,
   and we say so. Independent derivation reaching the same answer is
   verification. Divergence from convention is only taken when a
   requirement forces it — and that is exactly where novelty lives.
4. **Nothing is claimed without a measurement or a proof.**

## Derivation chain (so far)

**Language → C.** Requirement: the project's core thesis is minimizing
bytes moved per token on a consumer machine (12 GB VRAM at ~504 GB/s,
64 GB DDR5 behind a ~25 GB/s PCIe link). Therefore the language must
expose every allocation and every byte, and must extend directly to
CUDA kernels for the target GPU. C satisfies both; nothing hidden.

**Learning signal → reverse-mode differentiation.** Requirement:
adjust millions of parameters against one scalar loss. Chain rule,
applied backward over the computation graph, yields *all* parameter
gradients for approximately the cost of one forward pass; forward-mode
would cost one pass *per parameter*. The math picks reverse mode.
(Declared convergence: this is backpropagation. We take it from the
chain rule, not from a framework.)

**Backward rules, derived by hand and proven:**
- matmul: dA = dY·Bᵀ, dB = Aᵀ·dY
- broadcast bias: db_j = Σ_i dY_ij
- tanh: dx = (1 − y²)·dy
- softmax+cross-entropy fused: ∂L/∂z_ij = (p_ij − 1[j=t_i])/m,
  from d(logsumexp)/dz_j = p_j. (Declared convergence: classical
  result; re-derived, not imported.)

**Randomness → own xorshift64\*.** No dependence on libc's rand.

## Day-one proofs (measured, `day1.c`, gcc 13.3, double precision)

- **Gradient proof:** 24 sampled entries across all 4 parameter
  tensors, analytic vs central finite differences:
  max relative error **7.91e-08**. The only referee is calculus.
- **Learning proof:** first model (2,078 params, 2-char context,
  one hidden layer) trained by our engine on one sentence of our own:
  loss **2.6296 → 0.1790** nats/char.
- **Optimality proof:** the exact conditional entropy of the data
  (computed independently) is **0.1777** nats/char. The model landed
  within **0.0013 nats of information-theoretically perfect** — it
  extracted essentially all extractable structure.
- Honest artifact: greedy decoding cycles ("the weaves weaves…")
  because argmax at genuinely ambiguous contexts always takes the
  majority branch. The data has ambiguity; deterministic decoding
  turns ambiguity into loops. This is a property of decoding, not a
  training failure — and it is *why* sampling will be derived later,
  from this observed requirement.

## The ladder ahead (each rung: derive → build → prove → measure)
- **Day 2 — the optimizer.** Measure where plain gradient descent
  stalls on curved loss surfaces; derive per-parameter step-size
  adaptation from that geometry; implement ours; benchmark against
  plain GD. Declare any convergence with known optimizers.
- **Day 3 — sequence machinery.** Requirement: influence must travel
  between positions, content-addressed, order-aware. Derive the
  comparison operator and a positional scheme from scratch; prove
  causality bit-exactly as before.
- **Day 4 — the byte discipline.** float32, cache-aware matmul
  kernels, measured GB/s; then CUDA port for the RTX 4070.
- **Day 5+ —** rebuild the loop-core + external-memory thesis on this
  engine, then run the knowledge-starving experiment that decides
  whether knowledge can live outside the core at all.

## Standing honesty notes
- The designer (me) cannot un-know its training. The enforceable
  standard is not amnesia but audit: every line traces to a stated
  requirement or a proof, so nothing enters by authority.
- Small-scale success proves mechanism, never scaling. Each rung's
  claims are limited to what that rung measured.

## Appended 2026-07-07 — PURITY DECISION + supersession (append-only)

**Supersession.** A self-contained on-site prompt (native Windows host,
verified tree, machine gate already passed: RTX 4070 + Ryzen 9 7950X)
supersedes HANDOFF.md wherever they conflict. HANDOFF.md remains as a
historical brief. The canonical, self-briefing plan now lives in CLAUDE.md at
the repo root, read in order CLAUDE.md → ZERO.md → HANDOFF.md.

**Purity decision (permanent).** The `loom/` PyTorch prototype is RETIRED and
never executes on this machine; it is archived as `archive-track1-torch/`. No
PyTorch, no ML libraries, no numerics libraries, ever. Permitted foundations:
hardware, OS, C compilers, mathematics. `nvcc` and the CUDA runtime are
admitted as the C compiler + driver layer for the GPU; GPU libraries (cuBLAS,
cuDNN, thrust) remain forbidden — every kernel is hand-written. Track-1's
sandbox-era measurements (RESULTS.md, *.json) remain valid historical records;
no new results will come from that code.

**Environment (measured this session).** Native Windows 10 host; all builds
and runs driven through WSL2 Ubuntu (`wsl.exe -d Ubuntu`; default distro is
docker-desktop, so Ubuntu is named explicitly). Ubuntu: gcc 15.2.0, git,
python3 present; gh and nvcc initially absent. `nvidia-smi` inside Ubuntu
reports NVIDIA GeForce RTX 4070, 12282 MiB, driver 610.47 (WSL CUDA
passthrough working). FINGERPRINT check passed: full tree present, day1.c
carries the sentinel "the loom weaves what the weaver wills",
shakespeare.txt = 1,115,394 bytes.
