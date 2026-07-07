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

## Appended 2026-07-07 — DAY 2 optimizer: PRE-REGISTRATION (before any run)

Registered BEFORE building/running zero/day2.c. Predictions are locked; the
benchmark will publish hits AND misses against exactly these numbers.

**Testbed A — the quadratic bowl.** f(x) = 1/2 xᵀA x with A diagonal, two
eigenvalues placed at the extremes {λ_min = 1, λ_max = κ} (2-D captures the
worst case; the extreme modes govern the rate). Optimum x* = 0. Target:
reduce ‖x‖/‖x₀‖ to ≤ 1e-6, starting x₀ = (1,1).

**Derivation (plain gradient descent, fixed step).** Update x ← x − α∇f =
(I − αA)x, so mode i contracts by |1 − αλ_i| each step. The step that
minimises the worst mode is α* = 2/(λ_min+λ_max) = 2/(1+κ), which equalises
the two extreme modes at the same contraction factor
  ρ = (κ − 1)/(κ + 1).
Then ‖x_n‖ = ρⁿ‖x₀‖ exactly, so steps-to-target
  n(ε) = ln ε / ln ρ,  ε = 1e-6.
This is the strongest plain GD can do (oracle-tuned step); it is the bar our
method must beat. (Declared convergence: this is the classical steepest-
descent rate; re-derived here, not imported.)

**PRE-REGISTERED PREDICTIONS — plain GD, oracle step α*=2/(1+κ):**
| κ      | ρ=(κ−1)/(κ+1) | predicted steps to ‖x‖≤1e-6‖x₀‖ |
|--------|---------------|----------------------------------|
| 1      | 0             | 1                                |
| 100    | 0.980198      | 691                              |
| 10000  | 0.999800      | ~69,078                          |
Prediction: plain GD steps grow ~linearly in κ (n ≈ (κ/2)·ln(1/ε) for κ≫1).
Secondary prediction: with the *safe* untuned step α = 1/λ_max (what you use
when you don't know the spectrum), it is ~2× worse: n ≈ κ·ln(1/ε), i.e.
κ=100 → ~1,375; κ=10000 → ~138,150.

**P-bars for OUR derived optimizer (per-parameter step normalised by a
running estimate of each coordinate's gradient scale):**
- P1: on Testbed A it reaches the same ‖x‖≤1e-6 target in a step count that
  is essentially κ-INDEPENDENT — predict ≤ 60 steps for ALL of κ∈{1,100,10⁴}
  (a diagonal gradient-scale normaliser ≈ diagonal Newton on a diagonal
  quadratic, which removes the conditioning). 
- P2 (the win): at κ=10000, ours reaches target in < 1/50 of plain-GD-oracle
  steps (i.e. < ~1,380 vs ~69,078).
- KILL CRITERION: if ours is not strictly better than plain-GD-oracle at
  κ=10000 (steps_ours ≥ steps_GD), the adaptation is worthless — report it as
  a failure, keep the number, do not bury it.

**Testbed B — the day1 char-LM (2,078 params).** Same engine, real non-
quadratic loss. Metric: GD steps to reach loss ≤ 0.20 (day1 reached ~0.179
by step 3000 with its hand-tuned schedule). Prediction: our adaptive method
reaches loss ≤ 0.20 in FEWER steps than plain fixed-step GD at that GD's best
hand-swept constant step; predict ours ≤ 0.5× the GD step count. No oracle
Hessian here, so this bar is directional, not a locked integer.

**Convergence to declare in advance:** normalising each coordinate's step by a
running root-mean-square of its gradient is the RMSProp/Adam family. If our
from-geometry derivation lands there, we KEEP it (independent derivation =
verification) and say so; constants (decay of the running estimate, ε floor,
base step) will be picked by OUR OWN sweep on these two testbeds, not taken
from any paper.

## Appended 2026-07-07 — PHASE 0 (PURE): MEASURED RESULTS

All numbers executed on this machine (RTX 4070 + Ryzen 9 7950X) via WSL2
Ubuntu. Probes are hand-written C/CUDA in probes/ (instruments). Full machine
truth is in hw.json; highlights:
- Host RAM (STREAM triad): **47.19 GB/s**.
- CPU AVX2 FMA: **fp64 934 GFLOP/s, fp32 1876 GFLOP/s** (32 threads).
- VRAM streaming copy: **384.6 GB/s** (76% of the 504 spec; the measured
  number is the ceiling we design against).
- PCIe 4.0 pinned: **26.7 GB/s H2D / 26.3 GB/s D2H** (pageable 16.1 / 12.2).
  Confirms the brief's ~25 GB/s host<->device assumption by measurement.
- Hand-written fp32 matmul (N=2048): **naive 2107, tiled 2730 GFLOP/s**
  (tiled 1.30x naive, bit-identical). Far below the ~29 TFLOP arithmetic
  peak — simple kernels, a known future rung, not a defect.
- bf16 / tensor-core: **null, deferred** — recorded honestly, never faked.

**Day 2 optimizer — scored against the pre-registered bars (zero/day2.c):**
Carried-over engine gradcheck after all changes: max rel err **8.20e-07 <
1e-6 → PASS** (day1's own gradcheck also still passes; the value differs only
because day2 samples different random entries under a fixed seed).

Derived method = per-coordinate step ÷ running RMS of that coordinate's
gradient. DECLARED CONVERGENCE: this is the RMSProp family; kept as
independent re-derivation. Constants chosen by our own sweep (β=0.9, ε=1e-8,
base step swept).

Testbed A — quadratic bowl, steps to ‖x‖ ≤ 1e-6‖x₀‖:
| κ      | GD (oracle step), predicted | GD measured | OURS measured |
|--------|-----------------------------|-------------|---------------|
| 1      | 1                           | 1           | 4             |
| 100    | 691                         | 691         | 4             |
| 10000  | ~69,078                     | 69,078      | 4             |
- Plain-GD predictions HIT to the integer (1 / 691 / 69,078) — the pre-
  registered κ-linear degradation is exactly what the machine did.
- P1 (ours κ-independent, ≤60 steps ∀κ): **HIT** — 4 steps flat, for all κ.
- P2 (ours < 1/50 of GD-oracle at κ=10⁴, i.e. <~1,382): **HIT** — 4 vs
  69,078 (≈17,000× fewer steps).
- KILL criterion (ours ≥ GD at κ=10⁴): **not triggered**.
- Honesty note: the diagonal quadratic is the BEST case for a diagonal RMS
  normaliser (it exactly removes per-axis conditioning), so 4-steps-flat is
  the idealised win, not a general guarantee. Testbed B is the harder test.

Testbed B — day1 char-LM (2,078 params), steps to loss ≤ 0.20 (cap 5000):
- Plain GD best: **86 steps** at lr=2.0 (final loss 0.2000).
- OURS best: **30 steps** at lr=0.03 (final loss 0.1969).
- Prediction (ours ≤ 0.5× GD): **HIT** — 30/86 = 0.35×.

Verdict: every pre-registered bar met; no misses to report this rung. The
optimizer adaptation is real and measured, with the caveat that Testbed A's
result is the idealised diagonal case. Next rung (Day 3): sequence machinery.
