# LOOM — measured results (small-scale controlled run)

Everything below was measured by `train.py` on this run. Nothing is
estimated, projected, or copied from another source.

## Environment
Single CPU core · PyTorch 2.12.1+cu130 · fp32 · Python 3.12.3

## Setup
- Corpus: Tiny Shakespeare, 1,115,394 chars, char-level,
  vocab 65, 90/10 train/val split.
- Both models: batch 16 × context 96, 533 steps
  (818,688 training tokens each), AdamW lr 0.003
  cosine→0.0003, warmup 30, clip 1.0, identical data order,
  identical fixed val batches.
- LOOM: d=128, 2 unique blocks looped up to k_max=4 (ACT halting, τ=0.01),
  1,024-slot product-key memory (top-8/loop), per-position carry
  registers, rank-2 per-token per-loop hypernet deltas.
  **609,217 params.**
- Baseline: standard 3-layer pre-LN transformer, d=128, FFN 4×, tied
  head. **615,680 params
  (+1.06% vs LOOM).**

## Headline (val cross-entropy, nats/char — lower is better; chance = 4.174)

| model | val CE |
|---|---|
| LOOM (adaptive depth) | **2.0903** |
| Vanilla, param-matched | **2.1084** |

Δ(LOOM − vanilla) = -0.0181 nats/char.

## Compute accounting (the honest part of the trade)
Parameters are matched; compute is not. LOOM loops its core:

| | s/step | tokens/s (train) |
|---|---|---|
| LOOM | 0.42 | 3,661 |
| Vanilla | 0.107 | 14,417 |

LOOM cost 3.93× the baseline's
wall-clock per step here. The blueprint's efficiency claim is about weight
traffic at inference on bandwidth-bound hardware, which a 1-core fp32 CPU
run cannot measure — see "what this does not prove."

## Ablations (mechanisms removed from the TRAINED model at eval)

| variant | val CE | Δ vs full |
|---|---|---|
| full (adaptive) | 2.0903 | — |
| memory disabled | 2.0907 | +0.0004 |
| hypernet disabled | 2.1280 | +0.0377 |
| depth forced k=1 | 2.1304 | +0.0401 |
| depth forced k=4 | 4.0441 | +1.9538 |

## Adaptive halting behaviour (36,864 val tokens)
- mean steps 1.608 ± 0.505 (k_max 4);
  histogram over depths 1–4: [14738, 21828, 293, 5]
- Pearson r(per-token steps, per-token CE) = **0.129**
- batch-level early exit executed 3.21
  loops on average (per-token savings would need gather/scatter — not
  implemented).

## Memory usage on val
295/1,024 slots retrieved at least once; the 16
hottest slots account for 43.1% of
946,176 retrievals.

## Samples (360 chars, temp 0.8, top-k 40, seeded with newline)
LOOM:
```

BUUK:
Sus. you taw low the to mule ee.
Blamst frowe ar of il.
LUCrome, a sand sint I IN ford tharus shas deir heas
An I thaugh he is hon aster my amirteons heand as grean
Ond apbettem, be they and for chobend keacke,
If far till wer the, our glaiom novent mulle
What fith and the cop bur car thaput kis my and wither
Whe for neloth the hibs on sitt a gon whis 
```
Vanilla:
```


NUKIN INICK:
What the do ndacke the well that to thee spet a cone of
Ald the you that mon
Ske of you mise thre be asien on are for if tring;
Prey tied oithis to mand leat thatis as the sond the heeat.


ISone beank anliows sor the she mand
As the prarty plast be the nother dechp stold leeeests
sof ther they knot 'pell your of enot day's somper eewor hing of
```

## What this run proves
- The full architecture is implementable, causally correct (bit-exact
  test), and trainable end-to-end — all 11 proofs in `tests.py` pass.
- Recursion adds depth with zero added parameters (609,217 params at
  k_max=1 and k_max=8 alike).
- Whether each mechanism helped *at this scale* is exactly what the
  ablation table says — no more, no less.

## What this run does NOT prove
- Anything about 0.4B-scale behaviour, GPU/NPU bandwidth economics,
  knowledge editing, or the "knowledge-starving" objective (not
  implemented here). Those remain open hypotheses from the blueprint.
- One seed, one corpus, 533 steps on one CPU core. Small-scale
  results routinely fail to transfer.
