# DATA — corpus provenance & policy

Data tooling (`data/build_corpus.py`, future filters/scorers) are INSTRUMENTS:
stdlib only, never imported by artifact code, and marked as such. Training data
itself is not code, but its provenance is audited exactly like everything else.

## Provenance rule (non-negotiable)
Every corpus document is recorded in `data/manifest.json` with: source URL,
license, raw bytes, clean bytes, sha256 of the cleaned text, download date, and
the exact filters applied. **Nothing with unclear licensing ever enters
training.** The multi-gigabyte bytes are NOT tracked in git (`data/raw/`,
`data/clean/` are gitignored); the manifest is tracked and, with the builder
script, makes the corpus reproducible from scratch.

## Stage 1 — milestone pretrain corpus (this rung)
- Source: Project Gutenberg, public-domain English, biased toward simple /
  children's prose (fairy tales, classic children's novels, plain narrative).
- Pipeline (`data/build_corpus.py`): download plain-text UTF-8 → strip
  Gutenberg boilerplate (between the `*** START OF … ***` / `*** END OF … ***`
  markers) → normalize CRLF→LF, drop transcriber "Produced by…" block →
  exact-dedup by sha256 of cleaned content.
- Size target: < 1 GB (the curated set is a few MB — far under). Measured
  totals (docs, clean bytes) live in `data/manifest.json → totals`.
- License: each text is US public domain, distributed under the Project
  Gutenberg License; headers/footers carrying that license are stripped for
  training and the provenance kept in the manifest.
- chars-per-token: reported once the BPE tokenizer exists (Day 5d).

## Stage 2 — tier-2 plan (build pipeline + manifest now; do NOT download yet)
Target ~1–2B tokens from: Simple English Wikipedia dump + Gutenberg juvenile +
a FineWeb-Edu sample filtered by a readability scorer WE write (sentence-length
+ word-frequency based; no imported ML). Any single download > 5 GB: ASK the
human first. The pipeline and manifest schema are shared with stage 1.

**PRE-REGISTERED (locked, gates tier-2):** at equal token count, the
readability-FILTERED slice must beat the UNFILTERED slice in validation loss on
a small fixed model trained identically. If filtering does not help at equal
tokens, tier-2 filtering is not justified — report the miss and reconsider.

## Human data checkpoints
See STATUS.md → "Human checkpoints". The human personally tests the chat model
(`make talk`), a blind base-vs-chat A/B on 10 prompts, and 20 everyday
questions at tier-2. Verdicts are logged verbatim in ZERO.md.
