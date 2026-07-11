# Changelog

All notable changes to EvalGuard are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-07-11

First public release: open-data benchmark-contamination auditor.

### Added
- **Estimands** `ρ` (contamination rate) and `Δ` (score inflation).
- **Injection harness** that manufactures ground truth by inserting benchmark
  items into a clean corpus at a controlled rate and form (`verbatim`,
  `paraphrase`, `format_shift`, `answer_only`), deterministic by seed.
- **Matcher ensemble**: n-gram/MinHash, TF-IDF retrieval, paraphrase judge,
  answer-leak detector — each emits an evidence score plus a contamination path.
- **Data-driven calibration** (`evalguard calibrate`) — per-matcher and
  aggregator operating points selected from labeled data, with PR/F1 curves.
- **Two decision modes** (`--decision`): `global-tau` (default, conservative /
  precision-first — never falsely accuse, recovers `ρ`/`Δ`) and `per-form`
  (opt-in, coverage-first — an OR-gate over per-matcher thresholds that
  recovers weak forms a single global `τ` discards, at a precision cost).
- **CLI** `evalguard audit` with a JSON/text report and a CI gate (`--fail-over`).
- **Studies** reproducing the findings: scaling, distinctiveness, mode
  comparison, and multi-seed robustness (see `demo/` and `demo/FINDINGS.md`).
- 37 tests.

### Known limitations
- Embeddings use a TF-IDF stand-in; the LLM paraphrase judge is a rule-based
  stand-in (swap points marked `# SWAP:`).
- Open-data regime only — no gray-box (Min-K%++, exchangeability) or black-box
  methods yet.
- Results validated on a single benchmark×corpus pair (OpenBookQA × AG News).
