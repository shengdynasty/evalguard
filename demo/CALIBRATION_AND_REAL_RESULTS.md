# EvalGuard — Calibration curves & first real-data results

This document reports two additions to the v1 build:

- **Part A** — data-driven calibration that replaces the hand-tuned matcher
  thresholds (design §6: earned, not guessed).
- **Part B** — the first run on **real** public text instead of clean synthetic
  data (design §6/§7), with honest, degraded recovery numbers.

All numbers below are reproducible: `python demo/run_real_demo.py` and
`python -m evalguard.cli calibrate ...`.

---

## Part A — data-driven calibration

`evalguard/calibrate.py` takes a labeled set (the injection harness knows ground
truth by construction), sweeps a decision-threshold grid `[0.00 … 1.00]` step
0.01 for **each matcher** and for the **aggregated `c_i`**, computes
precision / recall / F1 at every point, and **selects operating points from
data**:

- the **F1-optimal** threshold,
- **precision-at-fixed-recall** (recall ≥ 0.90),
- **recall-at-fixed-precision** (precision ≥ 0.95).

Artifacts per run (a core research artifact from §6):

- `evalguard/calibration/{ngram,embedding,paraphrase,answer,aggregate}_pr.csv`
  — `threshold, precision, recall, f1, tp, fp, fn` per grid point.
- `evalguard/calibration/{…}.png` — PR curve + P/R/F1-vs-threshold, F1-optimal
  point marked.
- `evalguard/calibration/calibration.json` — the chosen operating points +
  metadata, loaded by `evalguard/config.py`.

The old `# SWAP:`-adjacent magic numbers (embedding `sim_floor=0.82`, paraphrase
`threshold=0.6`, answer `threshold=0.6`, and the aggregator `tau=0.5`) are gone
as literals: the score-shaping constants now live in `config.SHAPING_DEFAULTS`
and the **decision `tau` / per-matcher operating points are read from the
calibration JSON**. `rho`, `delta`, `contaminated_ids` default `tau=None` →
resolve to the calibrated value.

### Operating points chosen — SYNTHETIC (clean, separates easily)

| matcher   | F1-opt thr | P | R | F1 |
|-----------|-----------:|---:|---:|---:|
| ngram     | 0.01 | 1.000 | 0.350 | 0.518 |
| embedding | 0.01 | 1.000 | 0.725 | 0.841 |
| paraphrase| 0.06 | 1.000 | 0.850 | 0.919 |
| answer    | 0.01 | 1.000 | 1.000 | 1.000 |
| **aggregate (tau)** | **0.06** | **1.000** | **1.000** | **1.000** |

On synthetic data every matcher keeps P=1.0 at a tiny threshold, so the
F1-optimal `tau` collapses to ~0.06 — a symptom of data that separates *too*
easily, which motivated Part B.

### Operating points chosen — REAL (OpenBookQA × AG News, TRAIN split)

| matcher   | F1-opt thr | P | R | F1 |
|-----------|-----------:|---:|---:|---:|
| ngram     | 0.01 | 1.000 | 0.367 | 0.537 |
| embedding | 0.01 | 1.000 | 0.567 | 0.723 |
| paraphrase| 0.01 | 1.000 | 0.800 | 0.889 |
| answer    | 0.45 | 0.732 | 1.000 | 0.845 |
| **aggregate (tau)** | **0.86** | **1.000** | **0.800** | **0.889** |

Real text moves the calibration a lot: the **answer** matcher now needs
`thr≈0.45` (below it, real short answers produce false positives — P drops to
0.68 at 0.01), and the **aggregator `tau` jumps from 0.06 to 0.86** to hold
precision at 1.0. This is exactly the point of calibrating from data rather than
hand-tuning to synthetic.

### Does the calibrated point beat the old hardcoded `tau=0.5`?

Test `test_f1_optimal_tau_beats_hardcoded_on_heldout` calibrates `tau` on one
injection draw and evaluates on a **held-out** draw: the data-selected
F1-optimal `tau` is asserted to do **at least as well** as the old hardcoded 0.5
in F1, and it does. On synthetic data the hardcoded 0.5 already scored F1=1.0 so
the calibrated point ties it; the real value of calibration shows on real text,
where the correct `tau` is 0.86, not 0.5.

---

## Part B — first run on REAL text

### Data used (real, public)

| role | dataset | rows used | source |
|------|---------|-----------|--------|
| benchmark | **OpenBookQA** (`allenai/openbookqa`, config `main`, split `train`) | **300** MCQ items | HF datasets-server REST API |
| corpus | **AG News** (`fancyzhx/ag_news`, split `train`) | **200** news docs | HF datasets-server REST API |

Cached to `demo/real_data/{openbookqa,ag_news}.jsonl` (see that folder's README
for provenance). Each OpenBookQA item is `{question_stem, correct-choice text}`.

**How the data was obtained (honest path):** `datasets.load_dataset(...)` does
**not** work in this build sandbox — the outbound proxy returns **403 Forbidden**
for every `huggingface.co` host, so the library's HTTP/xet backend cannot reach
the CDN (`socksio` was installed to rule out the SOCKS-proxy error; the block is
upstream, not the client). The rows were instead pulled once through the
**`datasets-server` `/rows` JSON endpoint**, which *was* reachable, and
normalized to JSONL. This is genuinely the public OpenBookQA / AG News data.

### Real embeddings: attempted, fell back (and why)

`embedding.py` now has `try_sentence_transformer("all-MiniLM-L6-v2")` wired
behind the existing encoder interface, selected via `EmbeddingMatcher(use_real=True)`.
One honest attempt was made and **aborted**: `pip install sentence-transformers`
pulls **torch 2.13.0 (526 MB) + CUDA wheels (366 MB+)** — far too heavy for a
CPU-only box — and even installed, the model weights download from HF would hit
the same **403** block. Per the task's "one honest attempt, then fall back," we
kept the **TF-IDF stand-in** (`backend = "tfidf-stand-in"`). The real demo prints
which backend it used. All real numbers below are therefore with the TF-IDF
embedder, not MiniLM.

### Setup

- Benchmark split **150 train / 150 test** (seed 2024).
- Inject at **rate 0.20** across all four forms (verbatim / paraphrase /
  format_shift / answer_only) into the real corpus — known `rho`, known injected
  set, per-item form.
- **Calibrate on TRAIN only**, evaluate on **held-out TEST** at the calibrated
  `tau = 0.86`.
- Score model: contaminated items answered correctly w.p. 0.95, clean items
  w.p. 0.55 (a realistic MCQ base rate), giving a known true `Δ`.

### Detection on held-out TEST (calibrated tau = 0.86)

| form | recall | caught |
|------|-------:|--------|
| **OVERALL** | **P=1.000  R=0.667  F1=0.800** | tp 20 / fp 0 / fn 10 |
| verbatim | 1.000 | 5/5 |
| paraphrase | 1.000 | 6/6 |
| format_shift | 1.000 | 8/8 |
| **answer_only** | **0.091** | **1/11** |

### Estimand recovery on held-out TEST

| estimand | true | hat | abs err |
|----------|-----:|----:|--------:|
| ρ (rate) | 0.200 | 0.133 | 0.067 |
| Δ (inflation) | +0.083 | +0.051 | 0.032 |

(full score 0.667 → decontaminated 0.615, 20 items dropped.)

### What degraded vs the synthetic demo, and why

| | synthetic demo | real demo (held-out) |
|---|---|---|
| overall P / R / F1 | 1.00 / 1.00 / 1.00 | 1.00 / **0.667** / **0.80** |
| answer_only recall | 1.00 | **0.091** |
| ρ error | 0.000 | **0.067** |
| Δ error | 0.000 | **0.032** |

- **Precision held at 1.0** — no false positives on real background news text.
- **The whole loss is `answer_only`.** Synthetic answers are unique minted
  strings (`Thexesca-160`); **real OpenBookQA answers are short common words**
  ("fur", "water", "a mine", "energy") that occur throughout a real news corpus.
  A leak of only such an answer is genuinely **not distinguishable** from
  background text by IDF-weighted answer matching — the detector's IDF weighting
  correctly refuses to claim membership from a common string (design §4.1). So
  answer-only recall collapses from 1.00 → 0.09.
- Because the missed items are all one (hardest) form, **ρ and Δ are
  under-estimated**, not wrong in a random direction: the auditor reports **less**
  contamination than truly exists. That is the honest, conservative failure mode
  — it will never fabricate contamination, it will miss the hardest-to-prove kind.
- The strong forms (verbatim / paraphrase / format_shift) are **still recovered
  at recall 1.0 on real text**, so the ensemble's core claim survives contact
  with real data; only the intrinsically-hard answer-only case degrades, exactly
  as §4.1 predicts ("detectability of answer-only leakage is bounded by how
  distinctive the answer is").

### Honesty note (design §4.1)

This remains an **open-data** build: it reports per-item membership *evidence*
with the contamination path, never a certified accusation, and never a claim
stronger than the corpus evidence supports. The real-data run makes that
concrete — where the evidence is weak (common-word answers), the reported ρ/Δ
are correspondingly lower, with the gap stated plainly rather than hidden.
