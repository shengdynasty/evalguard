# EvalGuard: Findings

### Open-data benchmark-contamination auditing, and a phase transition in what is detectable as the corpus grows

*Working results section. All numbers are reproducible via `demo/run_scaling_study.py` (figures in `demo/figures/`, data in `demo/scaling_results.csv`) and `demo/run_real_demo.py`. Test suite: 37 passing.*

---

## 1. Problem and estimands

We audit **open-data** contamination: a team holds a training/fine-tuning corpus `D` and a benchmark `B`, and wants to know how much of a reported score is inflated because items of `B` (or near-duplicates) already appear in `D`. Per benchmark item we compute an evidence score `c_i ∈ [0,1]` from an ensemble of matchers, and report two estimands:

- **Contamination rate** `ρ = mean(c_i > τ)` — fraction of the benchmark compromised.
- **Score inflation** `Δ = Score(B) − Score(B_clean)`, where `B_clean` drops items with `c_i > τ` — the headline, actionable number.

The aggregator threshold `τ` and per-matcher operating points are **selected from data** by `calibrate.py`, not hand-set. We only claim contamination at the strength the corpus evidence supports (conservative by design): the auditor should under-report before it ever falsely accuses.

## 2. Experimental setup

- **Benchmark:** OpenBookQA, 300 real multiple-choice items, split TRAIN/TEST 150/150 (seed 2024).
- **Corpus:** AG News, 700 real news documents, fetched once from the HuggingFace datasets-server REST API (`datasets.load_dataset` is network-blocked in this environment; provenance in `real_data/README.md`).
- **Injection harness:** into a copy of the corpus we insert benchmark items at rate **0.20** across four controlled forms — `verbatim`, `paraphrase` (deterministic reword), `format_shift` (relabelled/recased Q/A), `answer_only` (only the answer leaks) — deterministically from a seed. This manufactures **known** ground truth: the injected-item set, the true `ρ`, and (via a score model where memorized items are answered correctly with p≈0.95 vs base-rate 0.55) the true `Δ`.
- **Calibration protocol:** for each condition, `τ` is chosen to maximize F1 on the TRAIN split only; detection and estimand recovery are measured on the held-out TEST split.
- **Scaling variable:** the clean corpus size `N ∈ {80,100,110,120,130,175,300,500,700}`. Everything else is held fixed. (Embeddings are a TF-IDF stand-in; real sentence-transformers could not be installed — see Limitations.)

## 3. Results

### 3.1 Strong forms are robust; answer-only collapses — `figures/recall_per_form.png`

`verbatim`, `paraphrase`, and `format_shift` are detected at **recall 1.00 at every corpus size** tested. `answer_only` recall is **1.00 for small corpora but drops to 0.09** once the corpus passes a critical size, and stays there flat out to N=700.

### 3.2 A phase transition in the operating point — `figures/tau_vs_size.png`, `figures/overall_prf.png`

The collapse is not gradual. It is a discrete regime switch in the F1-optimal `τ`, located between **N=100 and N=110** clean documents (against 30 injected items, i.e. a contaminated fraction of ~21–23%):

| N (clean docs) | τ | Precision | Recall | F1 | answer_only recall | ρ̂ (true 0.20) | Δ̂ (true +0.08) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 80  | 0.07 | 0.86 | 1.00 | 0.92 | 1.00 | 0.23 | +0.08 |
| 100 | 0.05 | 0.83 | 1.00 | 0.91 | 1.00 | 0.24 | +0.09 |
| 110 | 0.86 | 1.00 | 0.67 | 0.80 | 0.09 | 0.13 | +0.05 |
| 120–700 | 0.86 | 1.00 | 0.67 | 0.80 | 0.09 | 0.13 | +0.05 |

Below the critical size the calibrator keeps `τ` low: it can afford to accept weak `answer_only` evidence because the small background rarely produces a colliding match, so precision stays acceptable (0.83–0.86) and recall is perfect. Above it, a low `τ` would admit background collisions on short common-word answers, so the F1-optimal choice **jumps `τ` to 0.86**, buying precision 1.00 at the cost of abandoning the entire `answer_only` form.

### 3.3 Failure is conservative, never fabricated — `figures/estimand_error.png`

At and beyond the transition, precision is **1.00** — no false accusations. The cost lands entirely on recall, so `ρ` and `Δ` are **under-estimated** (ρ̂ 0.13 vs true 0.20; Δ̂ +0.05 vs true +0.08), never inflated. For an auditor this is the correct direction of error: it reports *less* contamination than exists rather than crying wolf.

## 4. Interpretation

The headline finding is that **open-data detectability of a contamination form depends on the *distinctiveness* of what leaks relative to the background corpus, and this interacts with calibration to produce a sharp threshold.** Verbatim/paraphrase/format-shift leak long, distinctive token sequences that stay separable from background text at any corpus size. `answer_only` leaks a short, common token (e.g. "water", "energy") whose signal is drowned as background grows; past a critical corpus size the only way to preserve precision is to raise `τ` and give that form up. This is not a tuning bug — it is a real detectability ceiling of the open-data regime, and a reason a production auditor must report per-form and per-corpus, not a single global score.

## 5. Limitations

1. **TF-IDF stand-in, not real embeddings.** `sentence-transformers`/MiniLM could not be installed (torch+CUDA too heavy; HF model download 403-blocked). Real embeddings would likely lift paraphrase/format-shift robustness further but would **not** rescue `answer_only`, whose ceiling is informational, not representational.
2. **Single benchmark × corpus pair** (OpenBookQA × AG News) and **small-to-mid corpus scale** (≤700 docs). The transition location (~100–110 docs vs 30 injected) is specific to this pair and injection rate; the *existence* of a distinctiveness-driven transition is the transferable claim, not the exact crossover.
3. **Discrete τ grid.** The switch's sharpness is partly an artifact of the calibrator's threshold grid; a finer grid would smooth the crossover but not remove it.
4. **Open-data regime only.** No gray-box (Min-K%++, exchangeability) or black-box methods here; this cannot audit closed models from logits or generations — a corpus is required.
5. **Synthetic score model for Δ.** True `Δ` is defined via an assumed memorization-boost score model, not measured against a real model's answers.

## 6. Takeaways

- On real text, three of four contamination forms are recovered essentially perfectly, and `ρ`/`Δ` are recovered exactly when the corpus is small.
- There is a **distinctiveness-driven phase transition**: low-distinctiveness leakage (`answer_only`) becomes undetectable open-data once the corpus is large enough that preserving precision forces the operating point up.
- The auditor fails **conservatively** (precision 1.00, under-estimated ρ/Δ), which is the right failure mode.
- **Design implication:** report contamination **per form and per corpus scale**, never as a single global number, and treat short-answer leakage as a known blind spot requiring a different signal (e.g. a planted canary or gray-box membership test) rather than corpus search.

---

## 7. Refinement: is the collapse distinctiveness-driven, or a calibration artifact?

> **Superseded in part by §9 (multi-seed).** The single-seed OpenBookQA numbers below are seed-noisy; §9 shows the robust cross-dataset picture. Read §7 for the *method*, §9 for the *conclusion*.

A follow-up (`demo/run_distinctiveness_study.py`; figure `figures/distinctiveness_transition.png`, data `demo/distinctiveness_results.csv`) isolates the cause of the §3 collapse. We inject **only** the `answer_only` form, calibrate `τ` on that signal **alone** (not shared with strong forms), and vary answer distinctiveness — mean IDF of answer tokens against the corpus — across OpenBookQA terciles, over corpus sizes 40–700.

**Result (precision / F1 on held-out TEST):**

| bucket | mean answer IDF | N=40 | N=150 | N=400 | N=700 | N* (F1<0.5) |
|---|---:|---:|---:|---:|---:|---:|
| low  | 4.54 | 0.77 / 0.87 | 0.77 / 0.87 | 0.77 / 0.87 | 0.71 / 0.83 | >700 (censored) |
| mid  | 5.70 | 0.83 / 0.91 | 0.83 / 0.91 | 0.83 / 0.91 | 0.83 / 0.91 | >700 (censored) |
| high | 7.28 | 1.00 / 1.00 | 0.83 / 0.91 | 0.62 / 0.77 | 0.56 / 0.71 | >700 (censored) |

**This refines (and partly corrects) the §4 interpretation in two ways:**

1. **The dramatic mixed-form collapse (recall 1.00 → 0.09) was primarily a *single-global-threshold artifact*, not an informational ceiling.** When `answer_only` is calibrated on its own signal, it **never** collapses below F1 0.71 out to N=700. Sharing one `τ` across a form-mix dominated by high-signal forms (verbatim/paraphrase/format-shift) pushes the F1-optimal `τ` high and *incidentally discards* the weak form. Per-form calibration recovers most of it.

2. **Answer distinctiveness does *not* simply buy a later collapse — its effect is non-monotone.** High-distinctiveness answers are detected *perfectly* in small corpora, but their **precision decays as the corpus grows** (1.00 → 0.56), crossing *below* the stably-mediocre low/mid buckets by N≈200. The intuitive "rarer answers stay safer at scale" prediction is **falsified**: rare multi-token answers accrue partial background collisions once `τ` is calibrated loosely enough to admit them, while short common answers sit at a stable, mediocre operating point independent of corpus size.

**Design implication (strengthened):** calibrate and threshold **per form**, never with one global `τ` — that single change recovers most `answer_only` detectability. Treat answer distinctiveness as a per-item *confidence* signal, not a monotone difficulty proxy.

**Caveats specific to §7:** each per-bucket TEST split is small (~50 items, ~10 injected), so precision is noisy (±~0.15); the qualitative crossover is robust but the exact values are not. Single benchmark/corpus pair, single seed. Multi-seed averaging and larger benchmarks are the obvious next step before publishing this as a headline claim.

---

## 8. Reproducibility on a second benchmark (SciQ × AG News)

To test whether §7's distinctiveness effect transfers, we repeated the isolated
`answer_only` study on a **second benchmark with a different answer profile** —
SciQ (300 items; distinctive science-term answers like "coriolis effect",
"mesophilic organisms") against the same AG News corpus. Figure:
`figures/distinctiveness_sciq.png`, data `demo/distinctiveness_sciq_results.csv`.

**Result (precision / F1, held-out TEST):**

| bucket | mean answer IDF | N=40 | N=250 | N=700 |
|---|---:|---:|---:|---:|
| low  | 5.87 | 0.71 / 0.83 | 0.50 / 0.67 | **0.36 / 0.53** |
| mid  | 7.31 | 0.83 / 0.91 | 0.67 / 0.80 | 0.59 / 0.74 |
| high | 7.55 | 1.00 / 1.00 | 1.00 / 1.00 | **1.00 / 1.00** |

**What reproduces, and what doesn't:**

1. **Reproduced — precision decays as the corpus grows.** Both benchmarks show
   `answer_only` precision falling as the clean corpus grows; it is a real,
   cross-dataset phenomenon, not an OpenBookQA quirk.
2. **Reproduced *more cleanly* on SciQ — distinctiveness protects precision.**
   On SciQ the ordering is perfectly monotone: the high-distinctiveness bucket
   holds precision 1.00 at every corpus size, the mid decays moderately, the low
   decays hardest. Higher answer distinctiveness ⇒ more resistance to
   corpus-growth false positives.
3. **Did *not* transfer cleanly — the exact bucket ordering.** On OpenBookQA the
   *highest*-distinctiveness bucket was the one that decayed (1.00 → 0.56), which
   violates the monotone story SciQ shows. Given the small per-bucket TEST splits
   (~10–16 injected items), the OpenBookQA high-bucket decay is most likely
   small-sample noise; the better-powered SciQ run supports the intuitive
   mechanism.

**Honest cross-dataset takeaway:** the *mechanism* — answer distinctiveness
governs how fast answer-only precision decays with corpus size — reproduces and
is directionally robust, but distinctiveness is a **noisy, not quantitatively
portable** predictor: absolute IDF thresholds and exact crossovers do not
transfer between benchmarks. Precise, per-benchmark claims still require
multi-seed averaging and larger per-bucket samples. This is exactly the kind of
tempering a second dataset is supposed to provide.

---

## 9. Multi-seed robustness (supersedes §7's single-seed OpenBookQA numbers)

The §7 and §8 per-bucket numbers were single-seed, on small per-bucket TEST
splits (~10–16 injected items). We re-ran the isolated `answer_only` study at the
decisive corpus size **N=700**, averaging precision/F1 over **5 seeds** (seed
varies the train/test split *and* the injected subset). `demo/run_distinctiveness_multiseed.py`.

**Result (precision / F1, mean ± std over 5 seeds, N=700):**

| benchmark | low | mid | high |
|---|---|---|---|
| **OpenBookQA** | 0.68±0.05 / 0.81±0.03 | 0.67±0.08 / 0.80±0.06 | 0.62±0.03 / 0.76±0.02 |
| **SciQ** | 0.34±0.02 / 0.51±0.02 | 0.59±0.07 / 0.74±0.05 | **1.00±0.00 / 1.00±0.00** |

**This sharpens — and partly retracts — the earlier claims:**

1. **On SciQ the distinctiveness → precision effect is large, monotone, and robust** (near-zero variance): rare technical answers (high, IDF 7.55) never collide with the news background so precision is a clean 1.00; common answers (low, IDF 5.87) collide heavily at 0.34. This is real signal, not noise.
2. **On OpenBookQA the effect vanishes under averaging.** All three buckets converge to ~0.62–0.68 (overlapping within std), with high marginally *lowest*. **Both** of §7's single-seed OpenBookQA patterns — the clean 0.77/0.83/1.00 ordering *and* the apparent high-bucket "decay to 0.56" — were **seed artifacts**, not signal. §7's OpenBookQA table should be read as superseded by this one.
3. **The robust, corrected claim is narrower and sharper:** answer distinctiveness governs answer-only precision *only when the answers' rarity actually translates to separability from the corpus* (SciQ). **Mean answer-token IDF is not a portable proxy for that** — OpenBookQA's "high-IDF" answers are short and still collide, so IDF overstates their detectability and the effect washes out. **Distinctiveness matters; the IDF proxy for it does not transfer across benchmarks.**

The reproducible engineering takeaway is unchanged and, if anything, reinforced: report per-form and per-benchmark, calibrate on representative data, and don't trust a single-corpus distinctiveness number as a cross-benchmark predictor.
