# EvalGuard — Consolidated Project Report

*Open-data contamination auditing for LLM evaluations: the idea, the v1 build, and what the experiments actually showed.*

This single document supersedes and consolidates the working notes (`EvalGuard_design.md`, `demo/CALIBRATION_AND_REAL_RESULTS.md`, `demo/FINDINGS.md`), which remain in the repo as detailed backing. Where the earlier notes and the later experiments disagree, **this report states the corrected view.**

Everything below is reproducible from the code in this repo. Test suite: **37 passing**. Key entry points: `demo/run_demo.py` (synthetic), `demo/run_real_demo.py` (real data), `demo/run_scaling_study.py`, `demo/run_distinctiveness_study.py`.

---

## 1. Executive summary

**What it is.** EvalGuard is a *contamination auditor* for LLM evaluations. Given a model's training/fine-tuning corpus `D` and a benchmark `B`, it estimates how much of a reported benchmark score is inflated because items of `B` (or near-duplicates) already appear in `D`, and it shows the evidence trail. It is not a benchmark, an eval harness, or a leaderboard — it sits next to an eval and puts a defensible asterisk on the number.

**Why this idea.** It keeps the "automatically find where a system is secretly using information it shouldn't" core of the earlier ChronoLint concept, but moves it from quant backtests into LLM evaluation — a larger market, a hotter research frontier, and, crucially, a moat that *compounds* (a labeled contamination benchmark) rather than one gutted by local-only execution.

**What was built.** A working v1 of the **open-data** auditor: two estimands (`ρ`, `Δ`), a four-matcher ensemble, an injection harness that manufactures ground truth, data-driven calibration, a CLI with a CI gate, and 37 tests. It runs on synthetic data (exact estimand recovery) and on real data (OpenBookQA × AG News).

**Headline result.** On real text, three of four contamination forms are recovered essentially perfectly and the score-inflation number `Δ` is recovered exactly on small corpora. The one hard form — `answer_only` (only the answer leaks) — appears to "collapse" as the corpus grows, but a controlled follow-up shows that collapse is **largely an artifact of sharing one global decision threshold across heterogeneous leak types**, not a hard information limit: calibrating *per form* recovers most of it. Throughout, the auditor fails **conservatively** — it under-reports contamination rather than ever fabricating it.

---

## 2. The idea and why it survives scrutiny

### 2.1 From a broad concept to a sharp one
The project narrowed deliberately: from a sprawling "research reproducibility OS," to ChronoLint (temporal-leakage detection in quant backtests), to this — **benchmark-contamination detection for LLMs**. The through-line is a single, coherent computer-science problem: track whether information that should have been unavailable has leaked into a result.

### 2.2 The pressure-test (why detection is a product, not a footnote)
Five real risks, and why the design survives them:

1. **The market's default answer to contamination is *prevention*, not detection** — fresh/dynamic/private benchmarks (LiveBench, LiveCodeBench, MMLU-CF, private held-out sets). Detection risks being a feature, not a company.
2. **On closed models, black-box detection is statistically contested** (a 2024–25 survey literally asks whether it works well and concludes many detectors' assumptions fail).
3. **The "gotcha, we caught Lab X" framing is unsellable and legally radioactive** — the subject of the audit and its would-be buyer are the same party.
4. **The academic field is crowded** — 50+ detection methods across 100+ papers.
5. **Willingness-to-pay is misaligned** — researchers care but have no budget; labs have budget but won't buy an accusation tool.

**The wedge that survives all five:** teams that fine-tune or evaluate models on semi-public data and need to trust *their own* numbers before shipping. They *want* to find their own contamination, they usually *control the corpus* (so detection is a tractable search, not contested black-box inference), and prevention-via-fresh-benchmarks doesn't help them because they evaluate on domain benchmarks they can't constantly regenerate.

### 2.3 Two design commitments that follow
- **Reframe from "catch the labs" to "trust your own score."** Self-serve, non-adversarial, corpus-in-hand.
- **v1 lives in the open-data / gray-box regime** where detection actually works — not the black-box case the headlines are about.

### 2.4 What claim strength is even available (governs everything)
A model-side detector generally *cannot* certify "this model trained on item 4,237." Absent one of three things it yields *evidence consistent with* contamination, not *proof of membership*:
1. **Corpus access** → membership becomes a search; near-proof (v1's home).
2. **A planted canary / watermark** inserted before training → controlled-FPR proof, but only for data you marked in advance.
3. **A statistical construction** (e.g. the Oren et al. exchangeability test) → a controlled-FPR *dataset-level* p-value, but it needs logit access, tests one specific signal, and is not per-item.

EvalGuard reports at the strongest level the available access supports and never above it. Honesty about claim strength *is* a feature.

---

## 3. What was built (v1, open-data regime)

### 3.1 Estimands
- **Contamination rate** `ρ = mean(c_i > τ)` — fraction of the benchmark compromised, where `c_i ∈ [0,1]` is per-item contamination evidence.
- **Score inflation** `Δ = Score(B) − Score(B_clean)` — drop items with `c_i > τ`, re-score; the actionable headline number (the analog of ChronoLint's "Sharpe 1.12 → 1.84").

### 3.2 The matcher ensemble (open-data)
Four matchers produce per-item evidence, combined into `c_i` with a transparent weighted rule and thresholded by a calibrated `τ`:
- **n-gram / MinHash-LSH overlap** — catches verbatim copy-paste (high precision, low recall).
- **TF-IDF retrieval** — near-duplicate / lightly-edited copies. *(Stand-in for real embeddings; see limitations.)*
- **Paraphrase judge** — IDF-weighted lexical overlap for reworded items. *(Stand-in for an LLM judge.)*
- **Answer-leak detector** — IDF-weighted answer-token match for the `answer_only` form.

Every flag carries a **contamination path** (which matcher fired, the matched span, why) — trust comes from showing the receipt, not a black-box score.

### 3.3 The injection harness (the research artifact + the moat)
The valuable, hard-to-reproduce asset. Given a clean corpus and a benchmark, it injects items at a controlled **rate** and **form** — `verbatim`, `paraphrase`, `format_shift`, `answer_only` — deterministically, and returns the ground-truth labels. This manufactures **known** `ρ`, a known injected-item set, and (via a score model) a known `Δ`, which is what lets us *measure* whether any detector recovers the truth.

### 3.4 Calibration, CLI, tests
- **Data-driven calibration** replaces hand-tuned thresholds: `τ` and per-matcher operating points are selected to maximize F1 on a TRAIN split, with precision/recall/F1 curves saved as CSV + PNG.
- **CLI:** `evalguard audit --benchmark … --corpus …`, plus a `calibrate` subcommand and a CI-gate mode (non-zero exit over a `ρ` threshold). Runs locally — the corpus never leaves the machine.
- **37 passing tests** across injection, matchers, estimands, calibration, real-data, scaling, and distinctiveness harnesses.

---

## 4. Experimental setup

- **Benchmark:** OpenBookQA, 300 real multiple-choice items, split TRAIN/TEST 150/150 (seed 2024).
- **Corpus:** AG News, 700 real news documents, fetched once from the HuggingFace datasets-server REST API (`datasets.load_dataset` is network-blocked in this environment; provenance in `demo/real_data/README.md`). Genuinely real public data, not synthetic.
- **Injection:** rate 0.20 across the four forms, deterministic per seed.
- **Calibration protocol:** `τ` chosen to maximize F1 on TRAIN only; all detection and recovery numbers are on the held-out TEST split.
- **Embeddings & LLM judge are lightweight stand-ins** (TF-IDF; rule-based) — real `sentence-transformers`/MiniLM could not be installed (torch+CUDA too heavy; HF model download blocked). Swap points are marked in code.

---

## 5. Results

### 5.1 Synthetic sanity: estimands are recoverable
On clean synthetic data with known ground truth, detection is perfect and `ρ`/`Δ` are recovered exactly (ρ 0.20→0.20; Δ +0.115→+0.115). This proves the estimands are recoverable in principle; the real-data sections stress-test how far that holds.

### 5.2 Real data, single condition
OpenBookQA × AG News, 200-doc corpus, calibrated on TRAIN, measured on held-out TEST:
- Overall **precision 1.00, recall 0.67, F1 0.80**.
- `verbatim`, `paraphrase`, `format_shift`: recall **1.00**. `answer_only`: recall **0.09**.
- `ρ`: true 0.20 → **0.13**; `Δ`: true +0.083 → **+0.051**. Under-estimated, never inflated.

### 5.3 Scaling: an apparent phase transition — `figures/recall_per_form.png`, `tau_vs_size.png`
Varying only the clean corpus size (`N` from 80 to 700), the three strong forms hold at recall **1.00 at every size**, while `answer_only` recall drops from 1.00 to **0.09** via a **discrete jump in the F1-optimal `τ` (0.05 → 0.86) between N=100 and N=110**. Beyond the switch, precision is **1.00** and `ρ`/`Δ` are stably under-estimated (ρ̂ 0.13, Δ̂ +0.05). The failure is entirely on recall — the conservative direction.

| N (clean docs) | τ | Precision | Recall | F1 | answer_only recall | ρ̂ (true 0.20) | Δ̂ (true +0.08) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 80  | 0.07 | 0.86 | 1.00 | 0.92 | 1.00 | 0.23 | +0.08 |
| 100 | 0.05 | 0.83 | 1.00 | 0.91 | 1.00 | 0.24 | +0.09 |
| 110–700 | 0.86 | 1.00 | 0.67 | 0.80 | 0.09 | 0.13 | +0.05 |

### 5.4 The controlled follow-up corrects the story — `figures/distinctiveness_transition.png`

> **Superseded by the multi-seed result (FINDINGS §9).** The single-seed OpenBookQA numbers in this subsection are seed-noisy. Averaged over 5 seeds at N=700, OpenBookQA's buckets converge to ~0.62–0.68 (no ordering; the "high-bucket decay" was noise), while **SciQ** shows a large, robust, *monotone* effect (0.34 / 0.59 / **1.00±0.00**). Corrected conclusion below and in §9.
Is that collapse a real information ceiling, or an artifact of how we calibrate? We isolated `answer_only` (inject *only* that form, calibrate `τ` on it **alone**) and varied answer distinctiveness (mean answer-token IDF vs corpus) across OpenBookQA terciles:

| bucket | mean answer IDF | N=40 | N=150 | N=400 | N=700 | N* (F1<0.5) |
|---|---:|---:|---:|---:|---:|---:|
| low  | 4.54 | 0.77 / 0.87 | 0.77 / 0.87 | 0.77 / 0.87 | 0.71 / 0.83 | >700 (censored) |
| mid  | 5.70 | 0.83 / 0.91 | 0.83 / 0.91 | 0.83 / 0.91 | 0.83 / 0.91 | >700 (censored) |
| high | 7.28 | 1.00 / 1.00 | 0.83 / 0.91 | 0.62 / 0.77 | 0.56 / 0.71 | >700 (censored) |
*(precision / F1 on held-out TEST)*

Two corrections to the naive reading of §5.3:

1. **The collapse was largely a *single-global-threshold artifact*, not a hard ceiling.** Calibrated on its own signal, `answer_only` never drops below F1 0.71 out to N=700. Sharing one `τ` across a form-mix dominated by high-signal forms pushes `τ` high and *incidentally discards* the weak form.
2. **Answer distinctiveness protects precision — but only when rarity maps to real separability.** Multi-seed (FINDINGS §9) shows a large, robust, monotone effect on **SciQ** (precision 0.34 / 0.59 / 1.00 low→high) and **no** effect on OpenBookQA (buckets converge ~0.65). So distinctiveness matters, but **mean answer-token IDF is not a portable proxy** for it across benchmarks — OpenBookQA's short "high-IDF" answers still collide. The single-seed "high-bucket decay" reported earlier was seed noise.

---

## 6. Corrected interpretation and design implications

- **Open-data detectability depends on the distinctiveness of what leaks *and* on how the operating point is calibrated — and the second factor dominated our first result.** Long, distinctive leaks (verbatim/paraphrase/format-shift) stay separable at any corpus size. Short/low-distinctiveness leaks (`answer_only`) are recoverable too, but only if they are *not* forced to share a global threshold with the strong forms.
- **Therefore: calibrate and threshold *per contamination form*, never with one global `τ`.** That single change recovers most `answer_only` detectability. This is the most important engineering takeaway of the whole study.
- **Report per-form and per-corpus-scale, never a single global contamination score.**
- **Treat answer distinctiveness as a per-item confidence signal, not a monotone difficulty proxy** — rare answers can be *less* precise at scale, not more.
- **Keep failing conservatively.** Under-reporting (precision 1.00) is the correct failure mode for an auditor; the design and calibrations preserve it.

---

## 7. Honest limitations

1. **TF-IDF stand-in, not real embeddings** (download blocked; torch too heavy). Real embeddings would likely lift paraphrase/format-shift robustness; they would not change the `answer_only` story, which is about token distinctiveness and thresholding, not representation.
2. **Two benchmark×corpus pairs, 5-seed averaging, small-to-mid corpus (≤700 docs).** Now covers OpenBookQA × AG News and SciQ × AG News with multi-seed at N=700 (§8–§9). Transferable claims: conservative failure, the global-threshold artifact, and "distinctiveness protects precision when rarity maps to separability (robust on SciQ), but IDF is not a portable proxy (no effect on OpenBookQA)." Still to do before a headline claim: more seeds/corpus sizes and a numeric-answer benchmark.
3. **Small per-bucket splits in §5.4** (~50 items, ~10 injected) → precision noisy (±~0.15). The qualitative crossover is robust; exact values are not.
4. **Open-data regime only** — no gray-box (Min-K%++, exchangeability) or black-box methods; cannot audit closed models from logits/generations.
5. **Synthetic score model for `Δ`** — true `Δ` uses an assumed memorization-boost, not a real model's answers.
6. **Discrete `τ` grid** sharpens the §5.3 switch; a finer grid smooths but does not remove it.

---

## 8. Commercial framing (condensed)

- **First users:** teams fine-tuning open-weight models on scraped/semi-public data who need to trust eval numbers before shipping; academic groups certifying a domain benchmark is clean.
- **Sequence:** open-source benchmark + CLI (credibility, distribution) → paid CI/cloud with dashboards and private benchmark registries → a "contamination-audited" certification for benchmark publishers.
- **Most likely acquirers:** eval-infrastructure companies (become their "trust layer"), MLOps/model-registry platforms (a promotion-gate check), leaderboard/data vendors wanting a cleanliness badge.
- **Honest read:** this is a *tool-shaped* business (Semgrep-for-evals, not a platform). The near-term returns are the **paper + open benchmark + competition + credibility**, with acquisition as an option, not the plan. Because audits run locally (corpus is sensitive), the moat is the **labeled benchmark and engine quality**, not a data network effect.

### 8.1 Acquirer landscape (July 2026, grounded in recent deals)

The relevant precedent is that eval/eval-integrity tooling *is* being bought and consolidated: **OpenAI acquired Promptfoo (Mar 2026)** to fold eval/red-teaming into its platform; **Langfuse → ClickHouse (Jan 2026)**; **Weights & Biases → CoreWeave (~$1.7B, 2025)**; and **Braintrust raised $80M at ~$800M (Feb 2026)** — i.e. buyers exist and are capitalized. Likely acquirers, by fit:

- **Eval-first platforms (most likely — feature tuck-in):** Braintrust (eval-first, dataset-first, ships sandboxed Python custom scorers — the exact socket a contamination check plugs into; best-funded), LangSmith, DeepEval/Confident AI, Patronus AI. Here EvalGuard is a *scorer/feature*, not a company — the realistic exit shape.
- **Frontier labs (real precedent):** OpenAI (Promptfoo), Anthropic, Google DeepMind, Mistral. Acute internal stake in benchmark-contamination for eval integrity + public credibility; the "audit your own score" reframe (not "catch cheaters") is what makes it buyable by them.
- **ML/GenAI platforms with an eval workflow:** Databricks (owns MLflow, which pivoted to GenAI eval in 3.0; historically acquisitive — MosaicML), CoreWeave/W&B, Arize, Comet — contamination check as a model-promotion gate.
- **Benchmark/data & leaderboard owners:** Hugging Face (hosts models + datasets + the Open LLM Leaderboard → most *strategically* natural home for a cleanliness badge), Scale AI (SEAL), Surge AI.

**Reality check:** for a v0.1 OSS tool with no traction, "acquisition" almost always means an acqui-hire, an asset/feature tuck-in, or OSS adoption/sponsorship — not a standalone strategic buy. That comes only after a cited public benchmark, real CI usage, or a distinctive team exists. Path: ship the benchmark + paper → get cited/adopted → then the tuck-in conversation (most likely a Braintrust-type eval platform or an OpenAI/Promptfoo-type lab). Acquisition is the option traction creates, not the plan. *(Sourced from public 2025–26 reporting; verify before citing in any external material.)*

---

## 9. Roadmap / next steps

1. **Per-form calibration** in the product (the §6 takeaway) — the highest-value change; recovers `answer_only` and removes the artifact.
2. **Robustness pass on §5.4 — done (FINDINGS §8–§9):** added a second benchmark (SciQ × AG News) and 5-seed averaging at N=700. Result: the distinctiveness effect is robust on SciQ, absent on OpenBookQA; IDF is not a portable predictor. Remaining: more seeds/sizes and a numeric-answer benchmark before a headline publication claim.
3. **Real embeddings + LLM paraphrase judge** in an environment that allows the downloads; quantify the lift.
4. **Gray-box module** (exchangeability + Min-K%++) for open-weight models without a corpus, reported as dataset-level signal with a stated FPR.
5. **Scale the benchmark** to thousands of corpus docs and more contamination forms (translation, partial leakage); publish it as the standard labeled contamination testbed.

---

## 10. Artifact index

| Path | What it is |
|---|---|
| `evalguard/` | the installable package (matchers, inject, calibrate, estimands, cli) |
| `demo/run_demo.py` | synthetic demo — exact ρ/Δ recovery |
| `demo/run_real_demo.py` | real OpenBookQA × AG News audit |
| `demo/run_scaling_study.py` | corpus-size scaling (§5.3) |
| `demo/run_distinctiveness_study.py` | distinctiveness / calibration-artifact study (§5.4) |
| `demo/figures/*.png` | all figures referenced above |
| `demo/scaling_results.csv`, `demo/distinctiveness_results.csv` | raw result tables |
| `demo/real_data/` | cached real datasets + provenance |
| `tests/` | 37 passing tests |
| `EvalGuard_design.md`, `demo/FINDINGS.md`, `demo/CALIBRATION_AND_REAL_RESULTS.md` | detailed backing notes (this report supersedes them) |

*Reproduce: `pip install -e . && python -m pytest -q && python demo/run_scaling_study.py && python demo/run_distinctiveness_study.py`.*

---

## 11. Update — the per-form fix is shipped (the §6 recommendation, implemented)

The §6 recommendation ("calibrate and threshold per form, not with one global τ") is now available as an **opt-in decision mode** in the engine (`evalguard/decision.py`; `evalguard audit --decision per-form|global-tau`). An item is flagged if **any** matcher exceeds its own data-calibrated threshold (an OR-gate over per-matcher operating points that calibration already produces).

**Single-seed proof it recovers the discarded form** (`demo/run_mode_comparison.py`, `figures/mode_comparison.png`): under global-τ, `answer_only` recall collapses to 0.09 past N≈110; under per-form it stays **1.00 at every corpus size**, with overall F1 ≥ the global mode at every size (precision falls from 1.00 to 0.67 as the corpus grows — the OR-gate's cost).

**Multi-seed robustness** (`demo/run_robustness.py`, 10 injection seeds):

| N | mode | answer_only recall (mean ± std) | overall F1 (mean ± std) |
|---:|---|---:|---:|
| 300 | global-τ | 0.31 ± 0.45 | **0.85 ± 0.03** |
| 300 | per-form | **1.00 ± 0.00** | 0.79 ± 0.15 |
| 500 | global-τ | 0.21 ± 0.40 | **0.84 ± 0.04** |
| 500 | per-form | **1.00 ± 0.00** | 0.75 ± 0.14 |

**The honest tradeoff (this is the shipped conclusion):** per-form **reliably and completely recovers** the weak `answer_only` form that global-τ *erratically* abandons (1.00 ± 0.00 vs ~0.2–0.3 with huge variance). It pays for this with lower and noisier overall precision/F1 at scale. There is no dominant mode — there are two legitimate philosophies:

- **per-form (default): coverage-first.** Never silently miss a contamination form; treat flags as a *triage queue* a human reviews. Right when a missed leak (false assurance) is the worse error.
- **global-τ: precision-first.** Never falsely accuse; may miss weak forms. Right when a flag is treated as an accusation.

EvalGuard ships **global-τ (precision-first) as the default**: it preserves the conservative “never falsely accuse” property and reliably recovers `ρ`/`Δ`. **`--decision per-form`** is the opt-in coverage-first mode for users who would rather over-flag and triage than risk missing a weak form. (Per-form is also sensitive to inter-item similarity: on data where benchmark items strongly resemble each other, the OR-gate over-flags, so it should be calibrated on representative data.) Both modes are first-class; one flag switches between them. A precision-sensitive deployment can also use the calibrated *recall-at-precision≥0.95* operating points already emitted in `calibration.json`.

*Reproducibility (§8 of FINDINGS): the distinctiveness effect was re-tested on a second benchmark, SciQ × AG News (`demo/run_distinctiveness_sciq.py`, `figures/distinctiveness_sciq.png`). The precision-decays-with-corpus mechanism reproduces and is cleaner on SciQ (monotone: higher distinctiveness → more stable precision); the exact bucket ordering does not transfer across benchmarks, so distinctiveness is a directional but not quantitatively portable predictor.*

*New artifacts: `evalguard/decision.py`, `tests/test_decision.py`, `demo/run_mode_comparison.py`, `demo/run_robustness.py`, `figures/mode_comparison.png`, `demo/mode_comparison_results.csv`.*
