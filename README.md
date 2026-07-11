# EvalGuard

**Open-data contamination auditing for LLM evaluations.** Given a model's
training/fine-tuning corpus and a benchmark, EvalGuard estimates how much of the
reported score is inflated because the benchmark already leaked into the corpus —
and shows the **evidence trail** behind every flag.

It is not a benchmark, an eval harness, or a leaderboard. It sits *next to* your
eval and puts a defensible asterisk on the number.

```
rho_hat (contamination rate) : 0.2000   (12 / 60 items compromised)
Delta_hat (score inflation)  : +0.1167   (full 0.5333 -> clean 0.4167, dropped 12)

  item-0023   c = 1.000
      - [ngram 1.00] 100% of 5-gram shingles found in corpus doc ...
      - [answer 1.00] answer tokens 100% present (IDF-weighted) in ...
```

> **Scope.** This is the v1 **open-data** auditor: you *have* the training corpus,
> so per-item membership is a search problem and detection is reliable. Gray-box
> (Min-K%++, exchangeability) and black-box methods are on the roadmap, not in
> this release. EvalGuard only ever claims **per-item membership evidence**, not
> certified proof — see *Honesty* below.

---

## Why

Benchmark contamination silently inflates reported scores: if test items leaked
into training, a model looks better than it is. The usual fix is *prevention*
(fresh/private benchmarks), but teams that fine-tune or evaluate on domain data
they can't constantly regenerate need to **detect** contamination in their own
pipeline before trusting a number. That's EvalGuard's wedge: self-serve,
corpus-in-hand, "trust your own score" — not "catch someone else cheating."

## Install

```bash
pip install -e .            # numpy, scikit-learn, scipy
pip install -e ".[dev]"     # + pytest, matplotlib (tests & figures)
```

Python 3.9+. Runs locally — your corpus never leaves the machine.

## Quickstart

EvalGuard measures two estimands:

- **`ρ` (contamination rate)** — fraction of the benchmark compromised.
- **`Δ` (score inflation)** — `Score(B) − Score(B_clean)`: how many points of the
  reported score are memorization rather than capability. **This is the number
  you act on.**

```bash
# 1. See it recover known ground truth on synthetic data (exact recovery):
python demo/run_demo.py

# 2. Audit your own data. Inputs:
#    benchmark.jsonl : {"id","question","answer"} per line
#    corpus.jsonl    : {"id","text"} per line  (or a .txt split on blank lines)
#    scores.json     : {item_id: correctness}  (optional, enables Delta)
evalguard audit \
    --benchmark examples/benchmark.jsonl \
    --corpus    examples/corpus.jsonl \
    --scores    examples/scores.json \
    --json report.json

# 3. Use it as a CI gate — non-zero exit if contamination exceeds a threshold:
evalguard audit --benchmark b.jsonl --corpus d.jsonl --fail-over 0.05
```

**Calibrate on your own data first** (recommended). Thresholds don't transfer
across corpora, so learn them on data representative of yours by injecting the
benchmark into a clean reference corpus:

```bash
evalguard calibrate --benchmark benchmark.jsonl --corpus clean_corpus.jsonl --rate 0.2
```

This writes per-matcher operating points + precision/recall/F1 curves you can
inspect, then `audit` uses them.

## How it works

1. **Injection harness** (`inject.py`) — manufactures ground truth by inserting
   benchmark items into a clean corpus at a controlled **rate** and **form**
   (`verbatim`, `paraphrase`, `format_shift`, `answer_only`), deterministic by
   seed. This is what lets EvalGuard *measure* whether detection recovers the truth.
2. **Matcher ensemble** (`matchers/`) — each emits an evidence score in `[0,1]`
   with a human-readable trail: `ngram` (MinHash n-gram containment), `embedding`
   (TF-IDF retrieval), `paraphrase` (reworded-match judge), `answer`
   (answer-token leak). No single metric is trusted.
3. **Aggregator** (`aggregate.py`) — combines evidence into a per-item score `c_i`
   via a transparent weighted noisy-OR, keeping the **contamination path**.
4. **Calibration** (`calibrate.py`) — selects thresholds *from data*, not by hand.
5. **Decision** (`decision.py`) — two modes (see below).
6. **Estimands + report** — `ρ`, `Δ`, and a ranked, evidence-backed report.

## Decision modes

| `--decision` | behavior | when to use |
|---|---|---|
| `global-tau` *(default)* | flag if combined `c_i > τ` | **precision-first / conservative** — never falsely accuse; recovers `ρ`/`Δ`. Can miss low-distinctiveness leaks (e.g. `answer_only`) on large corpora. |
| `per-form` | flag if **any** matcher exceeds its own calibrated threshold (OR-gate) | **coverage-first** — recovers weak forms a single `τ` discards, at a precision cost. Best when a missed leak is worse than an extra flag to triage. Sensitive to inter-item similarity; calibrate on representative data. |

This tradeoff is not a detail — it's a measured result (see `demo/FINDINGS.md`
§7 and `EvalGuard_REPORT.md` §11). The default is the conservative mode.

## Honesty

A corpus search yields **evidence of membership**, not proof. EvalGuard reports
at the strongest level its access supports and never above it: with the corpus,
per-item membership *evidence with a receipt*; it does not claim certified proof,
and it does not attempt to accuse closed models from generations alone. Weak
forms (e.g. a short, common leaked answer) are honestly hard — the tool reports
low evidence rather than pretending, and its default failure mode is to
**under-report** contamination rather than fabricate it.

## Reproduce the findings

```bash
pytest -q                              # 37 tests
python demo/run_demo.py                # synthetic: exact rho/Delta recovery
python demo/run_real_demo.py           # real data: OpenBookQA x AG News
python demo/run_scaling_study.py       # how detectability shifts with corpus size
python demo/run_distinctiveness_study.py   # what governs the answer_only blind spot
python demo/run_distinctiveness_sciq.py    # reproducibility on a 2nd benchmark (SciQ)
python demo/run_mode_comparison.py     # global-tau vs per-form
python demo/run_robustness.py          # multi-seed robustness
```

Figures land in `demo/figures/`. The full write-up is **`EvalGuard_REPORT.md`**;
the research-style findings are in **`demo/FINDINGS.md`**.

## Layout

```
evalguard/        package: inject, matchers, aggregate, calibrate, decision,
                  estimands, report, cli, config
demo/             runnable studies + figures + real data + FINDINGS.md
examples/         tiny benchmark/corpus/scores for the quickstart
tests/            37 tests
EvalGuard_REPORT.md   the consolidated project report
```

## Limitations (v0.1)

Embeddings use a TF-IDF stand-in and the paraphrase judge is rule-based (swap
points marked `# SWAP:`); open-data regime only (no gray-box/black-box);
validated on a single benchmark×corpus pair. See `EvalGuard_REPORT.md` §7 for the
full list and roadmap.

## License

MIT — see [LICENSE](LICENSE). Contributions welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
