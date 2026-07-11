# EvalGuard — Handoff for a terminal (Claude Code) agent

You are picking up **EvalGuard v0.1.0**, a working, launch-packaged open-source
project. This doc is everything you need; read `EvalGuard_REPORT.md` for depth
and `demo/FINDINGS.md` for the research results. Do not re-derive prior work.

## 1. What it is (one line)
An **open-data contamination auditor for LLM evaluations**: given a training/
fine-tuning corpus `D` and a benchmark `B`, it estimates how much of a reported
score is inflated because `B` leaked into `D`, and shows the evidence trail.
Two estimands: `ρ` (contamination rate) and `Δ` (score inflation). It is a
dev-tool / research artifact, NOT a benchmark or eval harness.

## 2. Current state
- **v0.1.0, 37 tests passing** (`pytest -q`, ~30s). Python 3.9+.
- Installable: `pip install -e ".[dev]"`; console script `evalguard`.
- Fully packaged for OSS: LICENSE (MIT), README, CHANGELOG, CONTRIBUTING,
  `.github/workflows/ci.yml`, `.gitignore`, `pyproject.toml`.
- **Not yet git-committed or pushed** (blocked in the build sandbox; see §7).

## 3. Repo layout
```
evalguard/            package
  corpus.py benchmark.py inject.py synth.py config.py
  matchers/  ngram.py embedding.py paraphrase.py answer.py  (+ Matcher proto)
  aggregate.py     evidence -> c_i (weighted noisy-OR) + contamination path
  calibrate.py     data-driven per-matcher + tau operating points (CSV/PNG curves)
  decision.py      global-tau (default) vs per-form OR-gate; MIN_EVIDENCE=0.05 floor
  estimands.py     rho, Delta
  report.py cli.py
  calibration/     shipped default operating points (calibration.json + curves)
demo/
  run_demo.py                    synthetic: exact rho/Delta recovery
  run_real_demo.py               OpenBookQA x AG News
  run_scaling_study.py           detectability vs corpus size
  run_distinctiveness_study.py   OBQA distinctiveness terciles (single seed)
  run_distinctiveness_sciq.py    SciQ replication
  run_distinctiveness_multiseed.py   5-seed robustness at fixed N (authoritative)
  run_mode_comparison.py         global-tau vs per-form
  run_robustness.py              multi-seed per-form vs global-tau
  real_data/    openbookqa.jsonl(300) ag_news.jsonl(700) sciq.jsonl(300)
                + _build_cache.py + README (provenance) + cached raw pages
  figures/      all PNGs;  *_results.csv    raw result tables
  FINDINGS.md   research-style results (READ THIS for conclusions)
examples/       benchmark.jsonl corpus.jsonl corpus_clean.jsonl scores.json
tests/          37 tests
EvalGuard_REPORT.md   consolidated project report (authoritative narrative)
EvalGuard_design.md   original design doc (historical)
```

## 4. How to run
```bash
pip install -e ".[dev]"
pytest -q                                   # 37 tests
python demo/run_demo.py                      # exact recovery on synthetic
# audit your own data (global-tau default):
evalguard audit --benchmark B.jsonl --corpus D.jsonl --scores S.json --json out.json
# calibrate thresholds on representative data FIRST (recommended):
evalguard calibrate --benchmark B.jsonl --corpus clean_D.jsonl --rate 0.2
# CI gate: non-zero exit over a contamination threshold:
evalguard audit --benchmark B.jsonl --corpus D.jsonl --fail-over 0.05
```
Formats: benchmark `{id,question,answer}` jsonl; corpus `{id,text}` jsonl (or .txt on blank lines); scores `{item_id: correctness}` json.

## 5. Key findings (current, CORRECTED — do not restate the superseded versions)
1. **Synthetic:** exact `ρ`/`Δ` recovery (sanity that estimands are recoverable).
2. **Real (OBQA × AG News, global-tau):** overall P 1.00 / R 0.67 / F1 0.80;
   `ρ` 0.20→0.13 (conservative under-estimate). verbatim/paraphrase/format_shift
   recall 1.00; `answer_only` recall 0.09.
3. **The `answer_only` "collapse" is largely a single-global-threshold artifact**,
   not an information ceiling. The **per-form OR-gate** (`decision.py`) recovers it
   (recall **1.00 ± 0.00** across seeds) at a precision cost. **Default is
   `global-tau` (conservative/precision-first); `per-form` is opt-in** via
   `--decision per-form`. Both are first-class.
4. **Distinctiveness effect (multi-seed, FINDINGS §9 — authoritative):** robust,
   large, monotone on **SciQ** (answer-only precision 0.34 / 0.59 / **1.00±0.00**
   low→high, N=700) but **absent on OpenBookQA** (buckets converge ~0.62–0.68).
   Conclusion: distinctiveness protects precision only when rarity maps to real
   separability; **mean answer-token IDF is NOT a portable cross-benchmark
   predictor.** (§7's single-seed OBQA numbers were seed noise — superseded by §9.)
5. Throughout, the auditor **fails conservatively** (under-reports, precision-first
   default) — the correct failure mode for an auditor.

## 6. Known stand-ins & limitations (marked `# SWAP:` in code)
- `matchers/embedding.py`: **TF-IDF stand-in**, not real embeddings. Swap `_Embedder`
  for sentence-transformers (`try_sentence_transformer` hook exists) — see §7.
- `matchers/paraphrase.py`: rule-based judge stand-in for an LLM "same item?" judge.
- Open-data regime ONLY: no gray-box (Min-K%++, exchangeability) or black-box.
- `Δ` uses a synthetic memorization-boost score model, not a real model's answers.
- Validated on 2 benchmark×corpus pairs (OBQA, SciQ), ≤700-doc corpus, ≤5 seeds.

## 7. Remaining work (prioritized). BLOCKED-here items need YOUR machine/creds.
**A. Ship it (blocked in build sandbox; trivial on a real machine):**
- `git init && git add . && git commit -m "EvalGuard v0.1.0" && git tag -a v0.1.0 -m v0.1.0`
- Push: `gh repo create <org>/evalguard --public --source=. --push` (needs `gh` auth).
- Replace `your-org` placeholder URLs in `pyproject.toml` with the real repo.
**B. Real embeddings (needs a working sentence-transformers/torch install — was
   blocked in sandbox):** wire `all-MiniLM-L6-v2` behind `matchers/embedding.py`
   `# SWAP:` / `try_sentence_transformer`, re-run `run_real_demo.py` +
   `run_scaling_study.py`, quantify the lift. Won't rescue `answer_only`.
**C. Strengthen the research (fully doable, no creds):**
- More seeds/sizes in `run_distinctiveness_multiseed.py`; add a **numeric-answer
  benchmark (e.g. GSM8K)** as a 3rd pair to re-test "IDF not portable."
- Add a `translation` contamination form to `inject.py`.
- Gray-box module (Min-K%++, exchangeability) — needs model logits, so needs a
  real (open-weight) model in the loop.

## 8. Environment gotchas from the build sandbox (mostly NOT present on your machine)
- **`datasets.load_dataset` was network-blocked**; real data was fetched via the
  HuggingFace **datasets-server REST `/rows` endpoint** and cached in
  `demo/real_data/` (see its README + `_build_cache.py`). On a normal machine
  `datasets` should work directly.
- **`sentence-transformers`/torch could not be installed** (too heavy / blocked) —
  hence the TF-IDF stand-in.
- There are **orphaned, permission-locked dirs** `evalguard/_test_calib*/` and
  `evalguard/calibration_real/` left by an earlier process — harmless, already in
  `.gitignore`; delete if your FS lets you.
- Tests all use `tmp_path`; nothing writes into the package on a clean run.

## 9. Decisions to respect (don't silently revert)
- **Default decision mode is `global-tau`** (conservative). `per-form` is the opt-in
  coverage mode. This was chosen after multi-seed showed per-form over-flags on
  high-similarity data and lowers/ destabilizes precision.
- **`MIN_EVIDENCE = 0.05`** floor in `decision.py` prevents a degenerate 0.0
  threshold from flagging everything — keep it.
- **Calibrate on representative data**; shipped `calibration.json` is illustrative,
  and per-matcher thresholds do NOT transfer across corpora.
- **Honesty rule:** never claim more than the corpus evidence supports; report
  membership *evidence*, not proof (see REPORT §2.4).
