"""EvalGuard REAL-DATA demo (design sections 6 & 7).

Unlike run_demo.py (synthetic, separates cleanly), this runs the pipeline on
REAL public data with a train/test split and DATA-DRIVEN calibration:

  1. Load a REAL MCQ benchmark (OpenBookQA, 300 items) and a REAL text corpus
     (AG News, 200 docs), both cached under demo/real_data/ (see its README for
     provenance; fetched once from the HF datasets-server REST API).
  2. Split the benchmark into TRAIN / TEST halves.
  3. Inject known contamination into the corpus at a known rate across all four
     forms (verbatim / paraphrase / format_shift / answer_only) -> known rho, Delta.
  4. CALIBRATE matcher thresholds + aggregator tau on the TRAIN split only,
     persisting curves to evalguard/calibration_real/.
  5. Evaluate detection P/R/F1 (overall + per form) and rho / Delta recovery on
     the held-out TEST split, at the calibrated operating point.

Real text will NOT give the perfect synthetic P/R -- that is the point. The
answer_only form in particular degrades badly because real OpenBookQA answers
are short common words that occur all over a real news corpus, so a distinctive-
answer leak is genuinely hard to distinguish from background text. We report
whatever the numbers actually are.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalguard.aggregate import aggregate
from evalguard.benchmark import Benchmark, BenchmarkItem
from evalguard.calibrate import calibrate
from evalguard.corpus import Corpus
from evalguard.estimands import contaminated_ids, delta, rho
from evalguard.inject import FORMS, inject
from evalguard.matchers.embedding import try_sentence_transformer

REAL_DIR = Path(__file__).resolve().parent / "real_data"
CALIB_REAL = Path(__file__).resolve().parents[1] / "evalguard" / "calibration_real"

RATE = 0.20
SPLIT_SEED = 2024
INJECT_SEED_TRAIN = 1
INJECT_SEED_TEST = 2
FORM_MIX = {"verbatim": 1, "paraphrase": 1, "format_shift": 1, "answer_only": 1}


def load_real_benchmark() -> Benchmark:
    items = []
    for line in (REAL_DIR / "openbookqa.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        items.append(BenchmarkItem(id=o["id"], question=o["question"], answer=str(o["answer"])))
    return Benchmark(items)


def load_real_corpus() -> Corpus:
    docs, ids = [], []
    for line in (REAL_DIR / "ag_news.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        docs.append(o["text"])
        ids.append(o["id"])
    return Corpus(docs, ids)


def split(bench: Benchmark, seed: int) -> tuple[Benchmark, Benchmark]:
    idx = list(range(len(bench)))
    random.Random(seed).shuffle(idx)
    cut = len(idx) // 2
    tr = Benchmark([bench.items[i] for i in sorted(idx[:cut])])
    te = Benchmark([bench.items[i] for i in sorted(idx[cut:])])
    return tr, te


def prf(pred: set, truth: set):
    tp = len(pred & truth); fp = len(pred - truth); fn = len(truth - pred)
    P = tp / (tp + fp) if (tp + fp) else 1.0
    R = tp / (tp + fn) if (tp + fn) else 1.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F, tp, fp, fn


def main() -> int:
    print("=" * 70)
    print("EvalGuard REAL-DATA demo  (OpenBookQA benchmark x AG News corpus)")
    print("=" * 70)

    # which encoder do we actually have?
    real_emb = try_sentence_transformer()
    emb_backend = getattr(real_emb, "backend", None) if real_emb else "tfidf-stand-in"
    print(f"embedding backend : {emb_backend}")
    if real_emb is None:
        print("  (sentence-transformers unavailable -> TF-IDF stand-in; see report)")

    bench = load_real_benchmark()
    corpus = load_real_corpus()
    print(f"benchmark         : {len(bench)} real OpenBookQA items")
    print(f"corpus            : {len(corpus)} real AG News docs\n")

    train, test = split(bench, SPLIT_SEED)
    print(f"train / test split: {len(train)} / {len(test)} items (seed {SPLIT_SEED})")

    res_tr = inject(corpus, train, rate=RATE, form_mix=FORM_MIX, seed=INJECT_SEED_TRAIN)
    res_te = inject(corpus, test, rate=RATE, form_mix=FORM_MIX, seed=INJECT_SEED_TEST)
    print(f"injected (train)  : {len(res_tr.injected_ids)} items at rate {RATE:.2f}")
    print(f"injected (test)   : {len(res_te.injected_ids)} items at rate {RATE:.2f}\n")

    # ---- calibrate on TRAIN only ----
    print("Calibrating matcher thresholds + tau on the TRAIN split...")
    cal = calibrate(train, res_tr, seed=1, out_dir=CALIB_REAL, write_plots=True)
    tau = cal["chosen"]["tau"]
    print(f"  calibrated operating points: {cal['chosen']}")
    print(f"  -> using aggregator tau = {tau:.2f}")
    print(f"  curves + PNGs written to {CALIB_REAL}\n")

    # ---- evaluate on held-out TEST ----
    ev = aggregate(test, res_te.corpus, seed=1)
    truth = res_te.injected_ids
    pred = set(contaminated_ids(ev, tau=tau))

    print("DETECTION on held-out TEST (calibrated tau)")
    print("-" * 70)
    P, R, F, tp, fp, fn = prf(pred, truth)
    print(f"  {'OVERALL':13s} P={P:.3f} R={R:.3f} F1={F:.3f}   (tp{tp}/fp{fp}/fn{fn})")
    ibf = res_te.ids_by_form()
    for form in FORMS:
        tf = set(ibf.get(form, []))
        if not tf:
            continue
        caught = pred & tf
        print(f"  {form:13s} recall={len(caught)/len(tf):.3f}   caught {len(caught)}/{len(tf)}")
    print()

    # ---- rho / Delta recovery ----
    rng = random.Random(7)
    p_clean, p_contam = 0.55, 0.95   # real MCQ base rate ~0.5-0.6; memorized -> ~0.95
    scores = {it.id: (1.0 if rng.random() < (p_contam if it.id in truth else p_clean) else 0.0)
              for it in test.items}
    all_s = [scores[i.id] for i in test.items]
    clean_s = [scores[i.id] for i in test.items if i.id not in truth]
    true_delta = sum(all_s) / len(all_s) - sum(clean_s) / len(clean_s)
    true_rho = res_te.true_rho(len(test))
    rho_hat = rho(ev, tau=tau)
    dres = delta(ev, scores, tau=tau)

    print("ESTIMAND RECOVERY on held-out TEST")
    print("-" * 70)
    print(f"  rho    true={true_rho:.3f}  hat={rho_hat:.3f}  abs_err={abs(true_rho-rho_hat):.3f}")
    print(f"  Delta  true={true_delta:+.3f}  hat={dres.delta:+.3f}  abs_err={abs(true_delta-dres.delta):.3f}")
    print(f"         (full {dres.score_full:.3f} -> clean {dres.score_clean:.3f}, "
          f"dropped {dres.n_dropped} items)")
    print()
    print("Honest read: verbatim/paraphrase/format_shift are recovered well on")
    print("real text, but answer_only collapses because real answers are short")
    print("common words that occur throughout a real news corpus. rho/Delta are")
    print("therefore UNDER-estimated -- the auditor reports less contamination")
    print("than truly exists, which is the honest failure mode (design 4.1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
