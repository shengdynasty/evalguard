"""Does PER-FORM calibration fix the answer_only collapse? (FINDINGS section 7 -> product)

Re-runs the mixed-form scaling setup under BOTH decision modes:
  * global-tau : flag if combined c_i > one calibrated tau (legacy)
  * per-form   : flag if ANY matcher exceeds its own calibrated threshold (new default)
and reports overall F1 + answer_only recall + rho/Delta recovery for each, as the
clean corpus grows. Prediction: per-form keeps answer_only detectable where
global-tau abandons it, at comparable precision.
"""
from __future__ import annotations

import csv
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalguard import decision
from evalguard.aggregate import aggregate
from evalguard.benchmark import Benchmark, BenchmarkItem
from evalguard.calibrate import calibrate
from evalguard.corpus import Corpus
from evalguard.estimands import contaminated_ids, delta, rho
from evalguard.inject import FORMS, inject

HERE = Path(__file__).resolve().parent
REAL_DIR = HERE / "real_data"
FIG_DIR = HERE / "figures"
CSV_PATH = HERE / "mode_comparison_results.csv"

RATE = 0.20
SPLIT_SEED = 2024
FORM_MIX = {"verbatim": 1, "paraphrase": 1, "format_shift": 1, "answer_only": 1}
SIZES = [80, 110, 175, 300, 500, 700]
P_CLEAN, P_CONTAM, SCORE_SEED = 0.55, 0.95, 7


def load_bench():
    import json
    return Benchmark([BenchmarkItem(id=o["id"], question=o["question"], answer=str(o["answer"]))
                      for o in (json.loads(l) for l in
                                (REAL_DIR / "openbookqa.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())])


def load_corpus():
    import json
    docs, ids = [], []
    for l in (REAL_DIR / "ag_news.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            o = json.loads(l); docs.append(o["text"]); ids.append(o["id"])
    return Corpus(docs, ids)


def split(bench, seed):
    idx = list(range(len(bench))); random.Random(seed).shuffle(idx); cut = len(idx) // 2
    return (Benchmark([bench.items[i] for i in sorted(idx[:cut])]),
            Benchmark([bench.items[i] for i in sorted(idx[cut:])]))


def prf(pred, truth):
    tp = len(pred & truth); fp = len(pred - truth); fn = len(truth - pred)
    P = tp / (tp + fp) if (tp + fp) else 1.0
    R = tp / (tp + fn) if (tp + fn) else 1.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F


def run_size(n, train, test, corpus):
    sub = Corpus(list(corpus.docs[:n]), list(corpus.doc_ids[:n]))
    res_tr = inject(sub, train, rate=RATE, form_mix=FORM_MIX, seed=1)
    res_te = inject(sub, test, rate=RATE, form_mix=FORM_MIX, seed=2)
    with tempfile.TemporaryDirectory() as td:
        cal = calibrate(train, res_tr, seed=1, out_dir=Path(td), write_plots=False)
    chosen = cal["chosen"]; tau = float(chosen["tau"])
    ev = aggregate(test, res_te.corpus, seed=1)
    truth = res_te.injected_ids
    ibf = res_te.ids_by_form(); ao = set(ibf.get("answer_only", []))

    rng = random.Random(SCORE_SEED)
    scores = {it.id: (1.0 if rng.random() < (P_CONTAM if it.id in truth else P_CLEAN) else 0.0)
              for it in test.items}
    true_rho = res_te.true_rho(len(test))
    all_s = [scores[i.id] for i in test.items]
    clean_s = [scores[i.id] for i in test.items if i.id not in truth]
    true_delta = sum(all_s) / len(all_s) - sum(clean_s) / len(clean_s)

    out = {"corpus_size": n, "true_rho": round(true_rho, 3), "true_delta": round(true_delta, 3)}
    # global-tau
    pg = set(contaminated_ids(ev, tau=tau)); P, R, F = prf(pg, truth)
    out.update({"global_tau": round(tau, 3), "global_P": round(P, 3), "global_R": round(R, 3),
                "global_F1": round(F, 3),
                "global_ans_recall": round(len(pg & ao) / len(ao), 3) if ao else float("nan"),
                "global_rho_hat": round(rho(ev, tau=tau), 3),
                "global_delta_hat": round(delta(ev, scores, tau=tau).delta, 3)})
    # per-form
    pp = set(decision.contaminated_ids(ev, chosen)); P, R, F = prf(pp, truth)
    out.update({"perform_P": round(P, 3), "perform_R": round(R, 3), "perform_F1": round(F, 3),
                "perform_ans_recall": round(len(pp & ao) / len(ao), 3) if ao else float("nan"),
                "perform_rho_hat": round(decision.rho(ev, chosen), 3),
                "perform_delta_hat": round(decision.delta(ev, scores, chosen).delta, 3)})
    return out


def make_figure(rows):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    xs = [r["corpus_size"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    a1.plot(xs, [r["global_ans_recall"] for r in rows], "o--", color="#e53e3e", label="global-tau")
    a1.plot(xs, [r["perform_ans_recall"] for r in rows], "^-", color="#38a169", label="per-form (new)")
    a1.set_title("answer_only recall vs corpus size"); a1.set_ylabel("answer_only recall (TEST)")
    a2.plot(xs, [r["global_F1"] for r in rows], "o--", color="#e53e3e", label="global-tau")
    a2.plot(xs, [r["perform_F1"] for r in rows], "^-", color="#38a169", label="per-form (new)")
    a2.set_title("overall F1 vs corpus size"); a2.set_ylabel("overall F1 (TEST)")
    for ax in (a1, a2):
        ax.set_ylim(-0.05, 1.05); ax.set_xlabel("clean corpus size (docs)")
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Per-form calibration recovers the form that global-tau discards", y=1.02)
    fig.tight_layout(); fig.savefig(FIG_DIR / "mode_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 74)
    print("EvalGuard MODE COMPARISON  global-tau  vs  per-form  (mixed-form scaling)")
    print("=" * 74)
    bench = load_bench(); corpus = load_corpus(); train, test = split(bench, SPLIT_SEED)
    rows = [run_size(n, train, test, corpus) for n in SIZES]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    make_figure(rows)
    h = f"{'N':>4} | {'ans_recall':>21} | {'overall_F1':>19} | {'rho_hat(true)':>16}"
    print(h); print(f"{'':>4} | {'global   per-form':>21} | {'global  per-form':>19} | {'g / p (true)':>16}")
    print("-" * len(h))
    for r in rows:
        print(f"{r['corpus_size']:>4} | {r['global_ans_recall']:>9.2f} {r['perform_ans_recall']:>11.2f} | "
              f"{r['global_F1']:>7.2f} {r['perform_F1']:>10.2f} | "
              f"{r['global_rho_hat']:>4.2f}/{r['perform_rho_hat']:>4.2f}({r['true_rho']:.2f})")
    print(f"\nCSV    -> {CSV_PATH}\nfigure -> {FIG_DIR}/mode_comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
