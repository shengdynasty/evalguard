"""EvalGuard CORPUS-SIZE SCALING STUDY (design sections 6 & 7).

Open question this answers: how do open-data contamination detectability and the
data-calibrated operating point (aggregator tau) change as the CLEAN corpus
grows? Bigger corpora change IDF statistics and raise the chance that a benchmark
item's tokens coincidentally appear in unrelated background text -- which should
hurt the low-distinctiveness `answer_only` form the most.

Protocol (fixed benchmark, fixed injection, only corpus size varies):
  * Real benchmark : OpenBookQA (300 real MCQ items), split TRAIN/TEST (150/150).
  * Real corpus    : AG News (700 real news docs), fetched once from the HF
                     datasets-server REST API (see real_data/README.md).
  * For each corpus size N in SIZES:
      - take the first N clean docs,
      - inject the SAME contamination (rate 0.20, all four forms, fixed seeds)
        into that corpus for TRAIN and TEST separately -> known rho, injected set,
      - CALIBRATE the aggregator tau on the TRAIN split only,
      - evaluate detection (overall P/R/F1 + per-form recall) and rho/Delta
        recovery on the held-out TEST split at that tau.
  * Emit demo/scaling_results.csv and demo/figures/*.png.

Everything is deterministic given the seeds. Honest expectation: verbatim /
paraphrase / format_shift hold up as N grows; answer_only stays near-undetectable
(short common-word answers), so rho/Delta stay conservatively UNDER-estimated.
"""
from __future__ import annotations

import csv
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalguard.aggregate import aggregate
from evalguard.benchmark import Benchmark, BenchmarkItem
from evalguard.calibrate import calibrate
from evalguard.corpus import Corpus
from evalguard.estimands import contaminated_ids, delta, rho
from evalguard.inject import FORMS, inject

HERE = Path(__file__).resolve().parent
REAL_DIR = HERE / "real_data"
FIG_DIR = HERE / "figures"
CSV_PATH = HERE / "scaling_results.csv"

RATE = 0.20
SPLIT_SEED = 2024
INJECT_SEED_TRAIN = 1
INJECT_SEED_TEST = 2
FORM_MIX = {"verbatim": 1, "paraphrase": 1, "format_shift": 1, "answer_only": 1}
SIZES = [80, 100, 110, 120, 130, 175, 300, 500, 700]

# score model for Delta (same as run_real_demo): real MCQ base-rate ~0.55,
# memorized items answered correctly ~0.95.
P_CLEAN, P_CONTAM = 0.55, 0.95
SCORE_SEED = 7


def load_real_benchmark() -> Benchmark:
    import json
    items = []
    for line in (REAL_DIR / "openbookqa.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        items.append(BenchmarkItem(id=o["id"], question=o["question"], answer=str(o["answer"])))
    return Benchmark(items)


def load_real_corpus() -> Corpus:
    import json
    docs, ids = [], []
    for line in (REAL_DIR / "ag_news.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        docs.append(o["text"]); ids.append(o["id"])
    return Corpus(docs, ids)


def split(bench: Benchmark, seed: int):
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


def run_one(size: int, train: Benchmark, test: Benchmark, full_corpus: Corpus) -> dict:
    sub = Corpus(list(full_corpus.docs[:size]), list(full_corpus.doc_ids[:size]))
    res_tr = inject(sub, train, rate=RATE, form_mix=FORM_MIX, seed=INJECT_SEED_TRAIN)
    res_te = inject(sub, test, rate=RATE, form_mix=FORM_MIX, seed=INJECT_SEED_TEST)

    with tempfile.TemporaryDirectory() as td:
        cal = calibrate(train, res_tr, seed=1, out_dir=Path(td), write_plots=False)
    tau = float(cal["chosen"]["tau"])

    ev = aggregate(test, res_te.corpus, seed=1)
    truth = res_te.injected_ids
    pred = set(contaminated_ids(ev, tau=tau))
    P, R, F, tp, fp, fn = prf(pred, truth)

    ibf = res_te.ids_by_form()
    per_form = {}
    for form in FORMS:
        tf = set(ibf.get(form, []))
        per_form[form] = (len(pred & tf) / len(tf)) if tf else float("nan")

    rng = random.Random(SCORE_SEED)
    scores = {it.id: (1.0 if rng.random() < (P_CONTAM if it.id in truth else P_CLEAN) else 0.0)
              for it in test.items}
    all_s = [scores[i.id] for i in test.items]
    clean_s = [scores[i.id] for i in test.items if i.id not in truth]
    true_delta = sum(all_s) / len(all_s) - sum(clean_s) / len(clean_s)
    true_rho = res_te.true_rho(len(test))
    rho_hat = rho(ev, tau=tau)
    dres = delta(ev, scores, tau=tau)

    return {
        "corpus_size": size,
        "tau": round(tau, 4),
        "precision": round(P, 4), "recall": round(R, 4), "f1": round(F, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "recall_verbatim": round(per_form["verbatim"], 4),
        "recall_paraphrase": round(per_form["paraphrase"], 4),
        "recall_format_shift": round(per_form["format_shift"], 4),
        "recall_answer_only": round(per_form["answer_only"], 4),
        "rho_true": round(true_rho, 4), "rho_hat": round(rho_hat, 4),
        "rho_abs_err": round(abs(true_rho - rho_hat), 4),
        "delta_true": round(true_delta, 4), "delta_hat": round(dres.delta, 4),
        "delta_abs_err": round(abs(true_delta - dres.delta), 4),
    }


def make_figures(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    xs = [r["corpus_size"] for r in rows]

    # (a) recall per form vs corpus size
    plt.figure(figsize=(7, 4.5))
    for form, mk in zip(FORMS, ["o-", "s-", "^-", "d-"]):
        plt.plot(xs, [r[f"recall_{form}"] for r in rows], mk, label=form)
    plt.ylim(-0.05, 1.05); plt.xlabel("clean corpus size (docs)"); plt.ylabel("recall (held-out TEST)")
    plt.title("Per-form detection recall vs corpus size")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(FIG_DIR / "recall_per_form.png", dpi=130); plt.close()

    # (b) overall precision & recall vs corpus size
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, [r["precision"] for r in rows], "o-", label="precision")
    plt.plot(xs, [r["recall"] for r in rows], "s-", label="recall")
    plt.plot(xs, [r["f1"] for r in rows], "^--", label="F1")
    plt.ylim(-0.05, 1.05); plt.xlabel("clean corpus size (docs)"); plt.ylabel("score")
    plt.title("Overall detection precision / recall / F1 vs corpus size")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(FIG_DIR / "overall_prf.png", dpi=130); plt.close()

    # (c) estimand error vs corpus size
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, [r["rho_abs_err"] for r in rows], "o-", label="|rho_true - rho_hat|")
    plt.plot(xs, [r["delta_abs_err"] for r in rows], "s-", label="|Delta_true - Delta_hat|")
    plt.xlabel("clean corpus size (docs)"); plt.ylabel("absolute error")
    plt.title("Estimand recovery error vs corpus size")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(FIG_DIR / "estimand_error.png", dpi=130); plt.close()

    # (d) calibrated tau vs corpus size
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, [r["tau"] for r in rows], "o-", color="purple")
    plt.xlabel("clean corpus size (docs)"); plt.ylabel("calibrated aggregator tau")
    plt.title("Data-selected operating point (tau) vs corpus size")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(FIG_DIR / "tau_vs_size.png", dpi=130); plt.close()


def main() -> int:
    print("=" * 72)
    print("EvalGuard CORPUS-SIZE SCALING STUDY  (OpenBookQA x AG News)")
    print("=" * 72)
    bench = load_real_benchmark()
    corpus = load_real_corpus()
    train, test = split(bench, SPLIT_SEED)
    print(f"benchmark {len(bench)} items -> train/test {len(train)}/{len(test)} "
          f"| corpus available {len(corpus)} docs | sizes {SIZES}\n")

    rows = [run_one(n, train, test, corpus) for n in SIZES]

    cols = list(rows[0].keys())
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

    hdr = f"{'N':>5} {'tau':>5} {'P':>5} {'R':>5} {'F1':>5} | {'verb':>5} {'para':>5} {'fmt':>5} {'ans':>5} | {'rhoT':>5} {'rhoH':>5} {'dT':>6} {'dH':>6}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['corpus_size']:>5} {r['tau']:>5.2f} {r['precision']:>5.2f} {r['recall']:>5.2f} "
              f"{r['f1']:>5.2f} | {r['recall_verbatim']:>5.2f} {r['recall_paraphrase']:>5.2f} "
              f"{r['recall_format_shift']:>5.2f} {r['recall_answer_only']:>5.2f} | "
              f"{r['rho_true']:>5.2f} {r['rho_hat']:>5.2f} {r['delta_true']:>+6.2f} {r['delta_hat']:>+6.2f}")

    make_figures(rows)
    print(f"\nCSV     -> {CSV_PATH}")
    print(f"figures -> {FIG_DIR}/  (recall_per_form, overall_prf, estimand_error, tau_vs_size).png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
