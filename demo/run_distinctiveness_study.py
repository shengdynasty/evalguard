"""Is the answer-only phase transition PREDICTED by answer distinctiveness?

The scaling study showed `answer_only` contamination collapsing past a critical
corpus size WHEN one global tau is calibrated across a mix of forms. This study
isolates the cause: inject ONLY the `answer_only` form, calibrate tau on that
signal alone, and vary answer distinctiveness (mean IDF of answer tokens vs the
corpus) across terciles of OpenBookQA. Everything else is fixed (one benchmark,
one AG News corpus, one harness).

For each tercile and corpus size we report answer_only precision / recall / F1
on a held-out split, and the critical size N* (first size where F1 < floor).
"""
from __future__ import annotations

import csv
import math
import os
import random
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalguard.aggregate import aggregate
from evalguard.benchmark import Benchmark, BenchmarkItem
from evalguard.calibrate import calibrate
from evalguard.corpus import Corpus
from evalguard.estimands import contaminated_ids
from evalguard.inject import inject

HERE = Path(__file__).resolve().parent
REAL_DIR = HERE / "real_data"
FIG_DIR = HERE / "figures"
CSV_PATH = HERE / "distinctiveness_results.csv"

RATE = 0.20
SPLIT_SEED = 2024
INJECT_SEED_TRAIN = 1
INJECT_SEED_TEST = 2
ANSWER_ONLY = {"answer_only": 1}
SIZES = [40, 80, 150, 250, 400, 550, 700]
RECALL_FLOOR = 0.5
N_TERCILES = 3


def _toks(s: str):
    return re.findall(r"[a-z0-9]+", s.lower())


def load_bench():
    import json
    items = []
    for line in (REAL_DIR / "openbookqa.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        items.append(BenchmarkItem(id=o["id"], question=o["question"], answer=str(o["answer"])))
    return Benchmark(items)


def load_corpus():
    import json
    docs, ids = [], []
    for line in (REAL_DIR / "ag_news.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        docs.append(o["text"]); ids.append(o["id"])
    return Corpus(docs, ids)


def answer_distinctiveness(bench, corpus):
    N = len(corpus)
    df = Counter()
    for d in corpus.docs:
        for w in set(_toks(d)):
            df[w] += 1
    def idf(w):
        return math.log((N + 1) / (df.get(w, 0) + 1)) + 1.0
    out = {}
    for it in bench.items:
        at = _toks(it.answer)
        out[it.id] = (sum(idf(w) for w in at) / len(at)) if at else 0.0
    return out


def terciles(bench, dist):
    ordered = sorted((it for it in bench.items if dist[it.id] > 0), key=lambda it: dist[it.id])
    k = len(ordered) // N_TERCILES
    buckets = []
    for t in range(N_TERCILES):
        lo = t * k
        hi = (t + 1) * k if t < N_TERCILES - 1 else len(ordered)
        items = ordered[lo:hi]
        mean_d = sum(dist[it.id] for it in items) / len(items)
        buckets.append((["low", "mid", "high"][t], Benchmark(items), mean_d))
    return buckets


def split(bench, seed):
    idx = list(range(len(bench)))
    random.Random(seed).shuffle(idx)
    cut = len(idx) // 2
    tr = Benchmark([bench.items[i] for i in sorted(idx[:cut])])
    te = Benchmark([bench.items[i] for i in sorted(idx[cut:])])
    return tr, te


def answer_only_metrics(size, train, test, full_corpus):
    sub = Corpus(list(full_corpus.docs[:size]), list(full_corpus.doc_ids[:size]))
    res_tr = inject(sub, train, rate=RATE, form_mix=ANSWER_ONLY, seed=INJECT_SEED_TRAIN)
    res_te = inject(sub, test, rate=RATE, form_mix=ANSWER_ONLY, seed=INJECT_SEED_TEST)
    with tempfile.TemporaryDirectory() as td:
        cal = calibrate(train, res_tr, seed=1, out_dir=Path(td), write_plots=False)
    tau = float(cal["chosen"]["tau"])
    ev = aggregate(test, res_te.corpus, seed=1)
    pred = set(contaminated_ids(ev, tau=tau))
    truth = res_te.injected_ids
    tp = len(pred & truth); fp = len(pred - truth); fn = len(truth - pred)
    P = tp / (tp + fp) if (tp + fp) else 1.0
    R = tp / (tp + fn) if (tp + fn) else 1.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F, tau


def critical_size(f1s):
    for n, f in f1s:
        if f < RECALL_FLOOR:
            return n
    return None


def make_figure(prec_series, f1_series, means):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    styles = {"low": "o-", "mid": "s-", "high": "^-"}
    fig, (axP, axF) = plt.subplots(1, 2, figsize=(12, 4.8))
    for name in prec_series:
        lbl = f"{name} (mean IDF {means[name]:.2f})"
        xs = [n for n, _ in prec_series[name]]
        axP.plot(xs, [v for _, v in prec_series[name]], styles.get(name, "o-"), label=lbl)
        axF.plot(xs, [v for _, v in f1_series[name]], styles.get(name, "o-"), label=lbl)
    for ax, ttl, yl in ((axP, "answer_only PRECISION vs corpus size", "precision"),
                        (axF, "answer_only F1 vs corpus size", "F1")):
        ax.axhline(RECALL_FLOOR, color="gray", ls=":", lw=1)
        ax.set_ylim(-0.05, 1.05); ax.set_xlabel("clean corpus size (docs)")
        ax.set_ylabel(f"{yl} (held-out TEST)"); ax.set_title(ttl)
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Answer-only detectability by answer distinctiveness (single-form calibration)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "distinctiveness_transition.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 72)
    print("EvalGuard DISTINCTIVENESS STUDY  (does distinctiveness predict N*?)")
    print("=" * 72)
    bench = load_bench(); corpus = load_corpus()
    dist = answer_distinctiveness(bench, corpus)
    buckets = terciles(bench, dist)
    print(f"benchmark {len(bench)} items | corpus {len(corpus)} docs | "
          f"terciles {[(b[0], len(b[1]), round(b[2], 2)) for b in buckets]}\n")

    prec_series = {}; f1_series = {}; means = {}; rows = []
    hdr = f"{'bucket':>6} {'meanIDF':>7} | " + " ".join(f"N={n:>3}" for n in SIZES) + f" | {'N*(F1)':>7}   (P/F1 per N)"
    print(hdr); print("-" * len(hdr))
    for name, bk, mean_d in buckets:
        tr, te = split(bk, SPLIT_SEED)
        precs, f1s = [], []
        for n in SIZES:
            P, R, F, tau = answer_only_metrics(n, tr, te, corpus)
            precs.append((n, round(P, 3))); f1s.append((n, round(F, 3)))
            rows.append({"bucket": name, "mean_idf": round(mean_d, 3), "corpus_size": n,
                         "precision": round(P, 3), "recall": round(R, 3),
                         "f1": round(F, 3), "tau": round(tau, 3)})
        prec_series[name] = precs; f1_series[name] = f1s; means[name] = mean_d
        nstar = critical_size(f1s)
        print(f"{name:>6} {mean_d:>7.2f} | " +
              " ".join(f"{p:>4.2f}/{f:>4.2f}" for (_, p), (_, f) in zip(precs, f1s)) +
              f" | {('>700' if nstar is None else nstar):>7}")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    make_figure(prec_series, f1_series, means)

    print(f"\nCritical corpus size N* (first size where answer_only F1 < {RECALL_FLOOR}) vs distinctiveness:")
    for name, _, mean_d in buckets:
        ns = critical_size(f1_series[name])
        print(f"  {name:>6}: mean IDF {mean_d:.2f} -> N* {'>700 (censored)' if ns is None else ns}")
    print(f"\nCSV     -> {CSV_PATH}")
    print(f"figure  -> {FIG_DIR}/distinctiveness_transition.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
