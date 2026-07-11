"""Multi-seed robustness of the per-form fix (hardens FINDINGS section 8).

At fixed corpus sizes, re-run the mixed-form audit over many injection seeds and
report mean +/- std of answer_only recall and overall F1 under both decision
modes. Confirms the per-form recovery is not a single-seed artifact.
"""
from __future__ import annotations

import os
import random
import statistics as st
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalguard import decision
from evalguard.aggregate import aggregate
from evalguard.benchmark import Benchmark, BenchmarkItem
from evalguard.calibrate import calibrate
from evalguard.corpus import Corpus
from evalguard.estimands import contaminated_ids
from evalguard.inject import inject

REAL_DIR = Path(__file__).resolve().parent / "real_data"
FORM_MIX = {"verbatim": 1, "paraphrase": 1, "format_shift": 1, "answer_only": 1}
SIZES = [300, 500]
SEEDS = list(range(1, 11))  # 10 injection seeds
RATE = 0.20


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
    return (2 * P * R / (P + R)) if (P + R) else 0.0


def main():
    print("=" * 70)
    print("EvalGuard MULTI-SEED ROBUSTNESS  (per-form vs global-tau, 10 seeds)")
    print("=" * 70)
    bench = load_bench(); corpus = load_corpus(); train, test = split(bench, 2024)
    print(f"{'N':>4} {'mode':>10} | {'answer_only recall':>22} | {'overall F1':>16}")
    print(f"{'':>4} {'':>10} | {'mean +/- std':>22} | {'mean +/- std':>16}")
    print("-" * 62)
    for n in SIZES:
        acc = {"global": {"ao": [], "f1": []}, "perform": {"ao": [], "f1": []}}
        for s in SEEDS:
            sub = Corpus(list(corpus.docs[:n]), list(corpus.doc_ids[:n]))
            res_tr = inject(sub, train, rate=RATE, form_mix=FORM_MIX, seed=s)
            res_te = inject(sub, test, rate=RATE, form_mix=FORM_MIX, seed=s + 100)
            with tempfile.TemporaryDirectory() as td:
                cal = calibrate(train, res_tr, seed=1, out_dir=Path(td), write_plots=False)
            chosen = cal["chosen"]; tau = float(chosen["tau"])
            ev = aggregate(test, res_te.corpus, seed=1)
            truth = res_te.injected_ids
            ao = set(res_te.ids_by_form().get("answer_only", []))
            g = set(contaminated_ids(ev, tau=tau)); p = set(decision.contaminated_ids(ev, chosen))
            acc["global"]["ao"].append(len(g & ao) / len(ao) if ao else float("nan"))
            acc["perform"]["ao"].append(len(p & ao) / len(ao) if ao else float("nan"))
            acc["global"]["f1"].append(prf(g, truth))
            acc["perform"]["f1"].append(prf(p, truth))
        for mode in ("global", "perform"):
            ao = acc[mode]["ao"]; f1 = acc[mode]["f1"]
            print(f"{n:>4} {mode:>10} | {st.mean(ao):>10.2f} +/- {st.pstdev(ao):<7.2f} | "
                  f"{st.mean(f1):>6.2f} +/- {st.pstdev(f1):<5.2f}")
        print("-" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
