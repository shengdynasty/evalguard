"""Distinctiveness study on GSM8K (numeric answers) x AG News.

GSM8K answers are short numbers ("72", "10", "5") — maximally low-distinctiveness.
This is the 3rd benchmark pair to re-test whether mean answer-token IDF is a
portable cross-benchmark predictor of answer_only detectability.
"""
from __future__ import annotations
import csv, math, os, random, re, sys, tempfile
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
CSV_PATH = HERE / "distinctiveness_gsm8k_results.csv"
RATE = 0.20; SPLIT_SEED = 2024; ANSWER_ONLY = {"answer_only": 1}
SIZES = [40, 80, 150, 250, 400, 550, 700]; RECALL_FLOOR = 0.5; N_TERCILES = 3


def _toks(s): return re.findall(r"[a-z0-9]+", s.lower())

def load_jsonl_bench(name):
    import json
    return Benchmark([BenchmarkItem(id=o["id"], question=o["question"], answer=str(o["answer"]))
                      for o in (json.loads(l) for l in (REAL_DIR/name).read_text(encoding="utf-8").splitlines() if l.strip())])

def load_corpus():
    import json
    docs, ids = [], []
    for l in (REAL_DIR/"ag_news.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            o = json.loads(l); docs.append(o["text"]); ids.append(o["id"])
    return Corpus(docs, ids)

def answer_distinctiveness(bench, corpus):
    N = len(corpus); df = Counter()
    for d in corpus.docs:
        for w in set(_toks(d)): df[w] += 1
    idf = lambda w: math.log((N+1)/(df.get(w,0)+1)) + 1.0
    out = {}
    for it in bench.items:
        at = _toks(it.answer); out[it.id] = (sum(idf(w) for w in at)/len(at)) if at else 0.0
    return out

def terciles(bench, dist):
    ordered = sorted((it for it in bench.items if dist[it.id] > 0), key=lambda it: dist[it.id])
    k = len(ordered)//N_TERCILES; buckets = []
    for t in range(N_TERCILES):
        lo = t*k; hi = (t+1)*k if t < N_TERCILES-1 else len(ordered)
        items = ordered[lo:hi]; mean_d = sum(dist[it.id] for it in items)/len(items)
        buckets.append((["low","mid","high"][t], Benchmark(items), mean_d))
    return buckets

def split(bench, seed):
    idx = list(range(len(bench))); random.Random(seed).shuffle(idx); cut = len(idx)//2
    return (Benchmark([bench.items[i] for i in sorted(idx[:cut])]),
            Benchmark([bench.items[i] for i in sorted(idx[cut:])]))

def metrics(size, train, test, corpus):
    sub = Corpus(list(corpus.docs[:size]), list(corpus.doc_ids[:size]))
    res_tr = inject(sub, train, rate=RATE, form_mix=ANSWER_ONLY, seed=1)
    res_te = inject(sub, test, rate=RATE, form_mix=ANSWER_ONLY, seed=2)
    with tempfile.TemporaryDirectory() as td:
        cal = calibrate(train, res_tr, seed=1, out_dir=Path(td), write_plots=False)
    tau = float(cal["chosen"]["tau"]); ev = aggregate(test, res_te.corpus, seed=1)
    pred = set(contaminated_ids(ev, tau=tau)); truth = res_te.injected_ids
    tp = len(pred&truth); fp = len(pred-truth); fn = len(truth-pred)
    P = tp/(tp+fp) if (tp+fp) else 1.0; R = tp/(tp+fn) if (tp+fn) else 1.0
    return P, R, (2*P*R/(P+R) if (P+R) else 0.0)

def critical_size(f1s):
    for n, f in f1s:
        if f < RECALL_FLOOR: return n
    return None

def make_figure(prec, f1, means):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    styles = {"low":"o-","mid":"s-","high":"^-"}
    fig, (aP, aF) = plt.subplots(1, 2, figsize=(12, 4.8))
    for name in prec:
        lbl = f"{name} (mean IDF {means[name]:.2f})"; xs = [n for n,_ in prec[name]]
        aP.plot(xs, [v for _,v in prec[name]], styles[name], label=lbl)
        aF.plot(xs, [v for _,v in f1[name]], styles[name], label=lbl)
    for ax, ttl, yl in ((aP,"answer_only PRECISION vs corpus size","precision"),(aF,"answer_only F1 vs corpus size","F1")):
        ax.axhline(RECALL_FLOOR, color="gray", ls=":", lw=1); ax.set_ylim(-0.05,1.05)
        ax.set_xlabel("clean corpus size (docs)"); ax.set_ylabel(f"{yl} (TEST)"); ax.set_title(ttl)
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("GSM8K x AG News: answer-only detectability by distinctiveness", y=1.02)
    fig.tight_layout(); fig.savefig(FIG_DIR/"distinctiveness_gsm8k.png", dpi=130, bbox_inches="tight"); plt.close(fig)

def main():
    print("="*72); print("DISTINCTIVENESS STUDY  GSM8K (numeric answers) x AG News"); print("="*72)
    bench = load_jsonl_bench("gsm8k.jsonl"); corpus = load_corpus()
    dist = answer_distinctiveness(bench, corpus); buckets = terciles(bench, dist)
    print(f"benchmark {len(bench)} GSM8K items | corpus {len(corpus)} | terciles {[(b[0],len(b[1]),round(b[2],2)) for b in buckets]}\n")
    prec, f1, means, rows = {}, {}, {}, []
    hdr = f"{'bucket':>6} {'meanIDF':>7} | " + " ".join(f"N={n:>3}" for n in SIZES) + f" | {'N*(F1)':>7}  (P/F1)"
    print(hdr); print("-"*len(hdr))
    for name, bk, md in buckets:
        tr, te = split(bk, SPLIT_SEED); ps, fs = [], []
        for n in SIZES:
            P, R, F = metrics(n, tr, te, corpus); ps.append((n, round(P,3))); fs.append((n, round(F,3)))
            rows.append({"bucket":name,"mean_idf":round(md,3),"corpus_size":n,"precision":round(P,3),"recall":round(R,3),"f1":round(F,3)})
        prec[name], f1[name], means[name] = ps, fs, md
        ns = critical_size(fs)
        print(f"{name:>6} {md:>7.2f} | " + " ".join(f"{p:>4.2f}/{f:>4.2f}" for (_,p),(_,f) in zip(ps,fs)) + f" | {('>700' if ns is None else ns):>7}")
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    make_figure(prec, f1, means)
    print(f"\nCSV -> {CSV_PATH}\nfigure -> {FIG_DIR}/distinctiveness_gsm8k.png")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
