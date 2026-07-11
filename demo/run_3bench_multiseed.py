"""3-benchmark multi-seed distinctiveness comparison.

Runs the isolated answer_only precision/F1 study across OpenBookQA, SciQ, and
GSM8K at a fixed corpus size (N=700) over multiple seeds. This is the
authoritative test of whether mean answer-token IDF predicts answer_only
detectability portably across benchmarks.

Usage:  python run_3bench_multiseed.py [n_seeds]   (default 5)
"""
from __future__ import annotations
import math, os, random, re, statistics as st, sys, tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evalguard.aggregate import aggregate
from evalguard.benchmark import Benchmark, BenchmarkItem
from evalguard.calibrate import calibrate
from evalguard.corpus import Corpus
from evalguard.estimands import contaminated_ids
from evalguard.inject import inject

REAL = Path(__file__).resolve().parent / "real_data"
RATE = 0.20; AO = {"answer_only": 1}; N_TERC = 3; SIZE = 700

def _tok(s): return re.findall(r"[a-z0-9]+", s.lower())

def load_bench(name):
    import json
    return Benchmark([BenchmarkItem(id=o["id"], question=o["question"], answer=str(o["answer"]))
        for o in (json.loads(l) for l in (REAL/name).read_text(encoding="utf-8").splitlines() if l.strip())])

def load_corpus():
    import json
    d, i = [], []
    for l in (REAL/"ag_news.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip(): o=json.loads(l); d.append(o["text"]); i.append(o["id"])
    return Corpus(d, i)

def distinct(bench, corpus):
    N=len(corpus); df=Counter()
    for doc in corpus.docs:
        for w in set(_tok(doc)): df[w]+=1
    idf=lambda w: math.log((N+1)/(df.get(w,0)+1))+1
    return {it.id:(sum(idf(w) for w in _tok(it.answer))/max(1,len(_tok(it.answer)))) for it in bench.items}

def terciles(bench, dist):
    o=sorted((it for it in bench.items if dist[it.id]>0), key=lambda it:dist[it.id]); k=len(o)//N_TERC; out=[]
    for t in range(N_TERC):
        lo=t*k; hi=(t+1)*k if t<N_TERC-1 else len(o); items=o[lo:hi]
        out.append((["low","mid","high"][t], Benchmark(items), sum(dist[x.id] for x in items)/len(items)))
    return out

def split(bench, seed):
    idx=list(range(len(bench))); random.Random(seed).shuffle(idx); c=len(idx)//2
    return Benchmark([bench.items[i] for i in sorted(idx[:c])]), Benchmark([bench.items[i] for i in sorted(idx[c:])])

def one(size, tr, te, corpus, seed):
    sub=Corpus(list(corpus.docs[:size]), list(corpus.doc_ids[:size]))
    rtr=inject(sub, tr, rate=RATE, form_mix=AO, seed=seed); rte=inject(sub, te, rate=RATE, form_mix=AO, seed=seed+500)
    with tempfile.TemporaryDirectory() as td:
        cal=calibrate(tr, rtr, seed=1, out_dir=Path(td), write_plots=False)
    tau=float(cal["chosen"]["tau"]); ev=aggregate(te, rte.corpus, seed=1)
    pred=set(contaminated_ids(ev, tau=tau)); truth=rte.injected_ids
    tp=len(pred&truth); fp=len(pred-truth); fn=len(truth-pred)
    P=tp/(tp+fp) if (tp+fp) else 1.0; R=tp/(tp+fn) if (tp+fn) else 1.0
    return P, (2*P*R/(P+R) if (P+R) else 0.0)

def run_bench(name, bench, corpus, nseeds):
    dist=distinct(bench, corpus); bks=terciles(bench, dist)
    results = []
    for bname, bk, md in bks:
        Ps, Fs = [], []
        for s in range(1, nseeds+1):
            tr, te=split(bk, 2024+s*7); P, F=one(SIZE, tr, te, corpus, seed=s); Ps.append(P); Fs.append(F)
        results.append((bname, md, st.mean(Ps), st.pstdev(Ps), st.mean(Fs), st.pstdev(Fs)))
    return results

def main():
    nseeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    benchmarks = [
        ("openbookqa.jsonl", "OpenBookQA", "science MCQ, word answers"),
        ("sciq.jsonl", "SciQ", "science, distinctive term answers"),
        ("gsm8k.jsonl", "GSM8K", "math, numeric answers"),
    ]
    corpus = load_corpus()
    print("=" * 76)
    print(f"3-BENCHMARK MULTI-SEED DISTINCTIVENESS COMPARISON  N={SIZE}  seeds={nseeds}")
    print("=" * 76)

    all_results = {}
    for fname, label, desc in benchmarks:
        bench = load_bench(fname)
        print(f"\n--- {label} ({desc}, {len(bench)} items) ---")
        print(f"{'bucket':>6} {'meanIDF':>7} | {'precision mean±std':>20} | {'F1 mean±std':>16}")
        print("-" * 58)
        results = run_bench(label, bench, corpus, nseeds)
        all_results[label] = results
        for bname, md, pm, ps, fm, fs in results:
            print(f"{bname:>6} {md:>7.2f} | {pm:>8.2f} ± {ps:<7.2f} | {fm:>6.2f} ± {fs:<5.2f}")

    print("\n" + "=" * 76)
    print("CROSS-BENCHMARK SUMMARY")
    print("=" * 76)
    print(f"\n{'Benchmark':<12} | {'low IDF':>10} | {'mid IDF':>10} | {'high IDF':>10} | {'monotone?':>9}")
    print("-" * 62)
    for label in all_results:
        res = all_results[label]
        precs = [r[2] for r in res]
        monotone = all(precs[i] <= precs[i+1] + 0.05 for i in range(len(precs)-1))
        vals = " | ".join(f"{r[2]:>8.2f}±{r[3]:.2f}" for r in res)
        print(f"{label:<12} | {vals} | {'YES' if monotone else 'NO':>9}")

    print("\nConclusion: if the monotone column is inconsistent across benchmarks,")
    print("mean answer-token IDF is NOT a portable cross-benchmark predictor of")
    print("answer_only precision.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
