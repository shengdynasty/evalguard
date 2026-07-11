"""End-to-end: synth -> inject -> save -> CLI audit -> report recovers estimands."""
import json
import random
import subprocess
import sys
from pathlib import Path

from evalguard.benchmark import save_benchmark
from evalguard.corpus import save_corpus
from evalguard.inject import inject
from evalguard.report import build_report, to_text
from evalguard.aggregate import aggregate
from evalguard.estimands import delta
from evalguard.synth import make_benchmark, make_clean_corpus

TAU = 0.5


def test_cli_audit_recovers_rho_and_delta(tmp_path: Path):
    bench = make_benchmark(n=150, seed=7)
    clean = make_clean_corpus(n_docs=400, seed=11)
    res = inject(
        clean, bench, rate=0.2,
        form_mix={"verbatim": 1, "paraphrase": 1, "format_shift": 1, "answer_only": 1},
        seed=42,
    )
    injected = res.injected_ids
    true_rho = res.true_rho(len(bench))

    rng = random.Random(0)
    scores = {
        it.id: 1.0 if rng.random() < (0.95 if it.id in injected else 0.4) else 0.0
        for it in bench.items
    }
    all_s = [scores[i.id] for i in bench.items]
    clean_s = [scores[i.id] for i in bench.items if i.id not in injected]
    true_delta = sum(all_s) / len(all_s) - sum(clean_s) / len(clean_s)

    bench_path = tmp_path / "bench.jsonl"
    corpus_path = tmp_path / "corpus.jsonl"
    scores_path = tmp_path / "scores.json"
    json_out = tmp_path / "report.json"
    save_benchmark(bench, bench_path)
    save_corpus(res.corpus, corpus_path)
    scores_path.write_text(json.dumps(scores))

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable, "-m", "evalguard.cli", "audit",
            "--benchmark", str(bench_path),
            "--corpus", str(corpus_path),
            "--scores", str(scores_path),
            "--tau", str(TAU),
            "--json", str(json_out),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "contamination audit" in proc.stdout

    report = json.loads(json_out.read_text())
    assert abs(report["rho_hat"] - true_rho) <= 0.03
    assert abs(report["delta"]["Delta_hat"] - true_delta) <= 0.03
    # report carries evidence trail
    assert report["top_contaminated"]
    assert report["top_contaminated"][0]["contamination_path"]


def test_cli_ci_gate_fails_over_threshold(tmp_path: Path):
    bench = make_benchmark(n=100, seed=7)
    clean = make_clean_corpus(n_docs=300, seed=11)
    res = inject(clean, bench, rate=0.2, form="verbatim", seed=42)

    bench_path = tmp_path / "b.jsonl"
    corpus_path = tmp_path / "c.jsonl"
    save_benchmark(bench, bench_path)
    save_corpus(res.corpus, corpus_path)

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable, "-m", "evalguard.cli", "audit",
            "--benchmark", str(bench_path),
            "--corpus", str(corpus_path),
            "--fail-over", "0.05",
        ],
        cwd=repo_root, capture_output=True, text=True,
    )
    # rho ~ 0.2 > 0.05 -> CI gate should fail (exit 2)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "CI GATE FAILED" in proc.stderr


def test_report_text_renders():
    bench = make_benchmark(n=60, seed=7)
    clean = make_clean_corpus(n_docs=150, seed=11)
    res = inject(clean, bench, rate=0.3, form="verbatim", seed=1)
    ev = aggregate(bench, res.corpus, seed=1)
    scores = {it.id: 1.0 for it in bench.items}
    dres = delta(ev, scores, tau=TAU)
    text = to_text(build_report(ev, tau=TAU, delta_result=dres))
    assert "rho_hat" in text
    assert "Delta_hat" in text
    assert "contamination path" in text.lower()
