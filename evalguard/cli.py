"""EvalGuard CLI (design section 5.4).

    evalguard audit --benchmark B.jsonl --corpus D.jsonl [--scores S.json] \
                    [--decision per-form|global-tau] [--tau 0.5] \
                    [--json out.json] [--fail-over RATE]

Runs locally; the corpus never leaves the machine. In CI mode (--fail-over),
exits non-zero when rho_hat exceeds the given threshold.

Decision mode defaults to ``per-form`` (recommended): an item is flagged if any
matcher exceeds its own data-calibrated threshold, which recovers weak
contamination forms that a single global tau discards (FINDINGS section 7).
Use ``--decision global-tau`` for the legacy single-threshold behaviour.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import decision as dec
from .aggregate import aggregate
from .benchmark import load_benchmark
from .corpus import load_corpus
from .estimands import delta as delta_global
from .report import build_report, to_json, to_text


def _load_scores(path: str) -> dict:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in obj.items()}


def cmd_audit(args: argparse.Namespace) -> int:
    benchmark = load_benchmark(args.benchmark)
    corpus = load_corpus(args.corpus)

    evidences = aggregate(benchmark, corpus, seed=args.seed)

    delta_result = None
    if args.scores:
        scores = _load_scores(args.scores)
        if args.decision == "per-form":
            delta_result = dec.delta(evidences, scores)
        else:
            delta_result = delta_global(evidences, scores, tau=args.tau)

    report = build_report(
        evidences, mode=args.decision, tau=args.tau,
        delta_result=delta_result, top_k=args.top_k,
    )

    print(to_text(report))
    if args.json:
        Path(args.json).write_text(to_json(report), encoding="utf-8")
        print(f"\n[wrote JSON report to {args.json}]")

    if args.fail_over is not None and report["rho_hat"] > args.fail_over:
        print(
            f"\nCI GATE FAILED: rho_hat {report['rho_hat']:.4f} "
            f"> threshold {args.fail_over:.4f}",
            file=sys.stderr,
        )
        return 2
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Calibrate matcher thresholds + tau from a labeled set built by injecting
    a benchmark into a corpus at a known rate/forms (design section 6)."""
    from .calibrate import calibrate, CALIB_DIR
    from .inject import inject

    benchmark = load_benchmark(args.benchmark)
    corpus = load_corpus(args.corpus)
    form_mix = {"verbatim": 1, "paraphrase": 1, "format_shift": 1, "answer_only": 1}
    injection = inject(corpus, benchmark, rate=args.rate, form_mix=form_mix, seed=args.seed)
    out_dir = args.out_dir or CALIB_DIR
    result = calibrate(benchmark, injection, seed=args.seed, out_dir=out_dir,
                       write_plots=not args.no_plots)
    print("Calibrated operating points (from data):")
    for k, v in result["chosen"].items():
        print(f"  {k:11s} threshold = {v:.2f}")
    print(f"\n[curves + PNGs + calibration.json written to {out_dir}]")
    print("Per-form decision uses the per-matcher thresholds above (not tau).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evalguard", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="audit a benchmark against a training corpus")
    a.add_argument("--benchmark", required=True, help="benchmark .jsonl {id,question,answer}")
    a.add_argument("--corpus", required=True, help="training corpus .jsonl {id,text} or .txt")
    a.add_argument("--scores", help="JSON map item_id -> per-item score/correctness")
    a.add_argument("--decision", choices=["global-tau", "per-form"], default="global-tau",
                   help="decision mode (default global-tau, conservative; per-form is coverage-first)")
    a.add_argument("--tau", type=float, default=0.5,
                   help="global-tau threshold (only used with --decision global-tau)")
    a.add_argument("--json", help="path to write the JSON report")
    a.add_argument("--top-k", type=int, default=20, help="how many contaminated items to show")
    a.add_argument("--seed", type=int, default=1, help="matcher RNG seed")
    a.add_argument("--fail-over", type=float, default=None, metavar="RATE",
                   help="CI mode: exit non-zero if rho_hat exceeds RATE")
    a.set_defaults(func=cmd_audit)

    c = sub.add_parser("calibrate", help="calibrate matcher thresholds + tau from data")
    c.add_argument("--benchmark", required=True, help="benchmark .jsonl {id,question,answer}")
    c.add_argument("--corpus", required=True, help="clean corpus .jsonl {id,text} or .txt")
    c.add_argument("--rate", type=float, default=0.2, help="injection rate for the labeled set")
    c.add_argument("--seed", type=int, default=1, help="injection + matcher seed")
    c.add_argument("--out-dir", default=None, help="where to write curves (default evalguard/calibration)")
    c.add_argument("--no-plots", action="store_true", help="skip PNG plots")
    c.set_defaults(func=cmd_calibrate)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
