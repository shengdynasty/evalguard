"""Data-driven calibration of matcher thresholds and the aggregator tau.

Design section 6: "Detector precision/recall/calibration curves per contamination
form - so your ensemble's weights are earned, not guessed."

Given a LABELED set (produced by the injection harness, which knows ground truth
by construction) this module:

  1. Runs each matcher over the (contaminated) corpus once, collecting the raw
     per-item score and the ground-truth label (injected or not).
  2. Sweeps a decision threshold over a grid for each matcher and for the
     aggregator's combined c_i, computing precision / recall / F1 at each point.
  3. SELECTS operating points FROM DATA:
       - the threshold that maximizes F1
       - precision-at-fixed-recall (recall >= a target)
       - recall-at-fixed-precision (precision >= a target)
  4. Persists per-matcher PR/F1 curves as CSV + PNG, and the chosen operating
     points as calibration.json (consumed by evalguard.config).

Everything is deterministic given a seed: the injection seed and matcher seed
fully determine the curves.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .aggregate import DEFAULT_WEIGHTS, _combine, build_matchers
from .benchmark import Benchmark
from .corpus import Corpus
from .inject import InjectionResult

CALIB_DIR = Path(__file__).resolve().parent / "calibration"

# Threshold grid: fine enough to find good operating points, coarse enough to be
# cheap and deterministic.
GRID = [round(i / 100.0, 2) for i in range(0, 101)]

# Targets for the "at-fixed-X" operating points (design section 6 asks for both
# precision-at-fixed-recall and recall-at-fixed-precision).
TARGET_RECALL = 0.90
TARGET_PRECISION = 0.95


@dataclass
class PRPoint:
    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


@dataclass
class MatcherCurve:
    name: str
    points: List[PRPoint] = field(default_factory=list)

    def best_f1(self) -> PRPoint:
        return max(self.points, key=lambda p: (p.f1, p.recall, -p.threshold))

    def precision_at_recall(self, target: float) -> PRPoint | None:
        cands = [p for p in self.points if p.recall >= target]
        return max(cands, key=lambda p: (p.precision, p.threshold)) if cands else None

    def recall_at_precision(self, target: float) -> PRPoint | None:
        cands = [p for p in self.points if p.precision >= target]
        return max(cands, key=lambda p: (p.recall, -p.threshold)) if cands else None


def _prf(scores: Sequence[float], labels: Sequence[int], thr: float) -> PRPoint:
    tp = fp = fn = 0
    for s, y in zip(scores, labels):
        pred = 1 if s >= thr else 0
        if pred and y:
            tp += 1
        elif pred and not y:
            fp += 1
        elif not pred and y:
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return PRPoint(thr, prec, rec, f1, tp, fp, fn)


def _sweep(scores: Sequence[float], labels: Sequence[int], name: str) -> MatcherCurve:
    return MatcherCurve(name, [_prf(scores, labels, t) for t in GRID])


def collect_scores(
    benchmark: Benchmark,
    injection: InjectionResult,
    seed: int = 1,
    weights: Dict[str, float] | None = None,
) -> Tuple[Dict[str, List[float]], List[float], List[int]]:
    """Run matchers once; return per-matcher scores, combined c_i, and labels.

    Returns:
        (matcher_scores {name: [score per item]}, combined [c_i per item],
         labels [1 if item injected else 0]).
    """
    weights = weights if weights is not None else DEFAULT_WEIGHTS
    matchers = build_matchers(seed=seed)
    for m in matchers:
        m.fit(injection.corpus)

    truth = injection.injected_ids
    matcher_scores: Dict[str, List[float]] = {m.name: [] for m in matchers}
    combined: List[float] = []
    labels: List[int] = []
    for item in benchmark.items:
        evs = [m.match(item) for m in matchers]
        for ev in evs:
            matcher_scores[ev.matcher].append(ev.score)
        combined.append(_combine(evs, weights))
        labels.append(1 if item.id in truth else 0)
    return matcher_scores, combined, labels


def calibrate(
    benchmark: Benchmark,
    injection: InjectionResult,
    seed: int = 1,
    out_dir: Path | str = CALIB_DIR,
    write_plots: bool = True,
) -> dict:
    """Full calibration pass. Returns the result dict and writes artifacts.

    Artifacts written to out_dir:
      {matcher}_pr.csv     per-threshold precision/recall/f1 for each matcher
      aggregate_pr.csv     same for the combined c_i (used to pick tau)
      {matcher}.png        PR + F1-vs-threshold plot (if matplotlib available)
      aggregate.png
      calibration.json     chosen operating points (F1-opt, P@R, R@P) + metadata
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matcher_scores, combined, labels = collect_scores(benchmark, injection, seed=seed)

    curves: Dict[str, MatcherCurve] = {}
    for name, scores in matcher_scores.items():
        curves[name] = _sweep(scores, labels, name)
    curves["aggregate"] = _sweep(combined, labels, "aggregate")

    # ---- write CSVs ----
    for name, curve in curves.items():
        csv_path = out_dir / f"{name}_pr.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["threshold", "precision", "recall", "f1", "tp", "fp", "fn"])
            for p in curve.points:
                w.writerow([f"{p.threshold:.2f}", f"{p.precision:.4f}",
                            f"{p.recall:.4f}", f"{p.f1:.4f}", p.tp, p.fp, p.fn])

    # ---- select operating points from data ----
    chosen: Dict[str, float] = {}
    op_detail: Dict[str, dict] = {}
    for name, curve in curves.items():
        bf = curve.best_f1()
        par = curve.precision_at_recall(TARGET_RECALL)
        rap = curve.recall_at_precision(TARGET_PRECISION)
        key = "tau" if name == "aggregate" else name
        chosen[key] = bf.threshold
        op_detail[name] = {
            "f1_optimal": {"threshold": bf.threshold, "precision": round(bf.precision, 4),
                           "recall": round(bf.recall, 4), "f1": round(bf.f1, 4)},
            f"precision_at_recall>={TARGET_RECALL}": (
                {"threshold": par.threshold, "precision": round(par.precision, 4),
                 "recall": round(par.recall, 4)} if par else None),
            f"recall_at_precision>={TARGET_PRECISION}": (
                {"threshold": rap.threshold, "precision": round(rap.precision, 4),
                 "recall": round(rap.recall, 4)} if rap else None),
        }

    result = {
        "seed": seed,
        "n_items": len(labels),
        "n_injected": int(sum(labels)),
        "grid": {"min": GRID[0], "max": GRID[-1], "step": round(GRID[1] - GRID[0], 3)},
        "targets": {"recall": TARGET_RECALL, "precision": TARGET_PRECISION},
        "chosen": chosen,
        "operating_points": op_detail,
    }
    (out_dir / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if write_plots:
        _write_plots(curves, out_dir)

    return result


def _write_plots(curves: Dict[str, MatcherCurve], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting is optional
        return

    for name, curve in curves.items():
        thr = [p.threshold for p in curve.points]
        prec = [p.precision for p in curve.points]
        rec = [p.recall for p in curve.points]
        f1 = [p.f1 for p in curve.points]
        bf = curve.best_f1()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # left: precision-recall curve
        order = sorted(range(len(rec)), key=lambda i: rec[i])
        ax1.plot([rec[i] for i in order], [prec[i] for i in order], "-", color="#2b6cb0")
        ax1.scatter([bf.recall], [bf.precision], color="#e53e3e", zorder=5,
                    label=f"F1-opt (thr={bf.threshold:.2f}, F1={bf.f1:.2f})")
        ax1.set_xlabel("recall"); ax1.set_ylabel("precision")
        ax1.set_xlim(-0.02, 1.02); ax1.set_ylim(-0.02, 1.02)
        ax1.set_title(f"{name}: PR curve"); ax1.legend(loc="lower left", fontsize=8)
        ax1.grid(alpha=0.3)

        # right: P/R/F1 vs threshold
        ax2.plot(thr, prec, label="precision", color="#2b6cb0")
        ax2.plot(thr, rec, label="recall", color="#38a169")
        ax2.plot(thr, f1, label="F1", color="#e53e3e")
        ax2.axvline(bf.threshold, color="#718096", linestyle="--", alpha=0.7)
        ax2.set_xlabel("decision threshold"); ax2.set_ylabel("score")
        ax2.set_xlim(0, 1); ax2.set_ylim(-0.02, 1.02)
        ax2.set_title(f"{name}: metrics vs threshold")
        ax2.legend(loc="lower center", fontsize=8); ax2.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=110)
        plt.close(fig)
