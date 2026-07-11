"""The two headline estimands (design section 4).

  rho   = (1/n) * sum 1[c_i > tau]                      contamination rate
  Delta = Score(M, B) - Score(M, B_clean)               score inflation

Delta is the money number: how many points of the reported score are memorization
rather than capability. B_clean drops the items whose c_i > tau.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .aggregate import ItemEvidence
from .config import calibrated_tau


def rho(evidences: Sequence[ItemEvidence], tau: float | None = None) -> float:
    """Contamination rate: fraction of items with c_i > tau.

    tau defaults to the DATA-CALIBRATED aggregator threshold (see
    evalguard.calibrate / evalguard.config) when not given explicitly.
    """
    if tau is None:
        tau = calibrated_tau()
    if not evidences:
        return 0.0
    hits = sum(1 for e in evidences if e.c > tau)
    return hits / len(evidences)


def contaminated_ids(evidences: Sequence[ItemEvidence], tau: float | None = None) -> List[str]:
    if tau is None:
        tau = calibrated_tau()
    return [e.item_id for e in evidences if e.c > tau]


@dataclass
class DeltaResult:
    delta: float          # Score(B) - Score(B_clean)
    score_full: float     # Score on the full benchmark
    score_clean: float    # Score on the decontaminated subset
    n_full: int
    n_clean: int
    n_dropped: int


def delta(
    evidences: Sequence[ItemEvidence],
    scores: Dict[str, float],
    tau: float | None = None,
) -> DeltaResult:
    """Score inflation from contamination.

    Args:
        evidences: per-item contamination evidence (gives c_i).
        scores: item_id -> per-item score/correctness (e.g. 1.0 correct, 0.0 wrong).
        tau: contamination threshold; items with c_i > tau are dropped for B_clean.

    Returns:
        DeltaResult with delta = mean(scores over B) - mean(scores over B_clean).
    """
    if tau is None:
        tau = calibrated_tau()
    all_scores = []
    clean_scores = []
    dropped = 0
    for e in evidences:
        if e.item_id not in scores:
            continue
        s = scores[e.item_id]
        all_scores.append(s)
        if e.c > tau:
            dropped += 1
        else:
            clean_scores.append(s)

    score_full = sum(all_scores) / len(all_scores) if all_scores else 0.0
    score_clean = sum(clean_scores) / len(clean_scores) if clean_scores else 0.0
    return DeltaResult(
        delta=score_full - score_clean,
        score_full=score_full,
        score_clean=score_clean,
        n_full=len(all_scores),
        n_clean=len(clean_scores),
        n_dropped=dropped,
    )
