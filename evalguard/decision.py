"""Decision layer: turn per-item evidence into contamination flags.

Two modes are supported:

  * ``global-tau`` (default, conservative): flag an item if the *combined* c_i
    exceeds one global threshold tau. Precision-first ("never falsely accuse")
    and recovers rho/Delta well, but a single tau tuned for the high-signal
    forms (verbatim / paraphrase / format_shift) can silently discard a weak
    form -- the ``answer_only`` collapse documented in FINDINGS section 7.

  * ``per-form`` (opt-in, coverage-first): flag an item if ANY matcher's own
    evidence exceeds *that matcher's* data-calibrated threshold (an OR-gate).
    Because each matcher specializes in a contamination form, per-matcher
    thresholding *is* per-form thresholding -- so a weak form keeps its own low
    operating point instead of being drowned by a global tau. It recovers weak
    forms at a precision cost, and is sensitive to inter-item similarity, so it
    should be calibrated on data representative of the corpus being audited.

The per-form thresholds come from the same data-driven calibration that already
selects a per-matcher F1-optimal threshold for every matcher
(``calibration.json`` -> ``chosen``); this layer simply *uses* them for the
decision instead of throwing them away.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from .aggregate import ItemEvidence
from .config import load_operating_points
from .estimands import DeltaResult

MATCHER_KEYS = ("ngram", "embedding", "paraphrase", "answer")

# A matcher only "fires" on meaningful evidence. This floor prevents two
# degeneracies of the OR-gate: (1) a calibrated threshold of exactly 0.0 would
# otherwise flag every item (score >= 0 is always true), and (2) tiny background
# similarity signals stacking across matchers into false positives. Evidence
# below the floor is treated as noise regardless of the calibrated threshold.
MIN_EVIDENCE = 0.05


def matcher_thresholds(thresholds: Dict[str, float] | None = None) -> Dict[str, float]:
    """Resolve per-matcher decision thresholds.

    If ``thresholds`` is given (e.g. the ``chosen`` dict returned by
    ``calibrate``), its per-matcher entries are used; otherwise the calibrated
    operating points persisted in calibration.json are loaded.
    """
    src = thresholds if thresholds is not None else load_operating_points()
    return {k: float(src[k]) for k in MATCHER_KEYS if k in src and src[k] is not None}


def item_contaminated(ev: ItemEvidence, thresholds: Dict[str, float] | None = None) -> bool:
    """Per-form OR-gate: True if any matcher fired above its own threshold."""
    th = matcher_thresholds(thresholds)
    for e in ev.evidence:
        t = th.get(e.matcher)
        if t is not None and e.score >= max(t, MIN_EVIDENCE):
            return True
    return False


def firing_matchers(ev: ItemEvidence, thresholds: Dict[str, float] | None = None) -> List[str]:
    """Which matchers fired above threshold, strongest first (evidence trail)."""
    th = matcher_thresholds(thresholds)
    fired = [(e.matcher, e.score) for e in ev.evidence
             if th.get(e.matcher) is not None and e.score >= max(th[e.matcher], MIN_EVIDENCE)]
    fired.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in fired]


def contaminated_ids(evidences: Sequence[ItemEvidence],
                     thresholds: Dict[str, float] | None = None) -> List[str]:
    return [ev.item_id for ev in evidences if item_contaminated(ev, thresholds)]


def rho(evidences: Sequence[ItemEvidence],
        thresholds: Dict[str, float] | None = None) -> float:
    if not evidences:
        return 0.0
    return sum(1 for ev in evidences if item_contaminated(ev, thresholds)) / len(evidences)


def delta(evidences: Sequence[ItemEvidence],
          scores: Dict[str, float],
          thresholds: Dict[str, float] | None = None) -> DeltaResult:
    """Score inflation under the per-form decision (drop items any matcher flags)."""
    all_scores: List[float] = []
    clean_scores: List[float] = []
    dropped = 0
    for ev in evidences:
        if ev.item_id not in scores:
            continue
        s = scores[ev.item_id]
        all_scores.append(s)
        if item_contaminated(ev, thresholds):
            dropped += 1
        else:
            clean_scores.append(s)
    score_full = sum(all_scores) / len(all_scores) if all_scores else 0.0
    score_clean = sum(clean_scores) / len(clean_scores) if clean_scores else 0.0
    return DeltaResult(delta=score_full - score_clean, score_full=score_full,
                       score_clean=score_clean, n_full=len(all_scores),
                       n_clean=len(clean_scores), n_dropped=dropped)
