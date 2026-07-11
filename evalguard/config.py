"""Calibrated operating points (design section 6).

The matcher decision thresholds and the aggregator tau are NOT hand-tuned magic
numbers any more: they are SELECTED FROM DATA by `evalguard.calibrate` and
persisted to `evalguard/calibration/calibration.json`. This module loads that
file (if present) so the rest of the package can consume the calibrated values.

If no calibration file exists yet, `DEFAULTS` are used. Those defaults are the
original conservative values; running the calibrator overwrites the JSON and the
loaded operating points then come from data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

CALIB_DIR = Path(__file__).resolve().parent / "calibration"
CALIB_JSON = CALIB_DIR / "calibration.json"

# Fallback operating points used only when no calibration.json is present.
# These are the pre-calibration values; the calibrator replaces them with
# data-selected ones (see calibrate.py).
DEFAULTS: Dict[str, float] = {
    "ngram": 0.5,
    "embedding": 0.5,
    "paraphrase": 0.5,
    "answer": 0.5,
    "tau": 0.5,
}


def load_operating_points() -> Dict[str, float]:
    """Return {matcher_name: decision_threshold, 'tau': tau}.

    Reads the calibrated operating points from calibration.json when available,
    otherwise falls back to DEFAULTS. Never raises on a missing file.
    """
    pts = dict(DEFAULTS)
    if CALIB_JSON.exists():
        try:
            obj = json.loads(CALIB_JSON.read_text(encoding="utf-8"))
            chosen = obj.get("chosen", {})
            for k in ("ngram", "embedding", "paraphrase", "answer", "tau"):
                if k in chosen and chosen[k] is not None:
                    pts[k] = float(chosen[k])
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
    return pts


def calibrated_tau(default: float = 0.5) -> float:
    """The data-selected aggregator threshold tau, or `default` if uncalibrated."""
    return load_operating_points().get("tau", default)


# --- matcher score-shaping constants -------------------------------------
# These rescale each matcher's raw signal into a [0,1] evidence score. They are
# NOT the decision threshold (that is the data-calibrated tau / per-matcher
# operating point). They were previously hardcoded magic numbers inside each
# matcher (marked "# SWAP:"); they now live here so calibration can persist
# tuned values and every matcher reads one source of truth.
SHAPING_DEFAULTS: Dict[str, float] = {
    "embedding_sim_floor": 0.82,   # cosine at/below -> ~0 evidence
    "paraphrase_floor": 0.6,       # IDF-content match below template level -> 0
    "answer_floor": 0.6,           # answer-token IDF match below this -> 0
}


def load_shaping() -> Dict[str, float]:
    """Return matcher score-shaping constants, calibrated values overriding
    the defaults when present in calibration.json under 'shaping'."""
    vals = dict(SHAPING_DEFAULTS)
    if CALIB_JSON.exists():
        try:
            obj = json.loads(CALIB_JSON.read_text(encoding="utf-8"))
            for k, v in obj.get("shaping", {}).items():
                if k in vals and v is not None:
                    vals[k] = float(v)
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
    return vals
