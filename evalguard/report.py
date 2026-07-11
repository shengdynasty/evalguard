"""Reporting (design section 5.4): JSON + human-readable report.

Reports rho, Delta, and the ranked contaminated items each WITH their
contamination path (the evidence trail). Honesty principle (section 4.1): this
is open-data per-item membership *evidence*; the report frames it as evidence,
not as a certified accusation.

Decision modes (see evalguard.decision):
  * ``per-form`` (default, recommended): an item is contaminated if ANY matcher
    exceeds its own data-calibrated threshold. Recovers weak forms that a single
    global threshold discards (FINDINGS section 7).
  * ``global-tau``: an item is contaminated if the combined c_i exceeds one tau.
"""
from __future__ import annotations

import json
from typing import Dict, Sequence

from .aggregate import ItemEvidence
from .config import calibrated_tau
from .estimands import DeltaResult, contaminated_ids as _contam_global, rho as _rho_global
from . import decision as _dec


def build_report(
    evidences: Sequence[ItemEvidence],
    mode: str = "global-tau",
    tau: float | None = None,
    thresholds: Dict[str, float] | None = None,
    delta_result: DeltaResult | None = None,
    top_k: int = 20,
) -> dict:
    ranked = sorted(evidences, key=lambda e: e.c, reverse=True)

    if mode == "per-form":
        thr = _dec.matcher_thresholds(thresholds)
        flagged = set(_dec.contaminated_ids(evidences, thresholds))
        rho_hat = _dec.rho(evidences, thresholds)
        decision_block = {"mode": "per-form",
                          "rule": "flag if any matcher exceeds its own calibrated threshold",
                          "matcher_thresholds": {k: round(v, 3) for k, v in thr.items()}}
    else:
        if tau is None:
            tau = calibrated_tau()
        flagged = set(_contam_global(evidences, tau))
        rho_hat = _rho_global(evidences, tau)
        decision_block = {"mode": "global-tau", "rule": "flag if combined c_i > tau", "tau": tau}

    contam = [e for e in ranked if e.item_id in flagged]
    items_out = []
    for e in contam[:top_k]:
        firing = _dec.firing_matchers(e, thresholds) if mode == "per-form" else None
        items_out.append({
            "item_id": e.item_id,
            "c": round(e.c, 4),
            "firing_matchers": firing,
            "contamination_path": e.path(),
        })

    report = {
        "regime": "open-data",
        "claim_level": "per-item membership evidence (not certified proof)",
        "decision": decision_block,
        "n_items": len(evidences),
        "rho_hat": round(rho_hat, 4),
        "n_contaminated": len(contam),
        "top_contaminated": items_out,
    }
    if delta_result is not None:
        report["delta"] = {
            "Delta_hat": round(delta_result.delta, 4),
            "score_full": round(delta_result.score_full, 4),
            "score_clean": round(delta_result.score_clean, 4),
            "n_full": delta_result.n_full,
            "n_clean": delta_result.n_clean,
            "n_dropped": delta_result.n_dropped,
        }
    return report


def to_json(report: dict) -> str:
    return json.dumps(report, indent=2)


def to_text(report: dict) -> str:
    d = report["decision"]
    lines = []
    lines.append("=" * 68)
    lines.append("EvalGuard contamination audit  (regime: open-data)")
    lines.append("=" * 68)
    lines.append(f"claim level : {report['claim_level']}")
    if d["mode"] == "per-form":
        thr = ", ".join(f"{k}={v}" for k, v in d["matcher_thresholds"].items())
        lines.append(f"decision    : per-form OR-gate  [{thr}]")
    else:
        lines.append(f"decision    : global tau = {d['tau']}")
    lines.append(f"benchmark   : {report['n_items']} items")
    lines.append("")
    lines.append(
        f"rho_hat (contamination rate) : {report['rho_hat']:.4f}   "
        f"({report['n_contaminated']} / {report['n_items']} items compromised)"
    )
    if "delta" in report:
        dd = report["delta"]
        lines.append(
            f"Delta_hat (score inflation)  : {dd['Delta_hat']:+.4f}   "
            f"(full {dd['score_full']:.4f} -> clean {dd['score_clean']:.4f}, "
            f"dropped {dd['n_dropped']})"
        )
    lines.append("")
    lines.append("Top contaminated items (with contamination path):")
    lines.append("-" * 68)
    if not report["top_contaminated"]:
        lines.append("  (none above threshold)")
    for it in report["top_contaminated"]:
        fired = f"  fired: {', '.join(it['firing_matchers'])}" if it.get("firing_matchers") else ""
        lines.append(f"  {it['item_id']}   c = {it['c']:.3f}{fired}")
        for step in it["contamination_path"]:
            lines.append(f"      - {step}")
    lines.append("=" * 68)
    return "\n".join(lines)
