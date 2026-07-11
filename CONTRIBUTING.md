# Contributing to EvalGuard

Thanks for your interest! EvalGuard is an open-data LLM benchmark-contamination
auditor. Contributions that add rigor (new contamination forms, real matchers,
more benchmark×corpus pairs, gray-box methods) are especially welcome.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test,plot]"
pytest -q          # 37 tests, ~30s
```

## Ground rules

- **Honesty over headline numbers.** Never let the tool claim more than the
  evidence supports (see the "claim strength" section of `EvalGuard_REPORT.md`).
  Detectors should fail *conservatively* and be explicit about blind spots.
- **Determinism.** Everything is seedable; new studies/tests must be reproducible.
- **Keep the evidence trail.** Any new matcher must return a human-readable
  contamination path, not just a score.
- **Tests required.** New behaviour ships with tests; keep the suite green.

## Good first issues

- Wire a real sentence-embedding backend behind `matchers/embedding.py`
  (`# SWAP:` point) and re-measure.
- Add a `translation` contamination form to the injection harness.
- Add a second benchmark×corpus pair and reproduce the distinctiveness study.

## PRs

Small, focused PRs with a clear description and passing tests. By contributing
you agree your work is licensed under the MIT License.
