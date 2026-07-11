"""EvalGuard - LLM benchmark-contamination auditor (open-data regime v1).

This package implements the OPEN-DATA regime from the EvalGuard design doc:
you have the training corpus D, so per-item membership is a search problem
and detection is reliable. It reports per-item contamination evidence c_i
WITH an evidence trail (which matcher fired, what it matched), and derives the
two headline estimands:

  rho   = fraction of the benchmark that is contaminated  (c_i > tau)
  Delta = Score(B) - Score(B_clean), the score inflation from contamination

Gray-box / black-box model-probability methods are intentionally NOT built here.
"""

__version__ = "0.1.0"

from .corpus import Corpus, load_corpus
from .benchmark import Benchmark, BenchmarkItem, load_benchmark
from .aggregate import aggregate, ItemEvidence
from .estimands import rho, delta

__all__ = [
    "Corpus",
    "load_corpus",
    "Benchmark",
    "BenchmarkItem",
    "load_benchmark",
    "aggregate",
    "ItemEvidence",
    "rho",
    "delta",
    "__version__",
]
