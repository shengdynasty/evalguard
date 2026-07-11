"""Deterministic synthetic data generator for the demo and tests.

Produces a clean benchmark B (~N items of factual Q/A) and a clean training
corpus D of distractor documents. Kept in the package so demo and tests share
one ground-truth source.

Design note: each benchmark item is given LEXICALLY DISTINCTIVE content (unique
invented entity names + a unique numeric answer), the way a real benchmark item
has its own subject matter. This matters: if every item were a near-identical
template ("capital of country {n}") then a clean item and a *different*
injected item would look like near-duplicates and contamination detection would
be ill-posed. Real benchmarks are not degenerate this way, and neither is this
generator.
"""
from __future__ import annotations

import random
from typing import List

from .benchmark import Benchmark, BenchmarkItem
from .corpus import Corpus

# Syllable pools to mint unique-looking proper nouns per item.
_A = ["Zor", "Tal", "Ven", "Kir", "Bral", "Nem", "Quil", "Fen", "Dris", "Ulm",
      "Pax", "Grud", "Mirr", "Osh", "Lann", "Vyr", "Cael", "Thex", "Yun", "Rho"]
_B = ["andor", "iska", "othep", "uvia", "arne", "ellis", "omir", "assk", "undt",
      "eyra", "ophon", "ixil", "urga", "esca", "ynth", "avor", "usia", "eqar"]

_QTEMPLATES = [
    ("In the {ent1} archives, who first catalogued the {ent2} manuscripts?",
     "{ent3} of {ent1}"),
    ("During the {ent1} accord, which delegation proposed the {ent2} clause?",
     "the {ent3} delegation"),
    ("What compound did the {ent1} lab isolate from {ent2} sediment?",
     "{ent3}-{num}"),
    ("Which navigator charted the {ent1} strait before the {ent2} expedition?",
     "captain {ent3}"),
    ("In {ent1} music theory, what interval defines the {ent2} mode?",
     "the {ent3} interval"),
]

_DISTRACTOR_TOPICS = [
    "photosynthesis converts light into chemical energy in plants",
    "the water cycle moves moisture between ocean land and atmosphere",
    "supply and demand set prices in a competitive market",
    "tectonic plates drift slowly over the mantle over millions of years",
    "an algorithm is a finite sequence of well defined instructions",
    "mitochondria are the powerhouse organelles of eukaryotic cells",
    "inflation is a sustained rise in the general price level",
    "gravity is the attraction between masses described by relativity",
    "a compiler translates source code into machine instructions",
    "antibodies are proteins the immune system uses to tag pathogens",
]


def _name(rng: random.Random, tag: str) -> str:
    # Append a per-item tag so entity names are globally UNIQUE across items.
    # Real benchmarks have distinctive subjects; without this, two items could
    # share an answer and ground-truth membership would be ill-defined.
    return rng.choice(_A) + rng.choice(_B) + tag


def make_benchmark(n: int = 200, seed: int = 7) -> Benchmark:
    rng = random.Random(seed)
    items: List[BenchmarkItem] = []
    for i in range(n):
        tmpl_q, tmpl_a = _QTEMPLATES[i % len(_QTEMPLATES)]
        ents = {f"ent{j}": _name(rng, f"{i}{chr(96 + j)}") for j in range(1, 4)}
        num = rng.randrange(100, 999)
        q = tmpl_q.format(num=num, **ents)
        a = tmpl_a.format(num=num, **ents)
        items.append(BenchmarkItem(id=f"item-{i:04d}", question=q, answer=a))
    return Benchmark(items)


def make_clean_corpus(n_docs: int = 600, seed: int = 11) -> Corpus:
    rng = random.Random(seed)
    docs: List[str] = []
    for i in range(n_docs):
        topic = _DISTRACTOR_TOPICS[i % len(_DISTRACTOR_TOPICS)]
        filler = " ".join(
            rng.choice(
                ["moreover", "generally", "in practice", "notably", "typically",
                 "as a result", "for example", "in short", "over time"]
            )
            for _ in range(rng.randrange(3, 8))
        )
        docs.append(f"Document {i}: {topic}. {filler}.")
    return Corpus(docs, [f"clean-{i:04d}" for i in range(n_docs)])
