"""The benchmark B = [{id, question, answer}] we are auditing for contamination."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class BenchmarkItem:
    id: str
    question: str
    answer: str

    def text(self) -> str:
        """Canonical 'item text' used by matchers when the whole item leaks."""
        return f"Q: {self.question}\nA: {self.answer}"


@dataclass
class Benchmark:
    items: List[BenchmarkItem] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def ids(self) -> List[str]:
        return [it.id for it in self.items]

    def by_id(self, item_id: str) -> BenchmarkItem:
        for it in self.items:
            if it.id == item_id:
                return it
        raise KeyError(item_id)


def load_benchmark(path: str | Path) -> Benchmark:
    """Load a benchmark from a .jsonl with {id, question, answer} per line."""
    path = Path(path)
    items = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(
                BenchmarkItem(
                    id=str(obj.get("id", f"item-{i}")),
                    question=obj.get("question", ""),
                    answer=str(obj.get("answer", "")),
                )
            )
    return Benchmark(items)


def save_benchmark(bench: Benchmark, path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        for it in bench.items:
            fh.write(
                json.dumps({"id": it.id, "question": it.question, "answer": it.answer})
                + "\n"
            )
