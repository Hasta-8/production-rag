"""Pipeline trace: every step records what it did, how long it took and what it found.

The same structure feeds the CLI (`rag ask --trace`), the API response, the UI trace
panel, recorded replays and the tests.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TraceStep:
    name: str
    ms: float
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    steps: list[TraceStep] = field(default_factory=list)

    @contextmanager
    def step(self, name: str, **data: Any) -> Iterator[dict[str, Any]]:
        """Time a block; the yielded dict collects the step's outputs."""
        record: dict[str, Any] = dict(data)
        start = time.perf_counter()
        try:
            yield record
        finally:
            ms = (time.perf_counter() - start) * 1000
            self.steps.append(TraceStep(name=name, ms=round(ms, 1), data=record))

    def get(self, name: str) -> TraceStep | None:
        return next((s for s in self.steps if s.name == name), None)

    @property
    def total_ms(self) -> float:
        return round(sum(s.ms for s in self.steps), 1)

    def to_list(self) -> list[dict[str, Any]]:
        return [asdict(s) for s in self.steps]
