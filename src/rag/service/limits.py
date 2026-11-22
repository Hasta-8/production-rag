"""Per-client rate limits and the generation gate (design 5.10, 9.4; D9).

* ``RateLimiter``: a sliding window per client key; over the limit it says how long to
  wait, which becomes the 429's ``Retry-After``.
* ``GenerationGate``: the CPU runs one answer at a time (llama.cpp's model object isn't
  safe to share between threads either). A few questions may wait; beyond that the caller
  gets "busy" straight away instead of an unbounded queue.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, limit: int, window_s: float = 60.0, clock=time.monotonic) -> None:
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """Record a request. None if allowed, else the seconds until one is allowed."""
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= now - self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(hits[0] + self.window_s - now, 0.0)
            hits.append(now)
            if len(self._hits) > 10_000:  # forget idle clients
                for k in [k for k, v in self._hits.items() if not v]:
                    del self._hits[k]
            return None

    @staticmethod
    def retry_after_header(seconds: float) -> str:
        return str(max(1, math.ceil(seconds)))


class Busy(Exception):
    """The generation queue is full."""


class GenerationGate:
    """One generation at a time; at most ``queue_size`` more may wait.

    ``reserve()`` is synchronous, so two requests can't both see a free place: either the
    caller gets a ticket (with its queue position) or ``Busy`` at once. The ticket is then
    held with ``async with`` for the whole generation.

    The slot is a ``threading`` lock, polled without blocking the event loop: generation
    runs in worker threads anyway, a cancelled waiter can't leave the lock acquired, and it
    works whichever event loop the request runs in.
    """

    POLL_S = 0.05

    def __init__(self, queue_size: int) -> None:
        self.queue_size = queue_size
        self.active = 0
        self.waiting = 0
        self._slot = threading.Lock()
        self._count = threading.Lock()

    @property
    def depth(self) -> int:
        return self.active + self.waiting

    def reserve(self) -> Ticket:
        with self._count:
            if self.depth >= 1 + self.queue_size:
                raise Busy
            position = self.depth  # 0: runs now, 1: next, ...
            self.waiting += 1
        return Ticket(self, position)


class Ticket:
    def __init__(self, gate: GenerationGate, position: int) -> None:
        self.gate = gate
        self.position = position

    async def __aenter__(self) -> Ticket:
        gate = self.gate
        try:
            while not gate._slot.acquire(blocking=False):
                await asyncio.sleep(gate.POLL_S)
        finally:
            with gate._count:
                gate.waiting -= 1
        with gate._count:
            gate.active += 1
        return self

    async def __aexit__(self, *exc) -> None:
        gate = self.gate
        with gate._count:
            gate.active -= 1
        gate._slot.release()
