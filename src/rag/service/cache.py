"""Exact-match answer cache with versioned keys (design 5.10).

The key includes the index version, the corpus revision (bumped by uploads and deletes),
the prompt version and the answer settings, so any change that could change an answer
also changes the key: stale entries are never served, they just expire.
In memory and per process; with several instances this would move to Redis behind the
same interface.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any


def normalise_question(question: str) -> str:
    return " ".join(question.lower().split()).rstrip("?!. ")


def cache_key(question: str, **parts: Any) -> str:
    payload = json.dumps({"q": normalise_question(question), **parts}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class TTLCache:
    def __init__(self, ttl_s: float, max_entries: int, clock=time.monotonic) -> None:
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._data.get(key)
            if item is None or item[0] <= self._clock():
                if item is not None:
                    del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return item[1]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (self._clock() + self.ttl_s, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
