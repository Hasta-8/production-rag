"""JSON logs with request IDs, and the metrics behind ``GET /metrics`` (design 5.10)."""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from typing import Any

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


def new_request_id(incoming: str | None = None) -> str:
    """Reuse a caller's X-Request-ID if it looks safe, else make one."""
    if incoming and _VALID_ID.match(incoming):
        return incoming
    return uuid.uuid4().hex[:16]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname.lower(),
            "logger": record.name,
            "request_id": request_id_var.get(),
            "msg": record.getMessage(),
        }
        data.update(getattr(record, "fields", {}))
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False, default=str)


logger = logging.getLogger("rag.service")


def setup_logging(level: int = logging.INFO) -> None:
    if any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def log(msg: str, **fields: Any) -> None:
    logger.info(msg, extra={"fields": fields})


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(round(p * (len(s) - 1))))], 1)


class Metrics:
    """Counters and recent latencies. Local models: the cost is $0 by construction, but
    tokens are tracked so a paid model's cost would be one multiplication away."""

    def __init__(self, window: int = 1000) -> None:
        self._lock = threading.Lock()
        self.started = time.time()
        self.requests: Counter[str] = Counter()
        self.status: Counter[str] = Counter()
        self.errors = 0  # failures: 5xx except 503 (busy / not ready are deliberate refusals)
        self.latency: defaultdict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self.events: Counter[str] = Counter()  # blocked, rate_limited, busy, cache_hit, ...
        self.answers = 0
        self.abstained = 0
        self.fallbacks = 0
        self.retries = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def record_request(self, route: str, status: int, ms: float) -> None:
        with self._lock:
            self.requests[route] += 1
            self.status[f"{status // 100}xx"] += 1
            self.errors += status >= 500 and status != 503
            self.latency[route].append(ms)

    def event(self, name: str) -> None:
        with self._lock:
            self.events[name] += 1

    def record_answer(
        self, *, abstained: bool, fallback: bool, retries: int, tokens_in: int, tokens_out: int
    ) -> None:
        with self._lock:
            self.answers += 1
            self.abstained += abstained
            self.fallbacks += fallback
            self.retries += retries
            self.tokens_in += tokens_in
            self.tokens_out += tokens_out

    def snapshot(self, **extra: Any) -> dict[str, Any]:
        with self._lock:
            total = sum(self.requests.values())
            errors = self.errors
            hits, misses = self.events.get("cache_hit", 0), self.events.get("cache_miss", 0)
            return {
                "uptime_s": round(time.time() - self.started),
                "requests": total,
                "requests_by_route": dict(self.requests),
                "responses_by_status": dict(self.status),
                "errors": errors,
                "error_rate": round(errors / total, 4) if total else 0.0,
                "latency_ms": {
                    route: {
                        "p50": _percentile(list(v), 0.50),
                        "p95": _percentile(list(v), 0.95),
                        "n": len(v),
                    }
                    for route, v in self.latency.items()
                },
                "cache": {
                    "hits": hits,
                    "misses": misses,
                    "hit_rate": round(hits / (hits + misses), 4) if hits + misses else None,
                },
                "answers": {
                    "generated": self.answers,
                    "abstention_rate": (
                        round(self.abstained / self.answers, 4) if self.answers else None
                    ),
                    "fallback_model_used": self.fallbacks,
                    "corrective_steps": self.retries,
                },
                "tokens": {"prompt": self.tokens_in, "completion": self.tokens_out},
                "estimated_cost_usd": 0.0,  # local models only (no API calls)
                "events": dict(self.events),
                **extra,
            }
