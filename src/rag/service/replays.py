"""Recorded demo runs (design 9.5): the fallback when the model is too slow or fails.

A replay is a real run of a preset question, recorded from the running service by
``rag demo record``: the stream's events with their arrival times (or the compare
result, or the blocked response). The UI plays it back step by step, always labelled
"replay". Each file stores the index and prompt versions it was recorded with, so a
replay that no longer matches the running service is flagged as stale.
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

_SAFE_ID = re.compile(r"^[a-z0-9_-]{1,40}$")


def replay_path(folder: Path, preset_id: str) -> Path | None:
    if not _SAFE_ID.match(preset_id):
        return None
    return folder / f"{preset_id}.json"


def load_replay(folder: Path, preset_id: str) -> dict | None:
    path = replay_path(folder, preset_id)
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text())


def staleness(replay: dict, index_version: str | None, prompt_version: str) -> str | None:
    """Why the replay no longer matches the running service, or None."""
    if index_version is None:
        return None  # the engine isn't loaded: nothing to compare with (the replay still plays)
    if replay.get("index_version") != index_version:
        return f"recorded on index {replay.get('index_version')}, now {index_version}"
    if replay.get("prompt_version") != prompt_version:
        return f"recorded with prompt v{replay.get('prompt_version')}, now v{prompt_version}"
    return None


# ------------------------------------------------------------------ recording


def _parse_sse(lines) -> Any:
    event, data = None, []
    for line in lines:
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif line == "" and event is not None:
            yield event, json.loads("\n".join(data))
            event, data = None, []


def record_preset(client: httpx.Client, preset: dict, ready: dict) -> dict:
    """Run one preset against the live service and capture what the UI would receive."""
    base = {
        "preset_id": preset["id"],
        "question": preset["question"],
        "view": preset.get("view", "single"),
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "index_version": ready["index_version"],
        "prompt_version": ready["prompt_version"],
        "model": ready["models"]["primary"],
    }
    if base["view"] == "compare":
        t0 = time.perf_counter()
        r = client.post(
            "/v1/compare",
            json={
                "question": preset["question"],
                "modes": ["dense", "hybrid", "hybrid_rerank"],
                "k": 10,
            },
        )
        r.raise_for_status()
        return {
            **base,
            "kind": "compare",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
            "compare": r.json(),
        }

    events: list[dict] = []
    t0 = time.perf_counter()
    with client.stream("POST", "/v1/query/stream", json={"question": preset["question"]}) as r:
        if r.status_code != 200:
            r.read()
            return {**base, "kind": "error", "error": {"status": r.status_code, **r.json()}}
        for event, data in _parse_sse(r.iter_lines()):
            events.append(
                {"t_ms": round((time.perf_counter() - t0) * 1000), "event": event, "data": data}
            )
    final = next((e["data"] for e in events if e["event"] == "final"), None)
    if final is None:
        raise RuntimeError(f"{preset['id']}: the stream ended without a final answer")
    return {**base, "kind": "stream", "events": events}


def record_all(base_url: str, presets: list[dict], folder: Path, only: set[str] | None = None):
    """Record presets in order (the "repeat" preset must follow the one it repeats)."""
    folder.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(10.0, read=600.0)) as client:
        ready = client.get("/ready")
        if ready.status_code != 200:
            raise RuntimeError(f"the service is not ready ({ready.status_code}); start it first")
        ready = ready.json()
        for preset in presets:
            if only and preset["id"] not in only:
                continue
            rec = record_preset(client, preset, ready)
            if rec["kind"] == "stream":
                final = rec["events"][-1]["data"]
                if final.get("cached") and preset["id"] != "repeat":
                    raise RuntimeError(
                        f"{preset['id']} was answered from the cache; restart the service so the "
                        "recording is a live run"
                    )
            path = replay_path(folder, preset["id"])
            assert path is not None
            path.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
            yield preset["id"], rec
