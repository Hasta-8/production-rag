"""The LLM boundary. The pipeline only sees ``LLMClient``; tests use a fake."""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

Messages = Sequence[Mapping[str, str]]
TokenCallback = Callable[[str], None]

NO_THINK = " /no_think"  # Qwen3 switch: skip the slow reasoning phase
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMError(RuntimeError):
    """The model could not be loaded or failed to generate."""


@dataclass
class Generation:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    ms: float
    first_token_ms: float | None


class LLMClient(Protocol):
    name: str

    def generate(
        self, messages: Messages, max_tokens: int, on_token: TokenCallback | None = None
    ) -> Generation: ...


def strip_reasoning(text: str) -> str:
    """Remove Qwen3's <think>...</think> block (empty when /no_think is used)."""
    return _THINK_BLOCK.sub("", text).strip()


class _ThinkFilter:
    """Pass streamed tokens through, except those inside a leading <think> block."""

    def __init__(self, forward: TokenCallback) -> None:
        self.forward = forward
        self.buffer = ""
        self.past_think = False  # the think block (if any) is behind us
        self.started = False  # real text has been forwarded

    def _emit(self, text: str) -> None:
        if not self.started:  # drop whitespace before the first real text
            text = text.lstrip()
            if not text:
                return
            self.started = True
        self.forward(text)

    def __call__(self, token: str) -> None:
        if self.past_think:
            self._emit(token)
            return
        self.buffer += token
        text = self.buffer.lstrip()
        if not text:
            return
        if text.startswith("<think>"):
            if "</think>" in text:  # block finished: forward whatever follows it
                self.past_think = True
                self._emit(text.split("</think>", 1)[1])
            return
        if "<think>".startswith(text):  # could still become "<think>"; wait
            return
        self.past_think = True  # no think block at all
        self._emit(text)


class LlamaCppClient:
    """Local GGUF model through llama-cpp-python, loaded on first use.

    One instance holds one model; a lock serialises calls, because a CPU can only run
    one generation at a time and llama.cpp state is not thread-safe.
    """

    def __init__(
        self,
        model_path: Path,
        name: str | None = None,
        n_ctx: int = 4096,
        n_threads: int = 8,
        temperature: float = 0.0,
        no_think: bool = True,
    ) -> None:
        self.model_path = model_path
        self.name = name or model_path.stem
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.temperature = temperature
        self.no_think = no_think
        self._llm = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def load(self) -> None:
        if self._llm is not None:
            return
        if not self.model_path.exists():
            raise LLMError(f"model file not found: {self.model_path}")
        try:
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=0,
                verbose=False,
            )
        except Exception as exc:  # llama.cpp raises plain exceptions on bad files
            raise LLMError(f"could not load {self.model_path.name}: {exc}") from exc

    def _count_tokens(self, text: str) -> int:
        return len(self._llm.tokenize(text.encode("utf-8"), add_bos=False))

    def generate(
        self, messages: Messages, max_tokens: int, on_token: TokenCallback | None = None
    ) -> Generation:
        msgs = [dict(m) for m in messages]
        if self.no_think and msgs and msgs[-1]["role"] == "user":
            msgs[-1]["content"] = msgs[-1]["content"] + NO_THINK
        with self._lock:
            self.load()
            forward = _ThinkFilter(on_token) if on_token else None
            start = time.perf_counter()
            first: float | None = None
            parts: list[str] = []
            try:
                stream = self._llm.create_chat_completion(
                    messages=msgs, max_tokens=max_tokens, temperature=self.temperature, stream=True
                )
                for chunk in stream:
                    delta = chunk["choices"][0]["delta"].get("content")
                    if not delta:
                        continue
                    if first is None:
                        first = time.perf_counter()
                    parts.append(delta)
                    if forward:
                        forward(delta)
            except Exception as exc:
                raise LLMError(f"generation failed: {exc}") from exc
            end = time.perf_counter()
            raw = "".join(parts)
            prompt_tokens = sum(self._count_tokens(m["content"]) for m in msgs)
            completion_tokens = self._count_tokens(raw) if raw else 0
        return Generation(
            text=strip_reasoning(raw),
            model=self.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ms=round((end - start) * 1000, 1),
            first_token_ms=round((first - start) * 1000, 1) if first else None,
        )
