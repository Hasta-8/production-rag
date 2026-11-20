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


class FirstTokenTimeout(LLMError):
    """The model produced no token within its first-token time limit (generation aborted)."""


@dataclass
class Generation:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    ms: float
    first_token_ms: float | None
    fallback_from: str | None = None  # the primary model, when the fallback answered instead
    fallback_reason: str | None = None


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
        first_token_timeout_s: float | None = None,
    ) -> None:
        self.model_path = model_path
        self.name = name or model_path.stem
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.temperature = temperature
        self.no_think = no_think
        self.first_token_timeout_s = first_token_timeout_s
        self._llm = None
        self._lock = threading.Lock()
        # Abort deadline for llama.cpp, checked between compute steps (see _install_abort).
        self._deadline: float | None = None
        self._abort_cb = None

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
        self._install_abort()

    def _install_abort(self) -> None:
        """Let a deadline interrupt llama.cpp mid-computation.

        Reading a long prompt is one blocking native call, so a Python-side timer can't stop
        it. llama.cpp calls this callback between compute steps; returning True aborts the
        decode, which llama-cpp-python surfaces as a RuntimeError.
        """
        import llama_cpp

        def should_abort(_data) -> bool:
            return self._deadline is not None and time.perf_counter() > self._deadline

        self._abort_cb = llama_cpp.ggml_abort_callback(should_abort)  # keep a reference
        llama_cpp.llama_set_abort_callback(self._llm._ctx.ctx, self._abort_cb, None)

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
            timeout = self.first_token_timeout_s
            self._deadline = start + timeout if timeout else None
            try:
                stream = self._llm.create_chat_completion(
                    messages=msgs, max_tokens=max_tokens, temperature=self.temperature, stream=True
                )
                for chunk in stream:
                    if first is None:
                        first = time.perf_counter()
                        self._deadline = None  # the limit covers the first token only
                    delta = chunk["choices"][0]["delta"].get("content")
                    if not delta:
                        continue
                    parts.append(delta)
                    if forward:
                        forward(delta)
            except Exception as exc:
                timed_out = first is None and timeout and time.perf_counter() - start >= timeout
                self._llm.reset()  # an aborted decode leaves the KV cache half-written
                if timed_out:
                    raise FirstTokenTimeout(
                        f"{self.name}: no first token within {timeout:g} s"
                    ) from exc
                raise LLMError(f"generation failed: {exc}") from exc
            finally:
                self._deadline = None
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


class FallbackLLM:
    """LLM gateway: use the primary model; if it fails or is too slow to start, the fallback.

    "Too slow" is the primary's own first-token timeout (``FirstTokenTimeout``). The answer
    says which model produced it, and why the fallback was used, so the trace shows it.
    """

    def __init__(self, primary: LLMClient, fallback: LLMClient) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = primary.name

    def generate(
        self, messages: Messages, max_tokens: int, on_token: TokenCallback | None = None
    ) -> Generation:
        try:
            return self.primary.generate(messages, max_tokens, on_token)
        except LLMError as exc:
            gen = self.fallback.generate(messages, max_tokens, on_token)
            gen.fallback_from = self.primary.name
            gen.fallback_reason = str(exc)
            return gen
