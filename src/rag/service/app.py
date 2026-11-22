"""The FastAPI service (Phase 5): the agent behind an API someone else could call.

Layers, outermost first: request ID + JSON access log + metrics, a general rate limit,
then per endpoint: validation, the stricter question rate limit, the security screen, the
answer cache, the generation gate (one answer at a time), the pipeline, output checks.

Run it with ``uv run rag serve`` (127.0.0.1:8000 by default).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag.config import Settings, get_settings
from rag.generation.answer import Answer
from rag.generation.prompts import PROMPT_VERSION, REFUSAL, SYSTEM_PROMPT
from rag.service.cache import TTLCache, cache_key
from rag.service.engine import Engine, build_engine
from rag.service.limits import Busy, GenerationGate, RateLimiter
from rag.service.monitoring import Metrics, log, new_request_id, request_id_var, setup_logging
from rag.service.schemas import CompareRequest, QueryRequest
from rag.service.security import leaks_prompt, mask_pii, scan_document, screen_input
from rag.tracing import Trace, TraceStep

# The leak check compares answers with the system prompt minus the refusal sentence: the
# prompt tells the model to reply with exactly that sentence, so it is not a leak.
LEAK_REFERENCE = SYSTEM_PROMPT.replace(REFUSAL, "")
# The UI is same-origin static files; PDF.js runs a worker and may decode images in wasm.
UI_CSP = (
    "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self' blob:; "
    "img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; frame-ancestors 'self'; "
    "base-uri 'none'; form-action 'self'"
)
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}
SNIPPET_CHARS = 300


class ApiError(Exception):
    def __init__(
        self, status: int, error: str, detail: str, headers: dict[str, str] | None = None
    ) -> None:
        self.status, self.error, self.detail, self.headers = status, error, detail, headers


def error_response(status: int, error: str, detail: str, headers=None) -> JSONResponse:
    body = {"error": error, "detail": detail, "request_id": request_id_var.get()}
    return JSONResponse(body, status_code=status, headers=headers)


@dataclass
class AppState:
    settings: Settings
    metrics: Metrics = field(default_factory=Metrics)
    engine: Engine | None = None
    loading: str = "starting"  # what /ready reports while the engine is being built
    load_failed: bool = False
    cache: TTLCache | None = None
    question_limiter: RateLimiter | None = None
    general_limiter: RateLimiter | None = None
    gate: GenerationGate | None = None
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    presets: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        s = self.settings
        self.cache = TTLCache(s.cache_ttl_s, s.cache_max_entries)
        self.question_limiter = RateLimiter(s.rate_limit_per_minute)
        self.general_limiter = RateLimiter(s.general_rate_limit_per_minute)
        self.gate = GenerationGate(s.queue_size)

    @property
    def max_question_chars(self) -> int:
        s = self.settings
        return (
            min(s.max_question_chars, s.demo_max_question_chars)
            if s.demo_mode
            else (s.max_question_chars)
        )

    def ready_engine(self) -> Engine:
        if self.engine is None:
            raise ApiError(503, "not_ready", f"The service is starting ({self.loading}).")
        return self.engine


def client_ip(request: Request, settings: Settings) -> str:
    """The direct peer, or the trusted proxy header if the peer is this machine."""
    peer = request.client.host if request.client else "unknown"
    header = settings.trusted_proxy_header
    if header and peer in LOCAL_HOSTS:
        forwarded = request.headers.get(header)
        if forwarded:
            return forwarded.split(",")[0].strip()
    return peer


def _hash_ip(ip: str) -> str:
    return hashlib.sha256(ip.encode()).hexdigest()[:12]  # logs never hold raw IPs


def load_presets(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    return list(data.get("presets", []))


# ------------------------------------------------------------------ answer helpers


def _mask_strings(value: Any) -> Any:
    """Mask PII in every string of a trace (a withdrawn answer or a rewrite may hold some)."""
    if isinstance(value, str):
        return mask_pii(value)[0]
    if isinstance(value, list):
        return [_mask_strings(v) for v in value]
    if isinstance(value, dict):
        return {k: _mask_strings(v) for k, v in value.items()}
    return value


def _step_dict(step: TraceStep) -> dict[str, Any]:
    return {"name": step.name, "ms": step.ms, "data": _mask_strings(step.data)}


def _build_response(
    engine: Engine, answer: Answer, req: QueryRequest, screen, latency_ms: float
) -> dict:
    """Output checks, then the response body (without per-request fields)."""
    text, out_masked = mask_pii(answer.text)
    abstained, reason = answer.abstained, answer.abstain_reason
    if not abstained and leaks_prompt(text, LEAK_REFERENCE):
        text, abstained, reason = REFUSAL, True, "the answer repeated the system prompt"
    cited = [] if abstained else answer.cited
    sources = []
    for s in answer.sources:
        entry = engine.index.info.documents.get(s.doc_id)
        source_file = entry.source if entry else ""
        sources.append(
            {
                "tag": s.tag,
                "doc_id": s.doc_id,
                "chunk_id": s.chunk_id,
                "title": s.title,
                "page": s.page,
                "score": s.score,
                "snippet": " ".join(s.text.split())[:SNIPPET_CHARS],
                "cited": s.tag in cited,
                "file_url": (
                    f"/v1/documents/{s.doc_id}/file#page={s.page}"
                    if source_file and engine.servable(source_file)
                    else None
                ),
            }
        )
    gens = [st for st in answer.trace.steps if st.name == "generate"]
    fallback = next(
        (g.data.get("fallback_reason") for g in gens if g.data.get("fallback_from")), None
    )
    return {
        "answer": text,
        "abstained": abstained,
        "abstain_reason": reason,
        "sources": sources,
        "cited": cited,
        "model_used": answer.model,
        "fallback_reason": fallback,
        "pipeline": answer.pipeline,
        "mode": req.mode or engine.default_mode,
        "retries": answer.retries,
        "index_version": engine.index.info.version,
        "prompt_version": PROMPT_VERSION,
        "security": {
            "masked": screen.masked,
            "stripped": len(screen.stripped),
            "output_masked": out_masked,
        },
        "trace": [_step_dict(st) for st in answer.trace.steps],
        "_tokens": (
            sum(g.data.get("prompt_tokens") or 0 for g in gens),
            sum(g.data.get("completion_tokens") or 0 for g in gens),
        ),
        "_generation_ms": latency_ms,
    }


class StreamGuard:
    """Masks PII in streamed tokens: text is held back a little, so a phone number or an
    email is complete (and masked) before any of it is sent. Stops streaming if the text
    starts repeating the system prompt; the final event carries the checked answer."""

    def __init__(self, emit, hold: int = 48) -> None:
        self.emit, self.hold = emit, hold
        self.buf, self.sent, self.stopped = "", 0, False

    def feed(self, token: str) -> None:
        self.buf += token
        if self.stopped:
            return
        if leaks_prompt(self.buf, LEAK_REFERENCE):
            self.stopped = True
            return
        masked = mask_pii(self.buf)[0]
        end = len(masked) - self.hold
        if end > self.sent:
            self.emit(masked[self.sent : end])
            self.sent = end

    def flush(self) -> None:
        if not self.stopped:
            masked = mask_pii(self.buf)[0]
            if len(masked) > self.sent:
                self.emit(masked[self.sent :])
            self.sent = len(masked)


# ------------------------------------------------------------------ the app


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging()
    state = AppState(settings, engine=engine, presets=load_presets(settings.presets_file))
    if engine is not None:
        state.loading = "ready"

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if state.engine is None:
            # Build in a thread: /health answers at once, /ready stays 503 until loaded (D8).
            def load() -> None:
                state.loading = "loading the index, reranker and language model"
                t0 = time.perf_counter()
                try:
                    state.engine = build_engine(settings)
                    state.loading = "ready"
                    log("engine ready", load_s=round(time.perf_counter() - t0, 1))
                except Exception:
                    state.load_failed = True
                    state.loading = "startup failed; see the server log"
                    import logging

                    logging.getLogger("rag.service").exception("engine failed to load")

            threading.Thread(target=load, name="engine-loader", daemon=True).start()
        yield

    demo = settings.demo_mode
    app = FastAPI(
        title="Production RAG",
        version="0.5.0",
        summary="Cited answers over AI research papers, fully local.",
        lifespan=lifespan,
        docs_url=None if demo else "/docs",
        redoc_url=None,
        openapi_url=None if demo else "/openapi.json",
    )
    app.state.rag = state

    # -------------------------------------------------------------- errors

    @app.exception_handler(ApiError)
    async def _api_error(_req: Request, exc: ApiError):
        return error_response(exc.status, exc.error, exc.detail, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(_req: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        detail = f"{where}: {first.get('msg', 'invalid value')}" if where else "invalid request"
        return error_response(422, "invalid_request", detail)  # never echoes the input

    @app.exception_handler(StarletteHTTPException)
    async def _http(_req: Request, exc: StarletteHTTPException):
        codes = {404: "not_found", 405: "method_not_allowed"}
        return error_response(
            exc.status_code, codes.get(exc.status_code, "http_error"), str(exc.detail)
        )

    @app.exception_handler(Exception)
    async def _unexpected(_req: Request, exc: Exception):
        import logging

        logging.getLogger("rag.service").exception("unhandled error")
        return error_response(
            500, "internal_error", "Something went wrong; it was logged with this request ID."
        )

    # -------------------------------------------------------------- middleware

    @app.middleware("http")
    async def _observe(request: Request, call_next):
        rid = new_request_id(request.headers.get("x-request-id"))
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        route = request.url.path
        ip = client_ip(request, settings)
        status = 500
        try:
            if route not in ("/health", "/ready"):
                wait = state.general_limiter.check(ip)
                if wait is not None:
                    state.metrics.event("rate_limited")
                    response = error_response(
                        429,
                        "rate_limited",
                        "Too many requests; slow down.",
                        {"Retry-After": RateLimiter.retry_after_header(wait)},
                    )
                    status = 429
                    return response
            try:
                response = await call_next(request)
            except Exception:  # noqa: BLE001  (the handler above logs it)
                response = await _unexpected(request, Exception())
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["X-Frame-Options"] = "SAMEORIGIN"
            if response.headers.get("content-type", "").startswith("text/html"):
                response.headers["Content-Security-Policy"] = UI_CSP
            return response
        finally:
            ms = (time.perf_counter() - t0) * 1000
            label = request.scope.get("route").path if request.scope.get("route") else "other"
            state.metrics.record_request(label, status, ms)
            log(
                "request",
                method=request.method,
                path=route,
                status=status,
                latency_ms=round(ms, 1),
                client=_hash_ip(ip),
            )
            request_id_var.reset(token)

    def limit_questions(request: Request) -> None:
        wait = state.question_limiter.check(client_ip(request, settings))
        if wait is not None:
            state.metrics.event("rate_limited")
            raise ApiError(
                429,
                "rate_limited",
                f"At most {settings.rate_limit_per_minute} questions per minute; try again soon.",
                {"Retry-After": RateLimiter.retry_after_header(wait)},
            )

    def check_question(question: str):
        if len(question) > state.max_question_chars:
            raise ApiError(
                422,
                "question_too_long",
                f"Questions are limited to {state.max_question_chars} characters.",
            )
        screen = screen_input(question)
        if screen.blocked:
            state.metrics.event("blocked_input")
            log("blocked input", pattern=screen.blocked)
            raise ApiError(
                400,
                "blocked_input",
                "This looks like a prompt-injection attempt "
                f"({screen.blocked.replace('_', ' ')}), so it was not sent to the model.",
            )
        if not screen.text.strip():
            raise ApiError(422, "invalid_request", "question: nothing left after cleaning")
        return screen

    def require_writes(request: Request) -> None:
        if demo:
            raise ApiError(403, "demo_mode", "Uploads and deletes are turned off in demo mode.")
        token = settings.admin_token.get_secret_value() if settings.admin_token else ""
        if token:
            given = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            if not secrets.compare_digest(given.encode(), token.encode()):
                raise ApiError(403, "forbidden", "A valid admin token is required.")
        elif (request.client.host if request.client else "") not in LOCAL_HOSTS:
            raise ApiError(403, "forbidden", "Writes are allowed from this machine only.")

    # -------------------------------------------------------------- answering

    def _key(engine: Engine, req: QueryRequest, text: str) -> str:
        return cache_key(
            text,
            mode=req.mode or engine.default_mode,
            pipeline=req.pipeline or engine.default_pipeline,
            index=engine.index.info.version,
            revision=engine.revision,
            prompt=PROMPT_VERSION,
            opts=repr(engine.opts),
        )

    def _finish(body: dict, cached: bool, t0: float, include_trace: bool) -> dict:
        out = {k: v for k, v in body.items() if not k.startswith("_")}
        out.update(
            cached=cached,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            request_id=request_id_var.get(),
        )
        if not include_trace:
            out["trace"] = None
        return out

    async def run_answer(engine, req, screen, events: asyncio.Queue | None, ticket) -> dict:
        """Hold the generation slot for the whole run, even if the client goes away: the
        CPU can only run one answer, and a finished answer is still worth caching."""
        loop = asyncio.get_running_loop()

        def push(kind: str, data: Any) -> None:
            if events is not None:
                loop.call_soon_threadsafe(events.put_nowait, (kind, data))

        if ticket.position:
            push("queued", {"position": ticket.position, "detail": "busy, you're next"})
        async with ticket:
            trace = Trace(listener=lambda st: push("step", _step_dict(st)))
            guard = StreamGuard(lambda text: push("token", {"text": text}))
            fn = engine.answerer(req.pipeline, req.mode)
            t0 = time.perf_counter()
            answer = await asyncio.to_thread(fn, screen.text, guard.feed, trace)
            guard.flush()
            body = _build_response(engine, answer, req, screen, (time.perf_counter() - t0) * 1000)
        tokens_in, tokens_out = body["_tokens"]
        state.metrics.record_answer(
            abstained=body["abstained"],
            fallback=body["fallback_reason"] is not None,
            retries=body["retries"],
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        if body["fallback_reason"] is None:  # a fallback answer is retried live next time
            state.cache.set(_key(engine, req, screen.text), body)
        log(
            "answer",
            pipeline=body["pipeline"],
            mode=body["mode"],
            model=body["model_used"],
            retries=body["retries"],
            abstained=body["abstained"],
            fallback=body["fallback_reason"] is not None,
            generation_ms=round(body["_generation_ms"], 1),
        )
        return body

    def _reserve():
        try:
            return state.gate.reserve()
        except Busy:
            state.metrics.event("busy")
            raise ApiError(
                503,
                "busy",
                "Another question is being answered (this laptop answers one at a time). "
                "Try again in a minute.",
                {"Retry-After": "60"},
            ) from None

    def _cached(engine, req, screen) -> dict | None:
        hit = state.cache.get(_key(engine, req, screen.text))
        state.metrics.event("cache_hit" if hit is not None else "cache_miss")
        return hit

    @app.post("/v1/query", tags=["answers"])
    async def query(req: QueryRequest, request: Request):
        """Ask a question; returns the answer, its sources and the pipeline trace."""
        t0 = time.perf_counter()
        limit_questions(request)
        engine = state.ready_engine()
        screen = check_question(req.question)
        hit = _cached(engine, req, screen)
        if hit is not None:
            return _finish(hit, True, t0, req.include_trace)
        ticket = _reserve()
        task = asyncio.create_task(run_answer(engine, req, screen, None, ticket))
        body = await asyncio.shield(task)
        return _finish(body, False, t0, req.include_trace)

    @app.post("/v1/query/stream", tags=["answers"])
    async def query_stream(req: QueryRequest, request: Request):
        """Server-sent events: ``queued``, ``step`` (each trace step as it finishes),
        ``token`` (answer text), then ``final`` (the checked answer, as from /v1/query)."""
        t0 = time.perf_counter()
        limit_questions(request)
        engine = state.ready_engine()
        screen = check_question(req.question)
        rid = request_id_var.get()

        def sse(kind: str, data: Any) -> str:
            return f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        hit = _cached(engine, req, screen)
        if hit is not None:

            async def replay_cached() -> AsyncIterator[str]:
                yield sse("meta", {"request_id": rid, "cached": True, "security": hit["security"]})
                yield sse("final", _finish(hit, True, t0, req.include_trace))

            return StreamingResponse(replay_cached(), media_type="text/event-stream")

        security = {"masked": screen.masked, "stripped": len(screen.stripped), "output_masked": []}
        ticket = _reserve()
        events: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(run_answer(engine, req, screen, events, ticket))

        async def stream() -> AsyncIterator[str]:
            token = request_id_var.set(rid)
            try:
                yield sse("meta", {"request_id": rid, "cached": False, "security": security})
                while True:
                    getter = asyncio.ensure_future(events.get())
                    done, _ = await asyncio.wait(
                        {getter, task}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if getter in done:
                        kind, data = getter.result()
                        yield sse(kind, data)
                        continue
                    getter.cancel()
                    while not events.empty():
                        kind, data = events.get_nowait()
                        yield sse(kind, data)
                    if task.exception() is not None:
                        import logging

                        logging.getLogger("rag.service").error(
                            "stream failed", exc_info=task.exception()
                        )
                        yield sse(
                            "error",
                            {
                                "error": "internal_error",
                                "detail": "Something went wrong; it was logged.",
                                "request_id": rid,
                            },
                        )
                    else:
                        yield sse("final", _finish(task.result(), False, t0, req.include_trace))
                    return
            finally:
                request_id_var.reset(token)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/compare", tags=["answers"])
    async def compare(req: CompareRequest, request: Request):
        """One question through several retrieval modes (retrieval only, no LLM call)."""
        limit_questions(request)
        engine = state.ready_engine()
        screen = check_question(req.question)
        out: dict[str, Any] = {}
        for mode in dict.fromkeys(req.modes):
            try:
                retriever = engine.retriever(mode)
            except ValueError as exc:
                raise ApiError(422, "invalid_request", str(exc)) from None
            t0 = time.perf_counter()
            hits = await asyncio.to_thread(retriever.search, screen.text, req.k)
            out[mode] = {
                "ms": round((time.perf_counter() - t0) * 1000, 1),
                "hits": [
                    {
                        **h.summary(),
                        "doc_id": h.doc_id,
                        "snippet": " ".join(h.text.split())[:SNIPPET_CHARS],
                    }
                    for h in hits[: req.k]
                ],
            }
        return {
            "question": screen.text,
            "k": req.k,
            "modes": out,
            "request_id": request_id_var.get(),
        }

    # -------------------------------------------------------------- documents

    def _doc_row(engine: Engine, entry) -> dict:
        meta = engine.doc_meta.get(entry.source, {})
        servable = engine.servable(entry.source)
        return {
            "doc_id": entry.doc_id,
            "title": entry.title,
            "source": entry.source,
            "pages": entry.pages,
            "chunks": entry.chunks,
            "licence": engine.licence(entry.source),
            "arxiv_id": meta.get("arxiv_id"),
            "abs_url": meta.get("abs_url"),
            "file_url": f"/v1/documents/{entry.doc_id}/file" if servable else None,
        }

    @app.get("/v1/documents", tags=["documents"])
    async def documents():
        engine = state.ready_engine()
        rows = [_doc_row(engine, e) for e in engine.index.info.documents.values()]
        return {"documents": sorted(rows, key=lambda r: r["title"].lower()), "count": len(rows)}

    @app.get("/v1/documents/{doc_id}/file", tags=["documents"])
    async def document_file(doc_id: str):
        """The PDF, for the citation viewer. Only licences that allow redistribution."""
        engine = state.ready_engine()
        entry = engine.index.info.documents.get(doc_id)
        if entry is None:
            raise ApiError(404, "not_found", "No document with this ID.")
        if engine.licence(entry.source) not in ("open", "noncommercial", "uploaded"):
            abs_url = engine.doc_meta.get(entry.source, {}).get("abs_url")
            raise ApiError(
                403, "not_redistributable", f"Its licence doesn't allow redistribution: {abs_url}"
            )
        path = engine.file_path(entry.source)
        if path is None:
            raise ApiError(404, "not_found", "The file is not on this machine.")
        return FileResponse(
            path,
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{entry.source}"'},
        )

    @app.post("/v1/documents", status_code=201, tags=["documents"])
    async def upload(request: Request, file: UploadFile = File(...)):  # noqa: B008
        """Upload and index a PDF (off in demo mode)."""
        require_writes(request)
        engine = state.ready_engine()
        name = Path(file.filename or "upload.pdf").name
        if not name.lower().endswith(".pdf"):
            raise ApiError(415, "unsupported_type", "Only PDF files can be uploaded.")
        limit = settings.max_upload_mb * 1024 * 1024
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise ApiError(413, "too_large", f"Files are limited to {settings.max_upload_mb} MB.")
        if not data.startswith(b"%PDF-"):
            raise ApiError(415, "unsupported_type", "The file is not a PDF.")
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)[-80:]
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        path = settings.uploads_dir / f"{hashlib.sha256(data).hexdigest()[:12]}-{safe}"
        async with state.write_lock:
            path.write_bytes(data)
            result, warnings = await asyncio.to_thread(_ingest_upload, engine, path)
            if result.status == "duplicate":
                path.unlink(missing_ok=True)
                raise ApiError(409, "duplicate", f"Already indexed as {result.doc_id}.")
            if result.status == "error":
                path.unlink(missing_ok=True)
                raise ApiError(422, "unreadable_document", "No usable text could be extracted.")
            await asyncio.to_thread(engine.refresh)
        state.cache.clear()
        log("document uploaded", doc_id=result.doc_id, chunks=result.chunks)
        entry = engine.index.info.documents[result.doc_id]
        return {**_doc_row(engine, entry), "status": result.status, "warnings": warnings}

    @app.delete("/v1/documents/{doc_id}", tags=["documents"])
    async def delete(doc_id: str, request: Request):
        """Remove a document: its chunks, then BM25 is rebuilt (off in demo mode)."""
        require_writes(request)
        engine = state.ready_engine()
        async with state.write_lock:
            entry = engine.index.info.documents.get(doc_id)
            if entry is None:
                raise ApiError(404, "not_found", "No document with this ID.")
            await asyncio.to_thread(_delete_document, engine, entry, settings.uploads_dir)
            await asyncio.to_thread(engine.refresh)
        state.cache.clear()
        log("document deleted", doc_id=doc_id, chunks=entry.chunks)
        return {"deleted": doc_id, "chunks": entry.chunks, "revision": engine.revision}

    # -------------------------------------------------------------- demo and ops

    @app.get("/v1/demo/presets", tags=["demo"])
    async def presets():
        return {"presets": state.presets, "demo_mode": demo}

    @app.get("/health", tags=["ops"])
    async def health():
        """Liveness: the process answers."""
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    async def ready():
        """Readiness: index open, reranker and main model loaded, index not empty."""
        engine = state.engine
        if engine is None:
            error = "startup_failed" if state.load_failed else "not_ready"
            return error_response(503, error, f"The service is starting ({state.loading}).")
        if not engine.index.info.documents:
            return error_response(503, "not_ready", "The index is empty; run `rag ingest`.")
        return {
            "status": "ready",
            "index_version": engine.index.info.version,
            "revision": engine.revision,
            "documents": len(engine.index.info.documents),
            "chunks": engine.index.info.total_chunks,
            "models": engine.models,
            "demo_mode": demo,
        }

    @app.get("/metrics", tags=["ops"])
    async def metrics():
        return state.metrics.snapshot(
            queue={"active": state.gate.active, "waiting": state.gate.waiting},
            cache_entries=len(state.cache),
        )

    # The built UI (Phase 6), mounted last so every API route keeps precedence.
    ui = settings.frontend_dir
    if (ui / "index.html").is_file():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(ui), html=True), name="ui")
    else:

        @app.get("/", include_in_schema=False)
        async def root():
            return {"service": "Production RAG", "ui": "not built (cd frontend && npm run build)"}

    return app


def _ingest_upload(engine: Engine, path: Path):
    from rag.ingestion.loaders import IngestionError, load_document
    from rag.ingestion.pipeline import ingest_file

    try:
        warnings = scan_document([p.text for p in load_document(path).pages])
    except IngestionError:
        warnings = []
    result = ingest_file(path, engine.index, lambda _p: {"licence_class": "uploaded"})
    flagged = [
        f"page {w['page']}: text that looks like instructions to a model ({w['pattern']})"
        for w in warnings
    ]
    return result, flagged


def _delete_document(engine: Engine, entry, uploads_dir: Path) -> None:
    from rag.ingestion.chunkers import chunk_id_for

    engine.index.store.delete(ids=[chunk_id_for(entry.doc_id, i) for i in range(entry.chunks)])
    engine.index.info.documents.pop(entry.doc_id, None)
    engine.index.save_info()
    uploaded = (uploads_dir / entry.source).resolve()
    if uploaded.parent == uploads_dir.resolve():
        uploaded.unlink(missing_ok=True)  # corpus PDFs stay; only uploads are removed
