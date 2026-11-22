# syntax=docker/dockerfile:1
# Production RAG: the API and the built UI in one image. Models, the index and the
# corpus PDFs are mounted at run time (docker-compose.yml), not baked in: the GGUF
# files alone are 2.5 GB, and the index is rebuilt with `rag ingest`.
#
# Built and tested 2026-11-22 (Docker 29): `docker compose up` passes the same
# browser check as the local demo (README, "Docker").

# ---- 1. the UI (Vite build) ------------------------------------------------------
FROM node:22-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npx tsc && npx vite build

# ---- 2. the service ---------------------------------------------------------------
FROM python:3.12-slim AS app
COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1
# curl for the HEALTHCHECK; libgomp1 for llama.cpp's OpenMP threads.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Dependencies first (cached layer), exactly as locked; then the project itself.
# uv's download cache lives in a BuildKit cache mount, not in the image layer.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY corpus/manifest.yaml corpus/candidates.yaml corpus/
COPY demo/ demo/
COPY --from=ui /ui/dist frontend/dist

# Non-root. uid 1000 matches the usual first desktop user, so mounted folders
# (indexes/, .cache/) stay writable without chown on the host. Only the folders
# the app writes to change owner: a recursive chown of /app would copy the whole
# virtual environment into a new layer (it cost 719 MB), and the code and
# dependencies are better left read-only for the app user anyway.
RUN useradd --uid 1000 --create-home rag \
 && mkdir -p indexes .cache data corpus/pdfs \
 && chown rag:rag indexes .cache data corpus/pdfs
USER rag

ENV PATH=/app/.venv/bin:$PATH \
    RAG_MODELS_DIR=/models
EXPOSE 8000
# Liveness only: /ready stays 503 while the model loads (up to minutes), which is
# expected, not unhealthy. The start period covers a cold load (217 s measured).
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1
# 0.0.0.0 inside the container; docker-compose publishes it on 127.0.0.1 only.
CMD ["rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
