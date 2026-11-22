// Types mirror the API (src/rag/service/schemas.py); functions are thin fetch wrappers.

export type Mode = "dense" | "bm25" | "hybrid" | "dense_rerank" | "hybrid_rerank";

export interface Source {
  tag: string;
  doc_id: string;
  chunk_id: string;
  title: string;
  page: number;
  score: number;
  snippet: string;
  cited: boolean;
  file_url: string | null;
}

export interface Security {
  masked: string[];
  stripped: number;
  output_masked: string[];
}

export interface TraceStep {
  name: string;
  ms: number;
  data: Record<string, unknown>;
}

export interface QueryResponse {
  answer: string;
  abstained: boolean;
  abstain_reason: string | null;
  sources: Source[];
  cited: string[];
  model_used: string | null;
  fallback_reason: string | null;
  pipeline: string;
  mode: string;
  retries: number;
  cached: boolean;
  replay?: boolean; // Phase 7: recorded runs, always labelled as such
  latency_ms: number;
  request_id: string;
  index_version: string;
  prompt_version: string;
  security: Security;
  trace: TraceStep[] | null;
}

export interface Hit {
  chunk_id: string;
  doc_id: string;
  title: string;
  page: number;
  rank: number;
  dense_rank: number | null;
  dense_score: number | null;
  bm25_rank: number | null;
  bm25_score: number | null;
  fused_rank: number | null;
  rerank_score: number | null;
  snippet?: string;
}

export interface CompareResponse {
  question: string;
  k: number;
  modes: Partial<Record<Mode, { ms: number; hits: Hit[] }>>;
  request_id: string;
}

export interface Preset {
  id: string;
  label: string;
  question: string;
  shows?: string;
  view?: "single" | "compare";
  verified?: boolean;
  replay?: boolean; // a recorded run exists (Phase 7)
}

/** A recorded run of a preset (design 9.5), served by /v1/demo/replays/{id}. */
export interface Replay {
  preset_id: string;
  question: string;
  view: "single" | "compare";
  recorded_at: string;
  index_version: string;
  prompt_version: string;
  model: string;
  kind: "stream" | "compare" | "error";
  events?: { t_ms: number; event: string; data: unknown }[];
  compare?: CompareResponse;
  error?: { status: number; error: string; detail: string; request_id?: string };
  stale: boolean;
  stale_reason: string | null;
}

export interface Ready {
  status: string;
  documents: number;
  chunks: number;
  index_version: string;
  models: { primary: string; fallback: string | null };
  demo_mode: boolean;
}

/** The API's one error shape, plus the HTTP status and Retry-After. */
export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    public detail: string,
    public requestId: string | null = null,
    public retryAfter: number | null = null,
  ) {
    super(detail);
  }
}

async function toApiError(r: Response): Promise<ApiError> {
  let body: { error?: string; detail?: string; request_id?: string } = {};
  try {
    body = await r.json();
  } catch {
    /* not JSON (e.g. a proxy error page) */
  }
  const retry = r.headers.get("retry-after");
  return new ApiError(
    r.status,
    body.error ?? "http_error",
    body.detail ?? `The server answered ${r.status}.`,
    body.request_id ?? null,
    retry ? Number(retry) : null,
  );
}

async function getJson<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  if (!r.ok) throw await toApiError(r);
  return (await r.json()) as T;
}

export const api = {
  ready: () => getJson<Ready>("/ready"),
  presets: () => getJson<{ presets: Preset[]; demo_mode: boolean }>("/v1/demo/presets"),
  replay: (id: string) => getJson<Replay>(`/v1/demo/replays/${encodeURIComponent(id)}`),
  compare: (question: string, modes: Mode[], k = 10) =>
    getJson<CompareResponse>("/v1/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, modes, k }),
    }),
};

// ------------------------------------------------------------------ streaming

export interface StreamEvent {
  event: string;
  data: unknown;
}

/** Incremental parser for server-sent events (a chunk may end mid-event). */
export class SSEParser {
  private buffer = "";

  feed(chunk: string): StreamEvent[] {
    this.buffer += chunk.replace(/\r\n/g, "\n");
    const out: StreamEvent[] = [];
    let end: number;
    while ((end = this.buffer.indexOf("\n\n")) !== -1) {
      const block = this.buffer.slice(0, end);
      this.buffer = this.buffer.slice(end + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      if (data.length) out.push({ event, data: JSON.parse(data.join("\n")) });
    }
    return out;
  }
}

export interface StreamHandlers {
  onEvent: (e: StreamEvent) => void;
}

/** POST /v1/query/stream (EventSource can't POST, so fetch + a reader). */
export async function streamQuery(
  question: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const r = await fetch("/v1/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!r.ok || !r.body) throw await toApiError(r);
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SSEParser();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    for (const e of parser.feed(decoder.decode(value, { stream: true }))) handlers.onEvent(e);
  }
}
