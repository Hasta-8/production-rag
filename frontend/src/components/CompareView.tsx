import { useEffect, useState } from "react";
import type { CompareResponse, Hit, Mode } from "../api";
import { api, ApiError } from "../api";

const MODES: { mode: Mode; label: string; score: (h: Hit) => string }[] = [
  { mode: "dense", label: "Dense only", score: (h) => `cos ${h.dense_score?.toFixed(3) ?? "–"}` },
  { mode: "hybrid", label: "Hybrid (BM25 + dense, RRF)", score: (h) => `fused #${h.fused_rank ?? "–"}` },
  {
    mode: "hybrid_rerank",
    label: "Hybrid + rerank",
    score: (h) => `rerank ${h.rerank_score?.toFixed(2) ?? "–"}`,
  },
];

/** Retrieval only, no LLM: dense vs hybrid vs hybrid + rerank, side by side. */
export function CompareView({
  question,
  onQuestion,
  runKey,
}: {
  question: string;
  onQuestion: (q: string) => void;
  runKey: number; // a preset bumps this to run the comparison at once
}) {
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);
  const [focus, setFocus] = useState<string | null>(null); // a chunk to follow across columns

  async function run(q = question) {
    if (!q.trim()) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await api.compare(q, MODES.map((m) => m.mode), 10));
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError(0, "network", "The service can't be reached."));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (runKey) void run();
  }, [runKey]);

  const rankIn = (mode: Mode, chunk: string) =>
    result?.modes[mode]?.hits.find((h) => h.chunk_id === chunk)?.rank ?? null;

  return (
    <section className="compare" aria-label="Compare retrieval modes">
      <form
        className="ask"
        onSubmit={(e) => {
          e.preventDefault();
          void run();
        }}
      >
        <input
          aria-label="Question to compare"
          value={question}
          onChange={(e) => onQuestion(e.target.value)}
          placeholder="A question that hinges on an exact name or number…"
        />
        <button type="submit" disabled={busy || !question.trim()}>
          {busy ? "Comparing…" : "Compare"}
        </button>
      </form>
      <p className="muted">
        Retrieval only (no model call), top 10 per mode. Hover a result to follow the same chunk across modes.
      </p>
      {error && (
        <p className="error" role="alert">
          {error.detail}
        </p>
      )}
      {result && (
        <div className="columns">
          {MODES.map(({ mode, label, score }) => {
            const col = result.modes[mode];
            return (
              <div className="column panel" key={mode} data-testid={`column-${mode}`}>
                <h3>
                  {label} <span className="muted">{col ? `${Math.round(col.ms)} ms` : ""}</span>
                </h3>
                <ol>
                  {col?.hits.map((h) => {
                    const others = MODES.filter((m) => m.mode !== mode)
                      .map((m) => [m.mode, rankIn(m.mode, h.chunk_id)] as const)
                      .map(([m, r]) => `${m.replace("_rerank", "+rr")} ${r ? `#${r}` : "–"}`);
                    return (
                      <li
                        key={h.chunk_id}
                        className={focus === h.chunk_id ? "focus" : ""}
                        onMouseEnter={() => setFocus(h.chunk_id)}
                        onMouseLeave={() => setFocus(null)}
                        title={h.snippet}
                      >
                        <div className="hit-title">
                          {h.title} · p. {h.page}
                        </div>
                        <div className="muted small">
                          {score(h)} · elsewhere: {others.join(", ")}
                        </div>
                      </li>
                    );
                  })}
                </ol>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
