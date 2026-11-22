import { useState } from "react";
import type { Hit, TraceStep } from "../api";
import type { RunState } from "../run";

const fmt = (v: unknown, digits = 2) =>
  typeof v === "number" ? v.toFixed(digits) : v === null || v === undefined ? "–" : String(v);
const ms = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`);

/** One line per step: what it did, in the terms the design's trace panel uses. */
export function summarise(step: TraceStep): string {
  const d = step.data as Record<string, any>;
  switch (step.name) {
    case "retrieve": {
      const hits = (d.hits as Hit[]) ?? [];
      const best = hits.reduce<number | null>(
        (b, h) => (h.rerank_score !== null && (b === null || h.rerank_score > b) ? h.rerank_score : b),
        null,
      );
      const q = d.query ? ` · query "${d.query}"` : "";
      return `${d.mode} · ${hits.length} chunks${best !== null ? ` · best rerank ${best.toFixed(2)}` : ""}${q}`;
    }
    case "grade":
      return `${d.verdict} · best ${d.signal} ${fmt(d.best_score)} vs ${fmt(d.threshold, 1)}`;
    case "rewrite":
      return `"${d.query}" · step ${d.retry}/${d.budget}${d.repeated ? " · repeated, loop stops" : ""}`;
    case "focus":
      return `${d.verdict} · best sentence ${fmt(d.best_score)}`;
    case "generate": {
      const fb = d.fallback_from ? ` · fallback from ${d.fallback_from}` : "";
      return `${d.model} · ${d.prompt_tokens ?? "?"} in / ${d.completion_tokens ?? "?"} out tokens · first token ${
        typeof d.first_token_ms === "number" ? ms(d.first_token_ms) : "–"
      } · $0 (local)${fb}`;
    }
    case "check":
      if (d.result !== "answered") return String(d.result);
      return `${(d.cited as string[])?.length ?? 0} citation(s) valid${
        (d.invalid as string[])?.length ? ` · ${(d.invalid as string[]).length} invalid removed` : ""
      }${d.inferred ? " · inferred" : ""}`;
    case "entities": {
      const names = (d.entities as string[]) ?? [];
      return `${d.verdict} · ${names.map((n) => `"${n}"`).join(", ")}`;
    }
    default:
      return "";
  }
}

function HitsTable({ hits }: { hits: Hit[] }) {
  return (
    <div className="hits-wrap">
    <table className="hits">
      <thead>
        <tr>
          <th>#</th>
          <th>paper · page</th>
          <th title="rank in dense (embedding) search">dense</th>
          <th title="rank in BM25 keyword search">BM25</th>
          <th title="rank after Reciprocal Rank Fusion">fused</th>
          <th title="cross-encoder score">rerank</th>
        </tr>
      </thead>
      <tbody>
        {hits.map((h) => (
          <tr key={h.chunk_id}>
            <td>{h.rank}</td>
            <td className="title-cell" title={h.title}>
              {h.title.slice(0, 34)}
              {h.title.length > 34 ? "…" : ""} · {h.page}
            </td>
            <td>{h.dense_rank ?? "–"}</td>
            <td>{h.bm25_rank ?? "–"}</td>
            <td>{h.fused_rank ?? "–"}</td>
            <td>{fmt(h.rerank_score)}</td>
          </tr>
        ))}
      </tbody>
    </table>
    </div>
  );
}

function StepRow({ step }: { step: TraceStep }) {
  const [open, setOpen] = useState(false);
  const hits = step.name === "retrieve" ? ((step.data.hits as Hit[]) ?? []) : [];
  return (
    <li className={`step step-${step.name}`} data-testid={`step-${step.name}`}>
      <div className="step-head">
        <span className="step-name">{step.name}</span>
        <span className="step-summary">{summarise(step)}</span>
        <span className="step-ms">{ms(step.ms)}</span>
      </div>
      {hits.length > 0 && (
        <>
          <button type="button" className="link" onClick={() => setOpen(!open)} aria-expanded={open}>
            {open ? "hide" : "show"} chunk ranks: dense | BM25 | fused | rerank
          </button>
          {open && <HitsTable hits={hits} />}
        </>
      )}
    </li>
  );
}

export function TracePanel({ run }: { run: RunState }) {
  if (run.status === "idle") return <aside className="panel trace empty">The pipeline trace appears here.</aside>;
  const sec = run.security;
  const blocked = run.error?.code === "blocked_input";
  const waitingFor =
    run.status !== "running"
      ? null
      : run.queuedPosition
        ? "waiting for the generation slot"
        : run.steps.at(-1)?.name === "grade" && (run.steps.at(-1)?.data as any)?.verdict === "relevant"
          ? "generating on the CPU…"
          : run.steps.length === 0
            ? "retrieving…"
            : "working…";
  return (
    <aside className="panel trace" aria-label="Pipeline trace">
      <h3>Trace</h3>
      <ol className="steps">
        <li className="step" data-testid="step-security">
          <div className="step-head">
            <span className="step-name">security</span>
            <span className="step-summary">
              {blocked
                ? `blocked · ${run.error?.detail}`
                : sec
                  ? [
                      "passed",
                      sec.masked.length ? `masked ${sec.masked.join(", ")}` : "no PII",
                      sec.stripped ? `${sec.stripped} delimiter trick(s) removed` : "",
                      sec.output_masked?.length ? `answer masked ${sec.output_masked.join(", ")}` : "",
                    ]
                      .filter(Boolean)
                      .join(" · ")
                  : run.error
                    ? "–"
                    : "checking…"}
            </span>
          </div>
        </li>
        {run.cached !== null && (
          <li className="step" data-testid="step-cache">
            <div className="step-head">
              <span className="step-name">cache</span>
              <span className="step-summary">{run.cached ? "hit · served without a model call" : "miss"}</span>
            </div>
          </li>
        )}
        {run.steps.map((s, i) => (
          <StepRow key={`${s.name}-${i}`} step={s} />
        ))}
        {waitingFor && <li className="step pending pulse">{waitingFor}</li>}
      </ol>
      {run.final && (
        <p className="muted mono">
          request {run.final.request_id} · index {run.final.index_version} · prompt v{run.final.prompt_version}
        </p>
      )}
    </aside>
  );
}
