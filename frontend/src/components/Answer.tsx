import type { QueryResponse, Source } from "../api";
import type { RunState } from "../run";
import { resultKind } from "../run";

const TAG = /(\[S\d+\])/g;

/** Answer text with each [S#] tag turned into a chip that opens the cited page. */
export function AnswerText({
  text,
  sources,
  onCite,
}: {
  text: string;
  sources: Source[];
  onCite: (s: Source) => void;
}) {
  const byTag = new Map(sources.map((s) => [s.tag, s]));
  return (
    <p className="answer-text">
      {text.split(TAG).map((part, i) => {
        const tag = part.match(/^\[(S\d+)\]$/)?.[1];
        const source = tag ? byTag.get(tag) : undefined;
        if (!source) return <span key={i}>{part}</span>;
        return (
          <button
            key={i}
            type="button"
            className="chip"
            title={`${source.title}, p. ${source.page}`}
            aria-label={`Open source ${tag}: ${source.title}, page ${source.page}`}
            onClick={() => onCite(source)}
          >
            {tag}
          </button>
        );
      })}
    </p>
  );
}

function StatusLine({ final }: { final: QueryResponse }) {
  const kind = resultKind(final);
  const secs = final.latency_ms >= 1000 ? `${(final.latency_ms / 1000).toFixed(1)} s` : `${Math.round(final.latency_ms)} ms`;
  return (
    <p className="status-line" data-testid="status-line">
      <span className={`kind kind-${kind}`}>{kind}</span> · {secs} · {final.model_used ?? "no model call"}
      {final.model_used ? " (local)" : ""} · {final.cached ? "from cache" : "not cached"}
      {final.retries ? ` · ${final.retries} corrective step${final.retries > 1 ? "s" : ""}` : ""}
      {final.fallback_reason ? ` · fallback model: ${final.fallback_reason}` : ""}
    </p>
  );
}

export function AnswerPanel({ run, onCite }: { run: RunState; onCite: (s: Source) => void }) {
  const { final, status } = run;
  if (status === "idle") {
    return (
      <section className="panel answer-panel empty">
        <p>Ask about the indexed AI/ML papers, or pick a preset above.</p>
      </section>
    );
  }
  if (status === "error" && run.error) {
    const e = run.error;
    const blocked = e.code === "blocked_input";
    return (
      <section className={`panel answer-panel ${blocked ? "blocked" : "failed"}`} role="alert">
        <h2>{blocked ? "Blocked by the security layer" : titleFor(e.code)}</h2>
        <p>{e.detail}</p>
        {blocked && <p className="muted">No retrieval or model call was made.</p>}
        {e.retryAfter ? <p className="muted">Try again in {e.retryAfter} s.</p> : null}
        {e.requestId && <p className="muted mono">request {e.requestId}</p>}
      </section>
    );
  }
  const text = final ? final.answer : run.streamed;
  const sources = final?.sources ?? [];
  return (
    <section className="panel answer-panel" aria-live="polite">
      <h2 className="question">{run.question}</h2>
      {final?.abstained && (
        <p className="declined" data-testid="declined">
          Declined: {final.abstain_reason}
        </p>
      )}
      {text ? (
        <AnswerText text={text} sources={sources} onCite={onCite} />
      ) : (
        <p className="muted pulse">
          {run.queuedPosition ? "Busy: you're next in the queue…" : "Retrieving, then generating on the CPU…"}
        </p>
      )}
      {status === "running" && text && <p className="muted pulse">generating (unchecked until it finishes)…</p>}
      {final && sources.length > 0 && (
        <div className="sources">
          <h3>Sources</h3>
          <ol>
            {sources.map((s) => (
              <li key={s.tag} className={s.cited ? "cited" : ""}>
                <button type="button" className="source-link" onClick={() => onCite(s)}>
                  <span className="tag">[{s.tag}]</span> {s.title} · p. {s.page} · {s.score.toFixed(2)}
                </button>
                {s.cited && <span className="badge">cited</span>}
              </li>
            ))}
          </ol>
        </div>
      )}
      {final && <StatusLine final={final} />}
    </section>
  );
}

function titleFor(code: string): string {
  return (
    {
      busy: "Busy: one question at a time",
      rate_limited: "Too many questions",
      not_ready: "Still starting",
      question_too_long: "Question too long",
      invalid_request: "Invalid question",
    }[code] ?? "Something went wrong"
  );
}
