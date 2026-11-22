import type { Preset, Ready } from "../api";
import type { RunState } from "../run";

export type View = "single" | "compare";

export function Banner({
  ready,
  readyError,
  run,
  view,
  onView,
}: {
  ready: Ready | null;
  readyError: string | null;
  run: RunState;
  view: View;
  onView: (v: View) => void;
}) {
  let queue = "idle";
  if (run.status === "running") queue = run.queuedPosition ? `busy · you're next (#${run.queuedPosition})` : "answering";
  if (run.error?.code === "busy") queue = "busy · another question is being answered";
  return (
    <header className="banner">
      <div className="brand">Production RAG</div>
      <div className="facts" data-testid="banner">
        {ready ? (
          <>
            {ready.demo_mode && <strong className="demo">DEMO MODE</strong>}
            <span>local models · {ready.models.primary}</span>
            <span>{ready.documents} papers</span>
            <span>one question at a time</span>
            {ready.demo_mode && <span>uploads off</span>}
            <span className={`queue ${queue.startsWith("busy") ? "warn" : ""}`}>queue: {queue}</span>
          </>
        ) : (
          <span className="warn">{readyError ?? "connecting…"}</span>
        )}
      </div>
      <div className="views" role="tablist" aria-label="View">
        {(["single", "compare"] as const).map((v) => (
          <button
            key={v}
            role="tab"
            aria-selected={view === v}
            className={view === v ? "active" : ""}
            onClick={() => onView(v)}
          >
            {v === "single" ? "Single" : "Compare"}
          </button>
        ))}
      </div>
    </header>
  );
}

export function PresetBar({
  presets,
  onPick,
  disabled,
}: {
  presets: Preset[];
  onPick: (p: Preset) => void;
  disabled: boolean;
}) {
  if (!presets.length) return null;
  return (
    <nav className="presets" aria-label="Preset questions">
      {presets.map((p) => (
        <button key={p.id} type="button" title={p.shows} disabled={disabled} onClick={() => onPick(p)}>
          {p.label}
        </button>
      ))}
    </nav>
  );
}
