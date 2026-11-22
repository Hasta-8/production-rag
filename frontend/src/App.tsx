import { lazy, Suspense, useCallback, useEffect, useReducer, useRef, useState } from "react";
import type { Preset, Ready, Source } from "./api";
import { api, ApiError, streamQuery } from "./api";
import { AnswerPanel } from "./components/Answer";
import { Banner, PresetBar, type View } from "./components/Banner";
import { CompareView } from "./components/CompareView";
import { TracePanel } from "./components/TracePanel";
import { initialRun, runReducer } from "./run";

const SourceViewer = lazy(() => import("./components/SourceViewer"));

export default function App() {
  const [ready, setReady] = useState<Ready | null>(null);
  const [readyError, setReadyError] = useState<string | null>(null);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [view, setView] = useState<View>("single");
  const [question, setQuestion] = useState("");
  const [run, dispatch] = useReducer(runReducer, initialRun);
  const [viewing, setViewing] = useState<Source | null>(null);
  const [compareKey, setCompareKey] = useState(0);
  const [hint, setHint] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  // /ready stays 503 while the model loads (D8): poll until it is up.
  useEffect(() => {
    let stop = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const r = await api.ready();
        if (!stop) {
          setReady(r);
          setReadyError(null);
        }
      } catch (e) {
        if (stop) return;
        setReadyError(e instanceof ApiError ? `starting: ${e.detail}` : "the service can't be reached");
        timer = setTimeout(poll, 3000);
      }
    };
    void poll();
    api
      .presets()
      .then((p) => setPresets(p.presets.filter((x) => x.verified !== false)))
      .catch(() => setPresets([]));
    return () => {
      stop = true;
      clearTimeout(timer);
    };
  }, []);

  const ask = useCallback(async (q: string) => {
    const text = q.trim();
    if (!text) return;
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    setViewing(null);
    dispatch({ type: "start", question: text, at: Date.now() });
    try {
      await streamQuery(text, { onEvent: (event) => dispatch({ type: "event", event }) }, ctrl.signal);
    } catch (e) {
      if (ctrl.signal.aborted) return;
      dispatch({
        type: "error",
        error: e instanceof ApiError ? e : new ApiError(0, "network", "The service can't be reached."),
      });
    }
  }, []);

  const pick = (p: Preset) => {
    setQuestion(p.question);
    setHint(p.shows ?? null);
    if (p.view === "compare") {
      setView("compare");
      setCompareKey((k) => k + 1);
    } else {
      setView("single");
      void ask(p.question);
    }
  };

  const running = run.status === "running";
  return (
    <div className="app">
      <Banner ready={ready} readyError={readyError} run={run} view={view} onView={setView} />
      <PresetBar presets={presets} onPick={pick} disabled={running || !ready} />
      {hint && <p className="hint">Shows: {hint}</p>}
      {view === "compare" ? (
        <CompareView question={question} onQuestion={setQuestion} runKey={compareKey} />
      ) : (
        <>
          <form
            className="ask"
            onSubmit={(e) => {
              e.preventDefault();
              setHint(null);
              void ask(question);
            }}
          >
            <input
              aria-label="Question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask about the indexed AI/ML papers…"
              maxLength={ready?.demo_mode ? 500 : 4000}
            />
            <button type="submit" disabled={running || !question.trim() || !ready}>
              {running ? "Answering…" : "Ask"}
            </button>
          </form>
          <main className={`workspace ${viewing ? "with-viewer" : ""}`}>
            <AnswerPanel run={run} onCite={setViewing} />
            <TracePanel run={run} />
            {viewing && (
              <Suspense fallback={<aside className="panel viewer">loading the viewer…</aside>}>
                <SourceViewer source={viewing} onClose={() => setViewing(null)} />
              </Suspense>
            )}
          </main>
        </>
      )}
      <footer className="muted small">
        Fully local: bge-small embeddings, BM25, a MiniLM cross-encoder and Qwen3-1.7B on this laptop's CPU. No
        data leaves the machine.
      </footer>
    </div>
  );
}
