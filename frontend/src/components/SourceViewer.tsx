// Citation viewer: PDF.js renders the cited page, and the cited passage is highlighted.
// Loaded lazily (PDF.js is the biggest dependency), and mocked in unit tests (jsdom has
// no canvas); the end-to-end check runs it in a real browser.

import * as pdfjs from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import { useEffect, useRef, useState } from "react";
import type { Source } from "../api";
import { findHighlight } from "../highlight";

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

interface Box {
  left: number;
  top: number;
  width: number;
  height: number;
}

export default function SourceViewer({ source, onClose }: { source: Source; onClose: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [page, setPage] = useState(source.page);
  const [pages, setPages] = useState<number | null>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const [found, setFound] = useState<boolean | null>(null);

  useEffect(() => setPage(source.page), [source]);

  useEffect(() => {
    if (!source.file_url) return;
    let cancelled = false;
    const url = source.file_url.split("#")[0];
    const task = pdfjs.getDocument({ url });
    setState("loading");
    (async () => {
      try {
        const doc = await task.promise;
        if (cancelled) return;
        setPages(doc.numPages);
        const p = await doc.getPage(Math.min(Math.max(page, 1), doc.numPages));
        const width = canvasRef.current?.parentElement?.clientWidth ?? 640;
        const base = p.getViewport({ scale: 1 });
        const viewport = p.getViewport({ scale: Math.min(2, (width - 8) / base.width) });
        const canvas = canvasRef.current;
        if (!canvas || cancelled) return;
        const ratio = window.devicePixelRatio || 1;
        canvas.width = Math.floor(viewport.width * ratio);
        canvas.height = Math.floor(viewport.height * ratio);
        canvas.style.width = `${viewport.width}px`;
        canvas.style.height = `${viewport.height}px`;
        await p.render({
          canvas,
          viewport,
          transform: ratio === 1 ? undefined : [ratio, 0, 0, ratio, 0, 0],
        }).promise;
        if (cancelled) return;
        // Highlight only on the cited page (the user may page away from it).
        if (page === source.page) {
          const content = await p.getTextContent();
          const items = content.items.filter((it): it is (typeof content.items)[number] & { str: string; transform: number[]; width: number; height: number } => "str" in it);
          const hit = findHighlight(items.map((it) => it.str), source.snippet);
          setFound(hit.length > 0);
          setBoxes(
            hit.map((i) => {
              const it = items[i];
              const [x, y] = [it.transform[4], it.transform[5]];
              const h = it.height || Math.hypot(it.transform[2], it.transform[3]);
              const [x1, y1] = viewport.convertToViewportPoint(x, y);
              const [x2, y2] = viewport.convertToViewportPoint(x + it.width, y + h);
              return {
                left: Math.min(x1, x2),
                top: Math.min(y1, y2),
                width: Math.abs(x2 - x1),
                height: Math.abs(y2 - y1),
              };
            }),
          );
        } else {
          setBoxes([]);
        }
        setState("ready");
      } catch {
        if (!cancelled) setState("error");
      }
    })();
    return () => {
      cancelled = true;
      task.destroy();
    };
  }, [source, page]);

  useEffect(() => {
    if (state === "ready" && boxes.length) {
      document.querySelector(".pdf-wrap .hl")?.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }, [state, boxes]);

  return (
    <aside className="panel viewer" aria-label="Source viewer" data-testid="viewer">
      <div className="viewer-head">
        <div>
          <strong>[{source.tag}]</strong> {source.title}
        </div>
        <button type="button" className="link" onClick={onClose} aria-label="Close the source viewer">
          close
        </button>
      </div>
      {!source.file_url ? (
        <p className="muted">This paper's licence doesn't allow redistribution, so it isn't served here.</p>
      ) : (
        <>
          <div className="viewer-nav">
            <button type="button" disabled={page <= 1} onClick={() => setPage(page - 1)}>
              ‹ prev
            </button>
            <span>
              page {page}
              {pages ? ` of ${pages}` : ""} {page === source.page ? "· cited page" : ""}
            </span>
            <button type="button" disabled={pages !== null && page >= pages} onClick={() => setPage(page + 1)}>
              next ›
            </button>
            {page !== source.page && (
              <button type="button" className="link" onClick={() => setPage(source.page)}>
                back to the cited page
              </button>
            )}
          </div>
          {page === source.page && found === false && state === "ready" && (
            <p className="muted">The passage couldn't be located in the PDF's text layer; it is on this page.</p>
          )}
          <div className="pdf-wrap">
            <canvas ref={canvasRef} />
            {boxes.map((b, i) => (
              <div key={i} className="hl" style={{ left: b.left, top: b.top, width: b.width, height: b.height }} />
            ))}
            {state === "loading" && <p className="muted overlay">loading page…</p>}
            {state === "error" && <p className="muted overlay">The PDF couldn't be loaded.</p>}
          </div>
        </>
      )}
    </aside>
  );
}
