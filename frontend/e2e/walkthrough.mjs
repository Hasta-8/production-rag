// Records a silent, captioned walkthrough video of a LIVE run of every preset (the last
// fallback in design 9.5, for when the laptop or the app fails entirely).
//
//   deploy/local-demo/demo.sh start      # fresh start, so the first answers are live
//   cd frontend && node e2e/walkthrough.mjs
//
// Needs Playwright's ffmpeg once: npx playwright-core install ffmpeg
// Writes e2e/video/walkthrough.webm and marks.json (git-ignored: share it from a file host).
// Long CPU generations are marked and the caption says they're sped up; then
// scripts/make_video.py speeds up only those stretches (×4) to fit the 3–5 minute target.

import { mkdirSync, readdirSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const BASE = process.env.BASE ?? "http://127.0.0.1:8000";
const OUT = fileURLToPath(new URL("./video/", import.meta.url));
const TMP = `${OUT}tmp/`;
mkdirSync(TMP, { recursive: true });

const browser = await chromium.launch({ executablePath: process.env.CHROME ?? "/usr/bin/google-chrome" });
const context = await browser.newContext({
  viewport: { width: 1440, height: 900 },
  recordVideo: { dir: TMP, size: { width: 1440, height: 900 } },
});
const page = await context.newPage();
const t0 = Date.now(); // the video starts with the page
const pause = (ms) => page.waitForTimeout(ms);
const marks = []; // [start_s, end_s] stretches to speed up
const SPEED = 4;
async function fast(note, waitFor) {
  await caption(`${note} (generation on the CPU, sped up ×${SPEED} in this video)`);
  const start = (Date.now() - t0) / 1000;
  await waitFor();
  marks.push([start, (Date.now() - t0) / 1000]);
}

/** A caption bar at the bottom of the page, standing in for narration. */
async function caption(text) {
  await page.evaluate((t) => {
    let el = document.getElementById("walkthrough-caption");
    if (!el) {
      el = document.createElement("div");
      el.id = "walkthrough-caption";
      Object.assign(el.style, {
        position: "fixed", left: "50%", bottom: "18px", transform: "translateX(-50%)",
        maxWidth: "1100px", padding: "10px 18px", borderRadius: "10px", zIndex: 9999,
        background: "rgba(20,28,38,0.92)", color: "#fff", font: "16px/1.4 system-ui, sans-serif",
        boxShadow: "0 4px 18px rgba(0,0,0,0.25)",
      });
      document.body.appendChild(el);
    }
    el.textContent = t;
  }, text);
}
const preset = (label) =>
  page.getByRole("navigation", { name: "Preset questions" }).getByRole("button", { name: label, exact: true }).click();
const done = () =>
  page.waitForFunction(
    () => document.querySelector('[data-testid="status-line"], [role="alert"], [data-testid="declined"]'),
    null,
    { timeout: 240_000 },
  );

await page.goto(BASE);
await page.getByText("one question at a time").waitFor({ timeout: 60_000 });
await caption("Production RAG: cited answers over 28 AI/ML papers. Everything runs locally on a laptop CPU (Qwen3-1.7B).");
await pause(5000);

await caption("1 · Cited answer (live). Retrieval and reranking appear in the trace within seconds; the CPU then generates.");
await preset("Cited answer");
await page.getByTestId("step-retrieve").waitFor({ timeout: 30_000 });
await pause(3000);
await page.getByRole("button", { name: /show chunk ranks/ }).first().click();
await caption("Each chunk's rank in dense search, BM25 and the fused list, and its cross-encoder score, while the answer streams.");
await pause(5000);
await fast("Qwen3-1.7B reads the 3 best chunks and writes a cited answer", done);
await caption("The checked answer cites [S1]. Click the chip: the PDF opens at the cited page with the passage highlighted.");
await pause(3000);
await page.locator(".chip").first().click();
await page.locator(".viewer .hl").first().waitFor({ timeout: 30_000 });
await pause(6000);
await page.getByRole("button", { name: "Close the source viewer" }).click();

await caption("2 · Exact term: compare mode, retrieval only. Dense search ranks LLaMA's data-sources page 7th; hybrid ranks it 1st.");
await preset("Exact term");
await page.getByTestId("column-hybrid_rerank").locator("li").first().waitFor({ timeout: 60_000 });
await pause(8000);
await page.getByRole("tab", { name: "Single" }).click();

await caption("3 · Vague question (live): the first retrieval is too weak, so the agent rewrites the query and retrieves again.");
await preset("Vague question");
await fast("Weak first retrieval: the agent rewrites the query", () => page.getByTestId("step-rewrite").waitFor({ timeout: 120_000 }));
await pause(3000);
await fast("Second retrieval, then the model answers", done);
await caption("Rewrite → second retrieval passes the gate (1.10 → 3.98) → answered from the Late Chunking paper.");
await pause(6000);

await caption("4 · Not in the corpus: two bounded rewrites, then a clean refusal with the low relevance score as the reason.");
await preset("Not in the corpus");
await fast("4 · Not in the corpus: weak evidence, two bounded rewrites", done);
await caption("4 · A clean refusal, with the low relevance score as the reason. No model answer was attempted.");
await pause(6000);

await caption("5 · Injection attempt: blocked by the security layer before any retrieval or model call.");
await preset("Injection attempt");
await page.getByRole("alert").waitFor({ timeout: 10_000 });
await pause(5000);

await caption("6 · Repeat: the first question again, served from the versioned answer cache in milliseconds.");
await preset("Repeat");
await done();
await pause(5000);

await caption("Fallback: Replay mode plays recorded runs, always labelled REPLAY, for when the model is too slow or down.");
await page.getByRole("button", { name: /Replay mode off/ }).click();
await preset("Cited answer");
await page.getByRole("button", { name: "skip to the end" }).click({ timeout: 10_000 }).catch(() => {});
await page.getByTestId("status-line").waitFor({ timeout: 60_000 });
await pause(5000);
await caption("Fully local: bge-small embeddings, BM25, a MiniLM cross-encoder and Qwen3-1.7B. No data leaves the machine.");
await pause(5000);

await context.close(); // finalises the video file
await browser.close();
const [file] = readdirSync(TMP).filter((f) => f.endsWith(".webm"));
renameSync(`${TMP}${file}`, `${OUT}walkthrough.webm`);
writeFileSync(`${OUT}marks.json`, JSON.stringify({ speed: SPEED, fast: marks }, null, 1));
rmSync(TMP, { recursive: true, force: true });
console.log(`saved ${OUT}walkthrough.webm`);
