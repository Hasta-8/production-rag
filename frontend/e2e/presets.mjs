// Phase 6 exit gate: every verified preset shows its intended behaviour end to end, in a
// real browser, against the real service (real models). Uses the installed Chrome.
//
//   cd frontend && npm run build && cd .. && uv run rag serve --demo   # wait for /ready
//   cd frontend && npm run e2e
//
// BASE (default http://127.0.0.1:8000), CHROME (default /usr/bin/google-chrome).
// Screenshots go to e2e/screenshots/. Exit code = number of failed checks.

import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const BASE = process.env.BASE ?? "http://127.0.0.1:8000";
const SHOTS = fileURLToPath(new URL("./screenshots/", import.meta.url)); // decodes "%20"
mkdirSync(SHOTS, { recursive: true });
const ANSWER_TIMEOUT = 240_000; // one CPU generation, plus up to two rewrites

let failed = 0;
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  (${detail})` : ""}`);
  if (!ok) failed++;
};

const browser = await chromium.launch({
  executablePath: process.env.CHROME ?? "/usr/bin/google-chrome",
  headless: true,
});
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const consoleErrors = [];
// Chrome logs every 4xx response; the injection preset's 400 is expected, nothing else is.
const expected = (text) => /Failed to load resource: .* status of 400/.test(text);
page.on("console", (m) => m.type() === "error" && !expected(m.text()) && consoleErrors.push(m.text()));
page.on("pageerror", (e) => consoleErrors.push(String(e)));

const presets = (await (await fetch(`${BASE}/v1/demo/presets`)).json()).presets.filter((p) => p.verified !== false);
const has = (id) => presets.some((p) => p.id === id);
const labelOf = (id) => presets.find((p) => p.id === id).label;
const shot = (name) => page.screenshot({ path: `${SHOTS}${name}.png`, fullPage: true });
const clickPreset = async (id) => {
  await page.getByRole("navigation", { name: "Preset questions" }).getByRole("button", { name: labelOf(id), exact: true }).click();
};
const waitDone = () =>
  page.waitForFunction(
    () => document.querySelector('[data-testid="status-line"], [role="alert"], [data-testid="declined"]'),
    null,
    { timeout: ANSWER_TIMEOUT },
  );
const stepNames = () => page.$$eval('.trace [data-testid^="step-"]', (els) => els.map((e) => e.dataset.testid.slice(5)));

await page.goto(BASE);
await page.getByText("one question at a time").waitFor({ timeout: 60_000 });
check("UI served from the API's origin, banner shows the service state", true);
const banner = await page.getByTestId("banner").innerText();
check("banner shows demo mode and the local model", /DEMO MODE/.test(banner) && /Qwen3/.test(banner), banner.replace(/\s+/g, " "));

if (has("cited")) {
  console.log("-- cited answer (a live generation)");
  const t0 = Date.now();
  await clickPreset("cited");
  // Trace steps should appear well before the answer finishes.
  await page.getByTestId("step-retrieve").waitFor({ timeout: 30_000 });
  const retrieveAt = Date.now() - t0;
  await waitDone();
  const total = Date.now() - t0;
  check("retrieval shows in the trace before the answer is done", retrieveAt < total / 2, `retrieve ${retrieveAt} ms, answer ${total} ms`);
  const answer = await page.locator(".answer-text").innerText();
  check("answer cites a source and says 18", /18/.test(answer) && (await page.locator(".chip").count()) > 0, answer);
  check("status line says live, not cached", /^LIVE/i.test(await page.getByTestId("status-line").innerText()));
  await shot("1-cited");
  await page.locator(".chip").first().click();
  await page.locator(".viewer canvas").waitFor({ timeout: 30_000 });
  await page.waitForFunction(() => document.querySelector(".viewer")?.innerText.includes("cited page") && !document.querySelector(".viewer .overlay"), null, { timeout: 30_000 });
  const boxes = await page.locator(".viewer .hl").count();
  const nav = await page.locator(".viewer-nav").innerText();
  check("chip opens the PDF at the cited page", /cited page/.test(nav), nav.replace(/\s+/g, " "));
  check("the cited passage is highlighted", boxes > 0, `${boxes} highlight boxes`);
  await page.getByRole("button", { name: /show chunk ranks/ }).first().click();
  check("chunk ranks table expands (dense | BM25 | fused | rerank)", (await page.locator("table.hits tbody tr").count()) > 0);
  await shot("2-viewer");
}

if (has("exact_term")) {
  console.log("-- exact term (compare mode, retrieval only)");
  await clickPreset("exact_term");
  await page.getByTestId("column-hybrid_rerank").locator("li").first().waitFor({ timeout: 60_000 });
  const firstEvidence = async (mode) => {
    const items = await page.getByTestId(`column-${mode}`).locator(".hit-title").allInnerTexts();
    const i = items.findIndex((t) => /LLaMA/.test(t) && /p\. 2$/.test(t.trim()));
    return i === -1 ? null : i + 1;
  };
  const [d, h, hr] = [await firstEvidence("dense"), await firstEvidence("hybrid"), await firstEvidence("hybrid_rerank")];
  check("dense ranks the evidence page lower than hybrid does", (d ?? 99) > (h ?? 99) && h !== null, `LLaMA p. 2: dense #${d}, hybrid #${h}, hybrid+rerank #${hr}`);
  await shot("3-compare");
  await page.getByRole("tab", { name: "Single" }).click();
}

if (has("vague")) {
  console.log("-- vague question (rewrite rescues it)");
  await clickPreset("vague");
  await waitDone();
  const steps = await stepNames();
  const declined = await page.getByTestId("declined").count();
  check("trace shows a rewrite and a second retrieval", steps.includes("rewrite") && steps.filter((s) => s === "retrieve").length >= 2, steps.join(" → "));
  check("the rewritten question is answered", declined === 0 && (await page.locator(".chip").count()) > 0);
  await shot("4-vague");
}

if (has("not_in_corpus")) {
  console.log("-- not in the corpus");
  await clickPreset("not_in_corpus");
  await waitDone();
  const reason = await page.getByTestId("declined").innerText().catch(() => "");
  check("declined, with the low relevance score as the reason", /below the threshold/.test(reason), reason);
  check("bounded rewrites are visible in the trace", (await stepNames()).includes("rewrite"), (await stepNames()).join(" → "));
  await shot("5-not-in-corpus");
}

if (has("injection")) {
  console.log("-- injection");
  await clickPreset("injection");
  await page.getByRole("alert").waitFor({ timeout: 10_000 });
  const text = await page.getByRole("alert").innerText();
  check("blocked by the security layer before any model call", /Blocked by the security layer/.test(text) && /No retrieval or model call/.test(text));
  check("trace shows the security block", /blocked/.test(await page.getByTestId("step-security").innerText()));
  await shot("6-injection");
}

if (has("repeat")) {
  console.log("-- repeat (cache)");
  const t0 = Date.now();
  await clickPreset("repeat");
  await waitDone();
  const status = await page.getByTestId("status-line").innerText();
  check("served from the cache in under a second", /^CACHED/i.test(status) && Date.now() - t0 < 3000, status);
  check("trace shows the cache hit", /hit/.test(await page.getByTestId("step-cache").innerText()));
  await shot("7-repeat");
}

check("no browser console errors", consoleErrors.length === 0, consoleErrors.slice(0, 3).join(" | "));
await browser.close();
console.log(`== ${failed === 0 ? "all checks passed" : `${failed} failed`}`);
process.exit(failed);
