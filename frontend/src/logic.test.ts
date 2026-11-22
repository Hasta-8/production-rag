import { describe, expect, it } from "vitest";
import { SSEParser } from "./api";
import { findHighlight, normalise } from "./highlight";
import { playReplay } from "./replay";
import { initialRun, resultKind, runReducer, type RunState } from "./run";

describe("SSEParser", () => {
  it("parses events split across chunks", () => {
    const p = new SSEParser();
    expect(p.feed('event: meta\ndata: {"cached": fal')).toEqual([]);
    const out = p.feed('se}\n\nevent: token\ndata: {"text": "Hi"}\n\nevent: tok');
    expect(out).toEqual([
      { event: "meta", data: { cached: false } },
      { event: "token", data: { text: "Hi" } },
    ]);
    expect(p.feed('en\ndata: {"text": "!"}\n\n')).toEqual([{ event: "token", data: { text: "!" } }]);
  });

  it("accepts CRLF line endings", () => {
    expect(new SSEParser().feed('event: final\r\ndata: {"a": 1}\r\n\r\n')).toEqual([
      { event: "final", data: { a: 1 } },
    ]);
  });
});

describe("findHighlight", () => {
  const items = ["Table 1: results.", "BEIR is a heteroge-", "neous benchmark of 18 re", "trieval datasets.", "Other text."];

  it("matches the snippet across items, ignoring hyphenation and spacing", () => {
    expect(findHighlight(items, "BEIR is a heterogeneous benchmark of 18 retrieval datasets")).toEqual([1, 2, 3]);
  });

  it("still finds the passage when its start is garbled", () => {
    const snippet = "B E I R  i5 a heterogeneous benchmark of 18 retrieval datasets for zero-shot evaluation";
    expect(findHighlight([...items, "for zero-shot evaluation"], snippet)).toContain(2);
  });

  it("returns nothing when the passage isn't on the page", () => {
    expect(findHighlight(items, "Mixtral routes each token to two experts")).toEqual([]);
  });

  it("normalises unicode and case", () => {
    expect(normalise("ﬁne-Tuning, 4-bit!")).toBe("finetuning4bit");
  });
});

describe("runReducer", () => {
  const step = (name: string, data = {}) => ({ type: "event" as const, event: { event: "step", data: { name, ms: 5, data } } });
  const token = (text: string) => ({ type: "event" as const, event: { event: "token", data: { text } } });

  it("restarts the streamed text when a retry generates again", () => {
    let s: RunState = runReducer(initialRun, { type: "start", question: "q", at: 0 });
    s = runReducer(s, token("I don't "));
    s = runReducer(s, token("know."));
    expect(s.streamed).toBe("I don't know.");
    s = runReducer(s, step("generate"));
    s = runReducer(s, step("focus"));
    s = runReducer(s, token("It is 42"));
    expect(s.streamed).toBe("It is 42");
  });

  it("shows a cached answer's recorded trace", () => {
    let s = runReducer(initialRun, { type: "start", question: "q", at: 0 });
    const final = { cached: true, trace: [{ name: "retrieve", ms: 1, data: {} }], security: { masked: [], stripped: 0, output_masked: [] } };
    s = runReducer(s, { type: "event", event: { event: "final", data: final } });
    expect(s.status).toBe("done");
    expect(s.steps.map((x) => x.name)).toEqual(["retrieve"]);
  });

  it("labels results live, cached or replay", () => {
    expect(resultKind(null)).toBeNull();
    expect(resultKind({ cached: false } as never)).toBe("live");
    expect(resultKind({ cached: true } as never)).toBe("cached");
    expect(resultKind({ cached: false, replay: true } as never)).toBe("replay");
  });
});

describe("playReplay", () => {
  it("replays events in order, marks the final answer as a replay, and can skip", async () => {
    const rec = {
      preset_id: "cited",
      events: [
        { t_ms: 0, event: "meta", data: {} },
        { t_ms: 60_000, event: "token", data: { text: "slow" } },
        { t_ms: 120_000, event: "final", data: { cached: false } },
      ],
    } as never;
    const seen: { event: string; data: unknown }[] = [];
    const started = Date.now();
    await playReplay(rec, (e) => seen.push(e), { skip: true });
    expect(Date.now() - started).toBeLessThan(1000);
    expect(seen.map((e) => e.event)).toEqual(["meta", "token", "final"]);
    expect(seen[2].data).toMatchObject({ replay: true, cached: false });
  });
});
