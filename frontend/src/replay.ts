// Plays a recorded run back through the same reducer as a live stream, with the recorded
// timing, so the trace and the answer build up exactly as they did live. Every final
// answer is marked `replay`, so the UI can never present it as live.

import type { Replay, StreamEvent } from "./api";

export interface PlayControl {
  skip: boolean; // set to true to play the rest at once
}

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve) => {
    if (ms <= 0 || signal?.aborted) return resolve();
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(t);
      resolve();
    });
  });

export async function playReplay(
  rec: Replay,
  onEvent: (e: StreamEvent) => void,
  control: PlayControl,
  signal?: AbortSignal,
): Promise<void> {
  let last = 0;
  for (const e of rec.events ?? []) {
    if (signal?.aborted) return;
    if (!control.skip) await sleep(e.t_ms - last, signal);
    last = e.t_ms;
    if (signal?.aborted) return;
    const data =
      e.event === "final" ? { ...(e.data as object), replay: true, request_id: `replay of ${rec.preset_id}` } : e.data;
    onEvent({ event: e.event, data });
  }
}
