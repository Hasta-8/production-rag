// The state of one question, built from stream events by a pure reducer.

import { ApiError } from "./api";
import type { QueryResponse, Security, StreamEvent, TraceStep } from "./api";

export type RunStatus = "idle" | "running" | "done" | "error";

export interface RunState {
  status: RunStatus;
  question: string;
  requestId: string | null;
  cached: boolean | null;
  security: Security | null;
  queuedPosition: number | null;
  steps: TraceStep[];
  streamed: string; // tokens of the current generation attempt (unchecked)
  attempt: number; // generations finished so far; a new attempt restarts `streamed`
  tokenAttempt: number;
  final: QueryResponse | null;
  error: ApiError | null;
  startedAt: number | null;
}

export const initialRun: RunState = {
  status: "idle",
  question: "",
  requestId: null,
  cached: null,
  security: null,
  queuedPosition: null,
  steps: [],
  streamed: "",
  attempt: 0,
  tokenAttempt: -1,
  final: null,
  error: null,
  startedAt: null,
};

export type RunAction =
  | { type: "start"; question: string; at: number }
  | { type: "event"; event: StreamEvent }
  | { type: "error"; error: ApiError };

export function runReducer(state: RunState, action: RunAction): RunState {
  switch (action.type) {
    case "start":
      return { ...initialRun, status: "running", question: action.question, startedAt: action.at };
    case "error":
      return { ...state, status: "error", error: action.error, queuedPosition: null };
    case "event":
      return applyEvent(state, action.event);
  }
}

function applyEvent(state: RunState, { event, data }: StreamEvent): RunState {
  const d = data as Record<string, unknown>;
  switch (event) {
    case "meta":
      return {
        ...state,
        requestId: (d.request_id as string) ?? null,
        cached: Boolean(d.cached),
        security: (d.security as Security) ?? null,
      };
    case "queued":
      return { ...state, queuedPosition: (d.position as number) ?? 1 };
    case "step": {
      const step = data as TraceStep;
      return {
        ...state,
        queuedPosition: null,
        steps: [...state.steps, step],
        attempt: state.attempt + (step.name === "generate" ? 1 : 0),
      };
    }
    case "token": {
      const text = (d.text as string) ?? "";
      // Tokens after a finished generation belong to a retry: start its text afresh.
      const fresh = state.tokenAttempt !== state.attempt;
      return {
        ...state,
        queuedPosition: null,
        streamed: (fresh ? "" : state.streamed) + text,
        tokenAttempt: state.attempt,
      };
    }
    case "final": {
      const final = data as QueryResponse;
      return {
        ...state,
        status: "done",
        final,
        cached: final.cached,
        security: final.security ?? state.security,
        // A cached answer arrives without step events: show its recorded trace.
        steps: state.steps.length ? state.steps : (final.trace ?? []),
        queuedPosition: null,
      };
    }
    case "error":
      return {
        ...state,
        status: "error",
        error: new ApiError(
          500,
          (d.error as string) ?? "internal_error",
          (d.detail as string) ?? "Something went wrong.",
          (d.request_id as string) ?? null,
        ),
      };
    default:
      return state;
  }
}

/** live, cached or replay: a replay is never presented as live. */
export function resultKind(final: QueryResponse | null): "live" | "cached" | "replay" | null {
  if (!final) return null;
  if (final.replay) return "replay";
  return final.cached ? "cached" : "live";
}
