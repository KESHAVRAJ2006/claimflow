"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import type { RunEvent, RunEventType } from "@/lib/types";

const EVENT_TYPES: RunEventType[] = [
  "run_queued",
  "node_started",
  "node_finished",
  "tool_call_started",
  "tool_call_finished",
  "run_completed",
  "run_failed",
  "run_unavailable",
];
const TERMINAL = new Set<RunEventType>(["run_completed", "run_failed", "run_unavailable"]);

export type StreamState = "connecting" | "open" | "closed" | "error";

/**
 * Subscribe to a claim's run over Server-Sent Events.
 *
 * EventSource reconnects by itself after a dropped connection and sends Last-Event-ID, so the server resends only
 * what was missed; events are still de-duplicated by (run, id) in case a reconnect overlaps. When the run ends, the
 * claim's cached data is refreshed so the decision card and risk panel pick up the final result.
 *
 * @param claimId  The claim to follow.
 * @param session  Change it to reconnect from scratch (e.g. after starting a new run).
 */
export function useRunStream(claimId: string, session: number) {
  const queryClient = useQueryClient();
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [state, setState] = useState<StreamState>("connecting");

  useEffect(() => {
    setEvents([]);
    setState("connecting");
    const seen = new Set<string>();
    const source = new EventSource(api.streamUrl(claimId));

    const onEvent = (message: MessageEvent<string>) => {
      const event = JSON.parse(message.data) as RunEvent;
      const key = `${event.run_id ?? "replay"}:${event.id}`;
      if (seen.has(key)) return;
      seen.add(key);
      setEvents((previous) => [...previous, event]);
      if (TERMINAL.has(event.type)) {
        // Close ourselves: otherwise EventSource would treat the server ending the stream as a drop and reconnect.
        source.close();
        setState("closed");
        void queryClient.invalidateQueries({ queryKey: ["claim", claimId] });
        void queryClient.invalidateQueries({ queryKey: ["claims"] });
        void queryClient.invalidateQueries({ queryKey: ["metrics"] });
      }
    };

    EVENT_TYPES.forEach((type) => source.addEventListener(type, onEvent as EventListener));
    source.onopen = () => setState("open");
    source.onerror = () => {
      // CONNECTING means the browser is retrying on its own; CLOSED means it gave up (e.g. a 404 from the server).
      if (source.readyState === EventSource.CLOSED) setState((current) => (current === "closed" ? current : "error"));
    };
    return () => source.close();
  }, [claimId, session, queryClient]);

  return { events, state };
}
