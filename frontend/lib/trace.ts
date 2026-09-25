/**
 * Turns the SSE event stream into the two-level trace the UI draws: nodes, and under the investigator, its tool
 * calls. A pure function of the events, so a live run and a replayed one render identically, and a reconnect that
 * re-delivers events can't duplicate entries.
 */
import type { JsonObject, RunEvent } from "@/lib/types";

export type NodeName = "intake" | "investigator" | "rules" | "decision" | "reflection" | "route_final";

export interface TraceTool {
  callId: string;
  tool: string;
  args: JsonObject;
  round: number;
  step: number;
  status: "running" | "ok" | "error";
  summary?: string;
  latencyMs?: number;
}

export interface TraceNode {
  key: string;
  node: NodeName;
  round: number;
  status: "running" | "done";
  latencyMs?: number;
  output?: JsonObject;
  tools: TraceTool[];
}

export interface TraceResult {
  nodes: TraceNode[];
  /** The terminal event, once the run has ended. */
  end: RunEvent | null;
  queued: boolean;
  replayed: boolean;
}

/** Which nodes are LLM agents and which are deterministic code: shown on every trace entry. */
export const NODE_KIND: Record<NodeName, "agent" | "deterministic"> = {
  intake: "agent",
  investigator: "agent",
  rules: "deterministic",
  decision: "agent",
  reflection: "agent",
  route_final: "deterministic",
};

export const NODE_LABEL: Record<NodeName, string> = {
  intake: "Intake",
  investigator: "Investigator",
  rules: "Rules engine",
  decision: "Decision",
  reflection: "Reflection",
  route_final: "Final routing",
};

const TERMINAL = new Set(["run_completed", "run_failed", "run_unavailable"]);

export function buildTrace(events: readonly RunEvent[]): TraceResult {
  const nodes: TraceNode[] = [];
  let end: RunEvent | null = null;
  let queued = false;
  let replayed = false;

  for (const event of [...events].sort((a, b) => a.id - b.id)) {
    replayed ||= Boolean(event.replayed);
    if (event.type === "run_queued") queued = true;
    else if (TERMINAL.has(event.type)) end = event;
    else if (event.type === "node_started" && event.node) {
      const round = event.round ?? 1;
      nodes.push({ key: `${event.node}-${round}-${nodes.length}`, node: event.node as NodeName, round, status: "running", tools: [] });
    } else if (event.type === "node_finished" && event.node) {
      const round = event.round ?? 1;
      let target = [...nodes].reverse().find((n) => n.node === event.node && n.round === round && n.status === "running");
      if (!target) {
        // Joined mid-node (no node_started seen): still show the finished node.
        target = { key: `${event.node}-${round}-${nodes.length}`, node: event.node as NodeName, round, status: "running", tools: [] };
        nodes.push(target);
      }
      target.status = "done";
      target.latencyMs = event.latency_ms;
      target.output = event.output;
    } else if ((event.type === "tool_call_started" || event.type === "tool_call_finished") && event.call_id) {
      const investigator = [...nodes].reverse().find((n) => n.node === "investigator");
      if (!investigator) continue;
      let tool = investigator.tools.find((t) => t.callId === event.call_id);
      if (!tool) {
        tool = {
          callId: event.call_id,
          tool: event.tool ?? "unknown",
          args: event.args ?? {},
          round: event.round ?? investigator.round,
          step: event.step ?? 1,
          status: "running",
        };
        investigator.tools.push(tool);
      }
      if (event.type === "tool_call_finished") {
        tool.status = event.status === "error" ? "error" : "ok";
        tool.summary = event.result_summary;
        tool.latencyMs = event.latency_ms;
      }
    }
  }
  return { nodes, end, queued, replayed };
}

// ---- one-line summaries of node outputs (outputs are untyped JSON; read defensively) ---------------------------

const num = (value: unknown) => (typeof value === "number" ? value : undefined);
const str = (value: unknown) => (typeof value === "string" ? value : undefined);
const arr = (value: unknown) => (Array.isArray(value) ? value : []);
const obj = (value: unknown) => (value && typeof value === "object" && !Array.isArray(value) ? (value as JsonObject) : {});

export function summarizeNode(node: NodeName, output: JsonObject | undefined): string | null {
  if (!output) return null;
  if (output.failed) return `Failed: ${str(arr(output.errors).at(-1)) ?? "see log"}`;
  switch (node) {
    case "intake": {
      const mismatches = arr(output.mismatches).length;
      const source = str(output.source) === "form_and_document" ? "form + document" : "form only";
      return `Extracted from ${source}; ${mismatches ? `${mismatches} mismatch(es) with the form` : "matches the form"}`;
    }
    case "investigator": {
      const calls = arr(output.tool_calls).length;
      const steps = num(output.steps_used);
      const early = output.stopped_early ? ` · stopped early: ${str(output.stop_reason)}` : "";
      return `${calls} tool call${calls === 1 ? "" : "s"} in ${steps ?? "?"} step${steps === 1 ? "" : "s"}${early}`;
    }
    case "rules": {
      const triggered = arr(output.results).filter((r) => obj(r).triggered).map((r) => str(obj(r).rule_id));
      return `Risk score ${num(output.risk_score) ?? 0}/100 · ${triggered.length ? `triggered ${triggered.join(", ")}` : "no rule triggered"}`;
    }
    case "decision": {
      const confidence = num(output.confidence);
      return `${output.covered ? "Covered" : "Not covered"} · agent confidence ${confidence != null ? Math.round(confidence * 100) : "?"}%`;
    }
    case "reflection": {
      const action = str(output.action);
      const problems = arr(output.citation_problems).length;
      const questions = arr(output.open_questions).length;
      const forReviewer = questions ? ` · ${questions} question${questions === 1 ? "" : "s"} left for the reviewer` : "";
      if (action === "accept") return `Grounding verified: every citation checks out${forReviewer}`;
      if (action === "retry_investigation") return `Sent back for more evidence${problems ? ` (${problems} citation problem(s))` : ""}`;
      return "Could not verify grounding: escalating to a human";
    }
    case "route_final":
      return `Recommendation: ${str(output.outcome) ?? "?"} · ${str(arr(output.reasons)[0]) ?? ""}`;
  }
}
