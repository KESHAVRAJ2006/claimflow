"use client";

import { AnimatePresence, motion } from "framer-motion";
import {
  Check,
  CircleDashed,
  FileText,
  Flag,
  Gavel,
  RotateCcw,
  Scale,
  Search,
  ShieldCheck,
  X,
  type LucideIcon,
} from "lucide-react";

import { formatLatency } from "@/lib/format";
import { fadeUp } from "@/lib/motion";
import { NODE_KIND, NODE_LABEL, summarizeNode, type NodeName, type TraceNode, type TraceTool } from "@/lib/trace";
import type { JsonObject } from "@/lib/types";
import { cn } from "@/lib/utils";

const NODE_ICON: Record<NodeName, LucideIcon> = {
  intake: FileText,
  investigator: Search,
  rules: Scale,
  decision: Gavel,
  reflection: ShieldCheck,
  route_final: Flag,
};

/** `{policy_number: "MOT-…", on_date: "2026-08-01"}` -> `policy_number=MOT-… on_date=2026-08-01`. */
function formatArgs(args: JsonObject): string {
  return Object.entries(args)
    .map(([key, value]) => `${key}=${typeof value === "string" ? value : JSON.stringify(value)}`)
    .join("  ");
}

function ToolCall({ call }: { call: TraceTool }) {
  const StatusIcon = call.status === "running" ? CircleDashed : call.status === "ok" ? Check : X;
  return (
    <motion.li variants={fadeUp} initial="hidden" animate="show" layout="position" className="relative pl-6">
      <StatusIcon
        aria-label={call.status === "running" ? "Running" : call.status === "ok" ? "Succeeded" : "Failed"}
        className={cn(
          "absolute left-0 top-0.5 size-4",
          call.status === "running" && "text-subtle motion-safe:animate-spin",
          call.status === "ok" && "text-approve",
          call.status === "error" && "text-reject",
        )}
      />
      <div className="flex items-baseline justify-between gap-3">
        <p className="min-w-0 font-mono text-table">
          <span className="text-subtle">{call.callId}</span> <span className="text-foreground">{call.tool}</span>
        </p>
        <span className="shrink-0 font-mono text-table text-muted">
          {call.latencyMs != null ? formatLatency(call.latencyMs) : "…"}
        </span>
      </div>
      <p className="break-all font-mono text-table text-muted">{formatArgs(call.args)}</p>
      {call.summary ? (
        <p className={cn("mt-0.5 text-table", call.status === "error" ? "text-reject" : "text-foreground")}>
          {call.summary}
        </p>
      ) : null}
    </motion.li>
  );
}

function TraceEntry({ entry, last }: { entry: TraceNode; last: boolean }) {
  const Icon = NODE_ICON[entry.node];
  const running = entry.status === "running";
  const kind = NODE_KIND[entry.node];
  const summary = summarizeNode(entry.node, entry.output);
  return (
    <motion.li variants={fadeUp} initial="hidden" animate="show" layout="position" className="relative flex gap-4 pb-6">
      {/* Connector to the next entry; it pulses while this node is still running. */}
      {!last || running ? (
        <span
          aria-hidden
          className={cn(
            "absolute left-[15px] top-9 h-[calc(100%-2.25rem)] w-px",
            running ? "bg-foreground motion-safe:animate-pulse-line" : "bg-border",
          )}
        />
      ) : null}
      <span
        className={cn(
          "relative z-10 flex size-8 shrink-0 items-center justify-center rounded-full border bg-surface",
          running ? "border-foreground text-foreground" : "border-border text-muted",
        )}
      >
        <Icon aria-hidden className="size-4" />
      </span>
      <div className="min-w-0 flex-1 pt-1">
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="text-body font-medium text-foreground">{NODE_LABEL[entry.node]}</h3>
          <span
            className={cn(
              "rounded px-1.5 py-px text-label normal-case tracking-normal ring-1 ring-inset",
              kind === "agent" ? "text-foreground ring-border" : "bg-surface-muted text-muted ring-transparent",
            )}
            title={kind === "agent" ? "An LLM decides here" : "Plain code: no LLM"}
          >
            {kind === "agent" ? "Agent" : "Deterministic"}
          </span>
          <span className="ml-auto font-mono text-table text-muted">
            {running ? <span className="text-foreground">running…</span> : formatLatency(entry.latencyMs)}
          </span>
        </div>
        {summary ? <p className="mt-1 text-table text-muted">{summary}</p> : null}
        {entry.node === "investigator" && entry.tools.length > 0 ? (
          <ol aria-label="Tool calls" className="mt-3 flex flex-col gap-3 border-l border-border pl-4">
            <AnimatePresence initial={false}>
              {entry.tools.map((call) => (
                <ToolCall key={call.callId} call={call} />
              ))}
            </AnimatePresence>
          </ol>
        ) : null}
      </div>
    </motion.li>
  );
}

function RoundDivider({ round }: { round: number }) {
  return (
    <motion.li variants={fadeUp} initial="hidden" animate="show" className="relative mb-6 flex items-center gap-3">
      <span className="flex size-8 shrink-0 items-center justify-center rounded-full border border-dashed border-escalate/60 bg-escalate-soft">
        <RotateCcw aria-hidden className="size-4 text-escalate" />
      </span>
      <div className="flex-1 border-t border-dashed border-escalate/40" />
      <span className="text-label uppercase text-escalate">Round {round} · reflection asked for more evidence</span>
      <div className="w-6 border-t border-dashed border-escalate/40" />
    </motion.li>
  );
}

/** The two-level reasoning trace: nodes, with the investigator's tool calls nested beneath it. */
export function AgentTrace({ nodes }: { nodes: TraceNode[] }) {
  return (
    <ol aria-label="Agent trace" aria-live="polite" className="flex flex-col">
      <AnimatePresence initial={false}>
        {nodes.flatMap((entry, index) => {
          const previous = nodes[index - 1];
          const items = [];
          // A reflection-triggered retry starts a new round: mark it so it reads as a second pass, not more steps.
          if (entry.node === "investigator" && entry.round > 1 && previous?.round !== entry.round) {
            items.push(<RoundDivider key={`round-${entry.round}`} round={entry.round} />);
          }
          items.push(<TraceEntry key={entry.key} entry={entry} last={index === nodes.length - 1} />);
          return items;
        })}
      </AnimatePresence>
    </ol>
  );
}
