"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Play, Workflow } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";

import { ActionBar } from "@/components/claim/action-bar";
import { AgentTrace } from "@/components/claim/agent-trace";
import { ClaimSummary } from "@/components/claim/claim-summary";
import { CitationSheet, useCitationSheet } from "@/components/claim/citations";
import { DecisionCard, rationaleFrom } from "@/components/claim/decision-card";
import { RiskPanel } from "@/components/claim/risk-panel";
import { EmptyState } from "@/components/empty-state";
import { ErrorState } from "@/components/error-state";
import { StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, api } from "@/lib/api";
import { formatMoney, humanize } from "@/lib/format";
import { useClaim } from "@/lib/queries";
import { buildTrace } from "@/lib/trace";
import { useRunStream, type StreamState } from "@/lib/use-run-stream";
import { cn } from "@/lib/utils";

function StreamIndicator({ state, replayed, ended }: { state: StreamState; replayed: boolean; ended: boolean }) {
  const [label, tone] = ended
    ? [replayed ? "Replayed from history" : "Run complete", "bg-subtle"]
    : state === "open"
      ? ["Live", "bg-approve motion-safe:animate-pulse"]
      : state === "error"
        ? ["Disconnected, retrying", "bg-reject"]
        : ["Connecting", "bg-subtle motion-safe:animate-pulse"];
  return (
    <span role="status" className="flex items-center gap-2 text-table text-muted">
      <span aria-hidden className={cn("size-1.5 rounded-full", tone)} />
      {label}
    </span>
  );
}

export default function ClaimDetailPage({ params }: { params: { id: string } }) {
  const queryClient = useQueryClient();
  const claim = useClaim(params.id);
  const [session, setSession] = useState(0);
  const stream = useRunStream(params.id, session);
  const trace = useMemo(() => buildTrace(stream.events), [stream.events]);
  const citations = useCitationSheet();

  const run = useMutation({
    mutationFn: () => api.runClaim(params.id),
    onSuccess: () => {
      setSession((value) => value + 1); // reconnect: the new run's events replace the "unavailable" replay
      void queryClient.invalidateQueries({ queryKey: ["claim", params.id] });
    },
  });

  if (claim.isError) {
    const notFound = claim.error instanceof ApiError && claim.error.status === 404;
    return notFound ? (
      <EmptyState
        icon={Workflow}
        title="This claim doesn't exist."
        action={
          <Button asChild variant="primary">
            <Link href="/claims">Back to claims</Link>
          </Button>
        }
      />
    ) : (
      <ErrorState error={claim.error} onRetry={() => claim.refetch()} />
    );
  }

  const data = claim.data;
  const running = !trace.end && trace.nodes.length > 0;
  const decisionOutput = [...(data?.runs ?? [])].reverse().find((r) => r.agent_name === "decision")?.output;
  const evidenceLabels = Object.fromEntries((data?.evidence ?? []).map((p) => [p.evidence_id, `${p.document} p.${p.page}`]));
  const unavailable = trace.end?.type === "run_unavailable";

  return (
    <div className="flex flex-col gap-6">
      <div>
        <Link href="/claims" className="inline-flex items-center gap-1 rounded text-table text-muted hover:text-foreground">
          <ArrowLeft aria-hidden className="size-3.5" />
          Claims
        </Link>
        {data ? (
          <div className="mt-2 flex flex-wrap items-center gap-3">
            <h1 className="font-mono text-title">{data.claim_number}</h1>
            <StatusBadge value={data.status} />
            <span className="text-body text-muted">
              {humanize(data.incident_type)} · {humanize(data.product_type)} ·{" "}
              <span className="font-mono text-foreground">{formatMoney(data.claimed_amount)}</span>
            </span>
          </div>
        ) : (
          <Skeleton className="mt-2 h-8 w-72" />
        )}
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[3fr_2fr]">
        {/* LEFT 60%: the live reasoning trace */}
        <section
          aria-labelledby="trace-heading"
          className="h-fit rounded-card border border-border bg-surface p-6 shadow-sm"
        >
          <div className="mb-6 flex items-center justify-between gap-3">
            <h2 id="trace-heading" className="text-section">
              Agent trace
            </h2>
            {!unavailable ? <StreamIndicator state={stream.state} replayed={trace.replayed} ended={Boolean(trace.end)} /> : null}
          </div>
          {unavailable && trace.nodes.length === 0 ? (
            <EmptyState
              icon={Workflow}
              title={
                data?.status === "failed"
                  ? "The last triage run failed. Run it again to get a recommendation."
                  : "This claim hasn't been triaged yet."
              }
              action={
                data && (data.status === "submitted" || data.status === "failed") ? (
                  <div className="flex flex-col items-center gap-2">
                    <Button variant="primary" onClick={() => run.mutate()} disabled={run.isPending}>
                      <Play aria-hidden />
                      Run triage
                    </Button>
                    {run.isError ? (
                      <p role="alert" className="text-table text-reject">
                        {run.error instanceof ApiError ? run.error.problem.detail : "Could not start triage."}
                      </p>
                    ) : null}
                  </div>
                ) : undefined
              }
            />
          ) : trace.nodes.length === 0 ? (
            <div className="flex flex-col gap-6" aria-busy>
              {Array.from({ length: 4 }, (_, index) => (
                <div key={index} className="flex gap-4">
                  <Skeleton className="size-8 rounded-full" />
                  <div className="flex flex-1 flex-col gap-2 pt-1">
                    <Skeleton className="h-4 w-32" />
                    <Skeleton className="h-3 w-2/3" />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <>
              <AgentTrace nodes={trace.nodes} />
              {trace.end?.type === "run_failed" ? (
                <p role="alert" className="rounded-md bg-reject-soft px-3 py-2 text-table text-reject">
                  Triage failed: {trace.end.reason ?? "see the audit log"}
                </p>
              ) : null}
            </>
          )}
        </section>

        {/* RIGHT 40%: recommendation, claim, risk, and the reviewer's actions */}
        <div className="flex flex-col gap-6">
          <DecisionCard
            recommendation={data?.recommendation ?? null}
            rationale={rationaleFrom(decisionOutput)}
            labels={evidenceLabels}
            pending={claim.isPending || running}
            onOpenCitation={citations.show}
          />
          {data ? <ClaimSummary claim={data} /> : <Skeleton className="h-64 w-full rounded-card" />}
          <RiskPanel report={data?.risk_report ?? null} pending={claim.isPending || running} />
          {data && !running ? <ActionBar claim={data} /> : null}
        </div>
      </div>

      <CitationSheet citation={citations.open} evidence={data?.evidence ?? []} onClose={citations.close} />
    </div>
  );
}
