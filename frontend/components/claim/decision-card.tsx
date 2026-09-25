"use client";

import { Info } from "lucide-react";

import { CitationChip, type CitationTarget } from "@/components/claim/citations";
import { StatusBadge } from "@/components/status-badge";
import { Skeleton } from "@/components/ui/skeleton";
import type { FinalRecommendation, JsonObject } from "@/lib/types";
import { cn } from "@/lib/utils";

// The routing threshold (backend app/rules/thresholds.py): below it a human decides.
const MIN_CONFIDENCE = 0.65;

interface RationalePoint {
  statement: string;
  basis: "policy_wording" | "claim_data" | "claim_form" | "rule_result";
  evidence_ids: string[];
  quote: string | null;
  tool_call_ids: string[];
}

const BASIS_LABEL: Record<RationalePoint["basis"], string> = {
  policy_wording: "Policy wording",
  claim_data: "Claim data",
  claim_form: "Claimant's account",
  rule_result: "Rule result",
};

/** Read the decision node's rationale from its stored output (untyped JSON). */
export function rationaleFrom(output: JsonObject | undefined): RationalePoint[] {
  const points = output && Array.isArray(output.rationale) ? output.rationale : [];
  return points.filter((point): point is RationalePoint => Boolean(point && typeof point === "object" && "statement" in point));
}

function ConfidenceBar({ label, value }: { label: string; value: string | null }) {
  const fraction = value == null ? null : Number(value);
  const below = fraction != null && fraction < MIN_CONFIDENCE;
  return (
    <div className="grid grid-cols-[96px_1fr_44px] items-center gap-3">
      <span className="text-table text-muted">{label}</span>
      <div
        role="meter"
        aria-label={`${label} confidence`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={fraction == null ? undefined : Math.round(fraction * 100)}
        className="relative h-1.5 rounded-full bg-surface-muted"
      >
        <div
          className={cn("h-full rounded-full", below ? "bg-escalate" : "bg-foreground")}
          style={{ width: `${Math.round((fraction ?? 0) * 100)}%` }}
        />
        {/* Threshold marker at 65%. */}
        <span aria-hidden className="absolute -top-1 h-3.5 w-px bg-muted" style={{ left: `${MIN_CONFIDENCE * 100}%` }} />
      </div>
      <span className="text-right font-mono text-table text-foreground">
        {fraction == null ? "—" : `${Math.round(fraction * 100)}%`}
      </span>
    </div>
  );
}

interface DecisionCardProps {
  recommendation: FinalRecommendation | null;
  rationale: RationalePoint[];
  labels: Record<string, string>;
  pending: boolean;
  onOpenCitation: (citation: CitationTarget) => void;
}

/** The AI recommendation: outcome, the confidence that routing compared, the reasons, and the cited rationale. */
export function DecisionCard({ recommendation, rationale, labels, pending, onOpenCitation }: DecisionCardProps) {
  return (
    <section aria-labelledby="decision-heading" className="rounded-card border border-border bg-surface p-6 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <h2 id="decision-heading" className="text-section">
          AI recommendation
        </h2>
        {recommendation ? <StatusBadge value={recommendation.outcome} /> : null}
      </div>

      {!recommendation ? (
        pending ? (
          <div className="mt-4 flex flex-col gap-3" aria-busy>
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-1.5 w-full" />
            <Skeleton className="h-1.5 w-full" />
            <Skeleton className="h-16 w-full" />
          </div>
        ) : (
          <p className="mt-3 text-body text-muted">No recommendation yet. It appears here when triage finishes.</p>
        )
      ) : (
        <>
          <p className="mt-3 text-body text-foreground">{recommendation.summary}</p>
          <div className="mt-4 flex flex-col gap-2">
            <ConfidenceBar label="Agent" value={recommendation.agent_confidence} />
            <ConfidenceBar label="Retrieval" value={recommendation.retrieval_confidence} />
            <ConfidenceBar label="Used" value={recommendation.confidence} />
          </div>
          <p className="mt-1 text-table text-muted">Routing uses the lower of the two; below 65% a human decides.</p>

          <h3 className="mt-5 text-label uppercase text-muted">Why this outcome</h3>
          <ul className="mt-2 flex list-disc flex-col gap-1 pl-4 text-body text-foreground">
            {recommendation.reasons.map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>

          {recommendation.open_questions.length > 0 ? (
            <>
              <h3 className="mt-5 text-label uppercase text-muted">Check with the claimant</h3>
              <p className="mt-1 text-table text-muted">No tool could answer these; confirm them before deciding.</p>
              <ul className="mt-2 flex list-disc flex-col gap-1 pl-4 text-body text-foreground">
                {recommendation.open_questions.map((question) => (
                  <li key={question}>{question}</li>
                ))}
              </ul>
            </>
          ) : null}

          {rationale.length > 0 ? (
            <>
              <h3 className="mt-5 text-label uppercase text-muted">Rationale</h3>
              <ol className="mt-2 flex flex-col gap-3">
                {rationale.map((point, index) => (
                  <li key={index} className="text-body">
                    <span className="mr-2 rounded bg-surface-muted px-1.5 py-px text-label normal-case tracking-normal text-muted">
                      {BASIS_LABEL[point.basis]}
                    </span>
                    {point.statement}
                    {point.evidence_ids.length || point.tool_call_ids.length ? (
                      <span className="mt-1.5 flex flex-wrap gap-1.5">
                        {point.evidence_ids.map((id) =>
                          labels[id] ? (
                            <CitationChip
                              key={id}
                              citation={{ evidenceId: id, label: labels[id]!, quote: point.quote }}
                              onOpen={onOpenCitation}
                            />
                          ) : null,
                        )}
                        {point.tool_call_ids.map((id) => (
                          <span key={id} className="rounded-md border border-border px-1.5 py-0.5 font-mono text-table text-muted">
                            {id}
                          </span>
                        ))}
                      </span>
                    ) : null}
                  </li>
                ))}
              </ol>
            </>
          ) : null}

          {recommendation.citations.length > 0 ? (
            <>
              <h3 className="mt-5 text-label uppercase text-muted">Sources cited</h3>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {recommendation.citations.map((citation) => (
                  <CitationChip
                    key={citation.evidence_id}
                    citation={{ evidenceId: citation.evidence_id, label: citation.label, quote: citation.quote }}
                    onOpen={onOpenCitation}
                  />
                ))}
              </div>
            </>
          ) : null}
        </>
      )}

      <p className="mt-5 flex gap-2 border-t border-border pt-4 text-table text-muted">
        <Info aria-hidden className="mt-0.5 size-3.5 shrink-0" />
        AI-assisted recommendation for a human reviewer, not an automated determination.
      </p>
    </section>
  );
}
