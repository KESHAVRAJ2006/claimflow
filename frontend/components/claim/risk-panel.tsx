import { ShieldAlert } from "lucide-react";

import { Skeleton } from "@/components/ui/skeleton";
import type { RiskReport, Severity } from "@/lib/types";
import { cn } from "@/lib/utils";

// Backend app/rules/thresholds.py: at or above this score a human decides.
const RISK_ESCALATION_THRESHOLD = 70;
const SEVERITY_WIDTH: Record<Severity, number> = { low: 25, medium: 50, high: 75, critical: 100 };
const SEVERITY_TONE: Record<Severity, string> = {
  low: "bg-subtle",
  medium: "bg-escalate",
  high: "bg-reject",
  critical: "bg-reject",
};

/** Deterministic rule results: the risk score against its threshold, triggered rules with severity, passed checks. */
export function RiskPanel({ report, pending }: { report: RiskReport | null; pending: boolean }) {
  const triggered = report?.results.filter((rule) => rule.triggered) ?? [];
  const passed = report?.results.filter((rule) => !rule.triggered) ?? [];
  return (
    <section aria-labelledby="risk-heading" className="rounded-card border border-border bg-surface p-6 shadow-sm">
      <div className="flex items-center justify-between">
        <h2 id="risk-heading" className="text-section">
          Risk
        </h2>
        <span className="rounded bg-surface-muted px-1.5 py-px text-label normal-case tracking-normal text-muted">
          Deterministic
        </span>
      </div>
      {!report ? (
        pending ? (
          <div className="mt-4 flex flex-col gap-3" aria-busy>
            <Skeleton className="h-8 w-20" />
            <Skeleton className="h-1.5 w-full" />
            <Skeleton className="h-10 w-full" />
          </div>
        ) : (
          <p className="mt-3 text-body text-muted">The rules engine runs as part of triage.</p>
        )
      ) : (
        <>
          <div className="mt-3 flex items-baseline gap-2">
            <span className="font-mono text-title">{report.risk_score}</span>
            <span className="font-mono text-table text-muted">/ 100</span>
          </div>
          <div
            role="meter"
            aria-label="Risk score"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={report.risk_score}
            className="relative mt-2 h-1.5 rounded-full bg-surface-muted"
          >
            <div
              className={cn(
                "h-full rounded-full",
                report.risk_score >= RISK_ESCALATION_THRESHOLD ? "bg-reject" : report.risk_score > 0 ? "bg-escalate" : "bg-approve",
              )}
              style={{ width: `${Math.max(report.risk_score, 2)}%` }}
            />
            <span aria-hidden className="absolute -top-1 h-3.5 w-px bg-muted" style={{ left: `${RISK_ESCALATION_THRESHOLD}%` }} />
          </div>
          <p className="mt-1 text-table text-muted">Escalates at {RISK_ESCALATION_THRESHOLD}. A hard block rejects regardless of score.</p>

          {triggered.length > 0 ? (
            <ul className="mt-4 flex flex-col gap-3">
              {triggered.map((rule) => (
                <li key={rule.rule_id} className="rounded-md border border-border p-3">
                  <div className="flex items-center gap-2">
                    <ShieldAlert aria-hidden className={cn("size-4", rule.severity === "low" ? "text-muted" : rule.severity === "medium" ? "text-escalate" : "text-reject")} />
                    <span className="font-mono text-table text-foreground">{rule.rule_id}</span>
                    <span className="truncate text-table text-foreground">{rule.name.replace(/_/g, " ")}</span>
                    {rule.hard_block ? (
                      <span className="ml-auto rounded bg-reject-soft px-1.5 py-px text-label normal-case tracking-normal text-reject">
                        Hard block
                      </span>
                    ) : (
                      <span className="ml-auto font-mono text-table text-muted">+{rule.score_contribution}</span>
                    )}
                  </div>
                  <div className="mt-2 grid grid-cols-[64px_1fr] items-center gap-2">
                    <span className="text-table capitalize text-muted">{rule.severity}</span>
                    <div className="h-1 rounded-full bg-surface-muted" aria-hidden>
                      <div className={cn("h-full rounded-full", SEVERITY_TONE[rule.severity])} style={{ width: `${SEVERITY_WIDTH[rule.severity]}%` }} />
                    </div>
                  </div>
                  <p className="mt-2 text-table text-muted">{rule.explanation}</p>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-4 text-body text-foreground">No fraud or eligibility rule was triggered.</p>
          )}

          <details className="group mt-4">
            <summary className="cursor-pointer text-table text-muted hover:text-foreground">
              {passed.length} check{passed.length === 1 ? "" : "s"} passed
            </summary>
            <ul className="mt-2 flex flex-col gap-1.5">
              {passed.map((rule) => (
                <li key={rule.rule_id} className="text-table text-muted">
                  <span className="font-mono text-foreground">{rule.rule_id}</span> {rule.explanation}
                </li>
              ))}
            </ul>
          </details>
        </>
      )}
    </section>
  );
}
