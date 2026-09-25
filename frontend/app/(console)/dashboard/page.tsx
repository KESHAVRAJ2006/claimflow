"use client";

import { motion } from "framer-motion";
import { AlertTriangle, Bot, CheckCircle2, FileStack, Inbox, Wrench } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";

import { claimColumns } from "@/components/claim-columns";
import { DataTable } from "@/components/data-table";
import { EmptyState } from "@/components/empty-state";
import { ErrorState } from "@/components/error-state";
import { KpiCard } from "@/components/kpi-card";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { VolumeChart } from "@/components/volume-chart";
import { formatRate } from "@/lib/format";
import { DURATION, EASE, STAGGER } from "@/lib/motion";
import { useClaims, useMetrics } from "@/lib/queries";

const WINDOW_DAYS = 30;

export default function DashboardPage() {
  const router = useRouter();
  const metrics = useMetrics(WINDOW_DAYS);
  const escalations = useClaims({ status: ["escalated"], sort: "created_at", order: "desc", page_size: 5 });
  const m = metrics.data;

  return (
    <div className="flex flex-col gap-8">
      <PageHeader
        title="Dashboard"
        description={`Triage activity over the last ${WINDOW_DAYS} days.`}
        actions={
          <Button asChild variant="primary">
            <Link href="/claims/new">Submit claim</Link>
          </Button>
        }
      />

      {metrics.isError ? (
        <ErrorState error={metrics.error} onRetry={() => metrics.refetch()} />
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {[
            { label: "Claims today", icon: FileStack, value: m?.claims_today, hint: m && `${m.claims_in_window} in ${WINDOW_DAYS} days` },
            {
              label: "Auto-decision rate",
              icon: CheckCircle2,
              value: formatRate(m?.auto_decision_rate),
              hint: m && `of ${m.triaged_in_window} triaged claims`,
            },
            {
              label: "Escalation rate",
              icon: AlertTriangle,
              value: formatRate(m?.escalation_rate),
              hint: m && `${m.pending_review} awaiting a reviewer`,
            },
            {
              label: "Avg tool calls / claim",
              icon: Wrench,
              value: m?.avg_tool_calls_per_claim?.toFixed(1) ?? "—",
              hint: "chosen by the investigator agent",
            },
          ].map((kpi, index) => (
            // Each card animates itself with a 60 ms stagger delay. (Parent-driven variant propagation left a card
            // stuck at opacity 0 when its content re-rendered mid-animation as the data arrived.)
            <motion.div
              key={kpi.label}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: DURATION.base, ease: EASE, delay: index * STAGGER }}
            >
              <KpiCard
                label={kpi.label}
                icon={kpi.icon}
                value={kpi.value ?? "—"}
                hint={kpi.hint || undefined}
                loading={metrics.isPending}
              />
            </motion.div>
          ))}
        </div>
      )}

      <section aria-labelledby="volume-heading" className="rounded-card border border-border bg-surface p-6 shadow-sm">
        <h2 id="volume-heading" className="mb-4 text-section">
          Claim volume
        </h2>
        {metrics.isPending ? (
          <Skeleton className="h-72 w-full" />
        ) : m && m.claims_in_window > 0 ? (
          <VolumeChart volume={m.volume} />
        ) : (
          <EmptyState
            icon={FileStack}
            title={`No claims were submitted in the last ${WINDOW_DAYS} days.`}
            action={
              <Button asChild variant="secondary">
                <Link href="/claims/new">Submit a claim</Link>
              </Button>
            }
          />
        )}
      </section>

      <section className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <h2 className="text-section">Recent escalations</h2>
          <Button asChild variant="ghost" size="sm">
            <Link href="/claims?status=escalated">View all</Link>
          </Button>
        </div>
        {escalations.isError ? (
          <ErrorState error={escalations.error} onRetry={() => escalations.refetch()} />
        ) : (
          <DataTable
            caption="The five most recent claims escalated to a human reviewer"
            columns={[
              claimColumns.claim!,
              claimColumns.incident!,
              claimColumns.amount!,
              claimColumns.risk!,
              claimColumns.submitted!,
            ]}
            rows={escalations.data?.items}
            rowKey={(claim) => claim.id}
            rowLabel={(claim) => `Claim ${claim.claim_number}, escalated`}
            onRowActivate={(claim) => router.push(`/claims/${claim.id}`)}
            loading={escalations.isPending}
            skeletonRows={5}
            empty={
              <EmptyState
                icon={Bot}
                title="No escalated claims. Claims the AI can't decide safely will appear here."
                action={
                  <Button asChild>
                    <Link href="/claims">
                      <Inbox aria-hidden />
                      Open the claims queue
                    </Link>
                  </Button>
                }
              />
            }
          />
        )}
      </section>
    </div>
  );
}
