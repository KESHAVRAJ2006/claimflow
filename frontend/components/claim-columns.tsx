import type { Column } from "@/components/data-table";
import { StatusBadge } from "@/components/status-badge";
import { formatConfidence, formatDate, formatMoney, humanize } from "@/lib/format";
import type { ClaimSummary } from "@/lib/types";

/** Column definitions shared by the claims queue and the dashboard's escalation table. Ids match API sort keys. */
export const claimColumns: Record<string, Column<ClaimSummary>> = {
  claim: {
    id: "claim_number",
    header: "Claim",
    cell: (claim) => <span className="font-mono text-foreground">{claim.claim_number}</span>,
  },
  policy: {
    id: "policy_number",
    header: "Policy",
    // Hidden below 2xl (1536px) so the full queue fits a laptop screen without horizontal scrolling.
    className: "hidden 2xl:table-cell",
    cell: (claim) => <span className="font-mono text-muted">{claim.policy_number}</span>,
  },
  incident: {
    id: "incident_type",
    header: "Incident",
    cell: (claim) => (
      <span>
        {humanize(claim.incident_type)}
        <span className="text-muted"> · {humanize(claim.product_type)}</span>
      </span>
    ),
  },
  incidentDate: {
    id: "incident_date",
    header: "Incident date",
    sortable: true,
    cell: (claim) => <span className="font-mono">{formatDate(claim.incident_date)}</span>,
  },
  amount: {
    id: "claimed_amount",
    header: "Amount",
    sortable: true,
    align: "right",
    cell: (claim) => <span className="font-mono">{formatMoney(claim.claimed_amount)}</span>,
  },
  risk: {
    id: "risk_score",
    header: "Risk",
    sortable: true,
    align: "right",
    cell: (claim) => <span className="font-mono">{claim.risk_score ?? "—"}</span>,
  },
  confidence: {
    id: "confidence",
    header: "Confidence",
    sortable: true,
    align: "right",
    cell: (claim) => <span className="font-mono">{formatConfidence(claim.confidence)}</span>,
  },
  recommendation: {
    id: "recommended_outcome",
    header: "AI recommendation",
    cell: (claim) =>
      claim.recommended_outcome ? (
        <StatusBadge value={claim.recommended_outcome} />
      ) : (
        <span className="text-subtle">—</span>
      ),
  },
  status: {
    id: "status",
    header: "Status",
    cell: (claim) => <StatusBadge value={claim.status} />,
  },
  submitted: {
    id: "created_at",
    header: "Submitted",
    sortable: true,
    cell: (claim) => <span className="font-mono text-muted">{formatDate(claim.created_at)}</span>,
  },
};
