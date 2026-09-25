import { formatDate, formatDateTime, formatMoney, humanize } from "@/lib/format";
import type { ClaimDetail } from "@/lib/types";

interface Mismatch {
  field: string;
  form_value: string;
  extracted_value: string;
}

function mismatchesOf(claim: ClaimDetail): Mismatch[] {
  const list = claim.extracted_fields?.mismatches;
  return Array.isArray(list) ? (list as Mismatch[]) : [];
}

/** The claim as submitted: identifiers and figures in mono, the claimant's description, intake discrepancies. */
export function ClaimSummary({ claim }: { claim: ClaimDetail }) {
  const rows: [string, string][] = [
    ["Claim", claim.claim_number],
    ["Policy", claim.policy_number],
    ["Product", humanize(claim.product_type)],
    ["Incident", humanize(claim.incident_type)],
    ["Incident date", formatDate(claim.incident_date)],
    ["Amount", formatMoney(claim.claimed_amount)],
    ["Submitted", formatDateTime(claim.created_at)],
    ["Document", claim.document_filename ?? "—"],
  ];
  const mismatches = mismatchesOf(claim);
  return (
    <section aria-labelledby="summary-heading" className="rounded-card border border-border bg-surface p-6 shadow-sm">
      <h2 id="summary-heading" className="text-section">
        Claim
      </h2>
      <dl className="mt-3 grid grid-cols-[112px_1fr] gap-x-4 gap-y-2">
        {rows.map(([label, value]) => (
          <div key={label} className="contents">
            <dt className="text-table text-muted">{label}</dt>
            <dd className="truncate font-mono text-table text-foreground">{value}</dd>
          </div>
        ))}
      </dl>
      <h3 className="mt-4 text-label uppercase text-muted">Claimant&apos;s description</h3>
      {/* Customer-written text: plain text only (React escapes it). */}
      <p className="mt-1 whitespace-pre-line text-body text-foreground">{claim.description}</p>
      {mismatches.length > 0 ? (
        <>
          <h3 className="mt-4 text-label uppercase text-escalate">Form vs document mismatches</h3>
          <ul className="mt-1 flex flex-col gap-1">
            {mismatches.map((m) => (
              <li key={m.field} className="font-mono text-table text-foreground">
                {m.field}: <span className="text-muted">form</span> {m.form_value}{" "}
                <span className="text-muted">document</span> {m.extracted_value}
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </section>
  );
}
