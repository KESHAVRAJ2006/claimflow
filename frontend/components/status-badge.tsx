import { cn } from "@/lib/utils";
import { humanize } from "@/lib/format";
import type { ClaimStatus, RecommendedOutcome } from "@/lib/types";

type Tone = "approve" | "reject" | "escalate" | "neutral" | "active";

/**
 * Status colours come from their own palette (emerald / red / amber), never the indigo accent, so a status can't be
 * mistaken for a button. A coloured dot plus the text label means colour is never the only signal.
 */
const TONE_BY_VALUE: Record<ClaimStatus | RecommendedOutcome, Tone> = {
  approve: "approve",
  approved: "approve",
  reject: "reject",
  rejected: "reject",
  failed: "reject",
  escalate: "escalate",
  escalated: "escalate",
  processing: "active",
  submitted: "neutral",
  awaiting_review: "neutral",
};

const TONE_CLASSES: Record<Tone, { badge: string; dot: string }> = {
  approve: { badge: "bg-approve-soft text-approve ring-approve/20", dot: "bg-approve" },
  reject: { badge: "bg-reject-soft text-reject ring-reject/20", dot: "bg-reject" },
  escalate: { badge: "bg-escalate-soft text-escalate ring-escalate/25", dot: "bg-escalate" },
  neutral: { badge: "bg-surface-muted text-muted ring-border", dot: "bg-subtle" },
  active: { badge: "bg-surface-muted text-foreground ring-border", dot: "bg-foreground motion-safe:animate-pulse" },
};

const LABEL_OVERRIDES: Partial<Record<ClaimStatus | RecommendedOutcome, string>> = {
  awaiting_review: "Awaiting review",
  approve: "Approve",
  reject: "Reject",
  escalate: "Escalate",
};

export function StatusBadge({ value, className }: { value: ClaimStatus | RecommendedOutcome; className?: string }) {
  const tone = TONE_CLASSES[TONE_BY_VALUE[value]];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 whitespace-nowrap rounded-md px-2 py-0.5 text-label normal-case tracking-normal ring-1 ring-inset",
        tone.badge,
        className,
      )}
    >
      <span aria-hidden className={cn("size-1.5 rounded-full", tone.dot)} />
      {LABEL_OVERRIDES[value] ?? humanize(value)}
    </span>
  );
}
