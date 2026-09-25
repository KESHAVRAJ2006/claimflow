import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

interface EmptyStateProps {
  icon: LucideIcon;
  /** One sentence saying what's missing. */
  title: string;
  /** The primary action that resolves the empty state (a Button). */
  action?: ReactNode;
  className?: string;
}

/** Empty states follow one pattern: an icon, one sentence, and a primary action. */
export function EmptyState({ icon: Icon, title, action, className }: EmptyStateProps) {
  return (
    <div className={cn("flex flex-col items-center justify-center gap-4 px-6 py-12 text-center", className)}>
      <div className="flex size-10 items-center justify-center rounded-card border border-border bg-surface-muted">
        <Icon aria-hidden className="size-5 text-muted" />
      </div>
      <p className="max-w-sm text-body text-muted">{title}</p>
      {action}
    </div>
  );
}
