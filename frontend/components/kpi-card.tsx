import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

interface KpiCardProps {
  label: string;
  value: ReactNode;
  icon: LucideIcon;
  /** One short line under the value, e.g. "of 42 triaged in 30 days". */
  hint?: ReactNode;
  loading?: boolean;
  className?: string;
}

/** A single headline number. Values render in the mono face with tabular figures so cards line up. */
export function KpiCard({ label, value, icon: Icon, hint, loading = false, className }: KpiCardProps) {
  return (
    <section
      aria-busy={loading}
      className={cn("rounded-card border border-border bg-surface p-6 shadow-sm", className)}
    >
      <div className="flex items-center justify-between gap-4">
        <h3 className="text-label uppercase text-muted">{label}</h3>
        <Icon aria-hidden className="size-4 text-subtle" />
      </div>
      {loading ? (
        <>
          <Skeleton className="mt-4 h-8 w-24" />
          <Skeleton className="mt-2 h-4 w-32" />
        </>
      ) : (
        <>
          <p className="mt-3 font-mono text-title tabular-nums text-foreground">{value}</p>
          {hint ? <p className="mt-1 text-table text-muted">{hint}</p> : null}
        </>
      )}
    </section>
  );
}
