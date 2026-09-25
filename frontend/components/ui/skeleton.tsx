import type { HTMLAttributes } from "react";

import { cn } from "@/lib/utils";

/**
 * Loading placeholder. The console never shows a bare spinner: every loading state is a skeleton shaped like the
 * content it replaces, so the layout doesn't jump when data arrives. `motion-safe:` stops the pulse for users who
 * prefer reduced motion.
 */
export function Skeleton({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div aria-hidden className={cn("rounded-md bg-surface-muted motion-safe:animate-pulse", className)} {...props} />
  );
}
