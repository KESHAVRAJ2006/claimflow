import { Workflow } from "lucide-react";

import { cn } from "@/lib/utils";

/** Product mark: icon plus wordmark (the wordmark hides in the collapsed rail). */
export function Brand({ collapsed = false }: { collapsed?: boolean }) {
  return (
    <div className={cn("flex h-14 items-center gap-2 px-4", collapsed && "justify-center px-0")}>
      <span className="flex size-7 items-center justify-center rounded-md bg-foreground text-background">
        <Workflow aria-hidden className="size-4" />
      </span>
      {collapsed ? <span className="sr-only">ClaimFlow</span> : <span className="text-section">ClaimFlow</span>}
    </div>
  );
}
