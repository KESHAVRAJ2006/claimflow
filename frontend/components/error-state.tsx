"use client";

import { AlertTriangle, RotateCw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ApiError } from "@/lib/api";
import { cn } from "@/lib/utils";

/**
 * Shown when a query fails. It surfaces the API's problem detail and request id (so a user can quote it in a support
 * request) and offers a retry.
 */
export function ErrorState({ error, onRetry, className }: { error: unknown; onRetry?: () => void; className?: string }) {
  const problem = error instanceof ApiError ? error.problem : null;
  return (
    <div role="alert" className={cn("flex flex-col items-center gap-3 px-6 py-12 text-center", className)}>
      <div className="flex size-10 items-center justify-center rounded-card border border-reject/20 bg-reject-soft">
        <AlertTriangle aria-hidden className="size-5 text-reject" />
      </div>
      <p className="max-w-md text-body text-foreground">
        {problem ? `${problem.title}. ${problem.detail}` : "Something went wrong while loading this data."}
      </p>
      {problem?.request_id ? (
        <p className="font-mono text-table text-muted">Request ID {problem.request_id}</p>
      ) : null}
      {onRetry ? (
        <Button onClick={onRetry}>
          <RotateCw aria-hidden />
          Try again
        </Button>
      ) : null}
    </div>
  );
}
