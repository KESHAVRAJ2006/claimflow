"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, CircleCheck, CircleX, HelpCircle, Loader2, Repeat2 } from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Field, Input, Textarea } from "@/components/ui/field";
import { ApiError, api } from "@/lib/api";
import { formatDateTime } from "@/lib/format";
import type { ClaimDetail, DecisionRequest } from "@/lib/types";
import { cn } from "@/lib/utils";

const REVIEWER_KEY = "claimflow.reviewer";
const REVIEWER_PATTERN = /^[a-z0-9][a-z0-9._-]{2,59}$/;
// Mirrors backend MIN_REASON_LENGTH: a one-word reason isn't a justification an auditor can use.
const MIN_REASON = 10;

type Mode = "accept" | "override" | "decide" | "request_info";

interface DialogSpec {
  title: string;
  description: string;
  confirm: string;
  reasonRequired: boolean;
  reasonLabel: string;
}

function specFor(mode: Mode, outcome: "approve" | "reject"): DialogSpec {
  switch (mode) {
    case "accept":
      return {
        title: outcome === "approve" ? "Approve this claim?" : "Confirm the rejection?",
        description: "You are agreeing with the AI recommendation. Your decision is binding and recorded in the audit log.",
        confirm: outcome === "approve" ? "Approve claim" : "Reject claim",
        reasonRequired: false,
        reasonLabel: "Note (optional)",
      };
    case "override":
      return {
        title: outcome === "approve" ? "Override: approve instead" : "Override: reject instead",
        description: "You are overriding the AI recommendation. Explain why; the reason is stored with the decision.",
        confirm: outcome === "approve" ? "Override and approve" : "Override and reject",
        reasonRequired: true,
        reasonLabel: "Reason for override",
      };
    case "decide":
      return {
        title: "Decide this escalated claim",
        description: "The AI could not decide safely. Choose the outcome and record your reasoning.",
        confirm: outcome === "approve" ? "Approve claim" : "Reject claim",
        reasonRequired: true,
        reasonLabel: "Reasoning",
      };
    case "request_info":
      return {
        title: "Request more information",
        description: "The claim stays open. Your question is recorded and sent to the notification workflow.",
        confirm: "Send request",
        reasonRequired: true,
        reasonLabel: "What is needed",
      };
  }
}

function DecisionDialog({
  claim,
  mode,
  initialOutcome,
  onClose,
}: {
  claim: ClaimDetail;
  mode: Mode | null;
  initialOutcome: "approve" | "reject";
  onClose: () => void;
}) {
  const queryClient = useQueryClient();
  const [reviewer, setReviewer] = useState("");
  const [reason, setReason] = useState("");
  const [outcome, setOutcome] = useState(initialOutcome);

  useEffect(() => {
    if (!mode) return;
    setReason("");
    setOutcome(initialOutcome);
    try {
      setReviewer(window.localStorage.getItem(REVIEWER_KEY) ?? "");
    } catch {
      // Storage blocked: the reviewer just types their id.
    }
  }, [mode, initialOutcome]);

  const decide = useMutation({
    mutationFn: (body: DecisionRequest) => api.decide(claim.id, body),
    onSuccess: () => {
      try {
        window.localStorage.setItem(REVIEWER_KEY, reviewer);
      } catch {
        // Not remembered; harmless.
      }
      void queryClient.invalidateQueries({ queryKey: ["claim", claim.id] });
      void queryClient.invalidateQueries({ queryKey: ["claims"] });
      void queryClient.invalidateQueries({ queryKey: ["metrics"] });
      onClose();
    },
  });

  if (!mode) return null;
  const spec = specFor(mode, outcome);
  const reviewerValid = REVIEWER_PATTERN.test(reviewer);
  const reasonLength = reason.trim().length;
  const reasonValid = !spec.reasonRequired || reasonLength >= MIN_REASON;
  const canConfirm = reviewerValid && reasonValid && !decide.isPending;

  const submit = () =>
    decide.mutate({
      action: mode === "request_info" ? "request_info" : outcome,
      reviewer,
      reason: reason.trim() || null,
    });

  return (
    <Dialog open onOpenChange={(open) => !open && !decide.isPending && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{spec.title}</DialogTitle>
          <DialogDescription>{spec.description}</DialogDescription>
        </DialogHeader>
        <form
          className="flex flex-col gap-4"
          onSubmit={(event) => {
            event.preventDefault();
            if (canConfirm) submit();
          }}
        >
          {mode === "decide" ? (
            <fieldset className="flex gap-2">
              <legend className="mb-1.5 text-label uppercase text-muted">Outcome</legend>
              {(["approve", "reject"] as const).map((value) => (
                <label
                  key={value}
                  className={cn(
                    "flex flex-1 cursor-pointer items-center gap-2 rounded-md border px-3 py-2 text-body transition-colors",
                    outcome === value ? "border-foreground bg-surface-muted" : "border-border hover:bg-surface-muted",
                  )}
                >
                  <input
                    type="radio"
                    name="outcome"
                    value={value}
                    checked={outcome === value}
                    onChange={() => setOutcome(value)}
                    className="accent-foreground"
                  />
                  {value === "approve" ? "Approve" : "Reject"}
                </label>
              ))}
            </fieldset>
          ) : null}
          <Field
            id="reviewer"
            label="Reviewer ID"
            error={reviewer && !reviewerValid ? "3–60 lowercase letters, digits, dots, dashes or underscores." : undefined}
            hint="Recorded as the decision maker, e.g. priya.nair"
          >
            <Input
              id="reviewer"
              value={reviewer}
              onChange={(event) => setReviewer(event.target.value.toLowerCase())}
              autoComplete="username"
              aria-invalid={Boolean(reviewer) && !reviewerValid}
              aria-describedby="reviewer-message"
            />
          </Field>
          <Field
            id="reason"
            label={spec.reasonLabel}
            hint={
              spec.reasonRequired ? (
                <span className={cn("font-mono", reasonValid ? "text-approve" : "text-muted")}>
                  {reasonLength}/{MIN_REASON} characters minimum
                </span>
              ) : undefined
            }
          >
            <Textarea
              id="reason"
              value={reason}
              onChange={(event) => setReason(event.target.value)}
              maxLength={2000}
              aria-required={spec.reasonRequired}
              aria-describedby="reason-message"
            />
          </Field>
          {decide.isError ? (
            <p role="alert" className="rounded-md bg-reject-soft px-3 py-2 text-table text-reject">
              {decide.error instanceof ApiError ? decide.error.problem.detail : "The decision could not be saved."}
            </p>
          ) : null}
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={onClose} disabled={decide.isPending}>
              Cancel
            </Button>
            <Button type="submit" variant={mode === "override" ? "danger" : "primary"} disabled={!canConfirm}>
              {decide.isPending ? <Loader2 aria-hidden className="motion-safe:animate-spin" /> : <Check aria-hidden />}
              {spec.confirm}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}

/** Sticky reviewer actions: accept the recommendation, override it (reason required), or request information. */
export function ActionBar({ claim }: { claim: ClaimDetail }) {
  const [mode, setMode] = useState<Mode | null>(null);
  const recommended = claim.recommended_outcome;
  const definite = recommended === "approve" || recommended === "reject" ? recommended : null;
  const opposite = definite === "approve" ? "reject" : "approve";
  const reviewable = claim.status === "awaiting_review" || claim.status === "escalated";

  if (claim.final_outcome) {
    return (
      <div className="sticky bottom-0 z-10 rounded-card border border-border bg-surface p-4 shadow-sm">
        <p className="flex items-center gap-2 text-body text-foreground">
          {claim.final_outcome === "approve" ? (
            <CircleCheck aria-hidden className="size-4 text-approve" />
          ) : (
            <CircleX aria-hidden className="size-4 text-reject" />
          )}
          {claim.final_outcome === "approve" ? "Approved" : "Rejected"} by{" "}
          <span className="font-mono">{claim.decided_by?.replace("reviewer:", "")}</span>
          <span className="text-muted">· {formatDateTime(claim.decided_at)}</span>
        </p>
        {claim.override_reason ? <p className="mt-1 text-table text-muted">Reason: {claim.override_reason}</p> : null}
      </div>
    );
  }
  if (!reviewable) return null;

  return (
    <>
      <div className="sticky bottom-0 z-10 flex flex-wrap items-center gap-2 rounded-card border border-border bg-surface p-4 shadow-sm">
        {definite ? (
          <>
            <Button variant="primary" onClick={() => setMode("accept")}>
              <Check aria-hidden />
              {definite === "approve" ? "Approve" : "Confirm rejection"}
            </Button>
            <Button onClick={() => setMode("override")}>
              <Repeat2 aria-hidden />
              Override
            </Button>
          </>
        ) : (
          <Button variant="primary" onClick={() => setMode("decide")}>
            <Check aria-hidden />
            Decide
          </Button>
        )}
        <Button variant="ghost" onClick={() => setMode("request_info")}>
          <HelpCircle aria-hidden />
          Request info
        </Button>
      </div>
      <DecisionDialog
        claim={claim}
        mode={mode}
        initialOutcome={mode === "override" ? opposite : (definite ?? "approve")}
        onClose={() => setMode(null)}
      />
    </>
  );
}
