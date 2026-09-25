"use client";

import { Search, X } from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/field";
import { humanize } from "@/lib/format";
import type { ClaimStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

export interface ClaimFilters {
  status: ClaimStatus[];
  q: string;
  date_from: string;
  date_to: string;
  amount_min: string;
  amount_max: string;
}

export const STATUS_OPTIONS: ClaimStatus[] = [
  "awaiting_review",
  "escalated",
  "processing",
  "submitted",
  "approved",
  "rejected",
  "failed",
];

/** A text value that commits after the user pauses typing, so each keystroke doesn't fire a request. */
function useDebounced(value: string, commit: (value: string) => void, delay = 300) {
  useEffect(() => {
    const timer = window.setTimeout(() => commit(value), delay);
    return () => window.clearTimeout(timer);
    // commit is recreated each render; only a change in value should restart the timer.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, delay]);
}

interface Props {
  filters: ClaimFilters;
  onChange: (changes: Partial<ClaimFilters>) => void;
  onClear: () => void;
}

/** One row of filters above the queue: status chips, search, submission date range, amount range. */
export function ClaimsFilterBar({ filters, onChange, onClear }: Props) {
  const [q, setQ] = useState(filters.q);
  const [amountMin, setAmountMin] = useState(filters.amount_min);
  const [amountMax, setAmountMax] = useState(filters.amount_max);

  // Keep local inputs in sync when the URL changes from elsewhere (Clear, back button).
  useEffect(() => setQ(filters.q), [filters.q]);
  useEffect(() => setAmountMin(filters.amount_min), [filters.amount_min]);
  useEffect(() => setAmountMax(filters.amount_max), [filters.amount_max]);

  useDebounced(q, (value) => value !== filters.q && onChange({ q: value }));
  useDebounced(amountMin, (value) => value !== filters.amount_min && onChange({ amount_min: value }));
  useDebounced(amountMax, (value) => value !== filters.amount_max && onChange({ amount_max: value }));

  const toggleStatus = (status: ClaimStatus) =>
    onChange({
      status: filters.status.includes(status) ? filters.status.filter((s) => s !== status) : [...filters.status, status],
    });

  const active =
    filters.status.length > 0 || filters.q || filters.date_from || filters.date_to || filters.amount_min || filters.amount_max;

  return (
    <div className="flex flex-col gap-3 rounded-card border border-border bg-surface p-4 shadow-sm">
      <div role="group" aria-label="Filter by status" className="flex flex-wrap gap-1.5">
        {STATUS_OPTIONS.map((status) => {
          const on = filters.status.includes(status);
          return (
            <button
              key={status}
              type="button"
              aria-pressed={on}
              onClick={() => toggleStatus(status)}
              className={cn(
                "rounded-md border px-2.5 py-1 text-table transition-colors",
                on ? "border-foreground bg-foreground text-background" : "border-border text-muted hover:bg-surface-muted hover:text-foreground",
              )}
            >
              {humanize(status)}
            </button>
          );
        })}
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-[1.4fr_1fr_1fr_1fr_1fr_auto]">
        <div className="relative">
          <Search aria-hidden className="pointer-events-none absolute left-3 top-2.5 size-4 text-subtle" />
          <Input
            aria-label="Search by claim or policy number"
            placeholder="Claim or policy number"
            value={q}
            onChange={(event) => setQ(event.target.value)}
            className="pl-9 font-mono"
          />
        </div>
        <Input
          type="date"
          aria-label="Submitted from"
          value={filters.date_from}
          max={filters.date_to || undefined}
          onChange={(event) => onChange({ date_from: event.target.value })}
        />
        <Input
          type="date"
          aria-label="Submitted to"
          value={filters.date_to}
          min={filters.date_from || undefined}
          onChange={(event) => onChange({ date_to: event.target.value })}
        />
        <Input
          inputMode="decimal"
          aria-label="Minimum amount"
          placeholder="Min ₹"
          value={amountMin}
          onChange={(event) => setAmountMin(event.target.value.replace(/[^\d.]/g, ""))}
          className="font-mono"
        />
        <Input
          inputMode="decimal"
          aria-label="Maximum amount"
          placeholder="Max ₹"
          value={amountMax}
          onChange={(event) => setAmountMax(event.target.value.replace(/[^\d.]/g, ""))}
          className="font-mono"
        />
        <Button variant="ghost" onClick={onClear} disabled={!active}>
          <X aria-hidden />
          Clear
        </Button>
      </div>
    </div>
  );
}
