"use client";

import { FilePlus2, SearchX } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback } from "react";

import { claimColumns } from "@/components/claim-columns";
import { ClaimsFilterBar, STATUS_OPTIONS, type ClaimFilters } from "@/components/claims-filter-bar";
import { DataTable, type SortState } from "@/components/data-table";
import { EmptyState } from "@/components/empty-state";
import { ErrorState } from "@/components/error-state";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import type { ClaimSort } from "@/lib/api";
import { useClaims } from "@/lib/queries";
import type { ClaimStatus } from "@/lib/types";

const PAGE_SIZE = 25;
const SORTS: ClaimSort[] = ["created_at", "claimed_amount", "incident_date", "risk_score", "confidence"];
const FILTER_KEYS = ["status", "q", "date_from", "date_to", "amount_min", "amount_max"] as const;
const DATE = /^\d{4}-\d{2}-\d{2}$/;
const AMOUNT = /^\d+(\.\d{1,2})?$/;

const COLUMNS = [
  claimColumns.claim!,
  claimColumns.policy!,
  claimColumns.incident!,
  claimColumns.amount!,
  claimColumns.recommendation!,
  claimColumns.confidence!,
  claimColumns.risk!,
  claimColumns.status!,
  claimColumns.submitted!,
];

/**
 * The claims queue. Filters, sort and page live in the URL, so a view can be bookmarked or shared and the back button
 * restores it. Values from the URL are validated before use; a hand-edited URL can't send nonsense to the API.
 */
function ClaimsQueue() {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();

  const sortParam = params.get("sort");
  const sort: ClaimSort = SORTS.includes(sortParam as ClaimSort) ? (sortParam as ClaimSort) : "created_at";
  const order = params.get("order") === "asc" ? "asc" : "desc";
  const page = Math.max(1, Number.parseInt(params.get("page") ?? "1", 10) || 1);
  const valid = (key: string, pattern: RegExp) => {
    const value = params.get(key) ?? "";
    return pattern.test(value) ? value : "";
  };
  const filters: ClaimFilters = {
    status: params.getAll("status").filter((value): value is ClaimStatus => STATUS_OPTIONS.includes(value as ClaimStatus)),
    q: (params.get("q") ?? "").slice(0, 40),
    date_from: valid("date_from", DATE),
    date_to: valid("date_to", DATE),
    amount_min: valid("amount_min", AMOUNT),
    amount_max: valid("amount_max", AMOUNT),
  };

  const claims = useClaims({ ...filters, sort, order, page, page_size: PAGE_SIZE });

  const navigate = useCallback(
    (next: URLSearchParams) => router.replace(`${pathname}?${next.toString()}`, { scroll: false }),
    [pathname, router],
  );

  const update = (changes: Record<string, string | string[]>) => {
    const next = new URLSearchParams(params);
    Object.entries(changes).forEach(([key, value]) => {
      next.delete(key);
      (Array.isArray(value) ? value : [value]).filter(Boolean).forEach((item) => next.append(key, item));
    });
    navigate(next);
  };

  // Any filter change goes back to page 1: page 7 of the old result set means nothing for the new one.
  const onFilterChange = (changes: Partial<ClaimFilters>) => update({ ...changes, page: "1" });
  const onClear = () => {
    const next = new URLSearchParams(params);
    FILTER_KEYS.forEach((key) => next.delete(key));
    next.set("page", "1");
    navigate(next);
  };
  const onSortChange = (next: SortState) => update({ sort: next.id, order: next.direction, page: "1" });
  const filtered = FILTER_KEYS.some((key) => params.has(key));

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Claims"
        description="Every claim with its AI recommendation. Open one to see the full reasoning trace."
        actions={
          <Button asChild variant="primary">
            <Link href="/claims/new">
              <FilePlus2 aria-hidden />
              Submit claim
            </Link>
          </Button>
        }
      />
      <ClaimsFilterBar filters={filters} onChange={onFilterChange} onClear={onClear} />
      {claims.isError ? (
        <ErrorState error={claims.error} onRetry={() => claims.refetch()} />
      ) : (
        <DataTable
          caption="Claims queue. Use the arrow keys to move between rows and Enter to open a claim."
          columns={COLUMNS}
          rows={claims.data?.items}
          rowKey={(claim) => claim.id}
          rowLabel={(claim) => `Claim ${claim.claim_number}, ${claim.status.replace(/_/g, " ")}`}
          onRowActivate={(claim) => router.push(`/claims/${claim.id}`)}
          sort={{ id: sort, direction: order }}
          onSortChange={onSortChange}
          pagination={{
            page,
            pageSize: PAGE_SIZE,
            total: claims.data?.total ?? 0,
            onPageChange: (next) => update({ page: String(next) }),
          }}
          loading={claims.isPending}
          fetching={claims.isFetching}
          empty={
            filtered ? (
              <EmptyState
                icon={SearchX}
                title="No claims match these filters."
                action={<Button onClick={onClear}>Clear filters</Button>}
              />
            ) : (
              <EmptyState
                icon={SearchX}
                title="No claims yet."
                action={
                  <Button asChild variant="primary">
                    <Link href="/claims/new">Submit a claim</Link>
                  </Button>
                }
              />
            )
          }
        />
      )}
    </div>
  );
}

export default function ClaimsPage() {
  // useSearchParams needs a Suspense boundary so the rest of the page can render before the URL is read.
  return (
    <Suspense
      fallback={
        <div className="flex flex-col gap-6">
          <Skeleton className="h-8 w-40" />
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-96 w-full" />
        </div>
      }
    >
      <ClaimsQueue />
    </Suspense>
  );
}
