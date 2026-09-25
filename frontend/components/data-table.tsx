"use client";

import { ArrowDown, ArrowUp, ChevronLeft, ChevronRight, ChevronsUpDown } from "lucide-react";
import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

export interface Column<T> {
  id: string;
  header: string;
  cell: (row: T) => ReactNode;
  sortable?: boolean;
  align?: "left" | "right";
  className?: string;
}

export interface SortState {
  id: string;
  direction: "asc" | "desc";
}

export interface Pagination {
  page: number;
  pageSize: number;
  total: number;
  onPageChange: (page: number) => void;
}

interface DataTableProps<T> {
  /** Screen-reader description of the table. */
  caption: string;
  columns: Column<T>[];
  rows: T[] | undefined;
  rowKey: (row: T) => string;
  /** Accessible name for a row, e.g. "Claim CLM-2026-000012". */
  rowLabel?: (row: T) => string;
  onRowActivate?: (row: T) => void;
  sort?: SortState;
  onSortChange?: (sort: SortState) => void;
  pagination?: Pagination;
  /** First load: shows skeleton rows. */
  loading?: boolean;
  /** Background refetch (new page or sort): keeps rows visible but dimmed. */
  fetching?: boolean;
  empty: ReactNode;
  skeletonRows?: number;
}

/**
 * Dense data table: server-side sorting and pagination, keyboard navigation and skeleton loading.
 *
 * Keyboard: Tab moves into the table onto one row (roving tabindex, so the table is a single tab stop). ↑/↓ move
 * between rows, Home/End jump to the first or last, Enter opens the row, and PageDown/PageUp change page.
 */
export function DataTable<T>({
  caption,
  columns,
  rows,
  rowKey,
  rowLabel,
  onRowActivate,
  sort,
  onSortChange,
  pagination,
  loading = false,
  fetching = false,
  empty,
  skeletonRows = 8,
}: DataTableProps<T>) {
  const [activeIndex, setActiveIndex] = useState(0);
  const rowRefs = useRef<(HTMLTableRowElement | null)[]>([]);
  const count = rows?.length ?? 0;

  // A new page resets focus to its first row.
  useEffect(() => setActiveIndex(0), [pagination?.page, count]);

  const focusRow = (index: number) => {
    const next = Math.max(0, Math.min(count - 1, index));
    setActiveIndex(next);
    rowRefs.current[next]?.focus();
  };

  const pageCount = pagination ? Math.max(1, Math.ceil(pagination.total / pagination.pageSize)) : 1;

  const onKeyDown = (event: KeyboardEvent<HTMLTableSectionElement>) => {
    const keys: Record<string, () => void> = {
      ArrowDown: () => focusRow(activeIndex + 1),
      ArrowUp: () => focusRow(activeIndex - 1),
      Home: () => focusRow(0),
      End: () => focusRow(count - 1),
      Enter: () => rows?.[activeIndex] && onRowActivate?.(rows[activeIndex]),
      PageDown: () => pagination && pagination.page < pageCount && pagination.onPageChange(pagination.page + 1),
      PageUp: () => pagination && pagination.page > 1 && pagination.onPageChange(pagination.page - 1),
    };
    const action = keys[event.key];
    if (action) {
      event.preventDefault(); // stop the page from scrolling as well
      action();
    }
  };

  const toggleSort = (column: Column<T>) => {
    if (!onSortChange) return;
    // A new column sorts descending first: largest amounts and newest claims are what reviewers look for.
    const direction = sort?.id === column.id && sort.direction === "desc" ? "asc" : "desc";
    onSortChange({ id: column.id, direction });
  };

  return (
    <div className="overflow-hidden rounded-card border border-border bg-surface shadow-sm">
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-left text-table">
          <caption className="sr-only">{caption}</caption>
          <thead className="border-b border-border bg-surface-muted/60">
            <tr>
              {columns.map((column) => {
                const sorted = sort?.id === column.id ? sort.direction : null;
                return (
                  <th
                    key={column.id}
                    scope="col"
                    aria-sort={sorted ? (sorted === "asc" ? "ascending" : "descending") : undefined}
                    className={cn(
                      "h-9 whitespace-nowrap px-4 text-label uppercase text-muted",
                      column.align === "right" && "text-right",
                      column.className,
                    )}
                  >
                    {column.sortable && onSortChange ? (
                      <button
                        type="button"
                        onClick={() => toggleSort(column)}
                        className={cn(
                          "-mx-1 inline-flex items-center gap-1 rounded px-1 uppercase transition-colors hover:text-foreground",
                          sorted && "text-foreground",
                        )}
                      >
                        {column.header}
                        {sorted === "asc" ? (
                          <ArrowUp aria-hidden className="size-3" />
                        ) : sorted === "desc" ? (
                          <ArrowDown aria-hidden className="size-3" />
                        ) : (
                          <ChevronsUpDown aria-hidden className="size-3 opacity-50" />
                        )}
                      </button>
                    ) : (
                      column.header
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody
            onKeyDown={onKeyDown}
            className={cn("transition-opacity duration-150", fetching && !loading && "opacity-60")}
          >
            {loading
              ? Array.from({ length: skeletonRows }, (_, index) => (
                  <tr key={index} className="border-b border-border last:border-0">
                    {columns.map((column) => (
                      <td key={column.id} className="h-11 px-4">
                        <Skeleton className={cn("h-4", column.align === "right" ? "ml-auto w-16" : "w-24")} />
                      </td>
                    ))}
                  </tr>
                ))
              : rows?.map((row, index) => (
                  <tr
                    key={rowKey(row)}
                    ref={(element) => {
                      rowRefs.current[index] = element;
                    }}
                    tabIndex={index === activeIndex ? 0 : -1}
                    aria-label={rowLabel?.(row)}
                    onClick={() => {
                      setActiveIndex(index);
                      onRowActivate?.(row);
                    }}
                    onFocus={() => setActiveIndex(index)}
                    className={cn(
                      "border-b border-border outline-none transition-colors duration-150 last:border-0",
                      "focus-visible:bg-surface-muted focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring focus-visible:ring-offset-0",
                      onRowActivate && "cursor-pointer hover:bg-surface-muted/70",
                    )}
                  >
                    {columns.map((column) => (
                      <td
                        key={column.id}
                        className={cn(
                          "h-11 whitespace-nowrap px-4 text-foreground",
                          column.align === "right" && "text-right",
                          column.className,
                        )}
                      >
                        {column.cell(row)}
                      </td>
                    ))}
                  </tr>
                ))}
          </tbody>
        </table>
        {!loading && count === 0 ? empty : null}
      </div>

      {pagination && pagination.total > 0 ? (
        <div className="flex items-center justify-between gap-4 border-t border-border px-4 py-2">
          <p className="text-table text-muted">
            <span className="font-mono text-foreground">
              {(pagination.page - 1) * pagination.pageSize + 1}–
              {Math.min(pagination.page * pagination.pageSize, pagination.total)}
            </span>{" "}
            of <span className="font-mono text-foreground">{pagination.total}</span>
          </p>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label="Previous page"
              disabled={pagination.page <= 1}
              onClick={() => pagination.onPageChange(pagination.page - 1)}
            >
              <ChevronLeft />
            </Button>
            <span className="px-1 font-mono text-table text-muted">
              {pagination.page} / {pageCount}
            </span>
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label="Next page"
              disabled={pagination.page >= pageCount}
              onClick={() => pagination.onPageChange(pagination.page + 1)}
            >
              <ChevronRight />
            </Button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
