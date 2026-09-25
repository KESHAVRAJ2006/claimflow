"use client";

import { useQuery } from "@tanstack/react-query";
import { BookOpenText, Search } from "lucide-react";
import { useState, type FormEvent } from "react";

import { EmptyState } from "@/components/empty-state";
import { ErrorState } from "@/components/error-state";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Input, Select } from "@/components/ui/field";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { humanize } from "@/lib/format";
import type { ProductType } from "@/lib/types";

const EXAMPLES = ["theft of a parked car", "waiting period for cataract surgery", "burst pipe while the home was empty"];

export default function PoliciesPage() {
  const [draft, setDraft] = useState("");
  const [product, setProduct] = useState<ProductType | "">("");
  // The committed search: queries run on submit, not on every keystroke.
  const [search, setSearch] = useState<{ q: string; product: ProductType | "" } | null>(null);

  const results = useQuery({
    queryKey: ["policy-search", search],
    queryFn: () => api.searchPolicies(search!.q, search!.product || undefined, 8),
    enabled: search != null,
  });

  const run = (q: string) => {
    const trimmed = q.trim();
    if (trimmed.length >= 3) setSearch({ q: trimmed, product });
  };
  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    run(draft);
  };

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Policies"
        description="Search the indexed policy wordings. Every passage shows the document and page it came from."
      />
      <form onSubmit={onSubmit} className="flex flex-col gap-3 sm:flex-row">
        <div className="relative flex-1">
          <Search aria-hidden className="pointer-events-none absolute left-3 top-2.5 size-4 text-subtle" />
          <Input
            aria-label="Question about the policy wording"
            placeholder="Ask about cover, exclusions, waiting periods…"
            value={draft}
            minLength={3}
            onChange={(event) => setDraft(event.target.value)}
            className="pl-9"
          />
        </div>
        <Select
          aria-label="Product"
          value={product}
          onChange={(event) => setProduct(event.target.value as ProductType | "")}
          className="sm:w-40"
        >
          <option value="">All products</option>
          {(["motor", "health", "home"] as const).map((value) => (
            <option key={value} value={value}>
              {humanize(value)}
            </option>
          ))}
        </Select>
        <Button type="submit" variant="primary" disabled={draft.trim().length < 3}>
          Search
        </Button>
      </form>

      {!search ? (
        <div className="rounded-card border border-border bg-surface shadow-sm">
          <EmptyState
            icon={BookOpenText}
            title="Ask a question in plain English to find the clauses that answer it."
            action={
              <div className="flex flex-wrap justify-center gap-2">
                {EXAMPLES.map((example) => (
                  <Button
                    key={example}
                    size="sm"
                    onClick={() => {
                      setDraft(example);
                      run(example);
                    }}
                  >
                    {example}
                  </Button>
                ))}
              </div>
            }
          />
        </div>
      ) : results.isError ? (
        <ErrorState error={results.error} onRetry={() => results.refetch()} />
      ) : results.isPending ? (
        <div className="flex flex-col gap-4" aria-busy>
          {Array.from({ length: 3 }, (_, index) => (
            <Skeleton key={index} className="h-36 w-full rounded-card" />
          ))}
        </div>
      ) : results.data.passages.length === 0 ? (
        <div className="rounded-card border border-border bg-surface shadow-sm">
          <EmptyState
            icon={Search}
            title="No passage matches that question."
            action={<Button onClick={() => setSearch(null)}>Start a new search</Button>}
          />
        </div>
      ) : (
        <section aria-label="Results" className="flex flex-col gap-4">
          <p className="text-table text-muted">
            {results.data.passages.length} passages · retrieval confidence{" "}
            <span className="font-mono text-foreground">{Math.round(results.data.retrieval_confidence * 100)}%</span>
            {results.data.retrieval_confidence < 0.65 ? " (weak match: rephrase or pick a product)" : ""}
          </p>
          {results.data.passages.map((passage) => (
            <article key={passage.chunk_id} className="rounded-card border border-border bg-surface p-6 shadow-sm">
              <div className="flex flex-wrap items-center gap-2">
                <span className="rounded-md border border-border px-2 py-0.5 font-mono text-table text-foreground">
                  {passage.document} · p.{passage.page}
                </span>
                <span className="text-table text-muted">{passage.section ?? humanize(passage.section_kind)}</span>
                <span className="ml-auto font-mono text-table text-muted">similarity {passage.similarity.toFixed(2)}</span>
              </div>
              <p className="mt-3 whitespace-pre-line text-body text-foreground">{passage.text}</p>
            </article>
          ))}
        </section>
      )}
    </div>
  );
}
