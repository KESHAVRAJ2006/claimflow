"use client";

import { BookOpenText } from "lucide-react";
import { useState } from "react";

import { Sheet, SheetContent, SheetDescription, SheetTitle } from "@/components/ui/sheet";
import { findQuote } from "@/lib/highlight";
import type { EvidencePassage } from "@/lib/types";

export interface CitationTarget {
  evidenceId: string;
  label: string;
  quote: string | null;
}

/** "Motor_Policy.pdf p.14" -> "Motor_Policy.pdf · p.14" */
const chipLabel = (label: string) => label.replace(/ p\.(\d+)$/, " · p.$1");

export function CitationChip({ citation, onOpen }: { citation: CitationTarget; onOpen: (c: CitationTarget) => void }) {
  return (
    <button
      type="button"
      onClick={() => onOpen(citation)}
      className="inline-flex items-center gap-1.5 rounded-md border border-border bg-surface px-2 py-0.5 font-mono text-table text-foreground shadow-sm transition-colors hover:bg-surface-muted"
    >
      <BookOpenText aria-hidden className="size-3.5 text-muted" />
      {chipLabel(citation.label)}
      <span className="text-subtle">{citation.evidenceId}</span>
    </button>
  );
}

/** Shows the retrieved passage behind a citation, with the quoted words highlighted. */
export function CitationSheet({
  citation,
  evidence,
  onClose,
}: {
  citation: CitationTarget | null;
  evidence: EvidencePassage[];
  onClose: () => void;
}) {
  const passage = citation ? evidence.find((item) => item.evidence_id === citation.evidenceId) : undefined;
  const range = passage ? findQuote(passage.text, citation?.quote) : null;
  return (
    <Sheet open={citation != null} onOpenChange={(open) => !open && onClose()}>
      <SheetContent side="right" className="overflow-y-auto p-6">
        <SheetTitle className="pr-8 font-mono text-section">{citation ? chipLabel(citation.label) : ""}</SheetTitle>
        <SheetDescription className="mt-1 text-table text-muted">
          {passage
            ? `${passage.section ?? "Unknown section"} · evidence ${passage.evidence_id} · match confidence ${Math.round(passage.retrieval_confidence * 100)}% · found by ${passage.found_by}`
            : "The passage for this citation is not available."}
        </SheetDescription>
        {passage ? (
          <>
            <p className="mt-4 text-label uppercase text-muted">Retrieved passage</p>
            {/* Untrusted document text: rendered as plain text (React escapes it), never as HTML. */}
            <p className="mt-2 whitespace-pre-line rounded-card border border-border bg-background p-4 text-body leading-relaxed text-foreground">
              {range ? (
                <>
                  {passage.text.slice(0, range[0])}
                  <mark className="rounded bg-foreground/10 px-0.5 text-foreground underline decoration-foreground/40 decoration-2 underline-offset-2">
                    {passage.text.slice(range[0], range[1])}
                  </mark>
                  {passage.text.slice(range[1])}
                </>
              ) : (
                passage.text
              )}
            </p>
            {citation?.quote && !range ? (
              <p className="mt-2 text-table text-muted">The cited quote could not be located in this passage.</p>
            ) : null}
          </>
        ) : null}
      </SheetContent>
    </Sheet>
  );
}

/** Controller for a list of chips sharing one sheet. */
export function useCitationSheet() {
  const [open, setOpen] = useState<CitationTarget | null>(null);
  return { open, show: setOpen, close: () => setOpen(null) };
}
