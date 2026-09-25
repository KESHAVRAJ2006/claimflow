"use client";

import { FileText, UploadCloud, X } from "lucide-react";
import { useId, useRef, useState, type DragEvent } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

// Mirrors the API limit. The server re-checks size and the PDF magic bytes; this only saves a pointless upload.
export const MAX_PDF_BYTES = 10 * 1024 * 1024;

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Client-side pre-check. Returns an error message, or null when the file looks acceptable. */
export function checkPdf(file: File): string | null {
  const looksLikePdf = file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
  if (!looksLikePdf) return "Only PDF documents are accepted.";
  if (file.size > MAX_PDF_BYTES) return `The file is ${formatBytes(file.size)}; the limit is 10 MB.`;
  if (file.size === 0) return "The file is empty.";
  return null;
}

interface Props {
  file: File | null;
  onChange: (file: File | null) => void;
  error?: string;
}

/** Drag-and-drop PDF zone: highlights while a file is dragged over it, and is also a plain file button for keyboards. */
export function PdfDropzone({ file, onChange, error }: Props) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const accept = (candidate: File | undefined) => {
    if (!candidate) return;
    const problem = checkPdf(candidate);
    setLocalError(problem);
    onChange(problem ? null : candidate);
  };

  const onDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragging(false);
    accept(event.dataTransfer.files[0]);
  };

  const message = localError ?? error;

  if (file) {
    return (
      <div className="flex items-center gap-3 rounded-card border border-border bg-surface px-4 py-3 shadow-sm">
        <span className="flex size-9 items-center justify-center rounded-md bg-surface-muted">
          <FileText aria-hidden className="size-4 text-muted" />
        </span>
        <div className="min-w-0 flex-1">
          <p className="truncate text-body font-medium text-foreground">{file.name}</p>
          <p className="font-mono text-table text-muted">{formatBytes(file.size)}</p>
        </div>
        <Button variant="ghost" size="icon-sm" aria-label={`Remove ${file.name}`} onClick={() => onChange(null)}>
          <X aria-hidden />
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1.5">
      <label
        htmlFor={inputId}
        onDragEnter={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          // Ignore leave events fired when moving between the zone's own children.
          if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false);
        }}
        onDrop={onDrop}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-card border border-dashed px-6 py-10 text-center transition-colors duration-150",
          dragging ? "border-foreground bg-surface-muted" : "border-border bg-surface hover:bg-surface-muted/60",
          message && "border-reject",
        )}
      >
        <UploadCloud aria-hidden className={cn("size-6", dragging ? "text-foreground" : "text-muted")} />
        <p className="text-body text-foreground">
          {dragging ? "Drop the PDF to attach it" : "Drag the claim document here, or click to choose"}
        </p>
        <p className="text-table text-muted">PDF with selectable text, up to 10 MB</p>
        <input
          ref={inputRef}
          id={inputId}
          type="file"
          accept="application/pdf,.pdf"
          className="sr-only"
          aria-invalid={Boolean(message)}
          aria-describedby={message ? `${inputId}-error` : undefined}
          onChange={(event) => {
            accept(event.target.files?.[0]);
            event.target.value = ""; // lets the same file be chosen again after removing it
          }}
        />
      </label>
      {message ? (
        <p id={`${inputId}-error`} role="alert" className="text-table text-reject">
          {message}
        </p>
      ) : null}
    </div>
  );
}
