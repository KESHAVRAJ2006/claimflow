"use client";

import { useMutation } from "@tanstack/react-query";
import { Loader2, Send } from "lucide-react";
import { useRouter } from "next/navigation";
import { useMemo, useState, type FormEvent } from "react";

import { PageHeader } from "@/components/page-header";
import { PdfDropzone } from "@/components/pdf-dropzone";
import { Button } from "@/components/ui/button";
import { Field, Input, Select, Textarea } from "@/components/ui/field";
import { ApiError, api } from "@/lib/api";
import { humanize } from "@/lib/format";
import type { ClaimSubmission, IncidentType, ProductType } from "@/lib/types";

// Mirrors backend app/domain/products.py: which incidents each product can cover.
const INCIDENTS: Record<ProductType, IncidentType[]> = {
  motor: ["collision", "theft", "vandalism", "natural_calamity"],
  health: ["hospitalisation", "surgery", "day_care"],
  home: ["fire", "burglary", "water_damage", "natural_calamity"],
};
const PRODUCT_BY_PREFIX: Record<string, ProductType> = { MOT: "motor", HLT: "health", HOM: "home" };
const POLICY_PATTERN = /^(MOT|HLT|HOM)-\d{4}-\d{6}$/;
const AMOUNT_PATTERN = /^\d{1,12}(\.\d{1,2})?$/;

type FormField = keyof ClaimSubmission | "document";
type Errors = Partial<Record<FormField, string>>;

function validate(form: ClaimSubmission, file: File | null): Errors {
  const errors: Errors = {};
  if (!POLICY_PATTERN.test(form.policy_number)) errors.policy_number = "Use the format MOT-2025-000123.";
  if (!form.incident_type) errors.incident_type = "Choose the type of incident.";
  if (!form.incident_date) errors.incident_date = "Enter the date of the incident.";
  if (!AMOUNT_PATTERN.test(form.claimed_amount) || Number(form.claimed_amount) <= 0)
    errors.claimed_amount = "Enter an amount greater than zero, with at most 2 decimals.";
  if (form.description.trim().length < 10) errors.description = "Describe what happened in at least 10 characters.";
  if (!file) errors.document = "Attach the supporting PDF.";
  return errors;
}

/** Map the API's validation errors (location ["body","claim","claimed_amount"]) onto the form's fields. */
function serverErrors(error: unknown): Errors {
  if (!(error instanceof ApiError)) return {};
  const fields: Errors = {};
  for (const item of error.problem.errors ?? []) {
    const field = item.location.at(-1);
    if (typeof field === "string") fields[field as FormField] = item.message;
  }
  const slug = error.problem.type.split(":").at(-1);
  if (slug === "unknown-policy") fields.policy_number = error.problem.detail;
  if (slug === "incident-type-not-covered-by-product") fields.incident_type = error.problem.detail;
  if (slug && ["not-a-pdf", "unreadable-pdf", "no-text-in-pdf", "too-many-pages", "file-too-large", "payload-too-large"].includes(slug))
    fields.document = error.problem.detail;
  return fields;
}

export default function SubmitClaimPage() {
  const router = useRouter();
  const [form, setForm] = useState<ClaimSubmission>({
    policy_number: "",
    incident_type: "" as IncidentType,
    incident_date: "",
    claimed_amount: "",
    description: "",
  });
  const [file, setFile] = useState<File | null>(null);
  const [errors, setErrors] = useState<Errors>({});
  const today = new Date().toISOString().slice(0, 10);

  const product = PRODUCT_BY_PREFIX[form.policy_number.slice(0, 3).toUpperCase()];
  const incidentOptions = useMemo(() => (product ? INCIDENTS[product] : []), [product]);

  const submit = useMutation({
    mutationFn: () => api.submitClaim(form, file!),
    // Straight to the claim: the reviewer watches the agents work in the live trace.
    onSuccess: (accepted) => router.push(`/claims/${accepted.claim_id}`),
    onError: (error) => setErrors(serverErrors(error)),
  });

  const set = <K extends keyof ClaimSubmission>(key: K, value: ClaimSubmission[K]) => {
    setForm((current) => ({ ...current, [key]: value }));
    setErrors((current) => ({ ...current, [key]: undefined }));
  };

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    const found = validate(form, file);
    setErrors(found);
    if (Object.keys(found).length === 0) submit.mutate();
  };

  const unmappedError =
    submit.error instanceof ApiError && Object.keys(serverErrors(submit.error)).length === 0 ? submit.error.problem : null;

  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col gap-6">
      <PageHeader
        title="Submit claim"
        description="The claim is stored, then triaged by the agents in the background. You'll see the live trace next."
      />
      <form noValidate onSubmit={onSubmit} className="flex flex-col gap-5 rounded-card border border-border bg-surface p-6 shadow-sm">
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
          <Field id="policy_number" label="Policy number" error={errors.policy_number} hint="e.g. MOT-2026-000001">
            <Input
              id="policy_number"
              value={form.policy_number}
              onChange={(event) => set("policy_number", event.target.value.toUpperCase().trim())}
              className="font-mono"
              autoComplete="off"
              aria-invalid={Boolean(errors.policy_number)}
              aria-describedby="policy_number-message"
            />
          </Field>
          <Field
            id="incident_type"
            label="Incident type"
            error={errors.incident_type}
            hint={product ? `Incidents a ${product} policy can cover` : "Enter the policy number first"}
          >
            <Select
              id="incident_type"
              value={form.incident_type}
              disabled={!product}
              onChange={(event) => set("incident_type", event.target.value as IncidentType)}
              aria-invalid={Boolean(errors.incident_type)}
              aria-describedby="incident_type-message"
            >
              <option value="">Choose…</option>
              {incidentOptions.map((type) => (
                <option key={type} value={type}>
                  {humanize(type)}
                </option>
              ))}
            </Select>
          </Field>
          <Field id="incident_date" label="Incident date" error={errors.incident_date}>
            <Input
              id="incident_date"
              type="date"
              max={today}
              value={form.incident_date}
              onChange={(event) => set("incident_date", event.target.value)}
              className="font-mono"
              aria-invalid={Boolean(errors.incident_date)}
              aria-describedby="incident_date-message"
            />
          </Field>
          <Field id="claimed_amount" label="Amount claimed (₹)" error={errors.claimed_amount}>
            <Input
              id="claimed_amount"
              inputMode="decimal"
              placeholder="42000.00"
              value={form.claimed_amount}
              onChange={(event) => set("claimed_amount", event.target.value.replace(/[^\d.]/g, ""))}
              className="font-mono"
              aria-invalid={Boolean(errors.claimed_amount)}
              aria-describedby="claimed_amount-message"
            />
          </Field>
        </div>
        <Field id="description" label="What happened" error={errors.description}>
          <Textarea
            id="description"
            value={form.description}
            maxLength={2000}
            onChange={(event) => set("description", event.target.value)}
            aria-invalid={Boolean(errors.description)}
            aria-describedby="description-message"
          />
        </Field>
        <div className="flex flex-col gap-1.5">
          <span className="text-label uppercase text-muted">Supporting document</span>
          <PdfDropzone
            file={file}
            onChange={(next) => {
              setFile(next);
              setErrors((current) => ({ ...current, document: undefined }));
            }}
            error={errors.document}
          />
        </div>
        {unmappedError ? (
          <p role="alert" className="rounded-md bg-reject-soft px-3 py-2 text-table text-reject">
            {unmappedError.title}. {unmappedError.detail}
          </p>
        ) : null}
        <div className="flex items-center justify-between gap-4 border-t border-border pt-5">
          <p className="text-table text-muted">The AI recommends; a human reviewer always decides.</p>
          <Button type="submit" variant="primary" disabled={submit.isPending}>
            {submit.isPending ? <Loader2 aria-hidden className="motion-safe:animate-spin" /> : <Send aria-hidden />}
            Submit claim
          </Button>
        </div>
      </form>
    </div>
  );
}
