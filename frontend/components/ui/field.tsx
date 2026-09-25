import {
  forwardRef,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
} from "react";

import { cn } from "@/lib/utils";

const control =
  "w-full rounded-md border border-border bg-surface px-3 text-body text-foreground shadow-sm transition-colors " +
  "placeholder:text-subtle hover:border-subtle disabled:cursor-not-allowed disabled:opacity-50 " +
  "aria-[invalid=true]:border-reject aria-[invalid=true]:ring-1 aria-[invalid=true]:ring-reject/30";

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => <input ref={ref} className={cn(control, "h-9", className)} {...props} />,
);
Input.displayName = "Input";

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(
  ({ className, ...props }, ref) => <textarea ref={ref} className={cn(control, "min-h-24 py-2", className)} {...props} />,
);
Textarea.displayName = "Textarea";

/** Native select: accessible and keyboard-friendly on every platform, styled to match the inputs. */
export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(
  ({ className, ...props }, ref) => <select ref={ref} className={cn(control, "h-9 pr-8", className)} {...props} />,
);
Select.displayName = "Select";

interface FieldProps {
  id: string;
  label: string;
  error?: string;
  hint?: ReactNode;
  children: ReactNode;
  className?: string;
}

/**
 * Label + control + hint/error. The control must use the same `id`, and `aria-describedby={`${id}-message`}` so
 * screen readers read the error with the field.
 */
export function Field({ id, label, error, hint, children, className }: FieldProps) {
  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <label htmlFor={id} className="text-label uppercase text-muted">
        {label}
      </label>
      {children}
      {error ? (
        <p id={`${id}-message`} role="alert" className="text-table text-reject">
          {error}
        </p>
      ) : hint ? (
        <p id={`${id}-message`} className="text-table text-muted">
          {hint}
        </p>
      ) : null}
    </div>
  );
}
