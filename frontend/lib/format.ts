/**
 * Display formatting. Money arrives as a decimal string and is formatted WITHOUT converting to a float: the integer
 * part goes through BigInt, so "123456789012.35" can never display as "...012.34" from binary rounding.
 */

const INR_GROUPING = new Intl.NumberFormat("en-IN");

/** "42000.5" -> "₹42,000.50" (Indian digit grouping: 1,00,000). */
export function formatMoney(value: string | null | undefined): string {
  if (value == null || value === "") return "—";
  const negative = value.startsWith("-");
  const [integer = "0", fraction = ""] = value.replace("-", "").split(".");
  const grouped = INR_GROUPING.format(BigInt(integer || "0"));
  return `${negative ? "-" : ""}₹${grouped}.${fraction.padEnd(2, "0").slice(0, 2)}`;
}

/** A 0–1 rate as a percentage: 0.4231 -> "42.3%". */
export function formatRate(value: number | null | undefined): string {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

/** A decimal-string confidence as a percentage: "0.870" -> "87%". Display only. */
export function formatConfidence(value: string | null | undefined): string {
  if (value == null) return "—";
  return `${Math.round(Number(value) * 100)}%`;
}

/** Milliseconds, switching to seconds above one second: 850 -> "850 ms", 12400 -> "12.4 s". */
export function formatLatency(ms: number | null | undefined): string {
  if (ms == null) return "—";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

const DATE = new Intl.DateTimeFormat("en-GB", { day: "2-digit", month: "short", year: "numeric", timeZone: "UTC" });
const DATE_TIME = new Intl.DateTimeFormat("en-GB", {
  day: "2-digit",
  month: "short",
  hour: "2-digit",
  minute: "2-digit",
});

/** "2026-08-01" -> "01 Aug 2026". Parsed as UTC so a calendar date never shifts a day in western timezones. */
export function formatDate(value: string | null | undefined): string {
  if (!value) return "—";
  return DATE.format(new Date(value.length === 10 ? `${value}T00:00:00Z` : value));
}

/** A timestamp in the viewer's local time: "01 Aug, 14:05". */
export function formatDateTime(value: string | null | undefined): string {
  return value ? DATE_TIME.format(new Date(value)) : "—";
}

/** "awaiting_review" -> "Awaiting review". */
export function humanize(value: string): string {
  const spaced = value.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}
