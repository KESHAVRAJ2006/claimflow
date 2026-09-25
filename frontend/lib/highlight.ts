/**
 * Locate a quote inside a passage for highlighting, tolerant of the differences reflection's check also ignores:
 * letter case, runs of whitespace, curly vs straight quotes. With "…" in the quote, the longest fragment is used.
 */
const TYPOGRAPHY: Record<string, string> = { "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-" };

function normalise(text: string): { value: string; map: number[] } {
  let value = "";
  const map: number[] = [];
  let lastWasSpace = false;
  for (let index = 0; index < text.length; index += 1) {
    const raw = text[index]!;
    const char = (TYPOGRAPHY[raw] ?? raw).toLowerCase();
    if (/\s/.test(char)) {
      if (lastWasSpace) continue;
      value += " ";
      lastWasSpace = true;
    } else {
      value += char;
      lastWasSpace = false;
    }
    map.push(index);
  }
  return { value, map };
}

/** Returns [start, end) offsets into `text`, or null when the quote isn't found. */
export function findQuote(text: string, quote: string | null | undefined): [number, number] | null {
  if (!quote) return null;
  const fragment = quote
    .split(/\.\.\.|…/)
    .map((part) => part.trim().replace(/^["'“‘]+|["'”’.,;:]+$/g, ""))
    .sort((a, b) => b.length - a.length)[0];
  if (!fragment) return null;
  const haystack = normalise(text);
  const needle = normalise(fragment).value.trim();
  const start = haystack.value.indexOf(needle);
  if (start < 0 || !needle) return null;
  const first = haystack.map[start]!;
  const last = haystack.map[start + needle.length - 1]!;
  return [first, last + 1];
}
