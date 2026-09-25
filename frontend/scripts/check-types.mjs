/**
 * Verify lib/types.ts mirrors the API: every schema in /api/openapi.json must have a TypeScript interface (or string
 * union, for enums) with exactly the same field names or values. Run with the backend up: `npm run check:types`.
 *
 * Why a script instead of generating the types: hand-written types stay readable (comments, narrow unions such as
 * `status: "ok" | "error"`), and this check still makes drift fail loudly.
 */
import { readFileSync } from "node:fs";

const env = Object.fromEntries(
  (() => {
    try {
      return readFileSync(".env.local", "utf8").split(/\r?\n/).filter((l) => l.includes("=") && !l.startsWith("#")).map((l) => [l.slice(0, l.indexOf("=")), l.slice(l.indexOf("=") + 1)]);
    } catch {
      return [];
    }
  })(),
);
const backend = process.env.BACKEND_URL ?? env.BACKEND_URL ?? "http://localhost:8000";
const IGNORED = /^(HTTPValidationError|ValidationError|Body_)/;

const source = readFileSync("lib/types.ts", "utf8");
const interfaces = new Map();
for (const match of source.matchAll(/export interface (\w+)(?: extends (\w+))? \{([\s\S]*?)\n\}/g)) {
  const [, name, parent, body] = match;
  const fields = [...body.matchAll(/^ {2}(\w+)\??:/gm)].map((m) => m[1]);
  interfaces.set(name, { parent, fields });
}
const fieldsOf = (name) => {
  const entry = interfaces.get(name);
  return entry ? [...(entry.parent ? fieldsOf(entry.parent) : []), ...entry.fields] : null;
};
const unions = new Map();
for (const match of source.matchAll(/export type (\w+) =([^;]+);/g)) {
  unions.set(match[1], [...match[2].matchAll(/"([^"]+)"/g)].map((m) => m[1]));
}

let schemas;
try {
  schemas = (await (await fetch(`${backend}/api/openapi.json`)).json()).components.schemas;
} catch (error) {
  console.error(`check:types: cannot fetch ${backend}/api/openapi.json (${error.message}). Is the backend running?`);
  process.exit(2);
}

const problems = [];
let checked = 0;
for (const [name, schema] of Object.entries(schemas)) {
  if (IGNORED.test(name)) continue;
  checked += 1;
  const sorted = (items) => [...items].sort().join(", ");
  if (schema.enum) {
    const values = unions.get(name);
    if (!values) problems.push(`${name}: missing string union`);
    else if (sorted(values) !== sorted(schema.enum)) problems.push(`${name}: values [${sorted(values)}] != API [${sorted(schema.enum)}]`);
    continue;
  }
  const fields = fieldsOf(name);
  const expected = Object.keys(schema.properties ?? {});
  if (!fields) problems.push(`${name}: missing interface`);
  else if (sorted(fields) !== sorted(expected)) {
    const missing = expected.filter((f) => !fields.includes(f));
    const extra = fields.filter((f) => !expected.includes(f));
    problems.push(`${name}: missing [${missing.join(", ")}] extra [${extra.join(", ")}]`);
  }
}

if (problems.length) {
  console.error(`check:types: lib/types.ts has drifted from the API:\n  - ${problems.join("\n  - ")}`);
  process.exit(1);
}
console.log(`check:types: all ${checked} API schemas match lib/types.ts`);
