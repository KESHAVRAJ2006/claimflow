/**
 * HTTP Basic auth in front of the whole console, including the /api proxy, when CONSOLE_PASSWORD is set.
 *
 * Why: the proxy adds the FastAPI key server-side, so without this anyone who can reach a deployed console could
 * approve claims and spend LLM credits. Basic auth over HTTPS is a small, honest gate for a demo deployment; a
 * real insurer would put SSO here. Locally, leave CONSOLE_PASSWORD unset and nothing changes.
 *
 * Browsers remember the credentials for the origin and send them on every later request, fetch and EventSource
 * (the live trace) included, so the reviewer signs in once.
 */
import { NextResponse, type NextRequest } from "next/server";

const REALM = "ClaimFlow console";

/** Compare in constant time, so response timing reveals nothing about how much of a guess was right. */
function safeEqual(a: string, b: string): boolean {
  const left = new TextEncoder().encode(a);
  const right = new TextEncoder().encode(b);
  // Compare against itself on a length mismatch, so both paths do the same amount of work.
  const other = left.length === right.length ? right : left;
  let diff = left.length ^ right.length;
  for (let i = 0; i < left.length; i += 1) diff |= left[i]! ^ other[i]!;
  return diff === 0;
}

function credentials(header: string | null): { user: string; password: string } | null {
  if (!header?.startsWith("Basic ")) return null;
  let decoded: string;
  try {
    decoded = atob(header.slice(6));
  } catch {
    return null;
  }
  const colon = decoded.indexOf(":");
  if (colon < 0) return null;
  return { user: decoded.slice(0, colon), password: decoded.slice(colon + 1) };
}

export function middleware(request: NextRequest) {
  const password = process.env.CONSOLE_PASSWORD;
  if (!password) return NextResponse.next();
  const user = process.env.CONSOLE_USER || "reviewer";

  const given = credentials(request.headers.get("authorization"));
  // Both checks always run (no short-circuit), so a wrong user name takes as long as a wrong password.
  const userOk = safeEqual(given?.user ?? "", user);
  const passwordOk = safeEqual(given?.password ?? "", password);
  if (given && userOk && passwordOk) return NextResponse.next();

  return new NextResponse("Sign in to use the ClaimFlow console.", {
    status: 401,
    headers: { "WWW-Authenticate": `Basic realm="${REALM}", charset="UTF-8"`, "Cache-Control": "no-store" },
  });
}

export const config = {
  // Everything except Next's static assets (hashed file names, no data in them) and the platform's health probe.
  matcher: ["/((?!_next/static|_next/image|favicon.ico|healthz$).*)"],
};
