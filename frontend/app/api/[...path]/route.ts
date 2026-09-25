/**
 * Server-side proxy: browser -> Next.js /api/* -> FastAPI /api/*.
 *
 * Why: the FastAPI key must not reach the browser, where anyone could read it from the network tab. This handler runs
 * on the Next.js server, adds X-API-Key there, and streams the response straight back (including SSE, unbuffered).
 *
 * Security note: this hides the key but does not add a login. Anyone who can open this console can use the API
 * through it, so a deployed console sets CONSOLE_PASSWORD (middleware.ts adds HTTP Basic auth).
 */
import type { NextRequest } from "next/server";

// Never cache or statically render: every request is live, and SSE responses must stream.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const RAW_BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";
// Render's fromService gives "host:port" with no scheme; that traffic stays on the private network, as plain HTTP.
const BACKEND_URL = /^https?:\/\//.test(RAW_BACKEND_URL) ? RAW_BACKEND_URL : `http://${RAW_BACKEND_URL}`;
// Only these API areas are reachable through the proxy (no /docs, no /openapi.json, no path tricks).
const ALLOWED_ROOTS = new Set(["claims", "metrics", "policies", "health"]);
// Request headers forwarded upstream; cookies and anything else the browser sends are dropped.
// content-length is not forwarded: the body is streamed (chunked); FastAPI still counts the bytes it receives.
const FORWARD_REQUEST = ["accept", "content-type", "last-event-id", "x-request-id"];
const FORWARD_RESPONSE = ["content-type", "cache-control", "x-request-id", "x-accel-buffering", "retry-after"];

function problem(status: number, title: string, detail: string, instance: string, slug?: string): Response {
  const type = `urn:claimflow:problem:${slug ?? (status === 404 ? "not-found" : "bad-gateway")}`;
  return Response.json(
    { type, title, status, detail, instance, request_id: null },
    { status, headers: { "content-type": "application/problem+json" } },
  );
}

// A 401 from the API is never the reviewer's fault: the browser sends no key, this server adds it. So it always
// means the two sides were configured with different keys, and saying that beats showing a bare 401.
const KEY_MISMATCH_DETAIL = process.env.API_KEY
  ? "The API rejected this console's API_KEY. Make API_KEY for the console (frontend/.env.local, or the service's " +
    "environment) equal to the API's API_KEY (root .env), then restart the console: Next.js reads env files only at startup."
  : "The console has no API_KEY. Add API_KEY to frontend/.env.local (same value as the root .env), then restart the console.";

async function proxy(request: NextRequest, { params }: { params: { path: string[] } }): Promise<Response> {
  const [root] = params.path;
  // ".." is rejected outright; each segment is re-encoded so it can't smuggle a "/" or "?" either.
  if (!root || !ALLOWED_ROOTS.has(root) || params.path.some((segment) => segment === ".." || segment === ".")) {
    return problem(404, "Not found", "No such API route.", request.nextUrl.pathname);
  }
  const target = new URL(`/api/${params.path.map(encodeURIComponent).join("/")}${request.nextUrl.search}`, BACKEND_URL);

  const headers = new Headers();
  for (const name of FORWARD_REQUEST) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  if (process.env.API_KEY) headers.set("x-api-key", process.env.API_KEY);

  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: request.method,
      headers,
      body: request.method === "GET" || request.method === "HEAD" ? undefined : request.body,
      // Required by Node's fetch to stream a request body (the multipart upload) instead of buffering it.
      duplex: "half",
      cache: "no-store",
      // If the browser closes the SSE stream, abort the upstream request too instead of leaking it.
      signal: request.signal,
    } as RequestInit & { duplex: "half" });
  } catch {
    return problem(502, "API unreachable", `Could not reach the ClaimFlow API at ${BACKEND_URL}.`, request.nextUrl.pathname);
  }

  if (upstream.status === 401) {
    await upstream.body?.cancel();
    return problem(502, "Console and API keys differ", KEY_MISMATCH_DETAIL, request.nextUrl.pathname, "api-key-mismatch");
  }

  const responseHeaders = new Headers();
  for (const name of FORWARD_RESPONSE) {
    const value = upstream.headers.get(name);
    if (value) responseHeaders.set(name, value);
  }
  // Passing the body stream through (not await .text()) is what keeps SSE events arriving one by one.
  return new Response(upstream.body, { status: upstream.status, headers: responseHeaders });
}

export { proxy as GET, proxy as POST };
