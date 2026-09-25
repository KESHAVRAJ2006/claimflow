/**
 * Liveness probe for the platform (Render's healthCheckPath). Says only that this Next.js server answers: it does
 * not call the API, so an API outage cannot restart the console, and it is exempt from the Basic auth in
 * middleware.ts because a load balancer has no credentials.
 */
export const dynamic = "force-dynamic";

export function GET(): Response {
  return new Response("ok", { headers: { "content-type": "text/plain", "cache-control": "no-store" } });
}
