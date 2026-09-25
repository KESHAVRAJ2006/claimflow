/**
 * Typed API client. Every call goes to this app's own /api/* route, which the Next.js server proxies to FastAPI
 * with the API key attached (app/api/[...path]/route.ts), so no secret is ever shipped to the browser.
 *
 * Errors are thrown as ApiError carrying the RFC 7807 problem document the API returns for every failure.
 */
import type {
  ClaimAccepted,
  ClaimDetail,
  ClaimPage,
  ClaimStatus,
  ClaimSubmission,
  DecisionRequest,
  DecisionResponse,
  HealthReport,
  Metrics,
  PolicySearchResponse,
  Problem,
  ProductType,
  RunStarted,
} from "@/lib/types";

export class ApiError extends Error {
  readonly problem: Problem;

  constructor(problem: Problem) {
    super(problem.detail || problem.title);
    this.name = "ApiError";
    this.problem = problem;
  }

  get status(): number {
    return this.problem.status;
  }
}

async function toProblem(response: Response): Promise<Problem> {
  try {
    const body = (await response.json()) as Partial<Problem>;
    if (body && typeof body.title === "string") return { ...body, status: response.status } as Problem;
  } catch {
    // Not JSON (e.g. a proxy error page): fall through to a generic problem.
  }
  return {
    type: "urn:claimflow:problem:http-error",
    title: response.statusText || "Request failed",
    status: response.status,
    detail: `The server answered ${response.status}.`,
    instance: new URL(response.url, "http://x").pathname,
    request_id: response.headers.get("x-request-id"),
  };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, { ...init, headers: { Accept: "application/json", ...init?.headers } });
  } catch {
    throw new ApiError({
      type: "urn:claimflow:problem:network-error",
      title: "Network error",
      status: 0,
      detail: "Could not reach the ClaimFlow server. Check your connection and try again.",
      instance: path,
      request_id: null,
    });
  }
  if (!response.ok) throw new ApiError(await toProblem(response));
  return (await response.json()) as T;
}

/** Build a query string, repeating array values (?status=a&status=b) and dropping empty ones. */
function query(params: Record<string, string | number | readonly string[] | null | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value == null || value === "") continue;
    if (Array.isArray(value)) value.forEach((item) => search.append(key, item));
    else search.set(key, String(value));
  }
  const text = search.toString();
  return text ? `?${text}` : "";
}

export type ClaimSort = "created_at" | "claimed_amount" | "incident_date" | "risk_score" | "confidence";

export interface ClaimListParams {
  status?: readonly ClaimStatus[];
  date_from?: string;
  date_to?: string;
  amount_min?: string;
  amount_max?: string;
  q?: string;
  sort?: ClaimSort;
  order?: "asc" | "desc";
  page?: number;
  page_size?: number;
}

export const api = {
  health: () => request<HealthReport>("/health"),
  metrics: (days = 30) => request<Metrics>(`/metrics${query({ days })}`),
  listClaims: (params: ClaimListParams = {}) => request<ClaimPage>(`/claims${query({ ...params })}`),
  getClaim: (id: string) => request<ClaimDetail>(`/claims/${encodeURIComponent(id)}`),
  submitClaim: (claim: ClaimSubmission, document: File) => {
    const form = new FormData();
    form.set("claim", JSON.stringify(claim));
    form.set("document", document);
    // No Content-Type header: the browser sets multipart/form-data with the boundary itself.
    return request<ClaimAccepted>("/claims", { method: "POST", body: form });
  },
  runClaim: (id: string) => request<RunStarted>(`/claims/${encodeURIComponent(id)}/run`, { method: "POST" }),
  decide: (id: string, body: DecisionRequest) =>
    request<DecisionResponse>(`/claims/${encodeURIComponent(id)}/decision`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  searchPolicies: (q: string, productType?: ProductType, limit = 5) =>
    request<PolicySearchResponse>(`/policies/search${query({ q, product_type: productType, limit })}`),
  /** URL for an EventSource; the proxy streams it through unbuffered. */
  streamUrl: (id: string) => `/api/claims/${encodeURIComponent(id)}/stream`,
};
