/**
 * TypeScript mirrors of the API's Pydantic models (backend/app/schemas, app/agents/state.py, app/rules/models.py).
 *
 * Conventions:
 * - Money, confidence and other Decimals arrive as strings ("42000.00", "0.870"). Never parse them into floats
 *   for arithmetic; lib/format.ts formats them for display without going through a float.
 * - Dates are ISO strings: "2026-08-01" (date) or "2026-08-01T10:00:00Z" (date-time).
 *
 * `npm run check:types` compares every interface here, field by field, with the live /api/openapi.json.
 */

// ---- enums -----------------------------------------------------------------------------------------------------

export type ClaimStatus =
  | "submitted"
  | "processing"
  | "awaiting_review"
  | "escalated"
  | "approved"
  | "rejected"
  | "failed";
export type RecommendedOutcome = "approve" | "reject" | "escalate";
export type FinalOutcome = "approve" | "reject";
export type ProductType = "motor" | "health" | "home";
export type IncidentType =
  | "collision"
  | "theft"
  | "vandalism"
  | "natural_calamity"
  | "hospitalisation"
  | "surgery"
  | "day_care"
  | "fire"
  | "burglary"
  | "water_damage";
export type Severity = "low" | "medium" | "high" | "critical";
export type SectionKind =
  | "coverage"
  | "limits"
  | "exclusions"
  | "waiting_periods"
  | "claims_procedure"
  | "definitions"
  | "conditions"
  | "general";

/** A JSON value whose shape the UI treats as opaque (node inputs/outputs, rule evidence). */
export type JsonObject = { [key: string]: unknown };

// ---- claims ----------------------------------------------------------------------------------------------------

export interface ClaimSubmission {
  policy_number: string;
  incident_type: IncidentType;
  incident_date: string;
  claimed_amount: string;
  description: string;
}

export interface ClaimAccepted {
  claim_id: string;
  claim_number: string;
  status: ClaimStatus;
  triage: "queued" | "unavailable";
  stream_url: string;
}

export interface ClaimSummary {
  id: string;
  claim_number: string;
  policy_number: string;
  product_type: ProductType;
  incident_type: IncidentType;
  incident_date: string;
  claimed_amount: string;
  status: ClaimStatus;
  recommended_outcome: RecommendedOutcome | null;
  confidence: string | null;
  risk_score: number | null;
  final_outcome: FinalOutcome | null;
  created_at: string;
}

export interface ClaimPage {
  items: ClaimSummary[];
  page: number;
  page_size: number;
  total: number;
}

export interface ClaimRunOut {
  id: string;
  run_id: string | null;
  agent_name: string;
  round: number;
  input: JsonObject;
  output: JsonObject;
  latency_ms: number;
  created_at: string;
}

export interface AuditEntryOut {
  id: number;
  actor: string;
  action: string;
  before: JsonObject | null;
  after: JsonObject | null;
  reason: string | null;
  created_at: string;
}

export interface ToolCallRecord {
  call_id: string;
  round: number;
  step: number;
  tool: string;
  args: JsonObject;
  status: "ok" | "error";
  result_summary: string;
  result_data: JsonObject | null;
  latency_ms: number;
  started_at: string;
}

export interface EvidencePassage {
  evidence_id: string;
  chunk_id: string;
  document: string;
  page: number;
  section: string | null;
  text: string;
  similarity: number;
  retrieval_confidence: number;
  found_by: string;
}

export interface RuleResult {
  rule_id: string;
  name: string;
  triggered: boolean;
  severity: Severity;
  hard_block: boolean;
  score_contribution: number;
  explanation: string;
  evidence: Record<string, string>;
}

export interface RiskReport {
  results: RuleResult[];
  triggered_rule_ids: string[];
  hard_blocks: string[];
  risk_score: number;
}

export interface ResolvedCitation {
  evidence_id: string;
  document: string;
  page: number;
  section: string | null;
  quote: string | null;
  label: string;
}

export interface FinalRecommendation {
  outcome: RecommendedOutcome;
  reasons: string[];
  covered: boolean | null;
  agent_confidence: string | null;
  retrieval_confidence: string;
  confidence: string;
  risk_score: number;
  hard_blocks: string[];
  triggered_rules: string[];
  summary: string;
  citations: ResolvedCitation[];
  escalation_flags: string[];
  open_questions: string[];
  requires_human_review: true;
  disclaimer: string;
}

export interface ClaimDetail extends ClaimSummary {
  description: string;
  document_filename: string | null;
  extracted_fields: JsonObject | null;
  decision_rationale: string | null;
  decided_by: string | null;
  decided_at: string | null;
  override_reason: string | null;
  run_id: string | null;
  runs: ClaimRunOut[];
  tool_call_log: ToolCallRecord[];
  evidence: EvidencePassage[];
  risk_report: RiskReport | null;
  recommendation: FinalRecommendation | null;
  audit: AuditEntryOut[];
  disclaimer: string;
}

export interface DecisionRequest {
  action: "approve" | "reject" | "request_info";
  reviewer: string;
  reason?: string | null;
}

export interface DecisionResponse {
  claim_id: string;
  status: ClaimStatus;
  final_outcome: FinalOutcome | null;
  overridden: boolean;
  decided_by: string | null;
  decided_at: string | null;
}

export interface RunStarted {
  claim_id: string;
  status: ClaimStatus;
  stream_url: string;
}

// ---- metrics, search, health ------------------------------------------------------------------------------------

export interface VolumePoint {
  date: string;
  submitted: number;
  auto_decided: number;
  escalated: number;
}

export interface Metrics {
  window_days: number;
  generated_at: string;
  claims_today: number;
  claims_in_window: number;
  triaged_in_window: number;
  pending_review: number;
  auto_decision_rate: number | null;
  escalation_rate: number | null;
  override_rate: number | null;
  avg_tool_calls_per_claim: number | null;
  latency_ms_p50: number | null;
  latency_ms_p95: number | null;
  volume: VolumePoint[];
}

export interface PolicyPassage {
  chunk_id: string;
  document: string;
  page: number;
  section: string | null;
  section_kind: SectionKind;
  label: string;
  text: string;
  similarity: number;
}

export interface PolicySearchResponse {
  query: string;
  product_type: ProductType | null;
  retrieval_confidence: number;
  passages: PolicyPassage[];
}

export interface ComponentHealth {
  status: "up" | "down";
  latency_ms: number;
  error?: string | null;
}

export interface HealthReport {
  status: "ok" | "degraded";
  version: string;
  environment: string;
  timestamp: string;
  checks: Record<string, ComponentHealth>;
}

// ---- errors and streaming -------------------------------------------------------------------------------------

/** RFC 7807 problem document: the shape of every API error. */
export interface Problem {
  type: string;
  title: string;
  status: number;
  detail: string;
  instance: string;
  request_id: string | null;
  errors?: { location: (string | number)[]; message: string; type: string }[];
}

export type RunEventType =
  | "run_queued"
  | "node_started"
  | "node_finished"
  | "tool_call_started"
  | "tool_call_finished"
  | "run_completed"
  | "run_failed"
  | "run_unavailable";

/** One Server-Sent Event from GET /api/claims/{id}/stream. Fields depend on `type`. */
export interface RunEvent {
  id: number;
  type: RunEventType;
  run_id?: string;
  replayed?: boolean;
  node?: string;
  round?: number;
  at?: string;
  latency_ms?: number;
  output?: JsonObject;
  // tool_call_* events carry a ToolCallRecord's fields
  call_id?: string;
  step?: number;
  tool?: string;
  args?: JsonObject;
  status?: string;
  result_summary?: string;
  outcome?: RecommendedOutcome | null;
  reason?: string | null;
  detail?: string;
}
