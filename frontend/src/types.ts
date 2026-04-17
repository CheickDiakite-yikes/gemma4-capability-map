export type SessionStatus =
  | "pending"
  | "warming"
  | "running"
  | "awaiting_approval"
  | "resuming"
  | "retrying"
  | "interrupted"
  | "completed"
  | "denied"
  | "failed";

export interface SystemProfile {
  system_id: string;
  display_name: string;
  short_label: string;
  backend: string;
  deployment: string;
  local: boolean;
  recommended: boolean;
  request_timeout_seconds: number;
}

export interface WorkflowCard {
  workflow_id: string;
  title: string;
  subtitle: string;
  description: string;
  role_family: string;
  category: string;
  lane: string;
  episode_id: string;
  supports_approval: boolean;
  preview_asset: string;
  recommended_system_id: string;
}

export interface ApprovalRequest {
  approval_id: string;
  session_id: string;
  title: string;
  reason: string;
  status: "pending" | "approved" | "denied";
  suggested_action: "approve" | "deny";
  created_at: string;
  note: string;
}

export interface ToolInvocation {
  stage_id?: string | null;
  task_id?: string | null;
  tool_name: string;
  arguments: Record<string, unknown>;
  validator_result: "pass" | "fail" | "unknown";
}

export interface RuntimeEvent {
  event_id: string;
  session_id: string;
  sequence: number;
  kind: string;
  message: string;
  created_at: string;
  payload: Record<string, unknown>;
}

export interface InstructionRecord {
  instruction_id: string;
  session_id: string;
  project_id: string;
  source: string;
  content: string;
  created_at: string;
  note: string;
}

export interface ArtifactRevisionRecord {
  artifact_revision_id: string;
  session_id: string;
  artifact_id: string;
  title: string;
  revision: number;
  file_path: string;
  review_feedback: string;
  created_at: string;
}

export interface RuntimeTrace {
  session_id: string;
  workflow_id: string;
  episode_id: string;
  output_dir: string;
  manifest_path?: string | null;
  summary_path?: string | null;
  episode_trace_path?: string | null;
  artifact_paths: string[];
  artifact_count: number;
  review_count: number;
  scorecard: Record<string, unknown>;
  runtime_bundle: Record<string, unknown>;
  warmup: Record<string, unknown>;
}

export interface AgentSession {
  session_id: string;
  title: string;
  project_id: string;
  workflow_id: string;
  workflow_title: string;
  workflow_category: string;
  workflow_tags: string[];
  episode_id: string;
  system_id: string;
  lane: string;
  status: SessionStatus;
  created_at: string;
  updated_at: string;
  attempt: number;
  parent_session_id?: string | null;
  lineage_root_session_id?: string | null;
  retry_of_session_id?: string | null;
  resumed_from_session_id?: string | null;
  human_request: string;
  latest_message: string;
  progress_label: string;
  last_activity_at: string;
  hold_reason?: string | null;
  resumable: boolean;
  active_approval_id?: string | null;
  last_event_sequence: number;
  preview_asset?: string | null;
  latest_instruction: string;
  instruction_history: InstructionRecord[];
  artifact_history: ArtifactRevisionRecord[];
  latest_artifact_title: string;
  latest_artifact_path: string;
  latest_revision_artifact_id: string;
  latest_review_feedback: string;
  artifact_paths: string[];
  tool_invocations: ToolInvocation[];
  approvals: ApprovalRequest[];
  runtime_trace?: RuntimeTrace | null;
  metrics: Record<string, string | number | boolean>;
  last_error?: string | null;
}

export interface SessionHistoryResponse {
  session: AgentSession;
  instruction_history: InstructionRecord[];
  artifact_history: ArtifactRevisionRecord[];
  events: RuntimeEvent[];
  runtime_trace?: RuntimeTrace | null;
}

export interface SessionStreamResponse {
  session: AgentSession;
  events: RuntimeEvent[];
  pending_approval?: ApprovalRequest | null;
}

export interface ProjectGroup {
  projectId: string;
  title: string;
  sessions: AgentSession[];
  latest: AgentSession;
  approvalCount: number;
  runningCount: number;
}
