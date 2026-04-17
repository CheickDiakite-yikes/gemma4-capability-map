import type {
  AgentSession,
  ApprovalRequest,
  SessionHistoryResponse,
  SessionStreamResponse,
  SystemProfile,
  WorkflowCard,
} from "./types";

const API_BASE = import.meta.env.VITE_MOONIE_API_BASE ?? "";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export function getHealth(signal?: AbortSignal) {
  return requestJson<{ ok: boolean }>("/health", { signal });
}

export function listProfiles(signal?: AbortSignal) {
  return requestJson<{ profiles: SystemProfile[] }>("/v1/profiles", { signal });
}

export function listWorkflows(lane: string, signal?: AbortSignal) {
  return requestJson<{ workflows: WorkflowCard[] }>(`/v1/workflows?lane=${encodeURIComponent(lane)}`, { signal });
}

export function listSessions(signal?: AbortSignal) {
  return requestJson<{ sessions: AgentSession[] }>("/v1/sessions", { signal });
}

export function listApprovals(includeAll = false, signal?: AbortSignal) {
  const suffix = includeAll ? "?all=true" : "";
  return requestJson<{ approvals: ApprovalRequest[] }>(`/v1/approvals${suffix}`, { signal });
}

export function getSessionHistory(sessionId: string, signal?: AbortSignal) {
  return requestJson<SessionHistoryResponse>(`/v1/sessions/${encodeURIComponent(sessionId)}/history`, { signal });
}

export function streamSession(sessionId: string, afterSequence = 0, timeoutSeconds = 2, signal?: AbortSignal) {
  return requestJson<SessionStreamResponse>(
    `/v1/sessions/${encodeURIComponent(sessionId)}/stream?after=${afterSequence}&timeout_s=${timeoutSeconds}`,
    { signal },
  );
}

export function launchSession(input: {
  workflow_id: string;
  system_id?: string;
  lane?: string;
  title?: string;
  human_request?: string;
  project_id?: string;
  background?: boolean;
}) {
  return requestJson<AgentSession>("/v1/sessions", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export function resumeSession(sessionId: string, note: string, background = true) {
  return requestJson<AgentSession>(`/v1/sessions/${encodeURIComponent(sessionId)}/resume`, {
    method: "POST",
    body: JSON.stringify({ note, background }),
  });
}

export function retrySession(sessionId: string, note: string, background = true) {
  return requestJson<AgentSession>(`/v1/sessions/${encodeURIComponent(sessionId)}/retry`, {
    method: "POST",
    body: JSON.stringify({ note, background }),
  });
}

export function resolveApproval(approvalId: string, decision: "approve" | "deny", note: string, resume = true) {
  return requestJson<AgentSession>(`/v1/approvals/${encodeURIComponent(approvalId)}/resolve`, {
    method: "POST",
    body: JSON.stringify({ decision, note, resume }),
  });
}

export function fileUrl(path: string) {
  return `${API_BASE}/v1/files?path=${encodeURIComponent(path)}`;
}
