from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import Field

from gemma4_capability_map.schemas import StrictModel


class SessionStatus(str, Enum):
    PENDING = "pending"
    WARMING = "warming"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    DENIED = "denied"
    FAILED = "failed"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"


class SystemProfile(StrictModel):
    system_id: str
    display_name: str
    short_label: str
    backend: str
    provider: str = ""
    capability_family: str = ""
    executor_mode: str = ""
    modality: str = ""
    deployment: str = ""
    local: bool = False
    reasoner: str = ""
    router: str = ""
    retriever: str = ""
    total_params_b: float = 0.0
    color: str = "#64748B"
    recommended: bool = False


class ToolInvocation(StrictModel):
    stage_id: str | None = None
    task_id: str | None = None
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    validator_result: Literal["pass", "fail", "unknown"] = "unknown"


class ApprovalRequest(StrictModel):
    approval_id: str
    session_id: str
    title: str
    reason: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    suggested_action: Literal["approve", "deny"] = "approve"
    required_by: str = "human"
    created_at: str
    resolved_at: str | None = None
    note: str = ""
    context: dict[str, Any] = Field(default_factory=dict)


class RuntimeEvent(StrictModel):
    event_id: str
    session_id: str
    sequence: int
    kind: Literal["created", "warming", "running", "approval_required", "completed", "approved", "denied", "failed"]
    message: str
    created_at: str
    payload: dict[str, Any] = Field(default_factory=dict)


class RuntimeTrace(StrictModel):
    session_id: str
    workflow_id: str
    episode_id: str
    output_dir: str
    manifest_path: str | None = None
    summary_path: str | None = None
    episode_trace_path: str | None = None
    artifact_paths: list[str] = Field(default_factory=list)
    scorecard: dict[str, Any] = Field(default_factory=dict)
    runtime_bundle: dict[str, Any] = Field(default_factory=dict)
    warmup: dict[str, Any] = Field(default_factory=dict)


class AgentSession(StrictModel):
    session_id: str
    title: str
    workflow_id: str
    workflow_title: str
    workflow_category: str
    workflow_tags: list[str] = Field(default_factory=list)
    episode_id: str
    system_id: str
    lane: str
    status: SessionStatus = SessionStatus.PENDING
    created_at: str
    updated_at: str
    human_request: str = ""
    latest_message: str = ""
    progress_label: str = ""
    preview_asset: str | None = None
    artifact_paths: list[str] = Field(default_factory=list)
    tool_invocations: list[ToolInvocation] = Field(default_factory=list)
    approvals: list[ApprovalRequest] = Field(default_factory=list)
    runtime_trace: RuntimeTrace | None = None
    metrics: dict[str, float | int | bool] = Field(default_factory=dict)
    last_error: str | None = None
