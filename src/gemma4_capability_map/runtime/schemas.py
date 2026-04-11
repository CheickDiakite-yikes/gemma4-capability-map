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
    RESUMING = "resuming"
    RETRYING = "retrying"
    INTERRUPTED = "interrupted"
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
    comparison_tier: str = ""
    publishable_default: bool = False
    local: bool = False
    reasoner: str = ""
    router: str = ""
    retriever: str = ""
    total_params_b: float = 0.0
    color: str = "#64748B"
    recommended: bool = False
    reasoner_max_new_tokens: int = 96
    request_timeout_seconds: float = 600.0
    run_timeout_seconds: float = 0.0


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
    kind: Literal[
        "created",
        "warming",
        "running",
        "artifacts_ready",
        "approval_required",
        "approval_resolved",
        "completed",
        "approved",
        "denied",
        "failed",
        "interrupted",
        "resumed",
        "retry_requested",
    ]
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
    artifact_count: int = 0
    review_count: int = 0
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
    attempt: int = 1
    parent_session_id: str | None = None
    lineage_root_session_id: str | None = None
    retry_of_session_id: str | None = None
    resumed_from_session_id: str | None = None
    human_request: str = ""
    latest_message: str = ""
    progress_label: str = ""
    last_activity_at: str = ""
    hold_reason: str | None = None
    resumable: bool = False
    active_approval_id: str | None = None
    last_event_sequence: int = 0
    preview_asset: str | None = None
    latest_artifact_title: str = ""
    latest_artifact_path: str = ""
    latest_revision_artifact_id: str = ""
    latest_review_feedback: str = ""
    artifact_paths: list[str] = Field(default_factory=list)
    tool_invocations: list[ToolInvocation] = Field(default_factory=list)
    approvals: list[ApprovalRequest] = Field(default_factory=list)
    runtime_trace: RuntimeTrace | None = None
    metrics: dict[str, float | int | bool] = Field(default_factory=dict)
    last_error: str | None = None
