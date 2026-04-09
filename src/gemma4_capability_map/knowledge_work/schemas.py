from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import Field

from gemma4_capability_map.schemas import RunTrace, StrictModel


class RoleFamily(str, Enum):
    EXECUTIVE_ASSISTANT = "executive_assistant"
    JOB_APPLICATION_OPS = "job_application_ops"
    FINANCE = "finance"


class BenchmarkLane(str, Enum):
    REPLAYABLE_CORE = "replayable_core"
    LIVE_WEB_STRESS = "live_web_stress"


class ArtifactKind(str, Enum):
    MEMO = "memo"
    DECK = "deck"
    SPREADSHEET = "spreadsheet"
    FORM_SUBMISSION = "form_submission"
    EMAIL = "email"
    SCHEDULE = "schedule"
    RESEARCH_NOTE = "research_note"
    MODEL = "model"


class ArtifactScoringContract(StrictModel):
    required_fragments: list[str] = Field(default_factory=list)
    forbidden_fragments: list[str] = Field(default_factory=list)
    required_sections: list[str] = Field(default_factory=list)
    required_table_rows: list[list[str]] = Field(default_factory=list)
    required_field_pairs: dict[str, str] = Field(default_factory=dict)
    required_slide_titles: list[str] = Field(default_factory=list)
    required_bullets: list[str] = Field(default_factory=list)
    minimum_citations: int = 0
    expected_format: str | None = None


class ArtifactSpec(StrictModel):
    artifact_id: str
    kind: ArtifactKind
    path_or_target: str
    scoring_contract: ArtifactScoringContract = Field(default_factory=ArtifactScoringContract)


class EpisodeStage(StrictModel):
    stage_id: str
    goal: str
    inputs: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    expected_state_delta: dict[str, Any] = Field(default_factory=dict)
    can_request_clarification: bool = False
    can_escalate: bool = False
    task_refs: list[str] = Field(default_factory=list)
    preferred_architecture: Literal["monolith", "hybrid", "modular"] | None = None
    browser_plan: list["BrowserStep"] = Field(default_factory=list)


class ReviewRound(StrictModel):
    review_id: str
    artifact_id: str
    feedback: str
    expected_improvements: list[str] = Field(default_factory=list)


class SuccessContract(StrictModel):
    required_artifacts: list[str] = Field(default_factory=list)
    min_stage_success: float = 1.0
    requires_collateral_damage_free: bool = True


class RiskGuardrails(StrictModel):
    no_public_side_effects: bool = True
    dry_run_only: bool = False
    escalation_required_for_high_risk: bool = False
    notes: list[str] = Field(default_factory=list)


class Episode(StrictModel):
    episode_id: str
    role_family: RoleFamily
    lane: BenchmarkLane
    workspace_id: str
    brief: str
    memory_scope: Literal["episode"] = "episode"
    tools: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactSpec] = Field(default_factory=list)
    stages: list[EpisodeStage] = Field(default_factory=list)
    review_rounds: list[ReviewRound] = Field(default_factory=list)
    success_contract: SuccessContract = Field(default_factory=SuccessContract)
    risk_guardrails: RiskGuardrails = Field(default_factory=RiskGuardrails)
    human_baseline_minutes: int = 30
    benchmark_tags: list[str] = Field(default_factory=list)


class ArtifactVersion(StrictModel):
    artifact_id: str
    revision: int
    content: str
    score: float = 0.0
    source_stage: str


class MemoryUpdate(StrictModel):
    stage_id: str
    key: str
    value: str


class BrowserAction(StrictModel):
    stage_id: str
    action: str
    target: str
    surface: Literal["workspace", "email", "calendar", "job_portal", "data_room", "spreadsheet", "presentation", "document", "public_web"] = "workspace"
    purpose: str = ""
    expected_signal: str = ""
    evidence: str = ""
    verification_checks: list[str] = Field(default_factory=list)
    verification_result: Literal["pass", "fail"] = "pass"
    captured_fields: list[str] = Field(default_factory=list)
    sandbox_endpoint: str | None = None
    status: Literal["dry_run", "replayed", "planned"] = "planned"


class BrowserStep(StrictModel):
    action: str
    target: str
    surface: Literal["workspace", "email", "calendar", "job_portal", "data_room", "spreadsheet", "presentation", "document", "public_web"] = "workspace"
    purpose: str
    expected_signal: str
    verification_checks: list[str] = Field(default_factory=list)
    captured_fields: list[str] = Field(default_factory=list)
    sandbox_endpoint: str | None = None
    allow_submission: bool = False


class StageTrace(StrictModel):
    stage_id: str
    task_traces: list[RunTrace] = Field(default_factory=list)
    artifact_updates: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class EpisodeScorecard(StrictModel):
    artifact_quality_score: float = 0.0
    browser_workflow_score: float = 0.0
    strict_interface_score: float = 0.0
    recovered_execution_score: float = 0.0
    revision_responsiveness: float = 0.0
    memory_retention_score: float = 0.0
    escalation_correctness: float = 0.0
    collateral_damage_free: float = 0.0
    human_time_ratio: float = 0.0
    role_readiness_score: float = 0.0


class EpisodeTrace(StrictModel):
    run_id: str
    episode_id: str
    role_family: RoleFamily
    lane: BenchmarkLane
    workspace_id: str
    benchmark_tags: list[str] = Field(default_factory=list)
    stage_traces: list[StageTrace] = Field(default_factory=list)
    browser_actions: list[BrowserAction] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    artifact_versions: list[ArtifactVersion] = Field(default_factory=list)
    memory_updates: list[MemoryUpdate] = Field(default_factory=list)
    review_history: list[ReviewRound] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    prompt_artifacts: dict[str, Any] = Field(default_factory=dict)
    scorecard: EpisodeScorecard = Field(default_factory=EpisodeScorecard)
