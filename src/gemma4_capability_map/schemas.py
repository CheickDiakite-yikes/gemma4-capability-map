from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "1.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Track(str, Enum):
    THINKING = "thinking"
    TOOL_ROUTING = "tool_routing"
    RETRIEVAL = "retrieval"
    FULL_STACK = "full_stack"


class Domain(str, Enum):
    FILES = "files"
    CALENDAR = "calendar"
    REPO = "repo"
    DOCS = "docs"
    SCREENSHOT = "screenshot"
    GENERAL = "general"


class StressorKind(str, Enum):
    LANGUAGE = "language"
    SCHEMA = "schema"
    CONTEXT = "context"
    EFFICIENCY = "efficiency"


class Message(StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    image_refs: list[str] = Field(default_factory=list)


class ToolSpec(StrictModel):
    name: str
    description: str
    json_schema: dict[str, Any] = Field(alias="schema", serialization_alias="schema")


class Document(StrictModel):
    doc_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExpectedEvent(StrictModel):
    event_type: Literal["tool_call", "retrieval", "message"]
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    expected_doc_ids: list[str] = Field(default_factory=list)
    summary: str | None = None
    parallel_group: str | None = None


class ScoringProfile(StrictModel):
    tool_match: bool = False
    arg_match: bool = False
    final_state_match: bool = False
    retrieval_match: bool = False
    answer_match: bool = False


class RealWorldProfile(StrictModel):
    job_role: str
    scenario: str
    autonomy_level: Literal["assistive", "bounded_autonomy", "zero_touch"]
    risk_tier: Literal["low", "medium", "high"]
    time_budget_minutes: int = 15
    human_equivalent_minutes: int | None = None
    requires_multistep_state: bool = False
    requires_recovery: bool = False
    requires_escalation_judgment: bool = False
    requires_multilingual: bool = False
    requires_multimodal_grounding: bool = False
    success_invariants: list[str] = Field(default_factory=list)
    failure_costs: list[str] = Field(default_factory=list)


class JudgmentMode(StrictModel):
    enabled: bool = True
    allowed_actions: list[Literal["proceed", "escalate", "defer", "clarify", "refuse"]] = Field(
        default_factory=lambda: ["proceed", "escalate", "defer", "clarify", "refuse"]
    )
    expected_action: Literal["proceed", "escalate", "defer", "clarify", "refuse"] | None = None
    requires_basis: bool = False
    basis_fragments: list[str] = Field(default_factory=list)


class Task(StrictModel):
    schema_version: str = SCHEMA_VERSION
    task_id: str
    track: Track
    domain: Domain
    input_modalities: list[Literal["text", "image"]] = Field(default_factory=lambda: ["text"])
    user_goal: str
    messages: list[Message]
    tool_specs: list[ToolSpec] = Field(default_factory=list)
    corpora: dict[str, list[Document]] = Field(default_factory=dict)
    initial_state: dict[str, Any] = Field(default_factory=dict)
    expected_events: list[ExpectedEvent] = Field(default_factory=list)
    expected_final_state: dict[str, Any] = Field(default_factory=dict)
    expected_answer_contains: list[str] = Field(default_factory=list)
    scoring_profile: ScoringProfile = Field(default_factory=ScoringProfile)
    image_refs: list[str] = Field(default_factory=list)
    benchmark_tags: list[str] = Field(default_factory=list)
    real_world_profile: RealWorldProfile | None = None
    judgment_mode: JudgmentMode | None = None


class VariantOverrides(StrictModel):
    messages_prefix: list[Message] = Field(default_factory=list)
    messages: list[Message] | None = None
    tool_specs: list[ToolSpec] | None = None
    corpora: dict[str, list[Document]] | None = None
    initial_state_patch: dict[str, Any] = Field(default_factory=dict)
    expected_events: list[ExpectedEvent] | None = None
    expected_final_state: dict[str, Any] | None = None
    expected_answer_contains: list[str] | None = None
    efficiency: dict[str, Any] = Field(default_factory=dict)


class Variant(StrictModel):
    schema_version: str = SCHEMA_VERSION
    variant_id: str
    base_task_id: str
    primary_stressor: StressorKind | None = None
    secondary_stressor: StressorKind | None = None
    stressors: dict[str, str | None] = Field(default_factory=dict)
    overrides: VariantOverrides = Field(default_factory=VariantOverrides)


class ToolCall(StrictModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    source_format: Literal["json", "python", "functiongemma", "raw", "oracle", "heuristic"]
    raw: str


class ModelTurn(StrictModel):
    raw_model_output: str
    normalized_tool_call: list[ToolCall] = Field(default_factory=list)
    final_answer: str = ""
    thinking_text: str = ""
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    runtime_metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalHit(StrictModel):
    doc_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(StrictModel):
    step: int
    selected_tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    validator_result: Literal["pass", "fail"] = "pass"
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    state_after: dict[str, Any] = Field(default_factory=dict)


class StateTransition(StrictModel):
    step: int
    tool_name: str
    before: dict[str, Any] = Field(default_factory=dict)
    after: dict[str, Any] = Field(default_factory=dict)
    diff: dict[str, Any] = Field(default_factory=dict)


class ModelBundleSpec(StrictModel):
    reasoner: str
    router: str | None = None
    retriever: str | None = None


class HardwareProfile(StrictModel):
    platform: str
    platform_version: str
    machine: str
    cpu_count: int
    memory_gb: float


class RunTrace(StrictModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    task_id: str
    variant_id: str
    track: Track
    architecture: Literal["monolith", "hybrid", "modular"]
    thinking_enabled: bool = False
    model_bundle: ModelBundleSpec
    backend: str
    stressors: dict[str, str | None] = Field(default_factory=dict)
    hardware_profile: HardwareProfile
    prompt_artifacts: dict[str, Any] = Field(default_factory=dict)
    retrieval_hits: list[RetrievalHit] = Field(default_factory=list)
    tool_steps: list[ToolResult] = Field(default_factory=list)
    state_transitions: list[StateTransition] = Field(default_factory=list)
    final_answer: str = ""
    image_refs: list[str] = Field(default_factory=list)
    benchmark_tags: list[str] = Field(default_factory=list)
    real_world_profile: RealWorldProfile | None = None
    metrics: dict[str, float | int | bool] = Field(default_factory=dict)
