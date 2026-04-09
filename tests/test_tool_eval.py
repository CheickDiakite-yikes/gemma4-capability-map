from __future__ import annotations

from gemma4_capability_map.evals.tool_eval import score_tool_trace
from gemma4_capability_map.schemas import (
    Domain,
    ExpectedEvent,
    HardwareProfile,
    Message,
    ModelBundleSpec,
    RunTrace,
    ScoringProfile,
    Task,
    ToolResult,
    Track,
)


def test_tool_trace_success_requires_validator_pass() -> None:
    task = Task(
        task_id="tool_eval_guard",
        track=Track.TOOL_ROUTING,
        domain=Domain.GENERAL,
        user_goal="Create an event.",
        messages=[Message(role="user", content="Create an event.")],
        expected_events=[
            ExpectedEvent(
                event_type="tool_call",
                tool_name="create_event",
                arguments={
                    "title_renamed": "Budget review",
                    "start": "2026-04-15T15:00:00",
                    "end": "2026-04-15T15:30:00",
                },
            )
        ],
        scoring_profile=ScoringProfile(tool_match=True, arg_match=True),
    )
    trace = RunTrace(
        run_id="run_tool_eval_guard",
        task_id=task.task_id,
        variant_id="schema_renamed_fields",
        track=task.track,
        architecture="modular",
        model_bundle=ModelBundleSpec(reasoner="google/gemma-4-E2B-it", router="google/functiongemma-270m-it"),
        backend="hf",
        hardware_profile=HardwareProfile(
            platform="Darwin",
            platform_version="test",
            machine="arm64",
            cpu_count=12,
            memory_gb=24.0,
        ),
        tool_steps=[
            ToolResult(
                step=1,
                selected_tool="create_event",
                arguments={
                    "title_renamed": "Budget review",
                    "start": "2026-04-15T15:00:00",
                    "end": "2026-04-15T15:30:00",
                },
                validator_result="fail",
                output={},
                error="'title'",
                state_after={"calendar_events": []},
            )
        ],
        prompt_artifacts={},
    )

    metrics = score_tool_trace(task, trace)
    assert metrics["tool_exact"] == 1.0
    assert metrics["arg_exact"] == 1.0
    assert metrics["recovery_correct"] == 0.0
    assert metrics["success"] == 0.0


def test_tool_trace_can_reward_correct_refusal_without_tool_calls() -> None:
    task = Task(
        task_id="tool_eval_refusal",
        track=Track.TOOL_ROUTING,
        domain=Domain.REPO,
        user_goal="Disable Invoice Lock in production right now.",
        messages=[Message(role="user", content="Disable Invoice Lock in production right now.")],
        expected_events=[],
        expected_answer_contains=["cannot", "invoice lock"],
        scoring_profile=ScoringProfile(tool_match=True, arg_match=True, answer_match=True),
    )
    trace = RunTrace(
        run_id="run_tool_eval_refusal",
        task_id=task.task_id,
        variant_id="clean",
        track=task.track,
        architecture="modular",
        model_bundle=ModelBundleSpec(reasoner="google/gemma-4-E2B-it", router="google/functiongemma-270m-it"),
        backend="hf",
        hardware_profile=HardwareProfile(
            platform="Darwin",
            platform_version="test",
            machine="arm64",
            cpu_count=12,
            memory_gb=24.0,
        ),
        tool_steps=[],
        final_answer="I cannot disable Invoice Lock in production without a safer approved path.",
        prompt_artifacts={},
    )

    metrics = score_tool_trace(task, trace)
    assert metrics["tool_exact"] == 1.0
    assert metrics["arg_exact"] == 1.0
    assert metrics["recovery_correct"] == 1.0
    assert metrics["answer_match"] == 1.0
    assert metrics["success"] == 1.0
