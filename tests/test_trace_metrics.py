from __future__ import annotations

from gemma4_capability_map.metrics.trace_metrics import derive_trace_metrics
from gemma4_capability_map.schemas import (
    Domain,
    HardwareProfile,
    Message,
    ModelBundleSpec,
    RunTrace,
    ScoringProfile,
    Task,
    Track,
)


def test_derive_trace_metrics_sums_prompt_and_latency_fields() -> None:
    task = Task(
        task_id="think_metrics",
        track=Track.THINKING,
        domain=Domain.GENERAL,
        user_goal="What is 19 plus 23?",
        messages=[Message(role="user", content="What is 19 plus 23?")],
        scoring_profile=ScoringProfile(answer_match=True),
    )
    trace = RunTrace(
        run_id="run_metrics",
        task_id=task.task_id,
        variant_id="clean",
        track=task.track,
        architecture="monolith",
        model_bundle=ModelBundleSpec(reasoner="google/gemma-4-E2B-it"),
        backend="hf",
        hardware_profile=HardwareProfile(
            platform="Darwin",
            platform_version="test",
            machine="arm64",
            cpu_count=12,
            memory_gb=24.0,
        ),
        prompt_artifacts={
            "planning_latency_ms": [120, 80],
            "planning_prompt_tokens": [30, 20],
            "planning_completion_tokens": [5, 4],
            "planning_repair_notes": [["canonicalized_tool:file_reader->read_repo_file"], []],
            "final_latency_ms": 200,
            "final_prompt_tokens": 10,
            "final_completion_tokens": 2,
        },
        final_answer="42",
    )
    metrics = derive_trace_metrics(task, trace)
    assert metrics["latency_ms"] == 400
    assert metrics["prompt_tokens"] == 60
    assert metrics["completion_tokens"] == 11
    assert metrics["controller_repair_count"] == 1
