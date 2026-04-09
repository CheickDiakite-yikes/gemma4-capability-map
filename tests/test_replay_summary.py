from __future__ import annotations

from gemma4_capability_map.schemas import (
    HardwareProfile,
    ModelBundleSpec,
    RunTrace,
    Track,
)
from gemma4_capability_map.traces.replay import summarize_traces


def _trace(
    run_id: str,
    *,
    success: float,
    tool_exact: float,
    arg_exact: float,
    recovery_correct: float,
    final_state_match: float,
    readiness: float,
) -> RunTrace:
    return RunTrace(
        run_id=run_id,
        task_id=run_id,
        variant_id="clean",
        track=Track.FULL_STACK,
        architecture="modular",
        model_bundle=ModelBundleSpec(reasoner="google/gemma-4-E2B-it"),
        backend="hf",
        hardware_profile=HardwareProfile(
            platform="Darwin",
            platform_version="test",
            machine="arm64",
            cpu_count=12,
            memory_gb=24.0,
        ),
        metrics={
            "success": success,
            "tool_exact": tool_exact,
            "arg_exact": arg_exact,
            "recovery_correct": recovery_correct,
            "final_state_match": final_state_match,
            "real_world_readiness_score": readiness,
        },
    )


def test_summarize_traces_exposes_strict_recovered_and_readiness_layers() -> None:
    summary = summarize_traces(
        [
            _trace("run_a", success=1.0, tool_exact=1.0, arg_exact=1.0, recovery_correct=1.0, final_state_match=1.0, readiness=0.95),
            _trace("run_b", success=1.0, tool_exact=0.0, arg_exact=0.0, recovery_correct=1.0, final_state_match=1.0, readiness=0.65),
        ]
    )
    assert summary["success_rate"] == 1.0
    assert summary["strict_interface_rate"] == 0.5
    assert summary["recovered_execution_rate"] == 1.0
    assert summary["real_world_readiness_avg"] == 0.8
