from __future__ import annotations

from gemma4_capability_map.metrics.failure_taxonomy import failure_tags, summarize_failure_tags
from gemma4_capability_map.schemas import HardwareProfile, ModelBundleSpec, RunTrace, Track


def _trace(**overrides) -> RunTrace:
    payload = {
        "run_id": "run-1",
        "task_id": "think_001_math",
        "variant_id": "think_001_math_clean",
        "track": Track.THINKING,
        "architecture": "monolith",
        "thinking_enabled": True,
        "model_bundle": ModelBundleSpec(reasoner="google/gemma-4-E2B-it"),
        "backend": "hf",
        "hardware_profile": HardwareProfile(
            platform="Darwin",
            platform_version="test",
            machine="arm64",
            cpu_count=12,
            memory_gb=24.0,
        ),
        "prompt_artifacts": {
            "final_raw_output": "<|channel>thought\nNeed more room.",
            "final_thinking_text": "Need more room.",
            "final_completion_tokens": 256,
            "final_max_new_tokens": 256,
        },
        "metrics": {
            "success": 0.0,
            "answer_match": 0.0,
        },
    }
    payload.update(overrides)
    return RunTrace(**payload)


def test_failure_tags_capture_thinking_overflow() -> None:
    trace = _trace(final_answer="")

    tags = failure_tags(trace)

    assert "failed" in tags
    assert "answer_missing" in tags
    assert "generation_truncated" in tags
    assert "thinking_overflow" in tags


def test_failure_tags_capture_image_grounding_miss() -> None:
    trace = _trace(
        thinking_enabled=False,
        final_answer="Off",
        image_refs=["data/assets/safe_mode_disabled.png"],
        prompt_artifacts={"final_raw_output": "Off", "final_thinking_text": "", "final_completion_tokens": 2, "final_max_new_tokens": 48},
    )

    tags = failure_tags(trace)

    assert "answer_mismatch" in tags
    assert "image_grounding_miss" in tags


def test_summarize_failure_tags_counts_across_runs() -> None:
    traces = [
        _trace(final_answer=""),
        _trace(
            run_id="run-2",
            thinking_enabled=False,
            final_answer="Off",
            image_refs=["data/assets/safe_mode_disabled.png"],
            prompt_artifacts={"final_raw_output": "Off", "final_thinking_text": "", "final_completion_tokens": 2, "final_max_new_tokens": 48},
        ),
    ]

    summary = summarize_failure_tags(traces)

    assert summary["failed"] == 2
    assert summary["answer_missing"] == 1
    assert summary["image_grounding_miss"] == 1
