from __future__ import annotations

import importlib.util
from pathlib import Path

from gemma4_capability_map.io import load_yaml
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
from gemma4_capability_map.reporting.knowledge_work_board import load_model_registry


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_knowledge_work_matrix.py"
SPEC = importlib.util.spec_from_file_location("run_knowledge_work_matrix_script", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

_build_run_specs = MODULE._build_run_specs
_collect_run_result = MODULE._collect_run_result
_collect_timeout_result = MODULE._collect_timeout_result
_system_run_args = MODULE._system_run_args


def test_system_run_args_use_registry_executor_mode_defaults() -> None:
    registry = load_model_registry()

    service_specialists = _system_run_args("hf_service_gemma4_specialists_cpu", registry)
    inprocess_reasoner = _system_run_args("hf_gemma4_e2b_reasoner_only", registry)
    gguf_reasoner = _system_run_args("llama_cpp_gemma4_31b_reasoner_only", registry)

    assert service_specialists["router_backend"] == "hf_service"
    assert service_specialists["retriever_backend"] == "hf_service"
    assert inprocess_reasoner["router_backend"] == "heuristic"
    assert inprocess_reasoner["retriever_backend"] == "heuristic"
    assert inprocess_reasoner["router"] == ""
    assert inprocess_reasoner["retriever"] == ""
    assert service_specialists["reasoner_max_new_tokens"] == 96
    assert service_specialists["request_timeout_seconds"] == 600.0
    assert service_specialists["run_timeout_seconds"] == 0.0
    assert _system_run_args("hf_service_gemma4_e4b_reasoner_only", registry)["reasoner_max_new_tokens"] == 64
    assert gguf_reasoner["backend"] == "llama_cpp"
    assert gguf_reasoner["reasoner"] == "google/gemma-4-31b-it"
    assert gguf_reasoner["router_backend"] == "heuristic"
    assert gguf_reasoner["retriever_backend"] == "heuristic"
    assert gguf_reasoner["reasoner_max_new_tokens"] == 96


def test_build_run_specs_filters_systems_and_lanes() -> None:
    registry = load_model_registry()
    matrix = {
        "systems": [
            "oracle_gemma4_e2b",
            "hf_service_gemma4_reasoner_only",
            "hf_service_gemma4_specialists_cpu",
        ],
        "lanes": ["replayable_core", "live_web_stress"],
        "lane_limits": {"replayable_core": 24, "live_web_stress": 18},
        "run_intent": "exploratory",
        "update_latest": False,
    }

    specs = _build_run_specs(
        matrix,
        registry,
        allowed_system_ids=["hf_service_gemma4_specialists_cpu"],
        allowed_lanes=["live_web_stress"],
    )

    assert len(specs) == 1
    assert specs[0]["system_id"] == "hf_service_gemma4_specialists_cpu"
    assert specs[0]["lane"] == "live_web_stress"
    assert specs[0]["limit"] == 18
    assert specs[0]["run_intent"] == "exploratory"


def test_build_run_specs_allow_full_lane_when_limits_are_omitted() -> None:
    registry = load_model_registry()
    matrix = {
        "systems": ["hf_service_gemma4_specialists_cpu"],
        "lanes": ["replayable_core"],
        "run_intent": "exploratory",
        "update_latest": False,
    }

    specs = _build_run_specs(matrix, registry)

    assert len(specs) == 1
    assert specs[0]["system_id"] == "hf_service_gemma4_specialists_cpu"
    assert specs[0]["lane"] == "replayable_core"
    assert specs[0]["limit"] is None


def test_experimental_matrix_includes_qwen_reasoner_only_system() -> None:
    registry = load_model_registry()
    config = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "knowledge_work_matrix_experimental.yaml")
    matrix = config.get("matrix", {})

    specs = _build_run_specs(
        matrix,
        registry,
        allowed_system_ids=["hf_qwen3_8b_reasoner_only"],
        allowed_lanes=["replayable_core"],
    )

    assert len(specs) == 1
    assert specs[0]["system_id"] == "hf_qwen3_8b_reasoner_only"
    assert specs[0]["backend"] == "hf"
    assert specs[0]["reasoner"] == "Qwen/Qwen3-8B"
    assert specs[0]["reasoner_backend"] == "hf"
    assert specs[0]["router_backend"] == "heuristic"
    assert specs[0]["retriever_backend"] == "heuristic"


def test_experimental_matrix_includes_llama_cpp_gemma31b_reasoner_only_system() -> None:
    registry = load_model_registry()
    config = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "knowledge_work_matrix_experimental.yaml")
    matrix = config.get("matrix", {})

    specs = _build_run_specs(
        matrix,
        registry,
        allowed_system_ids=["llama_cpp_gemma4_31b_reasoner_only"],
        allowed_lanes=["replayable_core"],
    )

    assert len(specs) == 1
    assert specs[0]["system_id"] == "llama_cpp_gemma4_31b_reasoner_only"
    assert specs[0]["backend"] == "llama_cpp"
    assert specs[0]["reasoner"] == "google/gemma-4-31b-it"
    assert specs[0]["reasoner_backend"] == "llama_cpp"
    assert specs[0]["router_backend"] == "heuristic"
    assert specs[0]["retriever_backend"] == "heuristic"


def test_tool_trace_reports_tool_family_and_intent_tags() -> None:
    task = Task(
        task_id="taxonomy_example",
        track=Track.TOOL_ROUTING,
        domain=Domain.GENERAL,
        user_goal="Inspect the repo file and revise the answer.",
        messages=[Message(role="user", content="Inspect the repo file and revise the answer.")],
        expected_events=[
            ExpectedEvent(
                event_type="tool_call",
                tool_name="read_repo_file",
                arguments={"path": "README.md"},
            )
        ],
        scoring_profile=ScoringProfile(tool_match=True, arg_match=True),
        benchmark_tags=["tool_family:function_call", "tool_intent:inspect"],
    )
    trace = RunTrace(
        run_id="run_taxonomy_example",
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
        tool_steps=[
            ToolResult(
                step=1,
                selected_tool="read_repo_file",
                arguments={"path": "README.md"},
                validator_result="pass",
                output={"path": "README.md"},
                state_after={},
            )
        ],
        prompt_artifacts={},
    )

    metrics = score_tool_trace(task, trace)
    assert metrics["tool_family"] == "function_call"
    assert metrics["tool_intent"] == "inspect"
    assert metrics["tool_taxonomy"] == "function_call:inspect"
    assert metrics["tool_taxonomy_source"] == "explicit"


def test_collect_run_result_marks_incomplete_progress_as_failure(tmp_path: Path) -> None:
    output_dir = tmp_path / "matrix_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        '{"episode_count": 24}\n',
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        '{"runs": 7}\n',
        encoding="utf-8",
    )
    (output_dir / "progress.json").write_text(
        '{"status": "running", "completed_runs": 7, "planned_runs": 24}\n',
        encoding="utf-8",
    )

    process = MODULE.subprocess.CompletedProcess(args=["python"], returncode=0, stdout="", stderr="")
    result = _collect_run_result(
        {
            "run_id": "hf_service_gemma4_e4b_reasoner_only__replayable_core",
            "system_id": "hf_service_gemma4_e4b_reasoner_only",
            "lane": "replayable_core",
        },
        output_dir,
        process,
    )

    assert result["returncode"] == 1
    assert result["validation_error"] == "incomplete_progress:running"


def test_collect_timeout_result_marks_run_as_timed_out(tmp_path: Path) -> None:
    output_dir = tmp_path / "matrix_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "progress.json").write_text(
        '{"status": "running", "completed_runs": 0, "planned_runs": 24}\n',
        encoding="utf-8",
    )

    timeout = MODULE.subprocess.TimeoutExpired(
        cmd=["python"],
        timeout=900.0,
        output="partial stdout",
        stderr="partial stderr",
    )
    result = _collect_timeout_result(
        {
            "run_id": "hf_service_gemma4_e4b_reasoner_only__replayable_core",
            "system_id": "hf_service_gemma4_e4b_reasoner_only",
            "lane": "replayable_core",
        },
        output_dir,
        timeout,
        timeout_seconds=900.0,
    )

    assert result["returncode"] == 124
    assert result["timed_out"] is True
    assert result["timeout_seconds"] == 900.0
    assert result["validation_error"] == "run_timeout"
