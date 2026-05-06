from __future__ import annotations

import csv
import json
from pathlib import Path

from gemma4_capability_map.knowledge_work.trace_analysis import (
    analyze_ablation_packet,
    compare_ablation_packets,
    write_packet_comparison,
    write_trace_analysis,
)


def test_analyze_ablation_packet_counts_notes_and_failures(tmp_path: Path) -> None:
    run_dir = tmp_path / "system_a__replayable_core"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"system_id": "system_a", "lane": "replayable_core"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "runs": 1,
                "real_world_readiness_avg": 0.72,
                "strict_interface_avg": 0.4,
                "recovered_execution_avg": 0.4,
                "controller_repair_avg": 0.5,
                "controller_fallback_avg": 1.0,
                "raw_planning_clean_rate_avg": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    trace = {
        "episode_id": "episode_1",
        "role_family": "finance",
        "stage_traces": [
            {
                "stage_id": "stage_1",
                "task_traces": [
                    {
                        "task_id": "task_a",
                        "prompt_artifacts": {
                            "planning_raw_outputs": ["I cannot do that.", "<start_function_call>call:tool_name{}"],
                            "planning_repair_notes": [
                                ["controller_repair_disabled", "controller_fallback_planner"],
                                ["controller_repair_disabled"],
                            ],
                        },
                    }
                ],
            }
        ],
        "tool_calls": [
            {
                "task_id": "task_a",
                "tool_name": "tool_name",
                "validator_result": "fail",
            }
        ],
        "scorecard": {
            "role_readiness_score": 0.72,
            "strict_interface_score": 0.4,
            "recovered_execution_score": 0.4,
            "controller_repair_count": 0.5,
            "argument_repair_count": 0.0,
            "controller_fallback_count": 1.0,
            "intent_override_count": 0.0,
            "raw_planning_clean_rate": 0.0,
        },
    }
    (run_dir / "episode_traces.jsonl").write_text(json.dumps(trace) + "\n", encoding="utf-8")

    analysis = analyze_ablation_packet(tmp_path)

    assert analysis["system_count"] == 1
    assert analysis["episode_count"] == 1
    assert analysis["failure_candidate_count"] == 1
    notes = {(row["system_id"], row["note"]): row["count"] for row in analysis["note_counts"]}
    assert notes[("system_a", "controller_repair_disabled")] == 2
    assert notes[("system_a", "controller_fallback_planner")] == 1
    failure = analysis["failure_rows"][0]
    assert failure["failed_tools"] == "task_a:tool_name"
    assert failure["failure_modes"] == "raw_refusal;generic_tool_name;repair_disabled;fallback_planner"
    assert failure["repair_note_counts"] == "controller_fallback_planner=1;controller_repair_disabled=2"
    mode_counts = {row["failure_mode"]: row["count"] for row in analysis["failure_mode_counts"]}
    assert mode_counts == {
        "raw_refusal": 1,
        "generic_tool_name": 1,
        "repair_disabled": 1,
        "fallback_planner": 1,
    }


def test_write_trace_analysis_outputs_csv_and_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "system_b__replayable_core"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({"system_id": "system_b"}) + "\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"runs": 1}) + "\n", encoding="utf-8")
    (run_dir / "episode_traces.jsonl").write_text(
        json.dumps(
            {
                "episode_id": "episode_2",
                "stage_traces": [{"task_traces": [{"task_id": "task_b", "planning_repair_notes": [["intent_prior:inspect_or_lookup"]]}]}],
                "scorecard": {"strict_interface_score": 1.0, "recovered_execution_score": 1.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    paths = write_trace_analysis(tmp_path)

    assert Path(paths["summary"]).exists()
    with Path(paths["note_counts"]).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["system_id"] == "system_b"
    assert rows[0]["note"] == "intent_prior:inspect_or_lookup"
    assert Path(paths["failures"]).read_text(encoding="utf-8") == ""
    assert Path(paths["failure_modes"]).read_text(encoding="utf-8") == ""


def test_compare_ablation_packets_reports_system_and_failure_deltas(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_packet_run(
        baseline,
        system_id="system_a",
        readiness=0.72,
        strict=0.4,
        recovered=0.4,
        repair=0.5,
        fallback=1.0,
        raw_clean=0.0,
        notes=["controller_fallback_planner"],
        failed=True,
    )
    _write_packet_run(
        candidate,
        system_id="system_a",
        readiness=0.92,
        strict=1.0,
        recovered=1.0,
        repair=0.1,
        fallback=0.0,
        raw_clean=0.9,
        notes=["repaired_arguments:extract_layout"],
        failed=False,
    )

    comparison = compare_ablation_packets(baseline, candidate)

    assert comparison["deltas"]["shared_system_count"] == 1
    assert comparison["deltas"]["failure_candidate_count_delta"] == -1
    row = comparison["system_deltas"][0]
    assert row["system_id"] == "system_a"
    assert row["delta_real_world_readiness_avg"] == 0.2
    assert row["delta_controller_fallback_avg"] == -1.0
    note_deltas = {(item["system_id"], item["note"]): item["delta_count"] for item in comparison["note_deltas"]}
    assert note_deltas[("system_a", "controller_fallback_planner")] == -1.0
    assert note_deltas[("system_a", "repaired_arguments:extract_layout")] == 1.0
    mode_deltas = {item["failure_mode"]: item["delta_count"] for item in comparison["failure_mode_deltas"]}
    assert mode_deltas["fallback_planner"] == -1.0


def test_write_packet_comparison_outputs_json_and_csv(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_packet_run(baseline, system_id="system_b", readiness=0.5, strict=0.5, recovered=0.5)
    _write_packet_run(candidate, system_id="system_b", readiness=0.75, strict=1.0, recovered=1.0)

    paths = write_packet_comparison(baseline, candidate)

    assert Path(paths["summary"]).exists()
    with Path(paths["system_deltas"]).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["system_id"] == "system_b"
    assert rows[0]["delta_real_world_readiness_avg"] == "0.25"


def test_analyze_ablation_packet_labels_visual_stepwise_failures(tmp_path: Path) -> None:
    run_dir = tmp_path / "system_no_deterministic_visual_follow_on__replayable_core"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"system_id": "system_no_deterministic_visual_follow_on", "lane": "replayable_core"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(json.dumps({"runs": 1}) + "\n", encoding="utf-8")
    trace = {
        "episode_id": "visual_episode",
        "role_family": "executive_assistant",
        "stage_traces": [
            {
                "stage_id": "stage_visual",
                "task_traces": [
                    {
                        "task_id": "visual_task",
                        "prompt_artifacts": {
                            "planning_raw_outputs": [
                                "<start_function_call>call:extract_layout{image_id:<escape>img-dashboard<escape>,target_query:<escape>dashboard metric<escape>}<end_function_call>",
                                "<start_function_call>call:refine_selection{selection_id:<escape>sel-001<escape>,filter_query:<escape>needs review<escape>}<end_function_call>",
                                "I cannot proceed because the tools are not applicable.",
                            ],
                            "planning_repair_notes": [
                                [],
                                ["repaired_arguments:refine_selection"],
                                ["controller_fallback_planner"],
                            ],
                        },
                    }
                ],
            }
        ],
        "tool_calls": [
            {
                "task_id": "visual_task",
                "tool_name": "extract_layout",
                "arguments": {"image_id": "img-dashboard", "target_query": "dashboard metric"},
                "validator_result": "pass",
            },
            {
                "task_id": "visual_task",
                "tool_name": "refine_selection",
                "arguments": {"selection_id": "sel-001", "filter_query": "needs review"},
                "validator_result": "pass",
            },
            {
                "task_id": "visual_task",
                "tool_name": "refine_selection",
                "arguments": {"selection_id": "sel-001", "filter_query": "needs review"},
                "validator_result": "pass",
            },
        ],
        "scorecard": {
            "role_readiness_score": 0.84,
            "strict_interface_score": 0.625,
            "recovered_execution_score": 0.5,
            "controller_repair_count": 1.5,
            "argument_repair_count": 0.5,
            "controller_fallback_count": 0.5,
            "intent_override_count": 0.0,
            "raw_planning_clean_rate": 0.7,
        },
    }
    (run_dir / "episode_traces.jsonl").write_text(json.dumps(trace) + "\n", encoding="utf-8")

    analysis = analyze_ablation_packet(tmp_path)

    failure = analysis["failure_rows"][0]
    modes = set(failure["failure_modes"].split(";"))
    assert {
        "raw_refusal",
        "fallback_planner",
        "argument_repair",
        "visual_follow_on",
        "visual_stepwise_control",
        "visual_repeated_refinement",
        "visual_readback_missing",
    }.issubset(modes)


def _write_packet_run(
    packet_dir: Path,
    *,
    system_id: str,
    readiness: float,
    strict: float,
    recovered: float,
    repair: float = 0.0,
    fallback: float = 0.0,
    raw_clean: float = 1.0,
    notes: list[str] | None = None,
    failed: bool = False,
) -> None:
    run_dir = packet_dir / f"{system_id}__live_web_stress"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"system_id": system_id, "lane": "live_web_stress"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "runs": 1,
                "real_world_readiness_avg": readiness,
                "strict_interface_avg": strict,
                "recovered_execution_avg": recovered,
                "controller_repair_avg": repair,
                "controller_fallback_avg": fallback,
                "raw_planning_clean_rate_avg": raw_clean,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    trace = {
        "episode_id": "episode_1",
        "stage_traces": [
            {
                "stage_id": "stage_1",
                "task_traces": [
                    {
                        "task_id": "task_a",
                        "prompt_artifacts": {
                            "planning_raw_outputs": ["I cannot proceed because the tools are not applicable."],
                            "planning_repair_notes": [notes or []],
                        },
                    }
                ],
            }
        ],
        "tool_calls": [
            {
                "task_id": "task_a",
                "tool_name": "tool_name",
                "validator_result": "fail" if failed else "pass",
            }
        ],
        "scorecard": {
            "role_readiness_score": readiness,
            "strict_interface_score": strict,
            "recovered_execution_score": recovered,
            "controller_repair_count": repair,
            "argument_repair_count": 0.0,
            "controller_fallback_count": fallback,
            "intent_override_count": 0.0,
            "raw_planning_clean_rate": raw_clean,
        },
    }
    (run_dir / "episode_traces.jsonl").write_text(json.dumps(trace) + "\n", encoding="utf-8")
