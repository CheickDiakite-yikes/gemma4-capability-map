from __future__ import annotations

import csv
import json
from pathlib import Path

from gemma4_capability_map.knowledge_work.trace_analysis import analyze_ablation_packet, write_trace_analysis


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
