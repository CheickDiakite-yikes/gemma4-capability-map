from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_visual_tool_choice_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("analyze_visual_tool_choice_diagnostics_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_visual_tool_choice_diagnostics_classifies_wrong_tool_and_no_call(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packet"
    runs_dir = packet_dir / "runs"
    wrong_tool_dir = runs_dir / "visual_latest_filter_literal"
    no_call_dir = runs_dir / "visual_form_target_literal"
    wrong_tool_dir.mkdir(parents=True)
    no_call_dir.mkdir(parents=True)
    (packet_dir / "manifest.json").write_text(
        json.dumps(
            {
                "packet_run_id": "packet-v1",
                "system_id": "candidate-system",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (packet_dir / "live_replay_results.json").write_text(
        json.dumps(
            [
                {
                    "case_id": "visual_latest_filter_literal",
                    "family": "visual_referent_carryover",
                    "replay_failure_mode": "wrong_tool",
                    "replay_exact_match": False,
                    "replay_executable_match": None,
                    "expected_call_count": 1,
                    "replay_actual_call_count": 1,
                    "output_dir": str(wrong_tool_dir),
                },
                {
                    "case_id": "visual_form_target_literal",
                    "family": "visual_argument_copying",
                    "replay_failure_mode": "no_tool_call",
                    "replay_exact_match": False,
                    "replay_executable_match": False,
                    "expected_call_count": 1,
                    "replay_actual_call_count": 0,
                    "output_dir": str(no_call_dir),
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (wrong_tool_dir / "probe_results.json").write_text(
        json.dumps(
            [
                {
                    "expected_calls": [
                        {"name": "refine_selection", "arguments": {"selection_id": "sel-001", "filter_query": "latest"}}
                    ],
                    "actual_calls": [
                        {"name": "extract_layout", "arguments": {"image_id": "img-form", "target_query": "error"}}
                    ],
                    "raw_model_output": 'extract_layout(image_id="img-form", target_query="error")',
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (no_call_dir / "probe_results.json").write_text(
        json.dumps(
            [
                {
                    "expected_calls": [
                        {"name": "extract_layout", "arguments": {"image_id": "img-form", "target_query": "validation error"}}
                    ],
                    "actual_calls": [],
                    "raw_model_output": "I need the screenshot.",
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = SCRIPT.analyze_visual_tool_choice_diagnostics([packet_dir], output_dir=tmp_path / "out")
    rows = {row["case_id"]: row for row in payload["rows"]}

    assert rows["visual_latest_filter_literal"]["diagnosis"] == "wrong_visual_tool_selection"
    assert rows["visual_latest_filter_literal"]["packet_label"] == "packet-v1"
    assert rows["visual_latest_filter_literal"]["expected_tools"] == "refine_selection"
    assert rows["visual_latest_filter_literal"]["actual_tools"] == "extract_layout"
    assert "latest-selection filtering" in rows["visual_latest_filter_literal"]["next_diagnostic"]
    assert rows["visual_form_target_literal"]["diagnosis"] == "visual_tool_initiation_missing"
    assert payload["summary"]["diagnosis_counts"] == {
        "visual_tool_initiation_missing": 1,
        "wrong_visual_tool_selection": 1,
    }
    assert payload["summary"]["case_diagnosis_transitions"] == {
        "visual_form_target_literal": ["packet-v1:visual_tool_initiation_missing"],
        "visual_latest_filter_literal": ["packet-v1:wrong_visual_tool_selection"],
    }
    assert (tmp_path / "out" / "visual_tool_choice_diagnostics.csv").exists()
    assert (tmp_path / "out" / "visual_tool_choice_diagnostics.md").exists()


def test_visual_tool_choice_diagnostics_marks_catalog_argument_mismatch(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packet"
    runs_dir = packet_dir / "runs"
    latest_dir = runs_dir / "visual_latest_filter_literal"
    latest_dir.mkdir(parents=True)
    (packet_dir / "manifest.json").write_text(
        json.dumps(
            {
                "packet_run_id": "catalog-packet",
                "system_id": "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (packet_dir / "live_replay_results.json").write_text(
        json.dumps(
            [
                {
                    "case_id": "visual_latest_filter_literal",
                    "family": "visual_referent_carryover",
                    "replay_failure_mode": "argument_mismatch",
                    "replay_exact_match": False,
                    "replay_executable_match": None,
                    "expected_call_count": 1,
                    "replay_actual_call_count": 1,
                    "output_dir": str(latest_dir),
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (latest_dir / "probe_results.json").write_text(
        json.dumps(
            [
                {
                    "expected_calls": [
                        {"name": "refine_selection", "arguments": {"selection_id": "sel-001", "filter_query": "latest"}}
                    ],
                    "actual_calls": [
                        {
                            "name": "refine_selection",
                            "arguments": {"selection_id": "sel-001", "filter_query": "latest issue"},
                        }
                    ],
                    "raw_model_output": 'refine_selection(selection_id="sel-001", filter_query="latest issue")',
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = SCRIPT.analyze_visual_tool_choice_diagnostics([packet_dir], output_dir=tmp_path / "out")
    row = payload["rows"][0]

    assert row["packet_label"] == "visual_role_catalog_v1"
    assert row["diagnosis"] == "visual_literal_argument_mismatch"
    assert row["actual_tools"] == "refine_selection"
    assert "literal visual selector" in row["next_diagnostic"]
    assert payload["summary"]["case_diagnosis_transitions"] == {
        "visual_latest_filter_literal": ["visual_role_catalog_v1:visual_literal_argument_mismatch"],
    }
