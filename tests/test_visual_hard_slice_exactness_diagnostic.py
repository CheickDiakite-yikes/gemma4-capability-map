from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_visual_hard_slice_exactness.py"
SPEC = importlib.util.spec_from_file_location("analyze_visual_hard_slice_exactness_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_visual_hard_slice_exactness_diagnostic_splits_label_artifacts_from_failures(tmp_path: Path) -> None:
    packet_dir = tmp_path / "packet"
    v4_dir = packet_dir / "v4"
    v5_dir = packet_dir / "v5"
    v4_dir.mkdir(parents=True)
    v5_dir.mkdir(parents=True)
    (packet_dir / "manifest.json").write_text(
        json.dumps({"packet_run_id": "packet-v1", "system_ids": ["v4", "v5"]}) + "\n",
        encoding="utf-8",
    )
    (v4_dir / "probe_results.json").write_text(
        json.dumps(
            [
                {
                    "case_id": "visual_metric_panel_vs_table_selector",
                    "family": "visual_argument_copying",
                    "exact_match": False,
                    "executable_match": True,
                    "expected_execution": {"region_ids": ["region-1"]},
                    "actual_execution": [
                        {
                            "validator_result": "pass",
                            "output": {"region_ids": ["region-1"]},
                        }
                    ],
                    "expected_calls": [
                        {"name": "extract_layout", "arguments": {"image_id": "img", "target_query": "dashboard metric"}}
                    ],
                    "actual_calls": [
                        {"name": "extract_layout", "arguments": {"image_id": "img", "target_query": "metric panel"}}
                    ],
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (v5_dir / "probe_results.json").write_text(
        json.dumps(
            [
                {
                    "case_id": "visual_form_error_with_prior_selection_decoy",
                    "family": "visual_tool_routing",
                    "exact_match": False,
                    "executable_match": False,
                    "expected_execution": {"region_ids": ["region-2"]},
                    "actual_execution": [
                        {
                            "validator_result": "fail",
                            "output": {},
                            "error": "Selection not found",
                        }
                    ],
                    "expected_calls": [
                        {"name": "extract_layout", "arguments": {"image_id": "img", "target_query": "validation error"}}
                    ],
                    "actual_calls": [
                        {"name": "refine_selection", "arguments": {"selection_id": "sel-stale", "filter_query": "validation error"}}
                    ],
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = SCRIPT.analyze_visual_hard_slice_exactness(
        packet_dir=packet_dir,
        output_dir=tmp_path / "out",
        system_ids=["v4", "v5"],
    )
    rows = {row["system_id"]: row for row in payload["case_rows"]}
    summary = {row["system_id"]: row for row in payload["system_summary"]}

    assert rows["v4"]["exactness_diagnosis"] == "executable_selector_alias"
    assert rows["v4"]["research_interpretation"] == "benchmark_label_artifact_candidate"
    assert rows["v4"]["executor_target_match"] is True
    assert rows["v5"]["exactness_diagnosis"] == "wrong_tool_executor_failure"
    assert rows["v5"]["research_interpretation"] == "true_harness_failure"
    assert rows["v5"]["executor_target_match"] is False
    assert summary["v4"]["benchmark_label_artifact_candidate_count"] == 1
    assert summary["v5"]["true_harness_failure_count"] == 1
    assert (tmp_path / "out" / "exactness_diagnostic.md").exists()
    assert (tmp_path / "out" / "tables" / "visual_hard_slice_exactness_gaps.csv").exists()
