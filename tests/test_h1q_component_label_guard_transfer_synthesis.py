from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_h1q_component_label_guard_transfer_synthesis.py"
)
SPEC = importlib.util.spec_from_file_location("build_h1q_component_label_guard_transfer_synthesis_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h1q_component_label_guard_transfer_synthesis_marks_tradeoff(tmp_path: Path) -> None:
    payload = SCRIPT.build_h1q_component_label_guard_transfer_synthesis(output_dir=tmp_path)

    assert payload["manifest"]["packet_row_count"] == 18
    assert payload["manifest"]["profile_count"] == 6
    assert payload["manifest"]["total_case_count"] == 32
    assert payload["manifest"]["v11_exact_success_count"] == 26
    assert payload["manifest"]["v11_executor_success_count"] == 29

    aggregate = {row["profile_label"]: row for row in payload["aggregate_rows"]}
    assert aggregate["no_directive"]["exact_success_count"] == 10
    assert aggregate["no_directive"]["executor_success_count"] == 12
    assert aggregate["argument_hints_v2"]["exact_success_count"] == 21
    assert aggregate["hybrid_label_guard_v8"]["executor_success_count"] == 27
    assert aggregate["component_value_guard_v9"]["exact_success_count"] == 23
    assert aggregate["component_value_guard_v9"]["executor_success_count"] == 25
    assert aggregate["component_label_guard_v11"]["exact_rate"] == 0.8125
    assert aggregate["component_label_guard_v11"]["executor_rate"] == 0.90625

    packet_rows = {
        (row["packet_label"], row["profile_label"]): row
        for row in payload["packet_rows"]
    }
    assert packet_rows[("h1n_component_value", "component_label_guard_v11")][
        "exact_success_count"
    ] == 6
    assert packet_rows[("h1n_component_value", "component_value_guard_v9")][
        "exact_success_count"
    ] == 4
    assert packet_rows[("h1o_control_factorial", "component_label_guard_v11")][
        "executor_success_count"
    ] == 12
    assert packet_rows[("h1p_component_value", "component_label_guard_v11")][
        "executor_success_count"
    ] == 10
    assert packet_rows[("h1p_component_value", "component_value_guard_v9")][
        "executor_success_count"
    ] == 11

    failures = {
        (row["packet_label"], row["case_id"]): row
        for row in payload["v11_failure_rows"]
    }
    assert failures[("h1n_component_value", "component_value_owner_field_stale_selection_decoy")][
        "failure_mode"
    ] == "wrong_tool"
    assert failures[("h1o_control_factorial", "h1o_code_alert_s92_negated_toggle_decoy")][
        "executor_equivalence_match"
    ] is True
    assert failures[("h1p_component_value", "h1p_surface_mode_toggle_note_value_decoy")][
        "failure_mode"
    ] == "argument_mismatch"

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "component_label_guard_v11 at 26/32" in findings["aggregate_strict_upper_bound"]
    assert "component_label_guard_v11 at 29/32" in findings["aggregate_executor_upper_bound"]
    assert "v11 repairs the broad v9 regression" in findings["v11_repairs_v9_h1n_regressions"]
    assert "10/12 exact and 12/12 executor-equivalent" in findings["v11_sets_h1o_executor_ceiling"]
    assert "loses one executor-equivalent case" in findings["h1p_tradeoff_vs_v9"]
    assert "owner-field, state-tag, and mode-toggle failures" in findings["promotion_decision"]

    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "tables" / "h1q_component_label_guard_packet_summary.csv").exists()
