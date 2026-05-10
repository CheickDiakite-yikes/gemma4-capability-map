from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_h1s_component_residual_transfer_synthesis.py"
)
SPEC = importlib.util.spec_from_file_location("build_h1s_component_residual_transfer_synthesis_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_build_h1s_component_residual_transfer_synthesis_marks_negative_transfer(tmp_path: Path) -> None:
    payload = SCRIPT.build_h1s_component_residual_transfer_synthesis(output_dir=tmp_path)

    assert payload["manifest"]["packet_row_count"] == 12
    assert payload["manifest"]["transfer_case_count"] == 32
    assert payload["manifest"]["v11_transfer_exact_success_count"] == 26
    assert payload["manifest"]["v11_transfer_executor_success_count"] == 29
    assert payload["manifest"]["v12_transfer_exact_success_count"] == 27
    assert payload["manifest"]["v12_transfer_executor_success_count"] == 27
    assert payload["manifest"]["promotion_decision"] == "targeted_patch_not_global_default"

    packet_rows = {(row["packet_label"], row["profile_label"]): row for row in payload["packet_rows"]}
    assert packet_rows[("h1r_component_residual", "component_residual_guard_v12")][
        "exact_success_count"
    ] == 6
    assert packet_rows[("h1n_component_value", "component_residual_guard_v12")][
        "exact_success_count"
    ] == 5
    assert packet_rows[("h1o_control_factorial", "component_residual_guard_v12")][
        "executor_success_count"
    ] == 11
    assert packet_rows[("h1p_component_value", "component_residual_guard_v12")][
        "executor_success_count"
    ] == 11

    aggregate = {row["profile_label"]: row for row in payload["transfer_aggregate_rows"]}
    assert aggregate["component_label_guard_v11"]["executor_success_count"] == 29
    assert aggregate["component_residual_guard_v12"]["exact_success_count"] == 27
    assert aggregate["component_residual_guard_v12"]["executor_success_count"] == 27
    assert aggregate["no_directive"]["exact_success_count"] == 10

    comparisons = {row["comparison_dir"]: row for row in payload["comparison_rows"]}
    h1n_vs_v11 = next(row for key, row in comparisons.items() if "h1n_vs_component_label_guard" in key)
    h1p_vs_v11 = next(row for key, row in comparisons.items() if "h1p_vs_component_label_guard" in key)
    assert h1n_vs_v11["delta_exact_rate"] == -0.125
    assert h1n_vs_v11["delta_executor_equivalence_rate"] == -0.25
    assert h1p_vs_v11["delta_exact_rate"] > 0
    assert h1p_vs_v11["delta_executor_equivalence_rate"] > 0

    failures = {(row["packet_label"], row["case_id"]): row for row in payload["v12_failure_rows"]}
    assert failures[("h1p_component_value", "h1p_stale_phase_tile_archive_decoy")][
        "failure_mode"
    ] == "wrong_tool"
    assert failures[("h1n_component_value", "component_value_state_pill_note_decoy")][
        "failure_mode"
    ] == "argument_mismatch"

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "targeted residual patch" in findings["promotion_decision"]
    assert "27/32" in findings["v12_transfers_strict_but_not_executor"]
    assert "negative-transfer warning" in findings["negative_h1n_transfer"]

    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "tables" / "h1s_component_residual_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h1s_component_residual_transfer_aggregate.csv").exists()
    assert (tmp_path / "tables" / "h1s_component_residual_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h1s_component_residual_v12_failures.csv").exists()
    assert (tmp_path / "tables" / "h1s_component_residual_findings.csv").exists()
