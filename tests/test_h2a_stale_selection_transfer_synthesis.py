from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_h2a_stale_selection_transfer_synthesis.py"
SPEC = importlib.util.spec_from_file_location("build_h2a_stale_selection_transfer_synthesis_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h2a_stale_selection_transfer_synthesis_promotes_scoped_gate(tmp_path: Path) -> None:
    payload = SCRIPT.build_h2a_stale_selection_transfer_synthesis(output_dir=tmp_path)

    manifest = payload["manifest"]
    assert manifest["local_case_count"] == 10
    assert manifest["local_h2a_exact_success_count"] == 8
    assert manifest["local_h2a_executor_success_count"] == 8
    assert manifest["transfer_case_count"] == 40
    assert manifest["transfer_no_directive_exact_success_count"] == 12
    assert manifest["transfer_no_directive_executor_success_count"] == 14
    assert manifest["transfer_v11_exact_success_count"] == 33
    assert manifest["transfer_v11_executor_success_count"] == 36
    assert manifest["transfer_v12_exact_success_count"] == 35
    assert manifest["transfer_v12_executor_success_count"] == 35
    assert manifest["transfer_h2a_exact_success_count"] == 35
    assert manifest["transfer_h2a_executor_success_count"] == 38
    assert manifest["transfer_h2a_exact_delta_vs_v11_count"] == 2
    assert manifest["transfer_h2a_executor_delta_vs_v11_count"] == 2
    assert manifest["transfer_h2a_exact_delta_vs_v12_count"] == 0
    assert manifest["transfer_h2a_executor_delta_vs_v12_count"] == 3
    assert manifest["promotion_decision"] == (
        "promote_h2a_as_scoped_controller_helper_and_target_exact_alias_residuals"
    )

    aggregates = {
        (row["evaluation_split"], row["profile_label"]): row
        for row in payload["aggregate_rows"]
    }
    assert aggregates[("transfer_h1n_h1o_h1p_h1x", "no_directive")]["exact_success_count"] == 12
    assert aggregates[("transfer_h1n_h1o_h1p_h1x", "component_label_guard_v11")][
        "executor_success_count"
    ] == 36
    assert aggregates[("transfer_h1n_h1o_h1p_h1x", "component_residual_guard_v12")][
        "executor_success_count"
    ] == 35
    assert aggregates[("transfer_h1n_h1o_h1p_h1x", "h2a_visual_stale_selection_gate")][
        "executor_success_count"
    ] == 38

    packet_rows = {
        (row["slice_id"], row["profile_label"]): row
        for row in payload["packet_rows"]
    }
    assert packet_rows[("h1x_v11_breaker", "h2a_visual_stale_selection_gate")]["exact_success_count"] == 8
    assert packet_rows[("h1p_component_value", "h2a_visual_stale_selection_gate")][
        "executor_success_count"
    ] == 10
    assert packet_rows[("h1o_control_factorial", "h2a_visual_stale_selection_gate")][
        "executor_success_count"
    ] == 12
    assert packet_rows[("h1n_component_value", "h2a_visual_stale_selection_gate")][
        "executor_success_count"
    ] == 8

    residuals = {(row["slice_id"], row["case_id"]): row for row in payload["h2a_residual_rows"]}
    assert residuals[("h1n_component_value", "component_value_result_pill_log_decoy")][
        "failure_mode"
    ] == "executable_paraphrase"
    assert residuals[("h1p_component_value", "h1p_compact_state_tag_log_value_decoy")][
        "failure_mode"
    ] == "argument_mismatch"
    assert residuals[("h1o_control_factorial", "h1o_code_alert_s92_negated_toggle_decoy")][
        "executor_equivalence_match"
    ] is True

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "35/40" in findings["h2a_transfers_beyond_h1y"]
    assert "2 strict rows" in findings["h2a_beats_v11_on_transfer"]
    assert "beats v12 executor-equivalence by 3 rows" in findings[
        "h2a_ties_v12_strict_but_beats_executor_equivalence"
    ]
    assert "Promote H2a" in findings["promotion_decision"]

    assert (tmp_path / "tables" / "h2a_transfer_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2a_transfer_aggregate_summary.csv").exists()
    assert (tmp_path / "tables" / "h2a_transfer_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2a_transfer_residual_rows.csv").exists()
    assert (tmp_path / "tables" / "h2a_transfer_findings.csv").exists()
