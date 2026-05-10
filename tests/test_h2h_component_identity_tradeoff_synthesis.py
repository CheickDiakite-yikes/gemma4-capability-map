from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_h2h_component_identity_tradeoff_synthesis.py"
SPEC = importlib.util.spec_from_file_location("build_h2h_component_identity_tradeoff_synthesis_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h2h_component_identity_tradeoff_synthesis_rejects_global_promotion(tmp_path: Path) -> None:
    payload = SCRIPT.build_h2h_component_identity_tradeoff_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 9
    assert manifest["comparison_count"] == 6
    assert manifest["h2h_h2f_exact_success_count"] == 9
    assert manifest["h2h_h2f_executor_success_count"] == 9
    assert manifest["h2h_h2b_exact_success_count"] == 3
    assert manifest["h2h_h2b_executor_success_count"] == 3
    assert manifest["h2h_h1x_exact_success_count"] == 6
    assert manifest["h2h_h1x_executor_success_count"] == 6
    assert manifest["h2h_delta_exact_vs_h2e_on_h2f"] == 0.30000000000000004
    assert manifest["h2h_delta_exact_vs_h2e_on_h2b"] == -0.4
    assert manifest["h2h_delta_exact_vs_h2e_on_h1x"] == -0.25
    assert manifest["h2h_non_exact_count"] == 5
    assert manifest["promotion_decision"] == "reject_global_h2h_keep_as_h2f_scoped_repair"

    packet_rows = {(row["suite"], row["profile_label"]): row for row in payload["packet_rows"]}
    assert packet_rows[("h2f", "h2h_component_identity_negative_examples")]["exact_success_count"] == 9
    assert packet_rows[("h2b", "h2e_route_arbitration")]["exact_success_count"] == 5
    assert packet_rows[("h2b", "h2h_component_identity_negative_examples")]["exact_success_count"] == 3
    assert packet_rows[("h1x", "h2e_route_arbitration")]["exact_success_count"] == 8
    assert packet_rows[("h1x", "h2h_component_identity_negative_examples")]["exact_success_count"] == 6

    non_exact = {(row["suite"], row["case_id"]): row for row in payload["h2h_non_exact_rows"]}
    assert non_exact[("h2f", "h2f_state_marker_history_value_decoy")]["actual_target_query"] == (
        "lifecycle state marker"
    )
    assert non_exact[("h2b", "component_value_result_pill_log_decoy")]["actual_target_query"] == "result tile"
    assert non_exact[("h2b", "h1o_code_badge_c08_note_decoy")]["actual_target_query"] == "badge m31 c08"
    assert non_exact[("h1x", "h1x_resolution_chip_comment_result_decoy")]["actual_target_query"] == "result tile"
    assert non_exact[("h1x", "h1x_error_notice_history_activation_decoy")]["actual_target_query"] == "error notice"

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "H2h reaches 9/10 strict" in findings["h2h_repairs_fresh_h2f"]
    assert "falls to 3/5 on H2b" in findings["h2h_regresses_prior_transfer_gates"]
    assert "h2b:result pill->result tile" in findings["h2h_failure_boundary"]
    assert "conditional arbitration" in findings["next_slice"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2h_tradeoff_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2h_tradeoff_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2h_tradeoff_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2h_tradeoff_findings.csv").exists()
    assert (tmp_path / "figures" / "h2h_tradeoff_gate.svg").exists()
