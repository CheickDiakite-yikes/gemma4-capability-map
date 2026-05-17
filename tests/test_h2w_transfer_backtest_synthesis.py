from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H2L_MODULE_PATH = ROOT / "scripts" / "build_h2l_target_normalization_overreach_synthesis.py"
H2W_TRANSFER_MODULE_PATH = ROOT / "scripts" / "build_h2w_transfer_backtest_synthesis.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load_module("build_h2l_target_normalization_overreach_synthesis", H2L_MODULE_PATH)
H2W_TRANSFER_SCRIPT = _load_module("build_h2w_transfer_backtest_synthesis", H2W_TRANSFER_MODULE_PATH)


def test_h2w_transfer_backtest_preserves_older_packets_and_records_runtime_posture(tmp_path: Path) -> None:
    payload = H2W_TRANSFER_SCRIPT.build_h2w_transfer_backtest_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 36
    assert manifest["comparison_count"] == 24
    assert manifest["family_row_count"] == 49
    assert manifest["h2w_transfer_packet_count"] == 12
    assert manifest["h2w_transfer_case_count"] == 109
    assert manifest["h2w_transfer_exact_success_count"] == 109
    assert manifest["h2w_transfer_executor_success_count"] == 109
    assert manifest["h2w_non_h2q_transfer_case_count"] == 101
    assert manifest["h2w_non_h2q_transfer_exact_success_count"] == 101
    assert manifest["h2w_non_h2q_transfer_executor_success_count"] == 101
    assert manifest["h2w_non_exact_count"] == 0
    assert manifest["h2w_exact_delta_sum_vs_h2u"] == 0.0
    assert manifest["h2w_executor_delta_sum_vs_h2u"] == 0.0
    assert manifest["h2w_exact_delta_sum_vs_h2r"] == 0.19999999999999996
    assert manifest["h2w_executor_delta_sum_vs_h2r"] == 0.19999999999999996
    assert manifest["h2w_regression_count_vs_h2u"] == 0
    assert manifest["h2w_fixed_case_count_vs_h2u"] == 0
    assert manifest["h2w_fixed_case_count_vs_h2r"] == 2
    assert manifest["h2t_delta_exact_vs_h2r"] == 0.19999999999999996
    assert manifest["h2w_semantic_target_preservation_count"] == 5
    assert manifest["h2w_target_query_normalization_count"] == 30
    assert manifest["h2w_stale_selection_gate_count"] == 7
    assert manifest["h2w_composed_route_gating_blocked_count"] == 4
    assert "Metal GPU timeout" in manifest["runtime_posture_note"]
    assert manifest["promotion_decision"] == "h2w_transfer_backtest_passes_ready_for_packaged_workflow_gate"

    packet_pairs = {row["slice"]: row for row in payload["packet_pair_rows"]}
    assert set(packet_pairs) == set(H2W_TRANSFER_SCRIPT.TRANSFER_LABELS)
    assert packet_pairs["h2t"]["h2r_exact_success_count"] == 8
    assert packet_pairs["h2t"]["h2w_exact_success_count"] == 10
    for row in packet_pairs.values():
        assert row["h2w_exact_success_count"] == row["case_count"]
        assert row["h2w_executor_success_count"] == row["case_count"]
        assert row["h2w_delta_exact_vs_h2u"] == 0.0
        assert row["h2w_delta_executor_vs_h2u"] == 0.0

    fixed_vs_h2r = {
        row["case_id"]
        for row in payload["fixed_case_rows"]
        if row["comparison_label"] == "h2t_h2w_vs_h2r" and row["delta_exact_match"] > 0
    }
    assert fixed_vs_h2r == {
        "h2t_metric_panel_negation_scope_note",
        "h2t_summary_tile_negation_scope_caption",
    }

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "109/109 strict exactness" in findings["h2w_transfer_backtest_is_clean"]
    assert "zero exact-rate" in findings["h2w_ties_current_h2u_incumbent"]
    assert "Metal GPU timeout" in findings["h2w_runtime_posture_needs_low_concurrency"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2w_transfer_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2w_transfer_packet_pairs.csv").exists()
    assert (tmp_path / "tables" / "h2w_transfer_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2w_transfer_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h2w_transfer_intervention_rows.csv").exists()
    assert (tmp_path / "tables" / "h2w_transfer_fixed_case_rows.csv").exists()
    assert (tmp_path / "figures" / "h2w_transfer_backtest_gate.svg").exists()
