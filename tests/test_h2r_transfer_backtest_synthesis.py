from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H2L_MODULE_PATH = ROOT / "scripts" / "build_h2l_target_normalization_overreach_synthesis.py"
H2Q_MODULE_PATH = ROOT / "scripts" / "build_h2q_composed_surface_value_stale_synthesis.py"
H2R_MODULE_PATH = ROOT / "scripts" / "build_h2r_composed_route_gating_synthesis.py"
TRANSFER_MODULE_PATH = ROOT / "scripts" / "build_h2r_transfer_backtest_synthesis.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load_module("build_h2l_target_normalization_overreach_synthesis", H2L_MODULE_PATH)
_load_module("build_h2q_composed_surface_value_stale_synthesis", H2Q_MODULE_PATH)
_load_module("build_h2r_composed_route_gating_synthesis", H2R_MODULE_PATH)
TRANSFER_SCRIPT = _load_module("build_h2r_transfer_backtest_synthesis", TRANSFER_MODULE_PATH)


def test_h2r_transfer_backtest_synthesis_marks_positive_transfer_without_global_closure(tmp_path: Path) -> None:
    payload = TRANSFER_SCRIPT.build_h2r_transfer_backtest_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["origin_case_count"] == 8
    assert manifest["origin_exact_success_count"] == 8
    assert manifest["transfer_packet_count"] == 9
    assert manifest["transfer_case_count"] == 81
    assert manifest["transfer_exact_success_count"] == 81
    assert manifest["transfer_executor_success_count"] == 81
    assert manifest["all_packet_count"] == 10
    assert manifest["all_case_count"] == 89
    assert manifest["all_exact_success_count"] == 89
    assert manifest["non_exact_count"] == 0
    assert manifest["comparison_count"] == 20
    assert manifest["h2b_delta_exact_vs_h2h"] == 0.4
    assert manifest["h1x_delta_exact_vs_h2h"] == 0.25
    assert manifest["h1y_delta_exact_vs_h2a"] == 0.19999999999999996
    assert manifest["h1o_delta_exact_vs_h1s"] == 0.08333333333333337
    assert manifest["h1p_delta_exact_vs_h1s"] == 0.08333333333333337
    assert manifest["promotion_decision"] == "h2r_transfer_positive_current_packets_requires_fresh_h2s_holdout"

    packets = {row["profile_label"]: row for row in payload["packet_rows"]}
    assert packets["h2m_transfer_h2r"]["exact_success_count"] == 8
    assert packets["h2k_transfer_h2r"]["exact_success_count"] == 8
    assert packets["h2l_transfer_h2r"]["exact_success_count"] == 8
    assert packets["h2f_transfer_h2r"]["exact_success_count"] == 10
    assert packets["h2b_regression_h2r"]["exact_success_count"] == 5
    assert packets["h1x_regression_h2r"]["exact_success_count"] == 8
    assert packets["h1y_transfer_h2r"]["exact_success_count"] == 10
    assert packets["h1o_transfer_h2r"]["exact_success_count"] == 12
    assert packets["h1p_transfer_h2r"]["exact_success_count"] == 12

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "81/81 strict" in findings["h2r_transfer_preserves_current_gates"]
    assert "beating H2h" in findings["h2r_avoids_h2h_regression_pattern"]
    assert "closes older unsaturated packets" in findings["h2r_closes_older_unsaturated_packets"]
    assert "fresh H2s composition holdout" in findings["h2r_next_requires_fresh_h2s"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2r_transfer_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2r_transfer_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2r_transfer_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h2r_transfer_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2r_transfer_intervention_rows.csv").exists()
    assert (tmp_path / "tables" / "h2r_transfer_findings.csv").exists()
    assert (tmp_path / "figures" / "h2r_transfer_backtest_gate.svg").exists()
