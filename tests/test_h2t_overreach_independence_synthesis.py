from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H2L_MODULE_PATH = ROOT / "scripts" / "build_h2l_target_normalization_overreach_synthesis.py"
H2Q_MODULE_PATH = ROOT / "scripts" / "build_h2q_composed_surface_value_stale_synthesis.py"
H2R_MODULE_PATH = ROOT / "scripts" / "build_h2r_composed_route_gating_synthesis.py"
H2T_MODULE_PATH = ROOT / "scripts" / "build_h2t_overreach_independence_synthesis.py"


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
H2T_SCRIPT = _load_module("build_h2t_overreach_independence_synthesis", H2T_MODULE_PATH)


def test_h2t_overreach_independence_synthesis_marks_controller_tradeoff(tmp_path: Path) -> None:
    payload = H2T_SCRIPT.build_h2t_overreach_independence_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["h2t_case_count"] == 10
    assert manifest["h2t_h2e_exact_success_count"] == 6
    assert manifest["h2t_h2e_executor_success_count"] == 9
    assert manifest["h2t_h2j_exact_success_count"] == 8
    assert manifest["h2t_h2o_exact_success_count"] == 8
    assert manifest["h2t_h2p_exact_success_count"] == 8
    assert manifest["h2t_h2r_exact_success_count"] == 8
    assert manifest["h2t_h2r_executor_success_count"] == 8
    assert manifest["h2t_h2r_delta_exact_vs_h2e"] == 0.20000000000000007
    assert manifest["h2t_h2r_delta_executor_vs_h2e"] == -0.09999999999999998
    assert manifest["h2t_h2r_delta_exact_vs_h2j"] == 0.0
    assert manifest["h2t_h2r_delta_exact_vs_h2o"] == 0.0
    assert manifest["h2t_h2r_delta_exact_vs_h2p"] == 0.0
    assert manifest["h2t_h2r_target_query_normalization_count"] == 6
    assert manifest["h2t_h2r_bad_normalization_count"] == 2
    assert manifest["h2t_negation_scope_h2e_exact_count"] == 2
    assert manifest["h2t_negation_scope_h2r_exact_count"] == 0
    assert manifest["promotion_decision"] == "h2t_breaks_h2r_requires_h2u_negation_aware_normalization"

    bad_rows = {row["case_id"]: row for row in payload["bad_normalization_rows"]}
    assert set(bad_rows) == {
        "h2t_metric_panel_negation_scope_note",
        "h2t_summary_tile_negation_scope_caption",
    }
    assert bad_rows["h2t_metric_panel_negation_scope_note"]["raw_target_query"] == "metric panel"
    assert bad_rows["h2t_metric_panel_negation_scope_note"]["actual_target_query"] == "training note"
    assert bad_rows["h2t_summary_tile_negation_scope_caption"]["raw_target_query"] == "summary tile"
    assert bad_rows["h2t_summary_tile_negation_scope_caption"]["actual_target_query"] == "caption"

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "8/10 strict" in findings["h2t_breaks_h2r_topline_saturation"]
    assert "6/10 strict" in findings["h2t_exposes_h2e_tradeoff"]
    assert "delta 0.0 exact-rate" in findings["h2t_later_helpers_do_not_add_signal"]
    assert "raw model emitted the expected target" in findings["h2t_bad_normalization_is_controller_induced"]
    assert "H2u intervention" in findings["h2t_next_requires_h2u"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2t_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2t_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2t_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h2t_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2t_intervention_rows.csv").exists()
    assert (tmp_path / "tables" / "h2t_bad_normalization_rows.csv").exists()
    assert (tmp_path / "tables" / "h2t_findings.csv").exists()
    assert (tmp_path / "figures" / "h2t_overreach_independence_gate.svg").exists()
