from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H2L_MODULE_PATH = ROOT / "scripts" / "build_h2l_target_normalization_overreach_synthesis.py"
H2Q_MODULE_PATH = ROOT / "scripts" / "build_h2q_composed_surface_value_stale_synthesis.py"
H2R_MODULE_PATH = ROOT / "scripts" / "build_h2r_composed_route_gating_synthesis.py"
H2S_MODULE_PATH = ROOT / "scripts" / "build_h2s_fresh_composed_holdout_synthesis.py"


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
H2S_SCRIPT = _load_module("build_h2s_fresh_composed_holdout_synthesis", H2S_MODULE_PATH)


def test_h2s_fresh_composed_holdout_synthesis_marks_frozen_h2r_gain(tmp_path: Path) -> None:
    payload = H2S_SCRIPT.build_h2s_fresh_composed_holdout_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["h2s_case_count"] == 10
    assert manifest["h2s_h2j_exact_success_count"] == 1
    assert manifest["h2s_h2o_exact_success_count"] == 3
    assert manifest["h2s_h2p_exact_success_count"] == 3
    assert manifest["h2s_h2r_exact_success_count"] == 10
    assert manifest["h2s_h2r_executor_success_count"] == 10
    assert manifest["h2s_h2r_delta_exact_vs_h2p"] == 0.7
    assert manifest["h2s_h2r_delta_executor_vs_h2p"] == 0.7
    assert manifest["h2s_h2r_delta_exact_vs_h2o"] == 0.7
    assert manifest["h2s_h2r_delta_exact_vs_h2j"] == 0.9
    assert manifest["h2s_h2r_non_exact_count"] == 0
    assert manifest["h2s_h2r_composed_route_gating_count"] == 7
    assert manifest["h2s_h2r_value_bearing_synthesis_count"] == 2
    assert manifest["h2s_h2r_target_query_normalization_count"] == 4
    assert manifest["promotion_decision"] == "h2r_passes_fresh_h2s_holdout_requires_h2t_or_packaged_transfer"

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "10/10 strict" in findings["h2s_fresh_holdout_confirms_h2r_transfer"]
    assert "0.7 exact-rate" in findings["h2s_composed_route_gate_is_causal"]
    assert "7 composed route gates" in findings["h2s_h2r_mechanism_is_mixed_not_single_helper"]
    assert "0 recorded helper rows" in findings["h2s_clean_control_does_not_need_visual_helper"]
    assert "harder H2t holdout" in findings["h2s_next_requires_h2t_or_packaged_transfer"]

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h2s_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2s_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h2s_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h2s_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2s_intervention_rows.csv").exists()
    assert (tmp_path / "tables" / "h2s_findings.csv").exists()
    assert (tmp_path / "figures" / "h2s_fresh_composed_holdout_gate.svg").exists()
