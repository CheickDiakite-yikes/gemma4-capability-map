from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
H2L_MODULE_PATH = ROOT / "scripts" / "build_h2l_target_normalization_overreach_synthesis.py"
H2X_MODULE_PATH = ROOT / "scripts" / "build_h2x_cli_semantic_pressure_synthesis.py"
MODULE_PATH = ROOT / "scripts" / "build_h3a_transfer_backtest_synthesis.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load_module("build_h2l_target_normalization_overreach_synthesis", H2L_MODULE_PATH)
_load_module("build_h2x_cli_semantic_pressure_synthesis", H2X_MODULE_PATH)
H3A_TRANSFER_SCRIPT = _load_module("build_h3a_transfer_backtest_synthesis", MODULE_PATH)


def test_h3a_transfer_backtest_synthesis_marks_clean_broad_transfer(tmp_path: Path) -> None:
    payload = H3A_TRANSFER_SCRIPT.build_h3a_transfer_backtest_synthesis(output_dir=tmp_path)
    manifest = payload["manifest"]

    assert manifest["packet_row_count"] == 24
    assert manifest["comparison_count"] == 12
    assert manifest["transfer_packet_count"] == 12
    assert manifest["h2w_transfer_case_count"] == 109
    assert manifest["h3a_transfer_case_count"] == 109
    assert manifest["h2w_transfer_exact_success_count"] == 109
    assert manifest["h3a_transfer_exact_success_count"] == 109
    assert manifest["h3a_transfer_executor_success_count"] == 109
    assert manifest["h3a_exact_delta_sum_vs_h2w"] == 0.0
    assert manifest["h3a_executor_delta_sum_vs_h2w"] == 0.0
    assert manifest["h3a_fixed_case_count_vs_h2w"] == 0
    assert manifest["h3a_regression_count_vs_h2w"] == 0
    assert manifest["h3a_non_exact_count"] == 0
    assert manifest["h3a_stale_paraphrase_intervention_count"] == 0
    assert manifest["h3a_negative_value_intervention_count"] == 0
    assert manifest["h3a_new_helper_intervention_count"] == 0
    assert manifest["h3a_transfer_clean"] is True
    assert manifest["h3a_ties_h2w_transfer_gate"] is True
    assert manifest["h3a_new_helpers_do_not_overtrigger_on_transfer"] is True
    assert (
        manifest["promotion_decision"]
        == "h3a_passes_broad_h2w_transfer_backtest_next_harder_holdout_required"
    )

    assert len(payload["packet_pair_rows"]) == 12
    assert all(row["h3a_delta_exact_vs_h2w"] == 0.0 for row in payload["packet_pair_rows"])
    assert all(row["h3a_delta_executor_vs_h2w"] == 0.0 for row in payload["packet_pair_rows"])
    assert payload["fixed_case_rows"] == []
    assert payload["regression_rows"] == []

    h3a_interventions = [
        row for row in payload["intervention_rows"] if row["profile_label"].endswith("_h3a_boundary_combined")
    ]
    assert any(row["intervention_kind"] == "visual_semantic_target_preservation" for row in h3a_interventions)
    assert not any(
        row["intervention_kind"]
        in {"visual_stale_selection_paraphrase_guard", "visual_negative_value_component_target_preservation"}
        for row in h3a_interventions
    )

    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "figures" / "h3a_transfer_backtest_gate.svg").exists()
    assert (tmp_path / "tables" / "h3a_transfer_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h3a_transfer_packet_pairs.csv").exists()
    assert (tmp_path / "tables" / "h3a_transfer_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h3a_transfer_intervention_rows.csv").exists()
