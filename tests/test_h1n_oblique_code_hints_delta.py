from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_h1n_oblique_code_hints_delta.py"
SPEC = importlib.util.spec_from_file_location("analyze_h1n_oblique_code_hints_delta_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_h1n_oblique_code_hints_delta_tracks_gain_and_regression(tmp_path: Path) -> None:
    payload = SCRIPT.analyze_h1n_oblique_code_hints_delta(output_dir=tmp_path)

    assert payload["manifest"]["case_count"] == 6
    assert payload["manifest"]["gain_count"] == 2
    assert payload["manifest"]["loss_count"] == 1
    assert payload["manifest"]["preserved_success_count"] == 3
    assert payload["manifest"]["net_executor_equivalence_gain"] == 1

    cases = {row["case_id"]: row for row in payload["case_rows"]}
    assert cases["transfer_oblique_cell_r42_notice_decoy"]["transition"] == "repair_gain"
    assert cases["transfer_oblique_alert_p55_toggle_decoy"]["transition"] == "repair_gain"
    regression = cases["transfer_oblique_field_e19_old_selection_decoy"]
    assert regression["transition"] == "regression"
    assert regression["classification"] == "stale_selection_tool_attraction"
    assert regression["candidate_tool"] == "refine_selection"
    assert regression["candidate_selection_id"] == "sel-e19-archive"
    assert regression["candidate_filter_query"] == "not"

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "repairs 2 cases and regresses 1 case" in findings["net_gain_with_regression"]
    assert "transfer_oblique_field_e19_old_selection_decoy regresses" in findings["regression_case"]
    assert "earlier oracle/repeat packets" in findings["next_test"]

    assert (tmp_path / "diagnostic.md").exists()
    assert (tmp_path / "diagnostic.json").exists()
    assert (tmp_path / "tables" / "h1n_oblique_code_hints_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "h1n_oblique_code_hints_findings.csv").exists()
