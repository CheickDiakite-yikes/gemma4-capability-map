from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_h2b_residual_exactness_synthesis.py"
SPEC = importlib.util.spec_from_file_location("build_h2b_residual_exactness_synthesis_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h2b_residual_exactness_synthesis_identifies_v12_strict_winner(tmp_path: Path) -> None:
    payload = SCRIPT.build_h2b_residual_exactness_synthesis(output_dir=tmp_path)

    manifest = payload["manifest"]
    assert manifest["profile_count"] == 6
    assert manifest["case_count"] == 5
    assert manifest["v12_exact_success_count"] == 4
    assert manifest["v12_executor_success_count"] == 4
    assert manifest["v9_exact_success_count"] == 3
    assert manifest["v9_executor_success_count"] == 4
    assert manifest["h2a_exact_success_count"] == 0
    assert manifest["h2a_executor_success_count"] == 3
    assert manifest["strict_winner"] == "component_residual_guard_v12"
    assert manifest["executor_winners"] == ["component_residual_guard_v12", "component_value_guard_v9"]
    assert manifest["promotion_decision"] == "do_not_globalize_v12_use_h2c_scoped_residual_route"

    packets = {row["profile_label"]: row for row in payload["packet_rows"]}
    assert packets["no_directive"]["exact_success_count"] == 1
    assert packets["component_label_guard_v11"]["executor_success_count"] == 3
    assert packets["component_residual_guard_v12"]["exact_success_count"] == 4
    assert packets["code_label_exact_guard_v15"]["exact_success_count"] == 3
    assert packets["h2a_stale_selection_gate"]["exact_success_count"] == 0

    case_rows = {
        (row["profile_label"], row["case_id"]): row
        for row in payload["case_rows"]
    }
    assert case_rows[
        ("component_residual_guard_v12", "h1p_surface_mode_toggle_note_value_decoy")
    ]["exact_match"] is True
    assert case_rows[
        ("component_residual_guard_v12", "component_value_result_pill_log_decoy")
    ]["executor_equivalence_match"] is False
    assert case_rows[
        ("code_label_exact_guard_v15", "h1o_code_badge_c08_note_decoy")
    ]["exact_match"] is True

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "4/5 strict" in findings["v12_is_strict_winner"]
    assert "tying v12 on executor-equivalence" in findings["v9_ties_executor_but_not_exact"]
    assert "not solve the alias/code-label residual" in findings["h2a_is_not_residual_exactness_solution"]
    assert "H2c" in findings["next_slice"]

    assert (tmp_path / "tables" / "h2b_residual_exactness_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h2b_residual_exactness_case_matrix.csv").exists()
    assert (tmp_path / "tables" / "h2b_residual_exactness_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h2b_residual_exactness_findings.csv").exists()
