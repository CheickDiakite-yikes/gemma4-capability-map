from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_h1x_v11_breaker_synthesis.py"
SPEC = importlib.util.spec_from_file_location("build_h1x_v11_breaker_synthesis_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h1x_v11_breaker_synthesis_marks_v12_local_winner(tmp_path: Path) -> None:
    payload = SCRIPT.build_h1x_v11_breaker_synthesis(output_dir=tmp_path)

    assert payload["manifest"]["profile_count"] == 4
    assert payload["manifest"]["case_count"] == 8
    assert payload["manifest"]["no_directive_exact_success_count"] == 2
    assert payload["manifest"]["v11_exact_success_count"] == 7
    assert payload["manifest"]["v11_executor_success_count"] == 7
    assert payload["manifest"]["v12_exact_success_count"] == 8
    assert payload["manifest"]["v12_executor_success_count"] == 8
    assert payload["manifest"]["v15_exact_success_count"] == 6
    assert payload["manifest"]["v15_executor_success_count"] == 7
    assert (
        payload["manifest"]["promotion_decision"]
        == "component_residual_guard_is_h1x_local_winner_keep_transfer_gate"
    )

    packet_rows = {row["profile_label"]: row for row in payload["packet_rows"]}
    assert packet_rows["no_directive"]["exact_success_count"] == 2
    assert packet_rows["component_label_guard_v11"]["exact_success_count"] == 7
    assert packet_rows["component_residual_guard_v12"]["exact_success_count"] == 8
    assert packet_rows["code_label_exact_guard_v15"]["exact_success_count"] == 6
    assert packet_rows["code_label_exact_guard_v15"]["executor_success_count"] == 7

    family_rows = {(row["profile_label"], row["family"]): row for row in payload["family_rows"]}
    assert family_rows[("no_directive", "h1x_oblique_activation_no_call")]["exact_success_count"] == 2
    assert family_rows[("no_directive", "h1x_oblique_stale_field")]["exact_success_count"] == 0
    assert family_rows[("component_label_guard_v11", "h1x_oblique_stale_field")][
        "exact_success_count"
    ] == 1
    assert family_rows[("component_residual_guard_v12", "h1x_oblique_stale_field")][
        "exact_success_count"
    ] == 2
    assert family_rows[("code_label_exact_guard_v15", "h1x_oblique_surface_value")][
        "exact_success_count"
    ] == 1
    assert family_rows[("code_label_exact_guard_v15", "h1x_oblique_surface_value")][
        "executor_success_count"
    ] == 2

    comparisons = {row["comparison_dir"]: row for row in payload["comparison_rows"]}
    v11_vs_no = next(row for key, row in comparisons.items() if "component_label_guard_vs_no_directive" in key)
    v12_vs_no = next(row for key, row in comparisons.items() if "component_residual_guard_vs_no_directive" in key)
    v12_vs_v11 = next(
        row for key, row in comparisons.items() if "component_residual_guard_vs_component_label_guard" in key
    )
    v15_vs_v12 = next(
        row for key, row in comparisons.items() if "code_label_exact_guard_vs_component_residual_guard" in key
    )
    assert v11_vs_no["delta_exact_rate"] == 0.625
    assert v12_vs_no["delta_exact_rate"] == 0.75
    assert v12_vs_v11["delta_exact_rate"] == 0.125
    assert v15_vs_v12["delta_executor_equivalence_rate"] == -0.125

    failures = {(row["profile_label"], row["case_id"]): row for row in payload["non_exact_rows"]}
    assert failures[("component_label_guard_v11", "h1x_responsible_party_field_old_owner_memo_decoy")][
        "failure_mode"
    ] == "wrong_tool"
    assert failures[("code_label_exact_guard_v15", "h1x_resolution_chip_comment_result_decoy")][
        "failure_mode"
    ] == "executable_paraphrase"

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "2/8" in findings["h1x_breaks_no_directive"]
    assert "7/8" in findings["h1x_breaks_v11_saturation"]
    assert "8/8" in findings["v12_local_winner"]
    assert "routed residual helper" in findings["next_slice"]

    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "tables" / "h1x_v11_breaker_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h1x_v11_breaker_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h1x_v11_breaker_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h1x_v11_breaker_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h1x_v11_breaker_findings.csv").exists()
