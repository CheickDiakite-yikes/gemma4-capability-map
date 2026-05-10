from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_h1y_routed_residual_synthesis.py"
SPEC = importlib.util.spec_from_file_location("build_h1y_routed_residual_synthesis_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h1y_routed_residual_synthesis_marks_v16_negative_result(tmp_path: Path) -> None:
    payload = SCRIPT.build_h1y_routed_residual_synthesis(output_dir=tmp_path)

    assert payload["manifest"]["case_count"] == 10
    assert payload["manifest"]["no_directive_exact_success_count"] == 0
    assert payload["manifest"]["v11_exact_success_count"] == 5
    assert payload["manifest"]["v11_executor_success_count"] == 5
    assert payload["manifest"]["v12_exact_success_count"] == 7
    assert payload["manifest"]["v12_executor_success_count"] == 7
    assert payload["manifest"]["v16_exact_success_count"] == 5
    assert payload["manifest"]["v16_executor_success_count"] == 5
    assert payload["manifest"]["promotion_decision"] == "do_not_promote_v16_design_v17_selection_origin_guard"

    packet_rows = {row["profile_label"]: row for row in payload["packet_rows"]}
    assert packet_rows["component_residual_guard_v12"]["exact_success_count"] == 7
    assert packet_rows["routed_residual_guard_v16"]["exact_success_count"] == 5

    family_rows = {
        (row["profile_label"], row["family"]): row
        for row in payload["family_rows"]
    }
    assert family_rows[("component_label_guard_v11", "h1y_route_stale_field")]["exact_success_count"] == 0
    assert family_rows[("component_residual_guard_v12", "h1y_route_stale_field")]["exact_success_count"] == 2
    assert family_rows[("component_residual_guard_v12", "h1y_preserve_surface_value")]["exact_success_count"] == 1
    assert family_rows[("routed_residual_guard_v16", "h1y_preserve_surface_value")]["exact_success_count"] == 0

    comparisons = {row["comparison_dir"]: row for row in payload["comparison_rows"]}
    v16_vs_v11 = next(row for key, row in comparisons.items() if "routed_residual_guard_vs_component_label" in key)
    v16_vs_v12 = next(row for key, row in comparisons.items() if "routed_residual_guard_vs_component_residual" in key)
    assert v16_vs_v11["delta_exact_rate"] == 0.0
    assert v16_vs_v12["delta_exact_rate"] == -0.19999999999999996

    failures = {
        (row["profile_label"], row["case_id"]): row
        for row in payload["non_exact_rows"]
    }
    assert failures[
        ("routed_residual_guard_v16", "h1y_responsible_party_field_old_owner_memo_decoy")
    ]["actual_tool"] == "refine_selection"
    assert "Pending" in failures[("routed_residual_guard_v16", "h1y_status_pill_summary_value_holdout")][
        "actual_arguments"
    ]

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "0/10" in findings["h1y_is_harder_than_no_directive"]
    assert "7/10" in findings["v12_best_local_but_still_noisy"]
    assert "ties v11" in findings["v16_route_text_is_not_enough"]
    assert "Do not promote v16" in findings["next_slice"]

    assert (tmp_path / "tables" / "h1y_routed_residual_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h1y_routed_residual_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h1y_routed_residual_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h1y_routed_residual_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h1y_routed_residual_findings.csv").exists()
