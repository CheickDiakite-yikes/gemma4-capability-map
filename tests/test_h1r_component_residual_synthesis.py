from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_h1r_component_residual_synthesis.py"
SPEC = importlib.util.spec_from_file_location("build_h1r_component_residual_synthesis_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_build_h1r_component_residual_synthesis_writes_report(tmp_path: Path) -> None:
    payload = SCRIPT.build_h1r_component_residual_synthesis(output_dir=tmp_path)

    assert payload["manifest"]["profile_count"] == 3
    assert payload["manifest"]["case_count"] == 6
    assert payload["manifest"]["comparison_count"] == 3
    assert payload["manifest"]["v12_exact_success_count"] == 6
    assert payload["manifest"]["v12_executor_success_count"] == 6

    packet_rows = {row["profile_label"]: row for row in payload["packet_rows"]}
    assert packet_rows["no_directive"]["exact_success_count"] == 0
    assert packet_rows["no_directive"]["executor_success_count"] == 1
    assert packet_rows["component_label_guard_v11"]["exact_success_count"] == 5
    assert packet_rows["component_residual_guard_v12"]["exact_success_count"] == 6

    family_rows = {
        (row["profile_label"], row["family"]): row for row in payload["family_rows"]
    }
    assert family_rows[
        ("component_residual_guard_v12", "h1r_code_label_exactness")
    ]["exact_success_count"] == 2
    assert family_rows[
        ("component_label_guard_v11", "h1r_code_label_exactness")
    ]["exact_success_count"] == 1

    comparison_rows = {row["candidate_system_id"]: row for row in payload["comparison_rows"]}
    assert comparison_rows[
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard"
    ]["candidate_exact_rate"] == 1.0

    finding_ids = {row["finding_id"] for row in payload["finding_rows"]}
    assert "h1r_breaks_no_directive" in finding_ids
    assert "v12_saturates_h1r" in finding_ids

    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "tables" / "h1r_component_residual_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h1r_component_residual_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h1r_component_residual_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h1r_component_residual_findings.csv").exists()
