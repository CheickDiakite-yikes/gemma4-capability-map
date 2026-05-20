from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_h3b_saturation_breaker_synthesis.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("build_h3b_saturation_breaker_synthesis", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h3b_saturation_breaker_synthesis_writes_family_attributed_result(tmp_path: Path) -> None:
    payload = SCRIPT.build_h3b_saturation_breaker_synthesis(output_dir=tmp_path)

    manifest = payload["manifest"]
    assert manifest["phase"] == "h3b_saturation_breaker_synthesis"
    assert manifest["packet_row_count"] == 3
    assert manifest["comparison_count"] == 3
    assert manifest["h3b_case_count"] == 24
    assert manifest["h2w_exact_success_count"] == 11
    assert manifest["h2z_exact_success_count"] == 11
    assert manifest["h3a_exact_success_count"] == 11
    assert manifest["h3a_executor_success_count"] == 14
    assert manifest["h2z_delta_exact_vs_h2w"] == 0.0
    assert manifest["h3a_delta_exact_vs_h2z"] == 0.0
    assert manifest["h3a_delta_executor_vs_h2w"] == 0.0
    assert manifest["h3a_unexpected_tool_call_count"] == 4
    assert manifest["current_ladder_zero_delta"] is True
    assert manifest["h3b_breaks_current_ladder"] is True

    h3a_families = {
        row["family"]: row for row in payload["family_rows"] if row["profile_label"] == "h3b_h3a_boundary_combined"
    }
    assert h3a_families["h3b_unseen_stale_origin_paraphrase"]["exact_success_count"] == 4
    assert h3a_families["h3b_extended_negative_value_vocabulary"]["exact_success_count"] == 0
    assert h3a_families["h4_approval_stop_boundary"]["executor_success_count"] == 0

    failure_modes = {
        row["failure_mode"]: row["case_count"]
        for row in payload["failure_taxonomy_rows"]
        if row["profile_label"] == "h3b_h3a_boundary_combined"
    }
    assert failure_modes["exact"] == 11
    assert failure_modes["unexpected_tool_call"] == 4
    assert failure_modes["wrong_tool"] == 2

    finding_ids = {row["finding_id"] for row in payload["finding_rows"]}
    assert "h3b_breaks_h3a_saturation" in finding_ids
    assert "approval_stop_is_a_true_live_operator_boundary" in finding_ids

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h3b_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h3b_failure_taxonomy.csv").exists()
    assert (tmp_path / "tables" / "h3b_case_matrix.csv").exists()
    assert (tmp_path / "figures" / "h3b_saturation_breaker_family_pressure.svg").exists()
