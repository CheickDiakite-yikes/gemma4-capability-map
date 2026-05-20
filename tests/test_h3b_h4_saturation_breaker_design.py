from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_h3b_h4_saturation_breaker_design.py"
SPEC = importlib.util.spec_from_file_location("build_h3b_h4_saturation_breaker_design", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h3b_h4_saturation_breaker_design_writes_publishable_contract(tmp_path: Path) -> None:
    payload = SCRIPT.build_h3b_h4_saturation_breaker_design(output_dir=tmp_path)

    manifest = payload["manifest"]
    assert manifest["phase"] == "h3b_h4_saturation_breaker_design"
    assert manifest["planned_family_count"] == 6
    assert manifest["planned_case_count"] == 24
    assert manifest["current_candidate"] == "mlx_gemma4_e2b_reasoner_only_h3a_boundary_combined"
    assert manifest["first_execution_packet"] == "h3b_saturation_breaker_v27"
    assert "controller-dependence" in manifest["publication_standard"]

    family_ids = {row["family_id"] for row in payload["family_rows"]}
    assert "h3b_unseen_stale_origin_paraphrase" in family_ids
    assert "h3b_extended_negative_value_vocabulary" in family_ids
    assert "h4_approval_stop_boundary" in family_ids

    metric_ids = {row["metric_id"] for row in payload["score_rows"]}
    assert {
        "strict_exact",
        "executor_equivalence",
        "controller_trace",
        "regression_count",
        "helper_overtrigger",
        "live_operator_artifact",
    }.issubset(metric_ids)

    baseline_ids = {row["system_id"] for row in payload["baseline_rows"]}
    assert "mlx_gemma4_e2b_reasoner_only_h3a_boundary_combined" in baseline_ids
    assert "gemini_cli_external_baseline" in baseline_ids

    external_benchmarks = {row["external_benchmark"] for row in payload["external_alignment_rows"]}
    assert "Terminal-bench style" in external_benchmarks
    assert "Toolathlon style" in external_benchmarks
    assert "OSWorld-Verified style" in external_benchmarks

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "design.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "tables" / "h3b_h4_family_plan.csv").exists()
    assert (tmp_path / "tables" / "h3b_h4_score_contract.csv").exists()
    assert (tmp_path / "tables" / "h3b_h4_baseline_plan.csv").exists()
    assert (tmp_path / "tables" / "h3b_h4_external_benchmark_alignment.csv").exists()
    assert (tmp_path / "figures" / "h3b_h4_benchmark_pressure_map.svg").exists()
