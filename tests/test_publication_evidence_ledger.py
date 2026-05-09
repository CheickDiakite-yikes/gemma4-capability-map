from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_publication_evidence_ledger.py"
SPEC = importlib.util.spec_from_file_location("build_publication_evidence_ledger_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_publication_evidence_ledger_writes_claims_and_sources(tmp_path: Path) -> None:
    payload = SCRIPT.build_ledger(output_dir=tmp_path)

    assert payload["manifest"]["claim_count"] >= 6
    assert payload["manifest"]["missing_source_count"] == 0

    claims = {row["claim_id"]: row for row in payload["claims"]}
    assert claims["C2_final_tool_directive_causal_for_protocol"]["status"] == "supported_current_packets"
    assert claims["C6_split_selector_wording_is_negative_evidence"]["status"] == "negative_result_current_packets"
    assert claims["C8_visual_hard_slice_targets_remaining_uncertainty"]["status"] == "supported_current_packets"
    assert claims["C9_schema_literal_targets_v5_is_negative_evidence"]["status"] == "negative_result_current_packets"
    assert claims["C10_v4_exact_misses_are_executor_success_aliases"]["status"] == "supported_current_packets"
    assert claims["C11_h1l_packaged_visual_workflows_remain_saturated"]["status"] == "negative_result_current_packets"
    assert claims["C12_replay_shaped_live_preserves_visual_hard_slice_signal"]["status"] == "supported_current_packets"
    assert "7/8" in claims["C2_final_tool_directive_causal_for_protocol"]["primary_metric"]
    assert "v3 raw exact falls" in claims["C6_split_selector_wording_is_negative_evidence"]["primary_metric"]
    assert "schema-field hints reach 6/8 strict and 8/8 executor-equivalent" in claims[
        "C8_visual_hard_slice_targets_remaining_uncertainty"
    ]["primary_metric"]
    assert "v5 reaches 5/8 strict and 7/8 executor-equivalent" in claims[
        "C9_schema_literal_targets_v5_is_negative_evidence"
    ]["primary_metric"]
    assert "true harness failure count is 0" in claims["C10_v4_exact_misses_are_executor_success_aliases"]["primary_metric"]
    assert "H1l candidate rows tie" in claims[
        "C11_h1l_packaged_visual_workflows_remain_saturated"
    ]["primary_metric"]
    assert "schema-field hints is 1/2 strict and 2/2 executor-equivalent" in claims[
        "C12_replay_shaped_live_preserves_visual_hard_slice_signal"
    ]["primary_metric"]

    source_types = {row["artifact_type"] for row in payload["evidence_sources"]}
    assert "h1_ablation_packet" in source_types
    assert "visual_hard_slice_probe_packet" in source_types
    assert "visual_hard_slice_profile_comparison" in source_types
    assert "visual_hard_slice_exactness_diagnostic" in source_types
    assert "design_packet" in source_types
    assert "live_replay_decision" in source_types
    assert all(row["exists"] for row in payload["evidence_sources"])

    assert (tmp_path / "ledger.md").exists()
    assert (tmp_path / "ledger.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "tables" / "claim_ledger.csv").exists()
    assert (tmp_path / "tables" / "evidence_sources.csv").exists()
