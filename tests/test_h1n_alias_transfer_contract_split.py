from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_h1n_alias_transfer_contract_split.py"
SPEC = importlib.util.spec_from_file_location("analyze_h1n_alias_transfer_contract_split_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h1n_alias_transfer_contract_split_diagnoses_planner_contract_gap(tmp_path: Path) -> None:
    payload = SCRIPT.analyze_h1n_alias_transfer_contract_split(output_dir=tmp_path)

    assert payload["manifest"]["case_count"] == 6
    assert payload["manifest"]["run_count"] == 6
    assert payload["manifest"]["expected_call_contract_mismatch_count"] == 5
    assert payload["manifest"]["contracted_exact_non_executor_count"] == 4
    assert payload["manifest"]["argument_hints_executor_success_count"] == 6

    expected = {row["case_id"]: row for row in payload["expected_call_rows"]}
    assert expected["transfer_form_error_old_selection_chip_decoy"][
        "expected_call_classification"
    ] == "expected_call_reaches_executor_target"
    assert expected["transfer_review_tile_notice_table_decoy"][
        "expected_call_classification"
    ] == "expected_call_returns_empty_region_selection"
    assert expected["transfer_queue_badge_person_decoy"][
        "expected_call_classification"
    ] == "expected_call_invalid_empty_region_id"

    replay = {(row["label"], row["case_id"]): row for row in payload["replay_rows"]}
    assert replay[
        ("contracted", "transfer_review_tile_notice_table_decoy")
    ]["replay_classification"] == "exact_against_nonoracle_expected_call"
    assert replay[
        ("argument_hints_v2", "transfer_review_tile_notice_table_decoy")
    ]["replay_classification"] == "nonexact_executor_target_success"

    findings = {row["finding_id"]: row for row in payload["finding_rows"]}
    assert "5 / 6" in findings["expected_calls_are_not_oracle_calls"]["finding"]
    assert "4 exact rows" in findings["contracted_exactness_is_overstated_for_h1n"]["finding"]

    assert (tmp_path / "diagnostic.md").exists()
    assert (tmp_path / "diagnostic.json").exists()
    assert (tmp_path / "tables" / "h1n_expected_call_contract_audit.csv").exists()
    assert (tmp_path / "tables" / "h1n_replay_contract_split.csv").exists()
