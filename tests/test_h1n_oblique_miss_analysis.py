from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_h1n_oblique_misses.py"
SPEC = importlib.util.spec_from_file_location("analyze_h1n_oblique_misses_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_h1n_oblique_miss_analysis_classifies_argument_hints_misses(tmp_path: Path) -> None:
    payload = SCRIPT.analyze_h1n_oblique_misses(output_dir=tmp_path)

    assert payload["manifest"]["row_count"] == 12
    assert payload["manifest"]["miss_count"] == 5
    misses = {(row["label"], row["case_id"]): row for row in payload["miss_rows"]}
    assert misses[
        ("argument_hints_v2", "transfer_oblique_cell_r42_notice_decoy")
    ]["classification"] == "code_suffix_truncation"
    assert misses[
        ("argument_hints_v2", "transfer_oblique_alert_p55_toggle_decoy")
    ]["classification"] == "negated_or_semantic_decoy_selected"
    assert misses[
        ("schema_field_hints_v4", "transfer_oblique_node_q17_table_decoy")
    ]["classification"] == "semantic_broad_selection"

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "2 misses" in findings["argument_hints_miss_count"]
    assert "3 misses" in findings["schema_field_miss_count"]
    assert "short code suffixes" in findings["next_intervention_target"]

    assert (tmp_path / "diagnostic.md").exists()
    assert (tmp_path / "diagnostic.json").exists()
    assert (tmp_path / "tables" / "h1n_oblique_misses.csv").exists()
    assert (tmp_path / "tables" / "h1n_oblique_miss_findings.csv").exists()
