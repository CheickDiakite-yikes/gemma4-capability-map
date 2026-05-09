from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_visual_hard_slice_design.py"
SPEC = importlib.util.spec_from_file_location("build_visual_hard_slice_design_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_visual_hard_slice_design_writes_fresh_case_designs(tmp_path: Path) -> None:
    payload = SCRIPT.build_design(output_dir=tmp_path)

    assert payload["manifest"]["case_count"] == 8
    case_ids = {row["case_id"] for row in payload["case_designs"]}
    assert "visual_form_error_with_prior_selection_decoy" in case_ids
    assert "visual_selection_id_opaque_copy_with_filter" in case_ids

    discriminators = {row["primary_discriminator"] for row in payload["case_designs"]}
    assert "extract_layout_vs_refine_selection_when_no_real_selection_id" in discriminators
    assert "read_region_text_json_shape" in discriminators

    expected_tools = {row["expected_tool"] for row in payload["case_designs"]}
    assert {"extract_layout", "refine_selection", "read_region_text"}.issubset(expected_tools)

    assert (tmp_path / "design.md").exists()
    assert (tmp_path / "design.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_case_designs.csv").exists()
