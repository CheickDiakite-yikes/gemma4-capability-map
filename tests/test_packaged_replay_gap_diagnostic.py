from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_packaged_replay_gap.py"
SPEC = importlib.util.spec_from_file_location("analyze_packaged_replay_gap_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_packaged_replay_gap_diagnostic_writes_saturation_findings(tmp_path: Path) -> None:
    payload = SCRIPT.analyze_packaged_replay_gap(output_dir=tmp_path)

    assert payload["manifest"]["surface_count"] == 2
    assert payload["manifest"]["saturated_packaged_surface_count"] == 2
    surfaces = {row["surface_id"]: row for row in payload["surface_rows"]}
    assert surfaces["h1l_visual_executor_equivalence"]["replay_max_delta_executor_equivalence_rate"] == 1.0
    assert surfaces["h1l_visual_executor_equivalence"]["packaged_readiness_span"] == 0.0
    assert surfaces["h1m_visual_alias_repeat"]["replay_max_delta_executor_equivalence_rate"] == 0.375
    assert surfaces["h1m_visual_alias_repeat"]["packaged_strict_interface_span"] == 0.0
    assert (
        surfaces["h1m_visual_alias_repeat"]["classification"]
        == "positive_replay_saturated_packaged_surface"
    )
    recommendations = {row["recommendation_id"]: row for row in payload["recommendations"]}
    assert recommendations[
        "do_not_spend_helper_budget_on_saturated_packaged_visual_surfaces"
    ]["status"] == "active"
    assert (tmp_path / "diagnostic.md").exists()
    assert (tmp_path / "diagnostic.json").exists()
    assert (tmp_path / "tables" / "packaged_replay_gap_surfaces.csv").exists()
    assert (tmp_path / "tables" / "packaged_replay_gap_recommendations.csv").exists()
