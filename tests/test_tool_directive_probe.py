from __future__ import annotations

import json
from pathlib import Path

from gemma4_capability_map.runtime.tool_directive_probe import (
    _score_probe_case,
    build_tool_directive_probe_cases,
    run_tool_directive_probe,
)
from gemma4_capability_map.schemas import ModelTurn, ToolCall
from gemma4_capability_map.tools.planner import plan_tool_calls
from gemma4_capability_map.tools.registry import build_default_registry


def test_tool_directive_probe_cases_cover_cli_api_visual_and_parallel_families() -> None:
    cases = build_tool_directive_probe_cases()
    families = {case.family for case in cases}

    assert "cli_canonicalization" in families
    assert "api_canonicalization" in families
    assert "visual_argument_copying" in families
    assert "parallel_tool_calling" in families

    specs = build_default_registry().specs
    form_case = next(case for case in cases if case.case_id == "visual_form_target_literal")
    planned = plan_tool_calls(
        form_case.messages,
        form_case.media,
        [specs[name] for name in form_case.tool_names],
    )
    assert planned[0].name == "extract_layout"
    assert planned[0].arguments == {"image_id": "img-form-live-latest", "target_query": "validation error"}


def test_run_tool_directive_probe_with_heuristic_system_writes_outputs(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
systems:
  heuristic_probe:
    backend: heuristic
    reasoner: google/gemma-4-E2B-it
    reasoner_max_new_tokens: 64
    request_timeout_seconds: 30.0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "probe"
    result = run_tool_directive_probe(
        system_id="heuristic_probe",
        output_dir=output_dir,
        registry_path=registry_path,
        cases=build_tool_directive_probe_cases()[:2],
    )

    assert result["summary"]["case_count"] == 2
    assert result["summary"]["exact_match_rate"] == 1.0
    assert (output_dir / "manifest.json").exists()
    rows = json.loads((output_dir / "probe_results.json").read_text(encoding="utf-8"))
    assert rows[0]["exact_match"] is True
    assert (output_dir / "probe_results.csv").exists()


def test_tool_directive_probe_scores_visual_paraphrase_execution() -> None:
    specs = build_default_registry().specs
    case = next(case for case in build_tool_directive_probe_cases() if case.case_id == "visual_form_target_literal")
    tool_specs = [specs[name] for name in case.tool_names]
    expected_calls = plan_tool_calls(case.messages, case.media, tool_specs)
    turn = ModelTurn(
        raw_model_output='{"name":"extract_layout","arguments":{"image_id":"img-form-live-latest","target_query":"phone issue"}}',
        normalized_tool_call=[
            ToolCall(
                name="extract_layout",
                arguments={"image_id": "img-form-live-latest", "target_query": "phone issue"},
                source_format="json",
                raw='{"name":"extract_layout","arguments":{"image_id":"img-form-live-latest","target_query":"phone issue"}}',
            )
        ],
    )

    row = _score_probe_case(case, tool_specs, expected_calls, turn)

    assert row["exact_match"] is False
    assert row["executable_match"] is True
    assert row["actual_execution"][0]["output"]["region_ids"] == ["form-err-202"]
