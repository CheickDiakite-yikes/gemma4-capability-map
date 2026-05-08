from __future__ import annotations

import json
from pathlib import Path

from gemma4_capability_map.runtime.tool_directive_probe import (
    _score_probe_case,
    build_tool_directive_probe_cases,
    compare_tool_directive_probe_packets,
    run_tool_directive_probe,
    write_tool_directive_probe_comparison,
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


def test_run_tool_directive_probe_records_prompt_contract_controls(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
systems:
  heuristic_prompt_contract:
    backend: heuristic
    reasoner: google/gemma-4-E2B-it
    reasoner_max_new_tokens: 64
    request_timeout_seconds: 30.0
    research_controls:
      disable_tool_turn_directive: true
      tool_prompt_contract_id: schema_anchor_v1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "probe"
    result = run_tool_directive_probe(
        system_id="heuristic_prompt_contract",
        output_dir=output_dir,
        registry_path=registry_path,
        cases=build_tool_directive_probe_cases()[:1],
    )

    assert result["manifest"]["research_controls"] == {
        "disable_tool_turn_directive": True,
        "tool_prompt_contract_id": "schema_anchor_v1",
    }
    assert result["manifest"]["runtime_info"]["tool_turn_directive_enabled"] is False
    assert result["manifest"]["runtime_info"]["tool_prompt_contract_id"] == "schema_anchor_v1"


def test_run_tool_directive_probe_records_tool_catalog_profile_controls(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        """
systems:
  heuristic_tool_catalog_profile:
    backend: heuristic
    reasoner: google/gemma-4-E2B-it
    reasoner_max_new_tokens: 64
    request_timeout_seconds: 30.0
    research_controls:
      disable_tool_turn_directive: true
      tool_catalog_profile_id: visual_role_catalog_v1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "probe"
    result = run_tool_directive_probe(
        system_id="heuristic_tool_catalog_profile",
        output_dir=output_dir,
        registry_path=registry_path,
        cases=build_tool_directive_probe_cases()[4:5],
    )

    assert result["manifest"]["research_controls"] == {
        "disable_tool_turn_directive": True,
        "tool_catalog_profile_id": "visual_role_catalog_v1",
    }
    assert result["manifest"]["runtime_info"]["tool_turn_directive_enabled"] is False
    assert result["manifest"]["runtime_info"]["tool_catalog_profile_id"] == "visual_role_catalog_v1"


def test_tool_directive_probe_comparison_reports_case_and_family_deltas(tmp_path: Path) -> None:
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
    cases = build_tool_directive_probe_cases()[:2]
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    run_tool_directive_probe(system_id="heuristic_probe", output_dir=baseline_dir, registry_path=registry_path, cases=cases)
    run_tool_directive_probe(system_id="heuristic_probe", output_dir=candidate_dir, registry_path=registry_path, cases=cases)

    candidate_rows = json.loads((candidate_dir / "probe_results.json").read_text(encoding="utf-8"))
    candidate_rows[0]["exact_match"] = False
    candidate_rows[0]["actual_calls"][0]["arguments"]["query"] = "invoice-lock failure"
    (candidate_dir / "probe_results.json").write_text(json.dumps(candidate_rows, indent=2) + "\n", encoding="utf-8")
    candidate_manifest = json.loads((candidate_dir / "manifest.json").read_text(encoding="utf-8"))
    candidate_manifest["summary"]["exact_match_count"] = 1
    candidate_manifest["summary"]["exact_match_rate"] = 0.5
    (candidate_dir / "manifest.json").write_text(json.dumps(candidate_manifest, indent=2) + "\n", encoding="utf-8")

    comparison = compare_tool_directive_probe_packets(baseline_dir, candidate_dir)
    outputs = write_tool_directive_probe_comparison(baseline_dir, candidate_dir)

    assert comparison["shared_case_count"] == 2
    assert comparison["delta_exact_match_rate"] == -0.5
    first = next(row for row in comparison["case_deltas"] if row["case_id"] == cases[0].case_id)
    assert first["delta_exact_match"] == -1
    assert first["baseline_failure_mode"] == "exact"
    assert first["candidate_failure_mode"] == "argument_mismatch"
    family = next(row for row in comparison["family_deltas"] if row["family"] == cases[0].family)
    assert family["delta_exact_rate"] == -1.0
    assert Path(outputs["summary"]).exists()
    assert Path(outputs["case_deltas"]).exists()


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
