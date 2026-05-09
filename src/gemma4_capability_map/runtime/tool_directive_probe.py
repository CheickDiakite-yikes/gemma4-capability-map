from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_capability_map.models.gemma4_runner import Gemma4Runner
from gemma4_capability_map.reporting.knowledge_work_board import DEFAULT_REGISTRY_PATH, load_model_registry
from gemma4_capability_map.research_controls import ResearchControls
from gemma4_capability_map.schemas import Message, ModelTurn, ToolCall, ToolSpec
from gemma4_capability_map.tools.executor import DeterministicExecutor
from gemma4_capability_map.tools.planner import plan_tool_calls
from gemma4_capability_map.tools.registry import build_default_registry


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROBE_RESULTS_ROOT = ROOT / "results" / "tool_directive_probe"


@dataclass(frozen=True)
class ToolDirectiveProbeCase:
    case_id: str
    family: str
    messages: list[Message]
    media: list[str]
    tool_names: list[str]
    initial_state: dict[str, Any] = field(default_factory=dict)
    expected_execution: dict[str, Any] = field(default_factory=dict)


def build_tool_directive_probe_cases() -> list[ToolDirectiveProbeCase]:
    return [
        ToolDirectiveProbeCase(
            case_id="cli_invoice_lock_hyphen_query",
            family="cli_canonicalization",
            messages=[
                Message(
                    role="user",
                    content="Ignore the earlier publish plan. Search logs/billing.log for the latest invoice-lock failure and report it.",
                )
            ],
            media=[],
            tool_names=["cli_search_logs", "api_fetch_record"],
        ),
        ToolDirectiveProbeCase(
            case_id="cli_phone_patch_latest_only",
            family="cli_patch_copying",
            messages=[
                Message(
                    role="user",
                    content="Ignore the earlier work-authorization edit. The newest recruiter instruction is to patch only the phone validation config in config/job_form.yaml.",
                )
            ],
            media=[],
            tool_names=["cli_apply_patch", "read_repo_file"],
        ),
        ToolDirectiveProbeCase(
            case_id="api_invoice_lock_hold_update",
            family="api_canonicalization",
            messages=[
                Message(
                    role="user",
                    content="Do not reopen publication. The latest finance direction is to update billing record INV-204 so invoice_lock stays on hold.",
                )
            ],
            media=[],
            tool_names=["api_update_record", "api_fetch_record"],
        ),
        ToolDirectiveProbeCase(
            case_id="api_form_issue_fetch",
            family="api_canonicalization",
            messages=[
                Message(
                    role="user",
                    content="The first note mentioned work authorization, but the latest recruiter note says focus on the phone field. Fetch issue FORM-88 and identify the latest blocker.",
                )
            ],
            media=[],
            tool_names=["api_fetch_record", "api_update_record"],
        ),
        ToolDirectiveProbeCase(
            case_id="visual_form_target_literal",
            family="visual_argument_copying",
            messages=[
                Message(role="system", content="visual_image_ids: img-form-live-latest"),
                Message(
                    role="user",
                    content="Using the local visual executor path, respect the latest recruiter note, isolate the phone issue first, and read back that message.",
                ),
            ],
            media=["img-form-live-latest"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_form_live_latest_state(),
            expected_execution={"region_ids": ["form-err-202"]},
        ),
        ToolDirectiveProbeCase(
            case_id="visual_latest_filter_literal",
            family="visual_referent_carryover",
            messages=[
                Message(role="system", content="visual_image_ids: img-form-latest"),
                Message(
                    role="user",
                    content="Inspect the form errors, keep only the latest issue first, then narrow to the phone issue and read back the remaining message.",
                ),
                Message(
                    role="tool",
                    content='{"tool_name":"extract_layout","status":"pass","arguments":{"image_id":"img-form-latest","target_query":"validation error"},"output":{"selection_id":"sel-001","image_id":"img-form-latest","selection_kind":"regions","count":2,"region_ids":["form-err-201","form-err-202"],"region_id":"form-err-201"}}',
                ),
            ],
            media=["img-form-latest"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
        ),
        ToolDirectiveProbeCase(
            case_id="visual_readback_region_literal",
            family="visual_referent_carryover",
            messages=[
                Message(role="system", content="visual_image_ids: img-form-latest"),
                Message(
                    role="user",
                    content="Inspect the form errors, keep only the latest issue first, then narrow to the phone issue and read back the remaining message.",
                ),
                Message(
                    role="tool",
                    content='{"tool_name":"extract_layout","status":"pass","arguments":{"image_id":"img-form-latest","target_query":"validation error"},"output":{"selection_id":"sel-001","image_id":"img-form-latest","selection_kind":"regions","count":2,"region_ids":["form-err-201","form-err-202"],"region_id":"form-err-201"}}',
                ),
                Message(
                    role="tool",
                    content='{"tool_name":"refine_selection","status":"pass","arguments":{"selection_id":"sel-001","filter_query":"latest"},"output":{"selection_id":"sel-002","image_id":"img-form-latest","selection_kind":"regions","count":1,"region_ids":["form-err-202"]}}',
                ),
                Message(
                    role="tool",
                    content='{"tool_name":"refine_selection","status":"pass","arguments":{"selection_id":"sel-002","filter_query":"phone"},"output":{"selection_id":"sel-003","image_id":"img-form-latest","selection_kind":"regions","count":1,"region_ids":["form-err-202"]}}',
                ),
            ],
            media=["img-form-latest"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
        ),
        ToolDirectiveProbeCase(
            case_id="parallel_audit_array_literal",
            family="parallel_tool_calling",
            messages=[Message(role="user", content="Check both the screenshot and config/settings.yaml before you answer.")],
            media=["img-parallel"],
            tool_names=["inspect_image", "read_repo_file", "propose_patch"],
        ),
    ]


def run_tool_directive_probe(
    *,
    system_id: str = "mlx_gemma4_e2b_reasoner_only",
    output_dir: str | Path | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    cases: list[ToolDirectiveProbeCase] | None = None,
) -> dict[str, Any]:
    registry = load_model_registry(registry_path)
    systems = registry.get("systems", {})
    meta = systems.get(system_id)
    if meta is None:
        raise ValueError(f"Unknown system profile `{system_id}`.")
    if str(meta.get("backend", "")) not in {"heuristic", "hf", "hf_service", "mlx", "llama_cpp"}:
        raise ValueError(f"Unsupported probe backend for `{system_id}`: {meta.get('backend')}")

    controls = ResearchControls.from_mapping(meta.get("research_controls"))
    runner = Gemma4Runner(
        model_id=str(meta.get("reasoner") or "google/gemma-4-E2B-it"),
        backend=str(meta.get("backend") or "heuristic"),
        max_new_tokens=int(meta.get("reasoner_max_new_tokens", 64) or 64),
        request_timeout_seconds=float(meta.get("request_timeout_seconds", 300.0) or 300.0),
        tool_turn_directive_enabled=not controls.disable_tool_turn_directive,
        tool_prompt_contract_id=controls.tool_prompt_contract_id,
        tool_catalog_profile_id=controls.tool_catalog_profile_id,
    )
    tool_specs_by_name = build_default_registry().specs
    selected_cases = cases or build_tool_directive_probe_cases()
    created_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = Path(output_dir) if output_dir else DEFAULT_PROBE_RESULTS_ROOT / f"{created_at}_{system_id}"
    target.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in selected_cases:
        tool_specs = [tool_specs_by_name[name] for name in case.tool_names]
        expected_calls = plan_tool_calls(case.messages, case.media, tool_specs)
        turn = runner.generate(
            messages=case.messages,
            media=case.media,
            tool_specs=tool_specs,
            thinking=False,
            max_new_tokens=int(meta.get("reasoner_max_new_tokens", 64) or 64),
        )
        rows.append(_score_probe_case(case, tool_specs, expected_calls, turn))

    summary = _summarize_probe(rows)
    manifest = {
        "created_at": created_at,
        "system_id": system_id,
        "backend": str(meta.get("backend") or ""),
        "reasoner": str(meta.get("reasoner") or ""),
        "case_count": len(rows),
        "runtime_info": runner.runtime_info(),
        "research_controls": controls.manifest_payload(),
        "summary": summary,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (target / "probe_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_probe_csv(target / "probe_results.csv", rows)
    return {
        "output_dir": str(target.resolve()),
        "manifest": manifest,
        "summary": summary,
        "rows": rows,
    }


def compare_tool_directive_probe_packets(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
) -> dict[str, Any]:
    baseline_root = Path(baseline_dir)
    candidate_root = Path(candidate_dir)
    baseline_manifest = json.loads((baseline_root / "manifest.json").read_text(encoding="utf-8"))
    candidate_manifest = json.loads((candidate_root / "manifest.json").read_text(encoding="utf-8"))
    baseline_rows = {str(row["case_id"]): row for row in json.loads((baseline_root / "probe_results.json").read_text(encoding="utf-8"))}
    candidate_rows = {str(row["case_id"]): row for row in json.loads((candidate_root / "probe_results.json").read_text(encoding="utf-8"))}
    shared_case_ids = sorted(set(baseline_rows) & set(candidate_rows))

    case_deltas = []
    family_buckets: dict[str, dict[str, Any]] = {}
    for case_id in shared_case_ids:
        baseline = baseline_rows[case_id]
        candidate = candidate_rows[case_id]
        family = str(candidate.get("family") or baseline.get("family") or "")
        row = {
            "case_id": case_id,
            "family": family,
            "baseline_exact_match": bool(baseline.get("exact_match")),
            "candidate_exact_match": bool(candidate.get("exact_match")),
            "delta_exact_match": _bool_delta(candidate.get("exact_match"), baseline.get("exact_match")),
            "baseline_failure_mode": _probe_failure_mode(baseline),
            "candidate_failure_mode": _probe_failure_mode(candidate),
            "baseline_executable_match": _optional_bool(baseline.get("executable_match")),
            "candidate_executable_match": _optional_bool(candidate.get("executable_match")),
            "delta_executable_match": _optional_bool_delta(candidate.get("executable_match"), baseline.get("executable_match")),
            "baseline_executor_target_match": _executor_target_value(baseline),
            "candidate_executor_target_match": _executor_target_value(candidate),
            "delta_executor_target_match": _optional_bool_delta(_executor_target_value(candidate), _executor_target_value(baseline)),
            "baseline_actual_call_count": int(baseline.get("actual_call_count") or 0),
            "candidate_actual_call_count": int(candidate.get("actual_call_count") or 0),
            "delta_actual_call_count": int(candidate.get("actual_call_count") or 0) - int(baseline.get("actual_call_count") or 0),
        }
        case_deltas.append(row)
        bucket = family_buckets.setdefault(
            family,
            {
                "family": family,
                "case_count": 0,
                "baseline_exact_count": 0,
                "candidate_exact_count": 0,
                "baseline_executable_count": 0,
                "candidate_executable_count": 0,
                "shared_executable_case_count": 0,
                "baseline_executor_target_count": 0,
                "candidate_executor_target_count": 0,
                "shared_executor_target_case_count": 0,
            },
        )
        bucket["case_count"] += 1
        bucket["baseline_exact_count"] += int(bool(baseline.get("exact_match")))
        bucket["candidate_exact_count"] += int(bool(candidate.get("exact_match")))
        if baseline.get("executable_match") is not None and candidate.get("executable_match") is not None:
            bucket["shared_executable_case_count"] += 1
            bucket["baseline_executable_count"] += int(bool(baseline.get("executable_match")))
            bucket["candidate_executable_count"] += int(bool(candidate.get("executable_match")))
        baseline_target = _executor_target_value(baseline)
        candidate_target = _executor_target_value(candidate)
        if baseline_target is not None and candidate_target is not None:
            bucket["shared_executor_target_case_count"] += 1
            bucket["baseline_executor_target_count"] += int(bool(baseline_target))
            bucket["candidate_executor_target_count"] += int(bool(candidate_target))

    family_deltas = []
    for bucket in sorted(family_buckets.values(), key=lambda item: str(item["family"])):
        case_count = int(bucket["case_count"])
        executable_count = int(bucket["shared_executable_case_count"])
        executor_target_count = int(bucket["shared_executor_target_case_count"])
        family_deltas.append(
            {
                **bucket,
                "baseline_exact_rate": bucket["baseline_exact_count"] / case_count if case_count else 0.0,
                "candidate_exact_rate": bucket["candidate_exact_count"] / case_count if case_count else 0.0,
                "delta_exact_rate": (bucket["candidate_exact_count"] - bucket["baseline_exact_count"]) / case_count if case_count else 0.0,
                "baseline_executable_rate": bucket["baseline_executable_count"] / executable_count if executable_count else None,
                "candidate_executable_rate": bucket["candidate_executable_count"] / executable_count if executable_count else None,
                "delta_executable_rate": (
                    (bucket["candidate_executable_count"] - bucket["baseline_executable_count"]) / executable_count if executable_count else None
                ),
                "baseline_executor_target_rate": (
                    bucket["baseline_executor_target_count"] / executor_target_count if executor_target_count else None
                ),
                "candidate_executor_target_rate": (
                    bucket["candidate_executor_target_count"] / executor_target_count if executor_target_count else None
                ),
                "delta_executor_target_rate": (
                    (
                        bucket["candidate_executor_target_count"] - bucket["baseline_executor_target_count"]
                    )
                    / executor_target_count
                    if executor_target_count
                    else None
                ),
            }
        )

    baseline_summary = baseline_manifest.get("summary", {})
    candidate_summary = candidate_manifest.get("summary", {})
    return {
        "baseline_dir": str(baseline_root.resolve()),
        "candidate_dir": str(candidate_root.resolve()),
        "baseline_system_id": str(baseline_manifest.get("system_id", "")),
        "candidate_system_id": str(candidate_manifest.get("system_id", "")),
        "shared_case_count": len(shared_case_ids),
        "baseline_exact_match_rate": float(baseline_summary.get("exact_match_rate") or 0.0),
        "candidate_exact_match_rate": float(candidate_summary.get("exact_match_rate") or 0.0),
        "delta_exact_match_rate": float(candidate_summary.get("exact_match_rate") or 0.0)
        - float(baseline_summary.get("exact_match_rate") or 0.0),
        "baseline_executable_match_rate": baseline_summary.get("executable_match_rate"),
        "candidate_executable_match_rate": candidate_summary.get("executable_match_rate"),
        "baseline_executor_equivalence_match_rate": baseline_summary.get("executor_equivalence_match_rate"),
        "candidate_executor_equivalence_match_rate": candidate_summary.get("executor_equivalence_match_rate"),
        "delta_executor_equivalence_match_rate": _optional_rate_delta(
            candidate_summary.get("executor_equivalence_match_rate"),
            baseline_summary.get("executor_equivalence_match_rate"),
        ),
        "case_deltas": case_deltas,
        "family_deltas": family_deltas,
    }


def write_tool_directive_probe_comparison(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    comparison = compare_tool_directive_probe_packets(baseline_dir, candidate_dir)
    target = Path(output_dir) if output_dir else Path(candidate_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "probe_comparison.json"
    case_deltas_path = target / "probe_case_deltas.csv"
    family_deltas_path = target / "probe_family_deltas.csv"
    summary_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_dict_csv(case_deltas_path, comparison["case_deltas"])
    _write_dict_csv(family_deltas_path, comparison["family_deltas"])
    return {
        "summary": str(summary_path.resolve()),
        "case_deltas": str(case_deltas_path.resolve()),
        "family_deltas": str(family_deltas_path.resolve()),
    }


def _score_probe_case(
    case: ToolDirectiveProbeCase,
    tool_specs: list[ToolSpec],
    expected_calls: list[ToolCall],
    turn: ModelTurn,
) -> dict[str, Any]:
    exact_match = _calls_exact_match(turn.normalized_tool_call, expected_calls)
    executable_match, actual_execution = _score_executable_case(
        case=case,
        tool_specs=tool_specs,
        actual_calls=turn.normalized_tool_call,
    )
    executor_target_match = _score_executor_target_case(
        case=case,
        actual_execution=actual_execution,
    )
    expected_payload = [{"name": call.name, "arguments": call.arguments} for call in expected_calls]
    actual_payload = [{"name": call.name, "arguments": call.arguments} for call in turn.normalized_tool_call]
    return {
        "case_id": case.case_id,
        "family": case.family,
        "expected_call_count": len(expected_calls),
        "actual_call_count": len(turn.normalized_tool_call),
        "exact_match": exact_match,
        "executable_match": executable_match,
        "executor_target_match": executor_target_match,
        "expected_execution": case.expected_execution,
        "actual_execution": actual_execution,
        "expected_calls": expected_payload,
        "actual_calls": actual_payload,
        "raw_model_output": turn.raw_model_output,
        "prompt_tokens": turn.prompt_tokens,
        "completion_tokens": turn.completion_tokens,
        "latency_ms": turn.latency_ms,
    }


def _summarize_probe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    exact = sum(1 for row in rows if row["exact_match"])
    executable_rows = [row for row in rows if row.get("executable_match") is not None]
    executable = sum(1 for row in executable_rows if row["executable_match"])
    executor_target_rows = [row for row in rows if _executor_target_value(row) is not None]
    executor_target = sum(1 for row in executor_target_rows if _executor_target_value(row))
    by_family: dict[str, dict[str, Any]] = {}
    for row in rows:
        family = str(row["family"])
        bucket = by_family.setdefault(
            family,
            {
                "cases": 0,
                "exact": 0,
                "executable_cases": 0,
                "executable": 0,
                "executor_equivalence_cases": 0,
                "executor_equivalence": 0,
            },
        )
        bucket["cases"] += 1
        if row["exact_match"]:
            bucket["exact"] += 1
        if row.get("executable_match") is not None:
            bucket["executable_cases"] += 1
            if row["executable_match"]:
                bucket["executable"] += 1
        executor_target_match = _executor_target_value(row)
        if executor_target_match is not None:
            bucket["executor_equivalence_cases"] += 1
            if executor_target_match:
                bucket["executor_equivalence"] += 1
    for bucket in by_family.values():
        cases = int(bucket["cases"])
        bucket["exact_rate"] = bucket["exact"] / cases if cases else 0.0
        executable_cases = int(bucket["executable_cases"])
        bucket["executable_rate"] = bucket["executable"] / executable_cases if executable_cases else None
        executor_equivalence_cases = int(bucket["executor_equivalence_cases"])
        bucket["executor_equivalence_rate"] = (
            bucket["executor_equivalence"] / executor_equivalence_cases if executor_equivalence_cases else None
        )
    return {
        "case_count": total,
        "exact_match_count": exact,
        "exact_match_rate": exact / total if total else 0.0,
        "executable_evaluable_count": len(executable_rows),
        "executable_match_count": executable,
        "executable_match_rate": executable / len(executable_rows) if executable_rows else None,
        "executor_equivalence_evaluable_count": len(executor_target_rows),
        "executor_equivalence_match_count": executor_target,
        "executor_equivalence_match_rate": executor_target / len(executor_target_rows) if executor_target_rows else None,
        "family_summary": by_family,
    }


def _calls_exact_match(actual_calls: list[ToolCall], expected_calls: list[ToolCall]) -> bool:
    if len(actual_calls) != len(expected_calls):
        return False
    return all(
        actual.name == expected.name and actual.arguments == expected.arguments
        for actual, expected in zip(actual_calls, expected_calls, strict=False)
    )


def _score_executable_case(
    *,
    case: ToolDirectiveProbeCase,
    tool_specs: list[ToolSpec],
    actual_calls: list[ToolCall],
) -> tuple[bool | None, list[dict[str, Any]]]:
    if not case.initial_state:
        return None, []
    if not actual_calls:
        return False, []
    execution = _execute_calls(case.initial_state, tool_specs, actual_calls)
    if any(result.get("validator_result") != "pass" for result in execution):
        return False, execution
    if not case.expected_execution:
        return True, execution
    return _execution_satisfies_contract(execution, case.expected_execution), execution


def _score_executor_target_case(
    *,
    case: ToolDirectiveProbeCase,
    actual_execution: list[dict[str, Any]],
) -> bool | None:
    if not case.initial_state or not case.expected_execution:
        return None
    if not actual_execution:
        return False
    if any(result.get("validator_result") != "pass" for result in actual_execution):
        return False
    return _execution_satisfies_contract(actual_execution, case.expected_execution)


def _execute_calls(initial_state: dict[str, Any], tool_specs: list[ToolSpec], calls: list[ToolCall]) -> list[dict[str, Any]]:
    executor = DeterministicExecutor(tool_specs=tool_specs)
    state = deepcopy(initial_state)
    rows = []
    for step, call in enumerate(calls, start=1):
        result = executor.step(state, call, step=step)
        state = result.state_after
        rows.append(
            {
                "step": result.step,
                "selected_tool": result.selected_tool,
                "arguments": result.arguments,
                "validator_result": result.validator_result,
                "output": result.output,
                "error": result.error,
            }
        )
    return rows


def _execution_satisfies_contract(execution: list[dict[str, Any]], expected_execution: dict[str, Any]) -> bool:
    if "region_ids" in expected_execution:
        expected_region_ids = [str(region_id) for region_id in expected_execution["region_ids"]]
        actual_region_ids = _last_output_list(execution, "region_ids")
        return actual_region_ids == expected_region_ids
    if "region_id" in expected_execution:
        expected_region_id = str(expected_execution["region_id"])
        actual_region_ids = _last_output_list(execution, "region_ids")
        actual_region_id = _last_output_value(execution, "region_id")
        return actual_region_id == expected_region_id or actual_region_ids == [expected_region_id]
    return True


def _bool_delta(candidate: Any, baseline: Any) -> int:
    return int(bool(candidate)) - int(bool(baseline))


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _optional_bool_delta(candidate: Any, baseline: Any) -> int | None:
    if candidate is None or baseline is None:
        return None
    return _bool_delta(candidate, baseline)


def _optional_rate_delta(candidate: Any, baseline: Any) -> float | None:
    if candidate is None or baseline is None:
        return None
    return float(candidate or 0.0) - float(baseline or 0.0)


def _executor_target_value(row: dict[str, Any]) -> bool | None:
    if "executor_target_match" in row:
        return _optional_bool(row.get("executor_target_match"))
    if row.get("expected_execution"):
        return _optional_bool(row.get("executable_match"))
    return None


def _probe_failure_mode(row: dict[str, Any]) -> str:
    if row.get("exact_match"):
        return "exact"
    if row.get("executable_match") is True:
        return "executable_paraphrase"
    expected_calls = row.get("expected_calls") or []
    actual_calls = row.get("actual_calls") or []
    if not actual_calls:
        return "no_tool_call"
    if len(actual_calls) != len(expected_calls):
        return "call_count_mismatch"
    expected_names = [str(call.get("name", "")) for call in expected_calls]
    actual_names = [str(call.get("name", "")) for call in actual_calls]
    if expected_names != actual_names:
        return "wrong_tool"
    expected_args = [call.get("arguments", {}) for call in expected_calls]
    actual_args = [call.get("arguments", {}) for call in actual_calls]
    if expected_args != actual_args:
        return "argument_mismatch"
    if row.get("executable_match") is False:
        return "executable_miss"
    return "unknown_mismatch"


def _last_output_list(execution: list[dict[str, Any]], key: str) -> list[str]:
    for result in reversed(execution):
        value = result.get("output", {}).get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    return []


def _last_output_value(execution: list[dict[str, Any]], key: str) -> str:
    for result in reversed(execution):
        value = result.get("output", {}).get(key)
        if value is not None:
            return str(value)
    return ""


def _form_live_latest_state() -> dict[str, Any]:
    return {
        "visual_executor_mode": "local",
        "images": {
            "img-form-live-latest": {
                "entities": [],
                "layouts": [],
                "local_layouts": [
                    {
                        "region_id": "form-err-201",
                        "label": "validation error",
                        "text": "Work authorization required before submission",
                        "attributes": {"field": "work authorization", "priority": "earlier"},
                    },
                    {
                        "region_id": "form-err-202",
                        "label": "validation error",
                        "text": "Phone number format invalid",
                        "attributes": {"field": "phone", "priority": "latest"},
                    },
                ],
            }
        },
    }


def _write_probe_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [
        "case_id",
        "family",
        "expected_call_count",
        "actual_call_count",
        "exact_match",
        "executable_match",
        "executor_target_match",
        "expected_calls",
        "actual_calls",
        "expected_execution",
        "actual_execution",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in fieldnames},
                    "expected_calls": json.dumps(row["expected_calls"], ensure_ascii=False),
                    "actual_calls": json.dumps(row["actual_calls"], ensure_ascii=False),
                    "expected_execution": json.dumps(row["expected_execution"], ensure_ascii=False),
                    "actual_execution": json.dumps(row["actual_execution"], ensure_ascii=False),
                }
            )


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_cell(row.get(field, "")) for field in fieldnames})


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value
