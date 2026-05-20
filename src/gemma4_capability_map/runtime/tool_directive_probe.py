from __future__ import annotations

import json
import re
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
_VISUAL_COMPONENT_WORDS = {
    "alert",
    "badge",
    "banner",
    "caption",
    "chip",
    "field",
    "lane",
    "marker",
    "memo",
    "note",
    "notice",
    "panel",
    "pill",
    "tag",
    "tile",
    "toggle",
    "switch",
}
_VISUAL_BASE_NEGATED_VALUE_WORDS = {"no", "not", "without"}
_VISUAL_NEGATIVE_VALUE_WORDS = _VISUAL_BASE_NEGATED_VALUE_WORDS | {
    "disabled",
    "expired",
    "inactive",
    "missing",
    "paused",
    "rejected",
    "unassigned",
    "unresolved",
}


@dataclass(frozen=True)
class ToolDirectiveProbeCase:
    case_id: str
    family: str
    messages: list[Message]
    media: list[str]
    tool_names: list[str]
    initial_state: dict[str, Any] = field(default_factory=dict)
    expected_execution: dict[str, Any] = field(default_factory=dict)
    expected_calls: list[ToolCall] = field(default_factory=list)


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
        expected_calls = _expected_calls_for_case(case, tool_specs)
        turn = runner.generate(
            messages=case.messages,
            media=case.media,
            tool_specs=tool_specs,
            thinking=False,
            max_new_tokens=int(meta.get("reasoner_max_new_tokens", 64) or 64),
        )
        if controls.enable_visual_semantic_target_preservation:
            turn = _apply_visual_semantic_target_preservation(turn=turn, case=case, tool_specs=tool_specs)
        if controls.enable_visual_stale_selection_gate:
            turn = _apply_visual_stale_selection_gate(
                turn=turn,
                case=case,
                tool_specs=tool_specs,
                preserve_semantic_targets=controls.enable_visual_semantic_target_preservation,
                reject_negated_current_selection=controls.enable_visual_stale_selection_negation_guard,
                reject_paraphrased_current_selection=controls.enable_visual_stale_selection_paraphrase_guard,
            )
        if controls.enable_visual_target_query_normalization:
            turn = _apply_visual_target_query_normalization(
                turn=turn,
                case=case,
                tool_specs=tool_specs,
                preserve_negated_exact_layout_targets=controls.enable_visual_negation_aware_target_query_normalization,
                preserve_semantic_targets=controls.enable_visual_semantic_target_preservation,
            )
        if controls.enable_visual_scoped_target_query_normalization:
            turn = _apply_visual_target_query_normalization(
                turn=turn,
                case=case,
                tool_specs=tool_specs,
                preserve_value_bearing_targets=True,
                preserve_negated_exact_layout_targets=controls.enable_visual_negation_aware_target_query_normalization,
                preserve_semantic_targets=controls.enable_visual_semantic_target_preservation,
            )
        if controls.enable_visual_value_bearing_target_query_synthesis:
            turn = _apply_visual_value_bearing_target_query_synthesis(
                turn=turn,
                case=case,
                tool_specs=tool_specs,
                preserve_negated_exact_layout_targets=controls.enable_visual_negation_aware_target_query_normalization,
                preserve_semantic_targets=controls.enable_visual_semantic_target_preservation,
            )
        if (
            controls.enable_visual_negated_component_target_preservation
            or controls.enable_visual_negative_value_component_target_preservation
        ):
            turn = _apply_visual_negated_component_target_preservation(
                turn=turn,
                case=case,
                tool_specs=tool_specs,
                preserve_negative_value_targets=(
                    controls.enable_visual_negative_value_component_target_preservation
                ),
            )
        if controls.enable_visual_contextual_surface_alias_routing:
            turn = _apply_visual_contextual_surface_alias_routing(turn=turn, case=case, tool_specs=tool_specs)
        if controls.enable_visual_composed_route_gating:
            turn = _apply_visual_composed_route_gating(
                turn=turn,
                case=case,
                tool_specs=tool_specs,
                preserve_negated_exact_layout_targets=controls.enable_visual_negation_aware_target_query_normalization,
                preserve_semantic_targets=controls.enable_visual_semantic_target_preservation,
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
        actual_calls=turn.normalized_tool_call,
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
        "runtime_metadata": turn.runtime_metadata,
        "prompt_tokens": turn.prompt_tokens,
        "completion_tokens": turn.completion_tokens,
        "latency_ms": turn.latency_ms,
    }


def _expected_calls_for_case(case: ToolDirectiveProbeCase, tool_specs: list[ToolSpec]) -> list[ToolCall]:
    if case.expected_execution.get("no_tool_call") is True:
        return []
    if case.expected_calls:
        return case.expected_calls
    return plan_tool_calls(case.messages, case.media, tool_specs)


def _apply_visual_semantic_target_preservation(
    *,
    turn: ModelTurn,
    case: ToolDirectiveProbeCase,
    tool_specs: list[ToolSpec],
) -> ModelTurn:
    tool_names = {tool.name for tool in tool_specs}
    if "extract_layout" not in tool_names or turn.normalized_tool_call:
        return turn

    image_id = _visual_image_id(case)
    target_query = _visual_semantic_target_label_from_state(case)
    if not image_id or not target_query:
        return turn

    arguments = {"image_id": image_id, "target_query": target_query}
    payload = {
        "name": "extract_layout",
        "arguments": arguments,
        "controller": "visual_semantic_target_preservation",
        "reason": "no_call_clear_visual_target",
    }
    replacement = ToolCall(
        name="extract_layout",
        arguments=arguments,
        source_format="heuristic",
        raw=json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
    )
    metadata = dict(turn.runtime_metadata)
    metadata["visual_semantic_target_preservation"] = [
        {
            "from_tool": "",
            "from_arguments": {},
            "to_tool": replacement.name,
            "to_arguments": replacement.arguments,
            "preserved_target_query": target_query,
            "reason": "no_call_clear_visual_target",
        }
    ]
    return turn.model_copy(update={"normalized_tool_call": [replacement], "runtime_metadata": metadata})


def _apply_visual_stale_selection_gate(
    *,
    turn: ModelTurn,
    case: ToolDirectiveProbeCase,
    tool_specs: list[ToolSpec],
    preserve_semantic_targets: bool = False,
    reject_negated_current_selection: bool = False,
    reject_paraphrased_current_selection: bool = False,
) -> ModelTurn:
    tool_names = {tool.name for tool in tool_specs}
    if "extract_layout" not in tool_names or not turn.normalized_tool_call:
        return turn

    patched_calls: list[ToolCall] = []
    gate_rows: list[dict[str, Any]] = []
    negation_rows: list[dict[str, Any]] = []
    paraphrase_rows: list[dict[str, Any]] = []
    for call in turn.normalized_tool_call:
        replacement = _visual_stale_selection_replacement(
            call=call,
            case=case,
            preserve_semantic_targets=preserve_semantic_targets,
            reject_negated_current_selection=reject_negated_current_selection,
            reject_paraphrased_current_selection=reject_paraphrased_current_selection,
        )
        if replacement is None:
            patched_calls.append(call)
            continue
        replacement_call, row = replacement
        patched_calls.append(replacement_call)
        if row.get("reason") == "negated_current_selection_to_requested_surface":
            negation_rows.append(row)
        elif row.get("reason") == "paraphrased_stale_selection_to_requested_surface":
            paraphrase_rows.append(row)
        else:
            gate_rows.append(row)

    if not gate_rows and not negation_rows and not paraphrase_rows:
        return turn
    metadata = dict(turn.runtime_metadata)
    if gate_rows:
        metadata["visual_stale_selection_gate"] = gate_rows
    if negation_rows:
        metadata["visual_stale_selection_negation_guard"] = negation_rows
    if paraphrase_rows:
        metadata["visual_stale_selection_paraphrase_guard"] = paraphrase_rows
    return turn.model_copy(update={"normalized_tool_call": patched_calls, "runtime_metadata": metadata})


def _apply_visual_target_query_normalization(
    *,
    turn: ModelTurn,
    case: ToolDirectiveProbeCase,
    tool_specs: list[ToolSpec],
    preserve_value_bearing_targets: bool = False,
    preserve_negated_exact_layout_targets: bool = False,
    preserve_semantic_targets: bool = False,
) -> ModelTurn:
    tool_names = {tool.name for tool in tool_specs}
    if "extract_layout" not in tool_names or not turn.normalized_tool_call:
        return turn

    prompt_state_label = _visual_target_label_from_state(case, preserve_semantic_targets=preserve_semantic_targets)
    if not prompt_state_label:
        return turn

    patched_calls: list[ToolCall] = []
    gate_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    for call in turn.normalized_tool_call:
        scope_block = (
            _visual_target_query_normalization_scope_block(
                call=call,
                case=case,
                prompt_state_label=prompt_state_label,
            )
            if preserve_value_bearing_targets
            else None
        )
        if scope_block is not None:
            patched_calls.append(call)
            blocked_rows.append(scope_block)
            continue
        negation_scope_block = (
            _visual_target_query_normalization_negation_scope_block(
                call=call,
                case=case,
                prompt_state_label=prompt_state_label,
            )
            if preserve_negated_exact_layout_targets
            else None
        )
        if negation_scope_block is not None:
            patched_calls.append(call)
            blocked_rows.append(negation_scope_block)
            continue
        replacement = _visual_target_query_normalization_replacement(
            call=call,
            prompt_state_label=prompt_state_label,
        )
        if replacement is None:
            patched_calls.append(call)
            continue
        patched_calls.append(replacement)
        gate_rows.append(
            {
                "from_tool": call.name,
                "from_arguments": call.arguments,
                "to_tool": replacement.name,
                "to_arguments": replacement.arguments,
                "prompt_state_label": prompt_state_label,
            }
        )

    if not gate_rows and not blocked_rows:
        return turn
    metadata = dict(turn.runtime_metadata)
    if gate_rows:
        metadata["visual_target_query_normalization"] = gate_rows
    if blocked_rows:
        metadata["visual_target_query_normalization_blocked"] = blocked_rows
    return turn.model_copy(update={"normalized_tool_call": patched_calls, "runtime_metadata": metadata})


def _visual_target_query_normalization_scope_block(
    *,
    call: ToolCall,
    case: ToolDirectiveProbeCase,
    prompt_state_label: str,
) -> dict[str, Any] | None:
    if call.name != "extract_layout":
        return None
    target_query = str(call.arguments.get("target_query", "")).strip()
    if not target_query:
        return None
    if target_query == prompt_state_label:
        return None

    user_text = " ".join(message.content for message in case.messages if message.role == "user").lower()
    prompt_label = prompt_state_label.lower()
    for label in _visual_layout_labels(case.initial_state):
        label_lower = label.lower()
        prefix = f"{prompt_label} "
        if not label_lower.startswith(prefix):
            continue
        value_suffix = label_lower[len(prefix) :].strip()
        if not value_suffix:
            continue
        if label_lower in user_text or f"{value_suffix} {prompt_label}" in user_text:
            return {
                "from_tool": call.name,
                "from_arguments": call.arguments,
                "prompt_state_label": prompt_state_label,
                "preserved_target_query": target_query,
                "value_bearing_label": label,
                "value_suffix": value_suffix,
                "reason": "value_bearing_label_requested",
            }
    return None


def _visual_target_query_normalization_negation_scope_block(
    *,
    call: ToolCall,
    case: ToolDirectiveProbeCase,
    prompt_state_label: str,
) -> dict[str, Any] | None:
    if call.name != "extract_layout":
        return None
    target_query = str(call.arguments.get("target_query", "")).strip()
    if not target_query or target_query == prompt_state_label:
        return None

    target_row = _visual_layout_row_by_label(case.initial_state, target_query)
    prompt_row = _visual_layout_row_by_label(case.initial_state, prompt_state_label)
    if target_row is None or prompt_row is None:
        return None

    user_text = " ".join(message.content for message in case.messages if message.role == "user").lower()
    if not _visual_label_positively_requested(user_text=user_text, label=target_query):
        return None
    if not _visual_label_contextually_deprioritized(user_text=user_text, label=prompt_state_label):
        return None

    return {
        "from_tool": call.name,
        "from_arguments": call.arguments,
        "prompt_state_label": prompt_state_label,
        "preserved_target_query": target_query,
        "preserved_region_id": target_row["region_id"],
        "blocked_label": prompt_state_label,
        "blocked_region_id": prompt_row["region_id"],
        "reason": "negation_scope_exact_layout_label",
    }


def _apply_visual_value_bearing_target_query_synthesis(
    *,
    turn: ModelTurn,
    case: ToolDirectiveProbeCase,
    tool_specs: list[ToolSpec],
    preserve_negated_exact_layout_targets: bool = False,
    preserve_semantic_targets: bool = False,
) -> ModelTurn:
    tool_names = {tool.name for tool in tool_specs}
    if "extract_layout" not in tool_names or not turn.normalized_tool_call:
        return turn

    prompt_state_label = _visual_target_label_from_state(case, preserve_semantic_targets=preserve_semantic_targets)
    if not prompt_state_label:
        return turn

    patched_calls: list[ToolCall] = []
    synthesis_rows: list[dict[str, Any]] = []
    normalization_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    for call in turn.normalized_tool_call:
        synthesis = _visual_value_bearing_target_query_synthesis_replacement(
            call=call,
            case=case,
            prompt_state_label=prompt_state_label,
        )
        if synthesis is not None:
            replacement, synthesis_row = synthesis
            patched_calls.append(replacement)
            synthesis_rows.append(synthesis_row)
            continue
        negation_scope_block = (
            _visual_target_query_normalization_negation_scope_block(
                call=call,
                case=case,
                prompt_state_label=prompt_state_label,
            )
            if preserve_negated_exact_layout_targets
            else None
        )
        if negation_scope_block is not None:
            patched_calls.append(call)
            blocked_rows.append(negation_scope_block)
            continue
        semantic_preservation = (
            _visual_semantic_target_preservation_row(
                call=call,
                case=case,
                prompt_state_label=prompt_state_label,
            )
            if preserve_semantic_targets
            else None
        )
        if semantic_preservation is not None:
            patched_calls.append(call)
            semantic_rows.append(semantic_preservation)
            continue
        replacement = _visual_target_query_normalization_replacement(
            call=call,
            prompt_state_label=prompt_state_label,
        )
        if replacement is None:
            patched_calls.append(call)
            continue
        patched_calls.append(replacement)
        normalization_rows.append(
            {
                "from_tool": call.name,
                "from_arguments": call.arguments,
                "to_tool": replacement.name,
                "to_arguments": replacement.arguments,
                "prompt_state_label": prompt_state_label,
            }
        )

    if not synthesis_rows and not normalization_rows and not blocked_rows and not semantic_rows:
        return turn
    metadata = dict(turn.runtime_metadata)
    if synthesis_rows:
        metadata["visual_value_bearing_target_query_synthesis"] = synthesis_rows
    if normalization_rows:
        metadata["visual_target_query_normalization"] = normalization_rows
    if blocked_rows:
        metadata["visual_target_query_normalization_blocked"] = blocked_rows
    if semantic_rows:
        metadata["visual_semantic_target_preservation"] = semantic_rows
    return turn.model_copy(update={"normalized_tool_call": patched_calls, "runtime_metadata": metadata})


def _apply_visual_negated_component_target_preservation(
    *,
    turn: ModelTurn,
    case: ToolDirectiveProbeCase,
    tool_specs: list[ToolSpec],
    preserve_negative_value_targets: bool = False,
) -> ModelTurn:
    tool_names = {tool.name for tool in tool_specs}
    if "extract_layout" not in tool_names or not turn.normalized_tool_call:
        return turn

    target_label = _visual_negated_component_target_label_from_state(
        case,
        include_negative_values=preserve_negative_value_targets,
    )
    if not target_label:
        return turn

    patched_calls: list[ToolCall] = []
    preservation_rows: list[dict[str, Any]] = []
    negative_value_rows: list[dict[str, Any]] = []
    controller = (
        "visual_negative_value_component_target_preservation"
        if _visual_label_has_extended_negative_value(target_label)
        else "visual_negated_component_target_preservation"
    )
    for call in turn.normalized_tool_call:
        replacement = _visual_negated_component_target_preservation_replacement(
            call=call,
            target_label=target_label,
            controller=controller,
        )
        if replacement is None:
            patched_calls.append(call)
            continue
        replacement_call, row = replacement
        patched_calls.append(replacement_call)
        if controller == "visual_negative_value_component_target_preservation":
            negative_value_rows.append(row)
        else:
            preservation_rows.append(row)

    if not preservation_rows and not negative_value_rows:
        return turn
    metadata = dict(turn.runtime_metadata)
    if preservation_rows:
        metadata["visual_negated_component_target_preservation"] = preservation_rows
    if negative_value_rows:
        metadata["visual_negative_value_component_target_preservation"] = negative_value_rows
    return turn.model_copy(update={"normalized_tool_call": patched_calls, "runtime_metadata": metadata})


def _visual_negated_component_target_preservation_replacement(
    *,
    call: ToolCall,
    target_label: str,
    controller: str = "visual_negated_component_target_preservation",
) -> tuple[ToolCall, dict[str, Any]] | None:
    if call.name != "extract_layout":
        return None
    target_query = str(call.arguments.get("target_query", "")).strip()
    if not target_query or target_query.lower() == target_label.lower():
        return None
    if not _short_component_query_for_label(target_query=target_query, label=target_label):
        return None

    arguments = dict(call.arguments)
    arguments["target_query"] = target_label
    payload = {
        "name": "extract_layout",
        "arguments": arguments,
        "controller": controller,
        "from_target_query": target_query,
    }
    replacement = ToolCall(
        name="extract_layout",
        arguments=arguments,
        source_format="heuristic",
        raw=json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
    )
    return replacement, {
        "from_tool": call.name,
        "from_arguments": call.arguments,
        "to_tool": replacement.name,
        "to_arguments": replacement.arguments,
        "preserved_target_query": target_label,
        "blocked_label": target_query,
        "reason": (
            "negative_value_component_query_preserved"
            if controller == "visual_negative_value_component_target_preservation"
            else "negated_value_component_query_preserved"
        ),
    }


def _visual_value_bearing_target_query_synthesis_replacement(
    *,
    call: ToolCall,
    case: ToolDirectiveProbeCase,
    prompt_state_label: str,
) -> tuple[ToolCall, dict[str, Any]] | None:
    if call.name != "extract_layout":
        return None
    target_query = str(call.arguments.get("target_query", "")).strip()
    if not target_query:
        return None

    user_text = " ".join(message.content for message in case.messages if message.role == "user").lower()
    prompt_label = prompt_state_label.lower()
    for label in _visual_layout_labels(case.initial_state):
        label_lower = label.lower()
        prefix = f"{prompt_label} "
        if not label_lower.startswith(prefix):
            continue
        value_suffix = label_lower[len(prefix) :].strip()
        if not value_suffix:
            continue
        matched_phrase = _value_bearing_target_phrase(user_text, prompt_label, value_suffix, label_lower)
        if not matched_phrase:
            continue
        if target_query == label:
            return None
        arguments = dict(call.arguments)
        arguments["target_query"] = label
        payload = {
            "name": "extract_layout",
            "arguments": arguments,
        }
        replacement = ToolCall(
            name="extract_layout",
            arguments=arguments,
            source_format="heuristic",
            raw=json.dumps(payload, ensure_ascii=False),
        )
        return replacement, {
            "from_tool": call.name,
            "from_arguments": call.arguments,
            "to_tool": replacement.name,
            "to_arguments": replacement.arguments,
            "prompt_state_label": prompt_state_label,
            "value_bearing_label": label,
            "value_suffix": value_suffix,
            "matched_phrase": matched_phrase,
            "reason": "value_bearing_label_recoverable",
        }
    return None


def _value_bearing_target_phrase(
    user_text: str,
    prompt_label: str,
    value_suffix: str,
    label: str,
) -> str:
    phrases = (label, f"{value_suffix} {prompt_label}")
    for phrase in phrases:
        if phrase not in user_text:
            continue
        if _phrase_is_negated_or_deprioritized(user_text, phrase):
            continue
        return phrase
    return ""


def _phrase_is_negated_or_deprioritized(user_text: str, phrase: str) -> bool:
    prefixes = (
        "not",
        "not use",
        "not target",
        "do not use",
        "do not target",
        "ignore",
        "leave",
        "leaving",
        "avoid",
        "exclude",
    )
    articles = ("", "the ")
    return any(f"{prefix} {article}{phrase}" in user_text for prefix in prefixes for article in articles)


def _apply_visual_contextual_surface_alias_routing(
    *,
    turn: ModelTurn,
    case: ToolDirectiveProbeCase,
    tool_specs: list[ToolSpec],
) -> ModelTurn:
    tool_names = {tool.name for tool in tool_specs}
    if "extract_layout" not in tool_names or not turn.normalized_tool_call:
        return turn

    patched_calls: list[ToolCall] = []
    alias_rows: list[dict[str, Any]] = []
    for call in turn.normalized_tool_call:
        routed = _visual_contextual_surface_alias_routing_replacement(call=call, case=case)
        if routed is None:
            patched_calls.append(call)
            continue
        replacement, alias_row = routed
        patched_calls.append(replacement)
        alias_rows.append(alias_row)

    if not alias_rows:
        return turn
    metadata = dict(turn.runtime_metadata)
    metadata["visual_contextual_surface_alias_routing"] = alias_rows
    return turn.model_copy(update={"normalized_tool_call": patched_calls, "runtime_metadata": metadata})


def _apply_visual_composed_route_gating(
    *,
    turn: ModelTurn,
    case: ToolDirectiveProbeCase,
    tool_specs: list[ToolSpec],
    preserve_negated_exact_layout_targets: bool = False,
    preserve_semantic_targets: bool = False,
) -> ModelTurn:
    tool_names = {tool.name for tool in tool_specs}
    if "extract_layout" not in tool_names or not turn.normalized_tool_call:
        return turn

    patched_calls: list[ToolCall] = []
    route_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    for call in turn.normalized_tool_call:
        block = (
            _visual_composed_route_gating_negation_scope_block(call=call, case=case)
            if preserve_negated_exact_layout_targets
            else None
        )
        if block is not None:
            patched_calls.append(call)
            blocked_rows.append(block)
            continue
        routed = _visual_composed_route_gating_replacement(
            call=call,
            case=case,
            preserve_semantic_targets=preserve_semantic_targets,
        )
        if routed is None:
            patched_calls.append(call)
            continue
        replacement, route_row = routed
        patched_calls.append(replacement)
        route_rows.append(route_row)

    if not route_rows and not blocked_rows:
        return turn
    metadata = dict(turn.runtime_metadata)
    if route_rows:
        metadata["visual_composed_route_gating"] = route_rows
    if blocked_rows:
        metadata["visual_composed_route_gating_blocked"] = blocked_rows
    return turn.model_copy(update={"normalized_tool_call": patched_calls, "runtime_metadata": metadata})


def _visual_composed_route_gating_negation_scope_block(
    *,
    call: ToolCall,
    case: ToolDirectiveProbeCase,
) -> dict[str, Any] | None:
    if call.name != "extract_layout":
        return None
    target_query = str(call.arguments.get("target_query", "")).strip()
    if not target_query:
        return None
    target_row = _visual_layout_row_by_label(case.initial_state, target_query)
    if target_row is None:
        return None

    requested = _visual_composed_requested_layout(case)
    if requested is None or target_query == requested["label"]:
        return None

    user_text = " ".join(message.content for message in case.messages if message.role == "user").lower()
    if not _visual_label_positively_requested(user_text=user_text, label=target_query):
        return None
    if not _visual_label_contextually_deprioritized(user_text=user_text, label=str(requested["label"])):
        return None

    return {
        "from_tool": call.name,
        "from_arguments": call.arguments,
        "preserved_target_query": target_query,
        "preserved_region_id": target_row["region_id"],
        "blocked_label": requested["label"],
        "blocked_region_id": requested["region_id"],
        "reason": "negation_scope_exact_layout_label",
    }


def _visual_composed_route_gating_replacement(
    *,
    call: ToolCall,
    case: ToolDirectiveProbeCase,
    preserve_semantic_targets: bool = False,
) -> tuple[ToolCall, dict[str, Any]] | None:
    requested = _visual_composed_requested_layout(case, preserve_semantic_targets=preserve_semantic_targets)
    if requested is None:
        return None

    user_text = " ".join(message.content for message in case.messages if message.role == "user").lower()
    image_id = _visual_image_id(case)
    if call.name == "refine_selection":
        selection_id = str(call.arguments.get("selection_id", "")).strip()
        if not selection_id:
            return None
        if selection_id in _visual_selection_ids(case.initial_state) and not _selection_id_deprioritized(
            user_text=user_text, selection_id=selection_id
        ):
            return None
        return _visual_composed_route_replacement(
            call=call,
            image_id=image_id,
            target_query=requested["label"],
            reason="stale_selection_to_requested_surface",
            requested_row=requested,
        )

    if call.name != "extract_layout":
        return None
    target_query = str(call.arguments.get("target_query", "")).strip()
    if not target_query or target_query == requested["label"]:
        return None
    target_row = _visual_layout_row_by_label(case.initial_state, target_query)
    target_is_deprioritized = _visual_label_deprioritized(user_text=user_text, label=target_query)
    if target_row is not None:
        target_is_deprioritized = target_is_deprioritized or _visual_label_deprioritized(
            user_text=user_text, label=target_row["label"]
        )
    if not target_is_deprioritized and requested["score"] < 5.0:
        return None
    return _visual_composed_route_replacement(
        call=call,
        image_id=str(call.arguments.get("image_id") or image_id),
        target_query=requested["label"],
        reason="requested_surface_over_deprioritized_decoy",
        requested_row=requested,
    )


def _visual_composed_route_replacement(
    *,
    call: ToolCall,
    image_id: str,
    target_query: str,
    reason: str,
    requested_row: dict[str, Any],
) -> tuple[ToolCall, dict[str, Any]] | None:
    if not image_id or not target_query:
        return None
    arguments = {"image_id": image_id, "target_query": target_query}
    payload = {
        "name": "extract_layout",
        "arguments": arguments,
        "controller": "visual_composed_route_gating",
        "reason": reason,
    }
    replacement = ToolCall(
        name="extract_layout",
        arguments=arguments,
        source_format="heuristic",
        raw=json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
    )
    return replacement, {
        "from_tool": call.name,
        "from_arguments": call.arguments,
        "to_tool": replacement.name,
        "to_arguments": replacement.arguments,
        "requested_label": requested_row["label"],
        "requested_region_id": requested_row["region_id"],
        "reason": reason,
    }


def _visual_composed_requested_layout(
    case: ToolDirectiveProbeCase,
    *,
    preserve_semantic_targets: bool = False,
) -> dict[str, Any] | None:
    if preserve_semantic_targets:
        semantic_label = _visual_semantic_target_label_from_state(case)
        if semantic_label:
            semantic_row = _visual_layout_row_by_label(case.initial_state, semantic_label)
            if semantic_row is not None:
                candidate = dict(semantic_row)
                candidate["score"] = 10.0
                return candidate

    user_text = " ".join(message.content for message in case.messages if message.role == "user").lower()
    candidates: list[dict[str, Any]] = []
    for row in _visual_layout_rows(case.initial_state):
        label = row["label"]
        label_lower = label.lower()
        component = label_lower.split()[-1]
        base_tokens = label_lower.split()[:-1]
        score = 0.0
        if label_lower in user_text:
            score += 2.0 + len(base_tokens) * 0.25
        for phrase in (f"use the {label_lower}", f"work from the {label_lower}", f"from the {label_lower}"):
            if phrase in user_text:
                score += 4.0
        if _surface_component_requested(user_text=user_text, component=component) and all(
            token in user_text for token in base_tokens
        ):
            score += 5.0
        if _visual_label_deprioritized(user_text=user_text, label=label):
            score -= 8.0
        if score <= 0:
            continue
        candidate = dict(row)
        candidate["score"] = score
        candidates.append(candidate)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (-float(item["score"]), item["source_index"]))[0]


def _visual_layout_row_by_label(initial_state: dict[str, Any], label: str) -> dict[str, Any] | None:
    label_lower = label.lower()
    for row in _visual_layout_rows(initial_state):
        if row["label"].lower() == label_lower:
            return row
    return None


def _selection_id_deprioritized(*, user_text: str, selection_id: str) -> bool:
    selection = selection_id.lower()
    return any(
        fragment in user_text
        for fragment in (
            f"ignore old selection {selection}",
            f"ignore the old selection {selection}",
            f"ignore selection {selection}",
            f"old selection {selection}",
        )
    )


def _visual_label_deprioritized(*, user_text: str, label: str) -> bool:
    label_lower = label.lower()
    component = label_lower.split()[-1]
    positive_fragments = (
        f"use the {label_lower}",
        f"use {label_lower}",
        f"select the {label_lower}",
        f"select {label_lower}",
        f"locate the {label_lower}",
        f"locate {label_lower}",
        f"find the {label_lower}",
        f"find {label_lower}",
    )
    exact_negative_fragments = (
        f"do not use the {label_lower}",
        f"do not use {label_lower}",
        f"do not select the {label_lower}",
        f"do not select {label_lower}",
        f"do not locate the {label_lower}",
        f"do not locate {label_lower}",
        f"do not find the {label_lower}",
        f"do not find {label_lower}",
        f"not the {label_lower}",
        f"not {label_lower}",
        f"ignore the {label_lower}",
        f"avoid the {label_lower}",
    )
    if any(fragment in user_text for fragment in positive_fragments) and not any(
        fragment in user_text for fragment in exact_negative_fragments
    ):
        return False
    if any(fragment in user_text for fragment in exact_negative_fragments):
        return True
    direct_fragments = (
        f"{label_lower} is nearby context",
        f"{label_lower} are nearby context",
        f"not the {label_lower}",
        f"not {label_lower}",
        f"ignore the {label_lower}",
        f"avoid the {label_lower}",
        f"{label_lower} are adjacent",
        f"{label_lower} is adjacent",
        f"{label_lower} are not",
        f"{label_lower} is not",
    )
    if any(fragment in user_text for fragment in direct_fragments):
        return True
    component_fragments = (
        f"not the {component}",
        f"not {component}",
        f"not the surface to use",
        "nearby context",
        "adjacent controls",
    )
    for match in re.finditer(re.escape(label_lower), user_text):
        window = user_text[match.start() : match.end() + 96]
        if any(fragment in window for fragment in component_fragments):
            return True
    return False


def _visual_label_positively_requested(*, user_text: str, label: str) -> bool:
    label_lower = label.lower()
    positive_fragments = (
        f"use the {label_lower}",
        f"use {label_lower}",
        f"work from the {label_lower}",
        f"work from {label_lower}",
        f"select the {label_lower}",
        f"select {label_lower}",
        f"locate the {label_lower}",
        f"locate {label_lower}",
        f"find the {label_lower}",
        f"find {label_lower}",
        f"target the {label_lower}",
        f"target {label_lower}",
    )
    return any(fragment in user_text for fragment in positive_fragments)


def _visual_label_contextually_deprioritized(*, user_text: str, label: str) -> bool:
    if _visual_label_deprioritized(user_text=user_text, label=label):
        return True

    label_lower = label.lower()
    context_markers = (
        "old example",
        "old negative example",
        "prior screenshot",
        "previous screenshot",
        "prior image",
        "previous image",
        "prior example",
        "previous example",
        "nearby context",
        "not the current target",
        "not current target",
        "not the target",
    )
    for match in re.finditer(re.escape(label_lower), user_text):
        window = user_text[match.start() : match.end() + 160]
        if any(marker in window for marker in context_markers):
            return True
    return False


def _visual_contextual_surface_alias_routing_replacement(
    *,
    call: ToolCall,
    case: ToolDirectiveProbeCase,
) -> tuple[ToolCall, dict[str, Any]] | None:
    if call.name != "extract_layout":
        return None
    target_query = str(call.arguments.get("target_query", "")).strip()
    if not target_query:
        return None

    user_text = " ".join(message.content for message in case.messages if message.role == "user").lower()
    target_lower = target_query.lower()
    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in _visual_layout_rows(case.initial_state):
        label = row["label"]
        label_lower = label.lower()
        if target_lower == label_lower:
            continue
        text_lower = row["text"].lower()
        if target_lower not in text_lower:
            continue
        label_tokens = label_lower.split()
        if len(label_tokens) < 2:
            continue
        component = label_tokens[-1]
        if not _surface_component_requested(user_text=user_text, component=component):
            continue
        base_tokens = label_tokens[:-1]
        if any(token not in user_text for token in base_tokens):
            continue
        if _surface_component_deprioritized(user_text=user_text, component=component):
            continue
        score = 1.0 + len(base_tokens) + (2.0 if f"{component}-style" in user_text else 0.0)
        candidates.append((score, row))

    if not candidates:
        return None
    _, best = sorted(candidates, key=lambda item: (-item[0], item[1]["source_index"]))[0]
    arguments = dict(call.arguments)
    arguments["target_query"] = best["label"]
    payload = {
        "name": "extract_layout",
        "arguments": arguments,
    }
    replacement = ToolCall(
        name="extract_layout",
        arguments=arguments,
        source_format="heuristic",
        raw=json.dumps(payload, ensure_ascii=False),
    )
    return replacement, {
        "from_tool": call.name,
        "from_arguments": call.arguments,
        "to_tool": replacement.name,
        "to_arguments": replacement.arguments,
        "display_value": target_query,
        "surface_label": best["label"],
        "surface_text": best["text"],
        "surface_region_id": best["region_id"],
        "reason": "contextual_surface_alias_recoverable",
    }


def _surface_component_requested(*, user_text: str, component: str) -> bool:
    requested_fragments = (
        f"{component}-style",
        f"{component} style",
        f"{component} surface",
        f"{component} region",
        f"{component} area",
        f"{component} component",
        f"{component} control",
    )
    return any(fragment in user_text for fragment in requested_fragments)


def _surface_component_deprioritized(*, user_text: str, component: str) -> bool:
    deprioritized_fragments = (
        f"{component} is nearby context",
        f"{component} are nearby context",
        f"{component} and",
        f"not the {component}",
        f"not {component}",
        f"ignore the {component}",
        f"avoid the {component}",
    )
    return any(fragment in user_text for fragment in deprioritized_fragments)


def _visual_target_query_normalization_replacement(
    *,
    call: ToolCall,
    prompt_state_label: str,
) -> ToolCall | None:
    if call.name != "extract_layout":
        return None
    target_query = str(call.arguments.get("target_query", "")).strip()
    if not target_query:
        return None
    if target_query == prompt_state_label:
        return None

    arguments = dict(call.arguments)
    arguments["target_query"] = prompt_state_label
    payload = {
        "name": "extract_layout",
        "arguments": arguments,
        "controller": "visual_target_query_normalization",
        "from_target_query": target_query,
    }
    return ToolCall(
        name="extract_layout",
        arguments=arguments,
        source_format="heuristic",
        raw=json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
    )


def _visual_stale_selection_replacement(
    *,
    call: ToolCall,
    case: ToolDirectiveProbeCase,
    preserve_semantic_targets: bool = False,
    reject_negated_current_selection: bool = False,
    reject_paraphrased_current_selection: bool = False,
) -> tuple[ToolCall, dict[str, Any]] | None:
    if call.name != "refine_selection":
        return None
    selection_id = str(call.arguments.get("selection_id", "")).strip()
    if not selection_id:
        return None
    user_text = " ".join(message.content for message in case.messages if message.role == "user").lower()
    current_selection_ids = _visual_selection_ids(case.initial_state)
    selection_is_current = selection_id in current_selection_ids
    stale_reason = ""
    if selection_is_current:
        if reject_negated_current_selection and _selection_id_stale_or_negated_context(
            user_text=user_text,
            selection_id=selection_id,
        ):
            stale_reason = "negated_current_selection_to_requested_surface"
        elif reject_paraphrased_current_selection and _selection_id_stale_paraphrase_context(
            user_text=user_text,
            selection_id=selection_id,
        ):
            stale_reason = "paraphrased_stale_selection_to_requested_surface"
        else:
            return None
    image_id = _visual_image_id(case)
    target_query = _visual_target_label_from_state(case, preserve_semantic_targets=preserve_semantic_targets)
    if not image_id or not target_query:
        return None
    reason = stale_reason if selection_is_current else "missing_selection_to_layout_lookup"
    controller = (
        "visual_stale_selection_paraphrase_guard"
        if reason == "paraphrased_stale_selection_to_requested_surface"
        else "visual_stale_selection_negation_guard"
        if selection_is_current
        else "visual_stale_selection_gate"
    )
    payload = {
        "name": "extract_layout",
        "arguments": {
            "image_id": image_id,
            "target_query": target_query,
        },
        "controller": controller,
        "replaced_selection_id": selection_id,
        "reason": reason,
    }
    replacement = ToolCall(
        name="extract_layout",
        arguments=payload["arguments"],
        source_format="heuristic",
        raw=json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
    )
    return replacement, {
        "from_tool": call.name,
        "from_arguments": call.arguments,
        "to_tool": replacement.name,
        "to_arguments": replacement.arguments,
        "replaced_selection_id": selection_id,
        "reason": reason,
    }


def _selection_id_stale_or_negated_context(*, user_text: str, selection_id: str) -> bool:
    selection = selection_id.lower()
    if selection not in user_text:
        return False
    markers = (
        "stale selection",
        "old selection",
        "prior selection",
        "previous selection",
        "ignore",
        "do not use",
        "avoid",
    )
    for match in re.finditer(re.escape(selection), user_text):
        window = user_text[max(0, match.start() - 80) : min(len(user_text), match.end() + 120)]
        if any(marker in window for marker in markers):
            return True
    return False


def _selection_id_stale_paraphrase_context(*, user_text: str, selection_id: str) -> bool:
    selection = selection_id.lower()
    if selection not in user_text:
        return False
    markers = (
        "archived selector",
        "background context only",
        "belongs to a retired",
        "came from",
        "carried over",
        "carry-over selection",
        "discarded",
        "from billing history",
        "from planning",
        "leftover evidence",
        "remembered selection",
        "retired selection",
        "retired view",
        "shadow selection",
    )
    for match in re.finditer(re.escape(selection), user_text):
        window = user_text[max(0, match.start() - 96) : min(len(user_text), match.end() + 140)]
        if any(marker in window for marker in markers):
            return True
    return False


def _visual_selection_ids(initial_state: dict[str, Any]) -> set[str]:
    selections = initial_state.get("visual_selections", {})
    if not isinstance(selections, dict):
        return set()
    return {str(selection_id) for selection_id in selections}


def _visual_image_id(case: ToolDirectiveProbeCase) -> str:
    if case.media:
        return str(case.media[0])
    images = case.initial_state.get("images", {})
    if isinstance(images, dict) and images:
        return str(next(iter(images)))
    for message in case.messages:
        if message.role == "system" and "visual_image_ids:" in message.content:
            return message.content.split("visual_image_ids:", 1)[1].strip().split()[0]
    return ""


def _visual_semantic_target_label_from_state(case: ToolDirectiveProbeCase) -> str:
    user_text = " ".join(message.content for message in case.messages if message.role == "user").lower()
    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in _visual_layout_rows(case.initial_state):
        label = row["label"]
        phrases = _visual_semantic_label_phrases(label)
        score = _visual_semantic_positive_score(user_text=user_text, phrases=phrases)
        if score <= 0:
            continue
        if _visual_semantic_label_deprioritized(user_text=user_text, phrases=phrases):
            continue
        candidates.append((score - float(row["source_index"]) * 0.001, row))
    if not candidates:
        return ""
    return sorted(candidates, key=lambda item: (-item[0], item[1]["source_index"]))[0][1]["label"]


def _visual_negated_component_target_label_from_state(
    case: ToolDirectiveProbeCase,
    *,
    include_negative_values: bool = False,
) -> str:
    user_text = " ".join(message.content for message in case.messages if message.role == "user").lower()
    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in _visual_layout_rows(case.initial_state):
        label = row["label"]
        if not _visual_label_has_negated_value(label, include_negative_values=include_negative_values):
            continue
        phrases = _visual_negated_component_label_phrases(label)
        score = _visual_semantic_positive_score(user_text=user_text, phrases=phrases)
        if score <= 0:
            continue
        if _visual_semantic_label_deprioritized(user_text=user_text, phrases=phrases):
            continue
        candidates.append((score - float(row["source_index"]) * 0.001, row))
    if not candidates:
        return ""
    return sorted(candidates, key=lambda item: (-item[0], item[1]["source_index"]))[0][1]["label"]


def _visual_label_has_negated_value(label: str, *, include_negative_values: bool = False) -> bool:
    tokens = label.lower().split()
    if not tokens:
        return False
    negative_tokens = _VISUAL_NEGATIVE_VALUE_WORDS if include_negative_values else _VISUAL_BASE_NEGATED_VALUE_WORDS
    component_indices = [index for index, token in enumerate(tokens) if token in _VISUAL_COMPONENT_WORDS]
    if not component_indices:
        return any(token in negative_tokens for token in tokens)
    value_tokens = tokens[component_indices[-1] + 1 :]
    return any(token in negative_tokens for token in value_tokens)


def _visual_label_has_extended_negative_value(label: str) -> bool:
    return _visual_label_has_negated_value(label, include_negative_values=True) and not _visual_label_has_negated_value(
        label,
        include_negative_values=False,
    )


def _visual_negated_component_label_phrases(label: str) -> tuple[str, ...]:
    label_lower = label.lower().strip()
    phrases = {label_lower}
    tokens = label_lower.split()
    component_indices = [index for index, token in enumerate(tokens) if token in _VISUAL_COMPONENT_WORDS]
    if component_indices:
        component_index = component_indices[-1]
        if component_index < len(tokens) - 1:
            base = " ".join(tokens[: component_index + 1])
            component = tokens[component_index]
            value = " ".join(tokens[component_index + 1 :])
            phrases.add(f"{value} {base}")
            phrases.add(f"{value} {component}")
    return tuple(sorted(phrases, key=lambda phrase: (-len(phrase), phrase)))


def _short_component_query_for_label(*, target_query: str, label: str) -> bool:
    target_tokens = target_query.lower().strip().split()
    label_tokens = label.lower().strip().split()
    if not target_tokens or not label_tokens:
        return False
    if len(target_tokens) >= len(label_tokens):
        return False
    if not all(token in label_tokens for token in target_tokens):
        return False
    return any(token in _VISUAL_COMPONENT_WORDS for token in target_tokens)


def _visual_semantic_target_preservation_row(
    *,
    call: ToolCall,
    case: ToolDirectiveProbeCase,
    prompt_state_label: str,
) -> dict[str, Any] | None:
    if call.name != "extract_layout":
        return None
    target_query = str(call.arguments.get("target_query", "")).strip()
    if not target_query or target_query != prompt_state_label:
        return None

    legacy_label = _visual_target_label_from_state(case, preserve_semantic_targets=False)
    semantic_label = _visual_semantic_target_label_from_state(case)
    if not semantic_label or legacy_label == semantic_label:
        return None

    return {
        "from_tool": call.name,
        "from_arguments": call.arguments,
        "to_tool": call.name,
        "to_arguments": call.arguments,
        "prompt_state_label": semantic_label,
        "preserved_target_query": target_query,
        "blocked_label": legacy_label,
        "reason": "semantic_label_preserved_over_stale_context",
    }


def _visual_semantic_label_phrases(label: str) -> tuple[str, ...]:
    label_lower = label.lower().strip()
    phrases = {label_lower}
    tokens = label_lower.split()
    component_index = next((index for index, token in enumerate(tokens) if token in _VISUAL_COMPONENT_WORDS), -1)
    if component_index > 0 and component_index < len(tokens) - 1:
        base = " ".join(tokens[: component_index + 1])
        component = tokens[component_index]
        value = " ".join(tokens[component_index + 1 :])
        phrases.add(f"{value} {base}")
        phrases.add(f"{value} {component}")
    return tuple(sorted(phrases, key=lambda phrase: (-len(phrase), phrase)))


def _visual_semantic_positive_score(*, user_text: str, phrases: tuple[str, ...]) -> float:
    best = 0.0
    for phrase in phrases:
        phrase_pattern = re.escape(phrase)
        direct_pattern = re.compile(
            rf"\b(use|select|locate|find|target|inspect|read)\s+"
            rf"(?:the\s+)?(?:current\s+|visible\s+|actual\s+|requested\s+)?{phrase_pattern}\b"
        )
        if direct_pattern.search(user_text):
            best = max(best, 8.0 + len(phrase.split()) * 0.25)
        target_pattern = re.compile(
            rf"\b(current target|target|actual component|visible component|layout label|field label)\s+"
            rf"(?:is|is still|remains)\s+(?:the\s+)?{phrase_pattern}\b"
        )
        if target_pattern.search(user_text):
            best = max(best, 7.0 + len(phrase.split()) * 0.25)
    return best


def _visual_semantic_label_deprioritized(*, user_text: str, phrases: tuple[str, ...]) -> bool:
    prefixes = (
        "do not use",
        "do not select",
        "do not locate",
        "do not find",
        "do not target",
        "not",
        "ignore",
        "avoid",
    )
    for phrase in phrases:
        phrase_pattern = re.escape(phrase)
        for prefix in prefixes:
            pattern = re.compile(rf"\b{re.escape(prefix)}\s+(?:the\s+)?{phrase_pattern}\b")
            for match in pattern.finditer(user_text):
                if not _visual_negative_match_is_stale_context(user_text=user_text, start=match.start(), end=match.end()):
                    return True
    return False


def _visual_negative_match_is_stale_context(*, user_text: str, start: int, end: int) -> bool:
    window = user_text[max(0, start - 96) : min(len(user_text), end + 140)]
    stale_markers = (
        "annotation saying",
        "belongs to an old",
        "caption belongs",
        "caption quotes",
        "caption says",
        "example note says",
        "note says",
        "old example",
        "old screenshot",
        "prior screenshot",
        "previous screenshot",
        "quoted",
        "quotes",
        "stale caption",
        "stale example",
        "training note",
    )
    return any(marker in window for marker in stale_markers)


def _visual_target_label_from_state(
    case: ToolDirectiveProbeCase,
    *,
    preserve_semantic_targets: bool = False,
) -> str:
    if preserve_semantic_targets:
        semantic_label = _visual_semantic_target_label_from_state(case)
        if semantic_label:
            return semantic_label

    user_text = " ".join(message.content for message in case.messages if message.role == "user").lower()
    labels = _visual_layout_labels(case.initial_state)
    if not labels:
        return ""
    matching_labels = [label for label in labels if label.lower() in user_text]
    scored_matches = [
        (
            _visual_prompt_label_score(
                label=label,
                user_text=user_text,
                component_words=_VISUAL_COMPONENT_WORDS,
                source_index=labels.index(label),
            ),
            label,
        )
        for label in matching_labels
    ]
    scored_matches = [item for item in scored_matches if item[0] > 0]
    if scored_matches:
        return sorted(scored_matches, key=lambda item: (-item[0], labels.index(item[1])))[0][1]
    if matching_labels:
        return sorted(matching_labels, key=lambda label: (-len(label), labels.index(label)))[0]
    return ""


def _visual_prompt_label_score(
    *,
    label: str,
    user_text: str,
    component_words: set[str],
    source_index: int,
) -> float:
    label_lower = label.lower()
    if label_lower not in user_text:
        return 0.0

    component = label_lower.split()[-1]
    score = 1.0 + (0.5 if component in component_words else 0.0)
    if re.search(r"\b[a-z]\d+\b", label_lower):
        score += 2.0

    for phrase in (
        "locate",
        "select",
        "find",
        "identify",
        "target is",
        "actual component is",
        "visible component is",
        "layout label is",
        "field label is",
    ):
        start = user_text.find(phrase)
        while start != -1:
            window = user_text[start : start + 140]
            offset = window.find(label_lower)
            if offset != -1:
                score += max(3.0, 10.0 - (offset / 20.0))
            start = user_text.find(phrase, start + 1)

    negative_fragments = (
        f"do not target {label_lower}",
        f"do not target the {label_lower}",
        f"do not target the {component}",
        f"do not use {label_lower}",
        f"do not use that {component}",
        f"not the {label_lower}",
        f"not any {label_lower}",
        f"not the {component}",
        f"before reading {label_lower}",
        f"before reading the {label_lower}",
        f"before reading the {component}",
    )
    if any(fragment in user_text for fragment in negative_fragments):
        score -= 8.0

    return score - (source_index * 0.001)


def _visual_layout_labels(initial_state: dict[str, Any]) -> list[str]:
    return [row["label"] for row in _visual_layout_rows(initial_state) if row["label"]]


def _visual_layout_rows(initial_state: dict[str, Any]) -> list[dict[str, Any]]:
    images = initial_state.get("images", {})
    if not isinstance(images, dict):
        return []
    rows_out: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    source_index = 0
    for image in images.values():
        if not isinstance(image, dict):
            continue
        for key in ("local_layouts", "layouts"):
            rows = image.get(key, [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                label = str(row.get("label", "")).strip()
                if not label or label in seen_labels:
                    continue
                seen_labels.add(label)
                rows_out.append(
                    {
                        "label": label,
                        "text": str(row.get("text", "")).strip(),
                        "region_id": str(row.get("region_id", "")).strip(),
                        "source_index": source_index,
                    }
                )
                source_index += 1
    return rows_out


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
    if case.expected_execution.get("no_tool_call") is True:
        if not actual_calls:
            return True, []
        if not case.initial_state:
            return False, []
        execution = _execute_calls(case.initial_state, tool_specs, actual_calls)
        return False, execution
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
    actual_calls: list[ToolCall],
    actual_execution: list[dict[str, Any]],
) -> bool | None:
    if case.expected_execution.get("no_tool_call") is True:
        return not actual_calls
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
    if not expected_calls:
        return "unexpected_tool_call"
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
