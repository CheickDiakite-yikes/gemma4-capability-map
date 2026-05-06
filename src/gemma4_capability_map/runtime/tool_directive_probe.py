from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_capability_map.models.gemma4_runner import Gemma4Runner
from gemma4_capability_map.reporting.knowledge_work_board import DEFAULT_REGISTRY_PATH, load_model_registry
from gemma4_capability_map.schemas import Message, ModelTurn, ToolCall, ToolSpec
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

    runner = Gemma4Runner(
        model_id=str(meta.get("reasoner") or "google/gemma-4-E2B-it"),
        backend=str(meta.get("backend") or "heuristic"),
        max_new_tokens=int(meta.get("reasoner_max_new_tokens", 64) or 64),
        request_timeout_seconds=float(meta.get("request_timeout_seconds", 300.0) or 300.0),
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


def _score_probe_case(
    case: ToolDirectiveProbeCase,
    tool_specs: list[ToolSpec],
    expected_calls: list[ToolCall],
    turn: ModelTurn,
) -> dict[str, Any]:
    del tool_specs
    exact_match = _calls_exact_match(turn.normalized_tool_call, expected_calls)
    expected_payload = [{"name": call.name, "arguments": call.arguments} for call in expected_calls]
    actual_payload = [{"name": call.name, "arguments": call.arguments} for call in turn.normalized_tool_call]
    return {
        "case_id": case.case_id,
        "family": case.family,
        "expected_call_count": len(expected_calls),
        "actual_call_count": len(turn.normalized_tool_call),
        "exact_match": exact_match,
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
    by_family: dict[str, dict[str, Any]] = {}
    for row in rows:
        family = str(row["family"])
        bucket = by_family.setdefault(family, {"cases": 0, "exact": 0})
        bucket["cases"] += 1
        if row["exact_match"]:
            bucket["exact"] += 1
    for bucket in by_family.values():
        cases = int(bucket["cases"])
        bucket["exact_rate"] = bucket["exact"] / cases if cases else 0.0
    return {
        "case_count": total,
        "exact_match_count": exact,
        "exact_match_rate": exact / total if total else 0.0,
        "family_summary": by_family,
    }


def _calls_exact_match(actual_calls: list[ToolCall], expected_calls: list[ToolCall]) -> bool:
    if len(actual_calls) != len(expected_calls):
        return False
    return all(
        actual.name == expected.name and actual.arguments == expected.arguments
        for actual, expected in zip(actual_calls, expected_calls, strict=False)
    )


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
        "expected_calls",
        "actual_calls",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in fieldnames},
                    "expected_calls": json.dumps(row["expected_calls"], ensure_ascii=False),
                    "actual_calls": json.dumps(row["actual_calls"], ensure_ascii=False),
                }
            )
