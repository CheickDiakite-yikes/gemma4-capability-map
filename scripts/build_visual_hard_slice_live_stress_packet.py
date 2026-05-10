from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_capability_map.reporting.knowledge_work_board import DEFAULT_REGISTRY_PATH
from gemma4_capability_map.runtime.tool_directive_probe import ToolDirectiveProbeCase
from gemma4_capability_map.schemas import Message
from gemma4_capability_map.tools.planner import plan_tool_calls
from gemma4_capability_map.tools.registry import build_default_registry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "tool_probe_replay_packets"
DEFAULT_REPLAY_SYSTEM_ID = "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints"
DEFAULT_SOURCE_SYSTEM_ID = "designed_visual_hard_slice_live_stress_v1"
DEFAULT_BASELINE_SYSTEM_ID = "mlx_gemma4_e2b_reasoner_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a harder replay-shaped visual live packet around executor-alias and stale-selection cases."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-group-id", default=None)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--replay-system-id", default=DEFAULT_REPLAY_SYSTEM_ID)
    parser.add_argument(
        "--suite",
        choices=[
            "v1",
            "alias_repeat_v2",
            "alias_transfer_v3",
            "alias_transfer_repeat_v4",
            "alias_transfer_oblique_v5",
            "alias_transfer_post_repair_v6",
            "alias_transfer_residual_v7",
            "component_value_v9",
            "h1o_control_factorial_v10",
            "h1p_component_value_holdout_v11",
            "h1r_component_label_residual_v12",
            "h1w_residual_overlap_v13",
            "h1x_v11_breaker_v14",
        ],
        default="v1",
    )
    parser.add_argument("--case-id", action="append", dest="case_ids", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet = build_visual_hard_slice_live_stress_packet(
        output_root=Path(args.output_root),
        run_group_id=args.run_group_id,
        registry_path=Path(args.registry),
        replay_system_id=args.replay_system_id,
        suite=args.suite,
        case_ids=args.case_ids,
    )
    print(json.dumps(packet["summary"], indent=2, ensure_ascii=False))


def build_visual_hard_slice_live_stress_packet(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_group_id: str | None = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    replay_system_id: str = DEFAULT_REPLAY_SYSTEM_ID,
    suite: str = "v1",
    case_ids: list[str] | None = None,
) -> dict[str, Any]:
    cases = _stress_cases_for_suite(suite)
    cases_by_id = {case.case_id: case for case in cases}
    selected_ids = case_ids or [case.case_id for case in cases]
    missing = [case_id for case_id in selected_ids if case_id not in cases_by_id]
    if missing:
        raise ValueError(f"Unknown visual hard-slice stress case id(s): {', '.join(missing)}")
    selected_cases = [cases_by_id[case_id] for case_id in selected_ids]

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    packet_run_id = run_group_id or f"{timestamp}_visual_hard_slice_live_stress_packet"
    packet_dir = output_root / packet_run_id
    case_dir = packet_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    registry = build_default_registry().specs
    replay_cases: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    for case in selected_cases:
        tool_specs = [registry[name] for name in case.tool_names]
        expected_calls = _expected_call_payloads(case=case, tool_specs=tool_specs, suite=suite)
        replay_case = {
            "case_id": case.case_id,
            "family": case.family,
            "messages": [message.model_dump(mode="json") for message in case.messages],
            "media": list(case.media),
            "tool_names": list(case.tool_names),
            "tool_specs": [tool.model_dump(mode="json", by_alias=True) for tool in tool_specs],
            "initial_state": case.initial_state,
            "expected_execution": case.expected_execution,
            "expected_calls": expected_calls,
            "source_system_id": DEFAULT_SOURCE_SYSTEM_ID,
            "source_failure_mode": _stress_failure_mode(case.family),
            "source_exact_match": False,
            "source_executable_match": False,
            "baseline_system_id": DEFAULT_BASELINE_SYSTEM_ID,
            "baseline_exact_match": "",
            "live_entrypoint_status": "visual_hard_slice_live_stress_packet_v1",
            "live_entrypoint_note": (
                "Designed replay-shaped stress cases for moonie-agent replay-live; no model evidence is claimed until executed."
            ),
        }
        replay_cases.append(replay_case)
        case_path = case_dir / f"{case.case_id}.json"
        _write_json(case_path, replay_case)
        commands.append(
            {
                "case_id": case.case_id,
                "family": case.family,
                "command": _live_replay_command(
                    packet_dir=packet_dir,
                    registry_path=registry_path,
                    replay_system_id=replay_system_id,
                    case_id=case.case_id,
                ),
            }
        )
        rows.append(
            {
                "case_id": case.case_id,
                "family": case.family,
                "source_failure_mode": _stress_failure_mode(case.family),
                "source_exact_match": False,
                "source_executable_match": False,
                "baseline_exact_match": "",
                "expected_call_count": len(expected_calls),
                "source_actual_call_count": "",
                "case_path": str(case_path.resolve()),
            }
        )

    summary = {
        "packet_run_id": packet_run_id,
        "packet_dir": str(packet_dir.resolve()),
        "source_system_id": DEFAULT_SOURCE_SYSTEM_ID,
        "baseline_system_id": DEFAULT_BASELINE_SYSTEM_ID,
        "replay_system_id": replay_system_id,
        "suite": suite,
        "case_count": len(rows),
        "family_counts": _count_by(rows, "family"),
        "failure_mode_counts": _count_by(rows, "source_failure_mode"),
        "dry_run": True,
        "executed_count": 0,
    }
    manifest = {
        **summary,
        "created_at": datetime.now(UTC).isoformat(),
        "registry_path": str(registry_path.resolve()),
        "case_ids": [row["case_id"] for row in rows],
        "operator_surface": "rich_cli_visual_hard_slice_live_stress_v1",
        "entrypoint": "moonie-agent replay-live",
        "suite": suite,
        "purpose": (
            "Repeat the visual hard-slice executor-alias and stale-selection mechanisms under fresh decoys before "
            "spending packaged H1 workflow budget."
        ),
    }
    _write_json(packet_dir / "manifest.json", manifest)
    _write_json(packet_dir / "summary.json", summary)
    _write_json(packet_dir / "commands.json", commands)
    _write_json(packet_dir / "replay_cases.json", replay_cases)
    _write_json(packet_dir / "replay_results.json", [])
    _write_csv(packet_dir / "replay_cases.csv", rows)
    _write_csv(packet_dir / "replay_results.csv", [])
    return {
        "packet_dir": str(packet_dir.resolve()),
        "summary": summary,
        "manifest": manifest,
        "rows": rows,
        "commands": commands,
        "replay_cases": replay_cases,
    }


def _stress_cases_for_suite(suite: str) -> list[ToolDirectiveProbeCase]:
    if suite == "v1":
        return _stress_cases()
    if suite == "alias_repeat_v2":
        return [*_stress_cases(), *_alias_repeat_cases_v2()]
    if suite == "alias_transfer_v3":
        return _alias_transfer_cases_v3()
    if suite == "alias_transfer_repeat_v4":
        return _alias_transfer_repeat_cases_v4()
    if suite == "alias_transfer_oblique_v5":
        return _alias_transfer_oblique_cases_v5()
    if suite == "alias_transfer_post_repair_v6":
        return _alias_transfer_post_repair_cases_v6()
    if suite == "alias_transfer_residual_v7":
        return _alias_transfer_residual_cases_v7()
    if suite == "component_value_v9":
        return _component_value_cases_v9()
    if suite == "h1o_control_factorial_v10":
        return _h1o_control_factorial_cases_v10()
    if suite == "h1p_component_value_holdout_v11":
        return _h1p_component_value_holdout_cases_v11()
    if suite == "h1r_component_label_residual_v12":
        return _h1r_component_label_residual_cases_v12()
    if suite == "h1w_residual_overlap_v13":
        return _h1w_residual_overlap_cases_v13()
    if suite == "h1x_v11_breaker_v14":
        return _h1x_v11_breaker_cases_v14()
    raise ValueError(f"Unknown visual live stress suite: {suite}")


def _expected_call_payloads(
    *,
    case: ToolDirectiveProbeCase,
    tool_specs: list[Any],
    suite: str,
) -> list[dict[str, Any]]:
    if suite in {
        "alias_transfer_v3",
        "alias_transfer_repeat_v4",
        "alias_transfer_oblique_v5",
        "alias_transfer_post_repair_v6",
        "alias_transfer_residual_v7",
        "component_value_v9",
        "h1o_control_factorial_v10",
        "h1p_component_value_holdout_v11",
        "h1r_component_label_residual_v12",
        "h1w_residual_overlap_v13",
        "h1x_v11_breaker_v14",
    }:
        return [_oracle_visual_extract_call(case)]
    return [
        {"name": call.name, "arguments": call.arguments}
        for call in plan_tool_calls(case.messages, case.media, tool_specs)
    ]


def _oracle_visual_extract_call(case: ToolDirectiveProbeCase) -> dict[str, Any]:
    target_region_id = _target_region_id(case.expected_execution)
    if not target_region_id:
        raise ValueError(f"Alias-transfer case {case.case_id} is missing an expected target region id.")
    image_id, region = _target_region(case.initial_state, target_region_id)
    target_query = str(region.get("label", "")).strip()
    if not target_query:
        raise ValueError(f"Alias-transfer case {case.case_id} target region {target_region_id} is missing a label.")
    return {
        "name": "extract_layout",
        "arguments": {
            "image_id": image_id,
            "target_query": target_query,
        },
    }


def _target_region_id(expected_execution: dict[str, Any]) -> str:
    region_ids = expected_execution.get("region_ids")
    if isinstance(region_ids, list) and region_ids:
        return str(region_ids[0])
    region_id = expected_execution.get("region_id")
    return str(region_id) if region_id is not None else ""


def _target_region(initial_state: dict[str, Any], target_region_id: str) -> tuple[str, dict[str, Any]]:
    images = initial_state.get("images", {})
    if not isinstance(images, dict):
        raise ValueError("Alias-transfer initial state is missing images.")
    for image_id, image in images.items():
        if not isinstance(image, dict):
            continue
        for key in ("local_layouts", "layouts"):
            layouts = image.get(key, [])
            if not isinstance(layouts, list):
                continue
            for region in layouts:
                if isinstance(region, dict) and str(region.get("region_id", "")) == target_region_id:
                    return str(image_id), region
    raise ValueError(f"Target region {target_region_id} not found in alias-transfer initial state.")


def _stress_cases() -> list[ToolDirectiveProbeCase]:
    return [
        ToolDirectiveProbeCase(
            case_id="stress_metric_panel_with_chart_table_decoys",
            family="visual_argument_copying_stress",
            messages=[
                Message(role="system", content="visual_image_ids: img-stress-metric-chart-table"),
                Message(
                    role="user",
                    content=(
                        "The dashboard includes a chart and an invoice table. Locate the dashboard metric panel "
                        "that needs review before reading either decoy."
                    ),
                ),
            ],
            media=["img-stress-metric-chart-table"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-stress-metric-chart-table",
                [
                    _region("stress-metric-2001", "dashboard metric", "Retention risk above threshold", area="metric panel"),
                    _region("stress-metric-2002", "trend chart", "Churn trend last 30 days", area="chart"),
                    _region("stress-metric-2003", "invoice totals table", "Q3 invoice totals", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["stress-metric-2001"]},
        ),
        ToolDirectiveProbeCase(
            case_id="stress_metric_panel_with_kpi_copy_decoy",
            family="visual_argument_copying_stress",
            messages=[
                Message(role="system", content="visual_image_ids: img-stress-kpi-copy"),
                Message(
                    role="user",
                    content=(
                        "The KPI copy says 'customer health'. Still locate the dashboard metric panel itself, not "
                        "the adjacent note or the table."
                    ),
                ),
            ],
            media=["img-stress-kpi-copy"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-stress-kpi-copy",
                [
                    _region("stress-kpi-2101", "dashboard metric", "Customer health requires review", area="metric panel"),
                    _region("stress-kpi-2102", "review note", "customer health", area="note"),
                    _region("stress-kpi-2103", "invoice totals table", "Open invoices", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["stress-kpi-2101"]},
        ),
        ToolDirectiveProbeCase(
            case_id="stress_form_error_stale_selection_status_decoy",
            family="visual_tool_routing_stress",
            messages=[
                Message(role="system", content="visual_image_ids: img-stress-form-status"),
                Message(
                    role="user",
                    content=(
                        "Do not refine old selection_id sel-archived. On the current form, locate the visible "
                        "validation error first and read that region."
                    ),
                ),
            ],
            media=["img-stress-form-status"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-stress-form-status",
                [
                    _region("stress-form-2201", "status message", "Profile saved successfully", source="system"),
                    _region("stress-form-2202", "validation error", "Routing code is missing", field="routing_code"),
                ],
            ),
            expected_execution={"region_ids": ["stress-form-2202"]},
        ),
        ToolDirectiveProbeCase(
            case_id="stress_form_error_stale_selection_warning_decoy",
            family="visual_tool_routing_stress",
            messages=[
                Message(role="system", content="visual_image_ids: img-stress-form-warning"),
                Message(
                    role="user",
                    content=(
                        "Ignore previous selection_id latest from another screenshot. On this form, locate the "
                        "visible validation error, not the yellow warning banner."
                    ),
                ),
            ],
            media=["img-stress-form-warning"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-stress-form-warning",
                [
                    _region("stress-form-2301", "warning banner", "Optional field incomplete", tone="warning"),
                    _region("stress-form-2302", "validation error", "SSN field failed checksum", field="ssn"),
                ],
            ),
            expected_execution={"region_ids": ["stress-form-2302"]},
        ),
    ]


def _alias_repeat_cases_v2() -> list[ToolDirectiveProbeCase]:
    return [
        ToolDirectiveProbeCase(
            case_id="stress_metric_panel_status_banner_decoy",
            family="visual_argument_copying_stress",
            messages=[
                Message(role="system", content="visual_image_ids: img-stress-metric-status"),
                Message(
                    role="user",
                    content=(
                        "The dashboard status banner mentions the same customer, but locate the dashboard metric "
                        "panel before inspecting the banner or table."
                    ),
                ),
            ],
            media=["img-stress-metric-status"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-stress-metric-status",
                [
                    _region("stress-metric-2401", "dashboard metric", "Escalation rate above limit", area="metric panel"),
                    _region("stress-metric-2402", "status banner", "Customer renewal review", area="banner"),
                    _region("stress-metric-2403", "invoice totals table", "Renewal invoices", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["stress-metric-2401"]},
        ),
        ToolDirectiveProbeCase(
            case_id="stress_metric_panel_summary_card_decoy",
            family="visual_argument_copying_stress",
            messages=[
                Message(role="system", content="visual_image_ids: img-stress-summary-card"),
                Message(
                    role="user",
                    content=(
                        "There is a summary card and a table nearby. Locate the dashboard metric panel itself "
                        "before reading the summary card."
                    ),
                ),
            ],
            media=["img-stress-summary-card"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-stress-summary-card",
                [
                    _region("stress-summary-2501", "dashboard metric", "Response SLA below target", area="metric panel"),
                    _region("stress-summary-2502", "summary card", "SLA review owner listed", area="card"),
                    _region("stress-summary-2503", "invoice totals table", "Open invoice exceptions", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["stress-summary-2501"]},
        ),
        ToolDirectiveProbeCase(
            case_id="stress_callout_warning_person_table_decoy",
            family="visual_argument_copying_stress",
            messages=[
                Message(role="system", content="visual_image_ids: img-stress-callout-person"),
                Message(
                    role="user",
                    content=(
                        "Morgan is named in the table. Locate the slide callout warning first and do not use "
                        "the table or person note as the target."
                    ),
                ),
            ],
            media=["img-stress-callout-person"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-stress-callout-person",
                [
                    _region("stress-callout-2601", "assignee note", "Morgan owns follow-up", person="Morgan"),
                    _region("stress-callout-2602", "slide callout", "Warning: launch approval is missing", tone="warning"),
                    _region("stress-callout-2603", "invoice totals table", "Launch invoices", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["stress-callout-2602"]},
        ),
        ToolDirectiveProbeCase(
            case_id="stress_callout_warning_risk_note_decoy",
            family="visual_argument_copying_stress",
            messages=[
                Message(role="system", content="visual_image_ids: img-stress-callout-risk"),
                Message(
                    role="user",
                    content=(
                        "The note says risk is high, but the target is the slide callout warning. Locate the "
                        "callout before reading any risk note."
                    ),
                ),
            ],
            media=["img-stress-callout-risk"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-stress-callout-risk",
                [
                    _region("stress-callout-2701", "risk note", "Risk is high for this account", tone="risk"),
                    _region("stress-callout-2702", "slide callout", "Warning: data freshness check failed", tone="warning"),
                    _region("stress-callout-2703", "dashboard metric", "Freshness below target", area="metric panel"),
                ],
            ),
            expected_execution={"region_ids": ["stress-callout-2702"]},
        ),
    ]


def _alias_transfer_cases_v3() -> list[ToolDirectiveProbeCase]:
    return [
        ToolDirectiveProbeCase(
            case_id="transfer_review_tile_notice_table_decoy",
            family="visual_argument_transfer",
            messages=[
                Message(role="system", content="visual_image_ids: img-transfer-review-tile"),
                Message(
                    role="user",
                    content=(
                        "The notice and the table both mention renewal risk. Locate the review tile itself first, "
                        "then read that region."
                    ),
                ),
            ],
            media=["img-transfer-review-tile"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-transfer-review-tile",
                [
                    _region("transfer-tile-3001", "review tile", "Renewal risk review required", area="tile"),
                    _region("transfer-tile-3002", "notice banner", "Renewal risk owner missing", area="banner"),
                    _region("transfer-tile-3003", "invoice table", "Renewal invoice exceptions", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["transfer-tile-3001"]},
        ),
        ToolDirectiveProbeCase(
            case_id="transfer_status_pill_chart_decoy",
            family="visual_argument_transfer",
            messages=[
                Message(role="system", content="visual_image_ids: img-transfer-status-pill"),
                Message(
                    role="user",
                    content=(
                        "Do not read the chart annotation. Locate the small status pill that marks the customer "
                        "as blocked."
                    ),
                ),
            ],
            media=["img-transfer-status-pill"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-transfer-status-pill",
                [
                    _region("transfer-pill-3101", "status pill", "Blocked", area="status pill"),
                    _region("transfer-pill-3102", "chart annotation", "Blocked accounts rising", area="chart"),
                    _region("transfer-pill-3103", "customer table", "Blocked account owner", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["transfer-pill-3101"]},
        ),
        ToolDirectiveProbeCase(
            case_id="transfer_error_banner_note_decoy",
            family="visual_argument_transfer",
            messages=[
                Message(role="system", content="visual_image_ids: img-transfer-error-banner"),
                Message(
                    role="user",
                    content=(
                        "A note repeats the error text, but the target is the visible error banner. Locate the "
                        "banner before reading anything else."
                    ),
                ),
            ],
            media=["img-transfer-error-banner"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-transfer-error-banner",
                [
                    _region("transfer-banner-3201", "support note", "Export failed for the workbook", area="note"),
                    _region("transfer-banner-3202", "error banner", "Export failed: missing approver", tone="error"),
                    _region("transfer-banner-3203", "settings table", "Approver routing", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["transfer-banner-3202"]},
        ),
        ToolDirectiveProbeCase(
            case_id="transfer_queue_badge_person_decoy",
            family="visual_argument_transfer",
            messages=[
                Message(role="system", content="visual_image_ids: img-transfer-queue-badge"),
                Message(
                    role="user",
                    content=(
                        "Taylor appears in the assignee note. Locate the queue badge, not Taylor's note, and read "
                        "the badge region."
                    ),
                ),
            ],
            media=["img-transfer-queue-badge"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-transfer-queue-badge",
                [
                    _region("transfer-queue-3301", "assignee note", "Taylor owns queue triage", person="Taylor"),
                    _region("transfer-queue-3302", "queue badge", "12 blocked items", area="badge"),
                    _region("transfer-queue-3303", "queue table", "Blocked item list", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["transfer-queue-3302"]},
        ),
        ToolDirectiveProbeCase(
            case_id="transfer_form_error_old_selection_chip_decoy",
            family="visual_tool_routing_transfer",
            messages=[
                Message(role="system", content="visual_image_ids: img-transfer-form-chip"),
                Message(
                    role="user",
                    content=(
                        "Ignore old selection_id sel-prior-chip from the prior screen. On the current form, locate "
                        "the visible validation error, not the saved status chip."
                    ),
                ),
            ],
            media=["img-transfer-form-chip"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-transfer-form-chip",
                [
                    _region("transfer-form-3401", "status chip", "Saved", state="saved"),
                    _region("transfer-form-3402", "validation error", "Tax ID has invalid length", field="tax_id"),
                ],
            ),
            expected_execution={"region_ids": ["transfer-form-3402"]},
        ),
        ToolDirectiveProbeCase(
            case_id="transfer_signature_warning_checkbox_decoy",
            family="visual_tool_routing_transfer",
            messages=[
                Message(role="system", content="visual_image_ids: img-transfer-signature-warning"),
                Message(
                    role="user",
                    content=(
                        "The checkbox is selected, but the actionable target is the signature warning text. Locate "
                        "the warning before reading the checkbox."
                    ),
                ),
            ],
            media=["img-transfer-signature-warning"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-transfer-signature-warning",
                [
                    _region("transfer-signature-3501", "checkbox", "I confirm authorization", checked=True),
                    _region("transfer-signature-3502", "signature warning", "Signature missing for approval", tone="warning"),
                    _region("transfer-signature-3503", "approval table", "Approver list", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["transfer-signature-3502"]},
        ),
    ]


def _alias_transfer_repeat_cases_v4() -> list[ToolDirectiveProbeCase]:
    return [
        ToolDirectiveProbeCase(
            case_id="transfer_repeat_audit_card_email_decoy",
            family="visual_argument_transfer_repeat",
            messages=[
                Message(role="system", content="visual_image_ids: img-transfer-repeat-audit-card"),
                Message(
                    role="user",
                    content=(
                        "The email preview repeats the audit language. Locate the audit card itself before reading "
                        "any email or table decoy."
                    ),
                ),
            ],
            media=["img-transfer-repeat-audit-card"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-transfer-repeat-audit-card",
                [
                    _region("transfer-repeat-a-4001", "audit card", "Audit review overdue", area="card"),
                    _region("transfer-repeat-a-4002", "email preview", "Audit review requested", area="email"),
                    _region("transfer-repeat-a-4003", "audit table", "Review owner list", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["transfer-repeat-a-4001"]},
        ),
        ToolDirectiveProbeCase(
            case_id="transfer_repeat_priority_tag_chart_decoy",
            family="visual_argument_transfer_repeat",
            messages=[
                Message(role="system", content="visual_image_ids: img-transfer-repeat-priority-tag"),
                Message(
                    role="user",
                    content=(
                        "The chart title says urgent, but the target is the compact priority tag. Locate the tag "
                        "and read that region."
                    ),
                ),
            ],
            media=["img-transfer-repeat-priority-tag"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-transfer-repeat-priority-tag",
                [
                    _region("transfer-repeat-priority-4101", "priority tag", "High priority", area="tag"),
                    _region("transfer-repeat-priority-4102", "chart title", "Urgent cases by week", area="chart"),
                    _region("transfer-repeat-priority-4103", "case table", "Priority queue", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["transfer-repeat-priority-4101"]},
        ),
        ToolDirectiveProbeCase(
            case_id="transfer_repeat_warning_toast_note_decoy",
            family="visual_argument_transfer_repeat",
            messages=[
                Message(role="system", content="visual_image_ids: img-transfer-repeat-warning-toast"),
                Message(
                    role="user",
                    content=(
                        "A note contains similar warning text. Locate the warning toast itself, then read the toast."
                    ),
                ),
            ],
            media=["img-transfer-repeat-warning-toast"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-transfer-repeat-warning-toast",
                [
                    _region("transfer-repeat-toast-4201", "operator note", "Missing approval noted", area="note"),
                    _region("transfer-repeat-toast-4202", "warning toast", "Approval missing before submit", tone="warning"),
                    _region("transfer-repeat-toast-4203", "approval table", "Pending approvers", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["transfer-repeat-toast-4202"]},
        ),
        ToolDirectiveProbeCase(
            case_id="transfer_repeat_latency_chip_person_decoy",
            family="visual_argument_transfer_repeat",
            messages=[
                Message(role="system", content="visual_image_ids: img-transfer-repeat-latency-chip"),
                Message(
                    role="user",
                    content=(
                        "Jordan is listed beside the metric, but the target is the latency chip. Locate the chip, "
                        "not Jordan's owner note."
                    ),
                ),
            ],
            media=["img-transfer-repeat-latency-chip"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-transfer-repeat-latency-chip",
                [
                    _region("transfer-repeat-latency-4301", "owner note", "Jordan owns latency follow-up", person="Jordan"),
                    _region("transfer-repeat-latency-4302", "latency chip", "SLA breach 18m", area="chip"),
                    _region("transfer-repeat-latency-4303", "latency table", "Incident timing", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["transfer-repeat-latency-4302"]},
        ),
        ToolDirectiveProbeCase(
            case_id="transfer_repeat_missing_field_old_selection_decoy",
            family="visual_tool_routing_transfer_repeat",
            messages=[
                Message(role="system", content="visual_image_ids: img-transfer-repeat-missing-field"),
                Message(
                    role="user",
                    content=(
                        "Ignore old selection_id sel-stale-owner from the last screen. On the current form, locate "
                        "the visible missing-field message."
                    ),
                ),
            ],
            media=["img-transfer-repeat-missing-field"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-transfer-repeat-missing-field",
                [
                    _region("transfer-repeat-field-4401", "owner chip", "Owner saved", state="saved"),
                    _region("transfer-repeat-field-4402", "missing field message", "Routing code is required", field="routing_code"),
                ],
            ),
            expected_execution={"region_ids": ["transfer-repeat-field-4402"]},
        ),
        ToolDirectiveProbeCase(
            case_id="transfer_repeat_consent_alert_toggle_decoy",
            family="visual_tool_routing_transfer_repeat",
            messages=[
                Message(role="system", content="visual_image_ids: img-transfer-repeat-consent-alert"),
                Message(
                    role="user",
                    content=(
                        "The consent toggle is on, but the target is the consent alert text. Locate the alert before "
                        "reading the toggle."
                    ),
                ),
            ],
            media=["img-transfer-repeat-consent-alert"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-transfer-repeat-consent-alert",
                [
                    _region("transfer-repeat-consent-4501", "consent toggle", "Consent enabled", checked=True),
                    _region("transfer-repeat-consent-4502", "consent alert", "Consent document expired", tone="alert"),
                    _region("transfer-repeat-consent-4503", "document table", "Consent document list", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["transfer-repeat-consent-4502"]},
        ),
    ]


def _alias_transfer_oblique_cases_v5() -> list[ToolDirectiveProbeCase]:
    return [
        ToolDirectiveProbeCase(
            case_id="transfer_oblique_node_q17_table_decoy",
            family="visual_argument_transfer_oblique",
            messages=[
                Message(role="system", content="visual_image_ids: img-transfer-oblique-node-q17"),
                Message(
                    role="user",
                    content=(
                        "The table mentions owner escalation too, but locate the visible node q17 region itself "
                        "before reading any table decoy."
                    ),
                ),
            ],
            media=["img-transfer-oblique-node-q17"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-transfer-oblique-node-q17",
                [
                    _region("transfer-oblique-node-5001", "node q17", "Owner escalation unresolved", area="node", state="unresolved"),
                    _region("transfer-oblique-node-5002", "owner table", "Owner escalation roster", area="table"),
                    _region("transfer-oblique-node-5003", "review note", "Escalation owner pending", area="note"),
                ],
            ),
            expected_execution={"region_ids": ["transfer-oblique-node-5001"]},
        ),
        ToolDirectiveProbeCase(
            case_id="transfer_oblique_badge_m88_chart_decoy",
            family="visual_argument_transfer_oblique",
            messages=[
                Message(role="system", content="visual_image_ids: img-transfer-oblique-badge-m88"),
                Message(
                    role="user",
                    content=(
                        "The chart title says blocked volume. Locate badge m88, not the chart title or the queue table."
                    ),
                ),
            ],
            media=["img-transfer-oblique-badge-m88"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-transfer-oblique-badge-m88",
                [
                    _region("transfer-oblique-badge-5101", "badge m88", "Blocked queue: 12", area="badge", state="blocked"),
                    _region("transfer-oblique-badge-5102", "chart title", "Blocked volume trend", area="chart"),
                    _region("transfer-oblique-badge-5103", "queue table", "Blocked item list", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["transfer-oblique-badge-5101"]},
        ),
        ToolDirectiveProbeCase(
            case_id="transfer_oblique_chip_z33_person_decoy",
            family="visual_argument_transfer_oblique",
            messages=[
                Message(role="system", content="visual_image_ids: img-transfer-oblique-chip-z33"),
                Message(
                    role="user",
                    content=(
                        "Riley's note repeats the latency warning. Locate chip z33 and read that chip region first."
                    ),
                ),
            ],
            media=["img-transfer-oblique-chip-z33"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-transfer-oblique-chip-z33",
                [
                    _region("transfer-oblique-chip-5201", "owner note", "Riley owns latency warning", person="Riley"),
                    _region("transfer-oblique-chip-5202", "chip z33", "Latency breach 18m", area="chip", tone="warning"),
                    _region("transfer-oblique-chip-5203", "latency table", "Incident timing", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["transfer-oblique-chip-5202"]},
        ),
        ToolDirectiveProbeCase(
            case_id="transfer_oblique_cell_r42_notice_decoy",
            family="visual_argument_transfer_oblique",
            messages=[
                Message(role="system", content="visual_image_ids: img-transfer-oblique-cell-r42"),
                Message(
                    role="user",
                    content=(
                        "The notice uses the same approval words. Locate cell r42, not the notice banner."
                    ),
                ),
            ],
            media=["img-transfer-oblique-cell-r42"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-transfer-oblique-cell-r42",
                [
                    _region("transfer-oblique-cell-5301", "notice banner", "Approval missing", area="banner"),
                    _region("transfer-oblique-cell-5302", "cell r42", "Approval owner blank", area="cell", state="missing"),
                    _region("transfer-oblique-cell-5303", "approval table", "Approver list", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["transfer-oblique-cell-5302"]},
        ),
        ToolDirectiveProbeCase(
            case_id="transfer_oblique_field_e19_old_selection_decoy",
            family="visual_tool_routing_transfer_oblique",
            messages=[
                Message(role="system", content="visual_image_ids: img-transfer-oblique-field-e19"),
                Message(
                    role="user",
                    content=(
                        "Ignore old selection_id sel-e19-archive from the last screen. On this form, locate field e19, "
                        "not the saved owner chip."
                    ),
                ),
            ],
            media=["img-transfer-oblique-field-e19"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-transfer-oblique-field-e19",
                [
                    _region("transfer-oblique-field-5401", "owner chip", "Owner saved", state="saved"),
                    _region("transfer-oblique-field-5402", "field e19", "Routing code is required", field="routing_code", state="missing"),
                ],
            ),
            expected_execution={"region_ids": ["transfer-oblique-field-5402"]},
        ),
        ToolDirectiveProbeCase(
            case_id="transfer_oblique_alert_p55_toggle_decoy",
            family="visual_tool_routing_transfer_oblique",
            messages=[
                Message(role="system", content="visual_image_ids: img-transfer-oblique-alert-p55"),
                Message(
                    role="user",
                    content=(
                        "The consent toggle is enabled, but locate alert p55 before reading the toggle."
                    ),
                ),
            ],
            media=["img-transfer-oblique-alert-p55"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-transfer-oblique-alert-p55",
                [
                    _region("transfer-oblique-alert-5501", "consent toggle", "Consent enabled", checked=True),
                    _region("transfer-oblique-alert-5502", "alert p55", "Consent document expired", tone="alert"),
                    _region("transfer-oblique-alert-5503", "document table", "Consent documents", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["transfer-oblique-alert-5502"]},
        ),
    ]


def _alias_transfer_post_repair_cases_v6() -> list[ToolDirectiveProbeCase]:
    return [
        ToolDirectiveProbeCase(
            case_id="post_repair_node_k21_chart_decoy",
            family="visual_argument_transfer_post_repair_code",
            messages=[
                Message(role="system", content="visual_image_ids: img-post-node-k21"),
                Message(
                    role="user",
                    content=(
                        "The chart repeats the same escalation words. Locate node k21 itself before reading the chart."
                    ),
                ),
            ],
            media=["img-post-node-k21"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-post-node-k21",
                [
                    _region("post-node-6001", "node k21", "Escalation queue stalled", area="node", state="stalled"),
                    _region("post-node-6002", "escalation chart", "Queue escalations rising", area="chart"),
                    _region("post-node-6003", "owner table", "Escalation owners", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["post-node-6001"]},
        ),
        ToolDirectiveProbeCase(
            case_id="post_repair_badge_t64_notice_decoy",
            family="visual_argument_transfer_post_repair_code",
            messages=[
                Message(role="system", content="visual_image_ids: img-post-badge-t64"),
                Message(
                    role="user",
                    content=(
                        "The notice banner also says blocked. Locate badge t64, not the notice banner or table."
                    ),
                ),
            ],
            media=["img-post-badge-t64"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-post-badge-t64",
                [
                    _region("post-badge-6101", "notice banner", "Blocked items need review", area="banner"),
                    _region("post-badge-6102", "badge t64", "Blocked: 7", area="badge"),
                    _region("post-badge-6103", "blocked table", "Blocked item list", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["post-badge-6102"]},
        ),
        ToolDirectiveProbeCase(
            case_id="post_repair_chip_l90_person_decoy",
            family="visual_argument_transfer_post_repair_code",
            messages=[
                Message(role="system", content="visual_image_ids: img-post-chip-l90"),
                Message(
                    role="user",
                    content=(
                        "Avery's note repeats the latency text. Locate chip l90, not Avery's owner note."
                    ),
                ),
            ],
            media=["img-post-chip-l90"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-post-chip-l90",
                [
                    _region("post-chip-6201", "owner note", "Avery owns latency issue", person="Avery"),
                    _region("post-chip-6202", "chip l90", "Latency breach 24m", area="chip", tone="warning"),
                    _region("post-chip-6203", "latency table", "Latency incidents", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["post-chip-6202"]},
        ),
        ToolDirectiveProbeCase(
            case_id="post_repair_field_b12_stale_selection_decoy",
            family="visual_tool_routing_transfer_post_repair",
            messages=[
                Message(role="system", content="visual_image_ids: img-post-field-b12"),
                Message(
                    role="user",
                    content=(
                        "Ignore old selection_id sel-b12-archive from the last screen. On this form, locate field b12, "
                        "not the saved owner chip."
                    ),
                ),
            ],
            media=["img-post-field-b12"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-post-field-b12",
                [
                    _region("post-field-6301", "owner chip", "Owner saved", state="saved"),
                    _region("post-field-6302", "field b12", "Cost center is required", field="cost_center", state="missing"),
                ],
            ),
            expected_execution={"region_ids": ["post-field-6302"]},
        ),
        ToolDirectiveProbeCase(
            case_id="post_repair_alert_c77_toggle_decoy",
            family="visual_tool_routing_transfer_post_repair",
            messages=[
                Message(role="system", content="visual_image_ids: img-post-alert-c77"),
                Message(
                    role="user",
                    content=(
                        "The access toggle is enabled, but locate alert c77 before reading the toggle or access table."
                    ),
                ),
            ],
            media=["img-post-alert-c77"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-post-alert-c77",
                [
                    _region("post-alert-6401", "access toggle", "Access enabled", checked=True),
                    _region("post-alert-6402", "alert c77", "Access review expired", tone="alert"),
                    _region("post-alert-6403", "access table", "Access review list", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["post-alert-6402"]},
        ),
        ToolDirectiveProbeCase(
            case_id="post_repair_status_pill_note_decoy",
            family="visual_argument_transfer_post_repair_noncode",
            messages=[
                Message(role="system", content="visual_image_ids: img-post-status-pill"),
                Message(
                    role="user",
                    content=(
                        "The owner note says blocked too. Locate the status pill itself, not the owner note."
                    ),
                ),
            ],
            media=["img-post-status-pill"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-post-status-pill",
                [
                    _region("post-status-6501", "owner note", "Blocked owner pending", area="note"),
                    _region("post-status-6502", "status pill", "Blocked", area="pill"),
                    _region("post-status-6503", "status table", "Blocked accounts", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["post-status-6502"]},
        ),
        ToolDirectiveProbeCase(
            case_id="post_repair_review_tile_table_decoy",
            family="visual_argument_transfer_post_repair_noncode",
            messages=[
                Message(role="system", content="visual_image_ids: img-post-review-tile"),
                Message(
                    role="user",
                    content=(
                        "The table repeats the renewal phrase. Locate the review tile first, not the table."
                    ),
                ),
            ],
            media=["img-post-review-tile"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-post-review-tile",
                [
                    _region("post-review-6601", "review tile", "Renewal risk review", area="tile"),
                    _region("post-review-6602", "renewal table", "Renewal risk items", area="table"),
                    _region("post-review-6603", "notice banner", "Renewal owner missing", area="banner"),
                ],
            ),
            expected_execution={"region_ids": ["post-review-6601"]},
        ),
        ToolDirectiveProbeCase(
            case_id="post_repair_warning_toast_email_decoy",
            family="visual_argument_transfer_post_repair_noncode",
            messages=[
                Message(role="system", content="visual_image_ids: img-post-warning-toast"),
                Message(
                    role="user",
                    content=(
                        "The email preview repeats the warning text. Locate the warning toast itself first."
                    ),
                ),
            ],
            media=["img-post-warning-toast"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-post-warning-toast",
                [
                    _region("post-email-6701", "email preview", "Approval warning sent", area="email"),
                    _region("post-toast-6702", "warning toast", "Approval missing before submit", tone="warning"),
                    _region("post-approval-6703", "approval table", "Approver list", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["post-toast-6702"]},
        ),
    ]


def _alias_transfer_residual_cases_v7() -> list[ToolDirectiveProbeCase]:
    return [
        ToolDirectiveProbeCase(
            case_id="residual_chip_n31_owner_note_decoy",
            family="visual_argument_transfer_residual_code",
            messages=[
                Message(role="system", content="visual_image_ids: img-residual-chip-n31"),
                Message(
                    role="user",
                    content=(
                        "Jordan's owner note repeats the queue delay. Locate chip n31 itself before reading the note."
                    ),
                ),
            ],
            media=["img-residual-chip-n31"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-residual-chip-n31",
                [
                    _region("residual-note-7001", "owner note", "Jordan owns queue delay", person="Jordan"),
                    _region("residual-chip-n31-7002", "chip n31", "Queue delay 18m", area="chip", tone="warning"),
                    _region("residual-queue-7003", "queue table", "Queue delay incidents", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["residual-chip-n31-7002"]},
        ),
        ToolDirectiveProbeCase(
            case_id="residual_chip_v82_chart_decoy",
            family="visual_argument_transfer_residual_code",
            messages=[
                Message(role="system", content="visual_image_ids: img-residual-chip-v82"),
                Message(
                    role="user",
                    content=(
                        "The chart repeats the retry language. Locate chip v82, not the chart or retry table."
                    ),
                ),
            ],
            media=["img-residual-chip-v82"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-residual-chip-v82",
                [
                    _region("residual-trend-7101", "retry chart", "Retries climbing", area="chart"),
                    _region("residual-chip-v82-7102", "chip v82", "Retry budget low", area="chip"),
                    _region("residual-retry-7103", "retry table", "Retry budget rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["residual-chip-v82-7102"]},
        ),
        ToolDirectiveProbeCase(
            case_id="residual_badge_q14_notice_decoy",
            family="visual_argument_transfer_residual_code",
            messages=[
                Message(role="system", content="visual_image_ids: img-residual-badge-q14"),
                Message(
                    role="user",
                    content=(
                        "The notice also says pending. Locate badge q14 directly before looking at the notice."
                    ),
                ),
            ],
            media=["img-residual-badge-q14"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-residual-badge-q14",
                [
                    _region("residual-message-7201", "notice banner", "Pending approvals changed", area="banner"),
                    _region("residual-badge-q14-7202", "badge q14", "Pending: 3", area="badge"),
                    _region("residual-approval-7203", "approval table", "Pending approval rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["residual-badge-q14-7202"]},
        ),
        ToolDirectiveProbeCase(
            case_id="residual_state_pill_note_decoy",
            family="visual_argument_transfer_residual_noncode",
            messages=[
                Message(role="system", content="visual_image_ids: img-residual-state-pill"),
                Message(
                    role="user",
                    content=(
                        "The note says on hold too. Locate the state pill itself, not the note."
                    ),
                ),
            ],
            media=["img-residual-state-pill"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-residual-state-pill",
                [
                    _region("residual-note-7301", "review note", "On hold until owner replies", area="note"),
                    _region("residual-state-pill-7302", "state pill", "On hold", area="pill"),
                    _region("residual-work-7303", "work table", "On hold tasks", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["residual-state-pill-7302"]},
        ),
        ToolDirectiveProbeCase(
            case_id="residual_phase_pill_ticket_decoy",
            family="visual_argument_transfer_residual_noncode",
            messages=[
                Message(role="system", content="visual_image_ids: img-residual-phase-pill"),
                Message(
                    role="user",
                    content=(
                        "The ticket summary repeats review. Locate the phase pill before reading the ticket."
                    ),
                ),
            ],
            media=["img-residual-phase-pill"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-residual-phase-pill",
                [
                    _region("residual-ticket-7401", "ticket summary", "Review requested by ops", area="ticket"),
                    _region("residual-phase-pill-7402", "phase pill", "Review", area="pill"),
                    _region("residual-board-7403", "review table", "Review queue", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["residual-phase-pill-7402"]},
        ),
        ToolDirectiveProbeCase(
            case_id="residual_notice_tile_email_decoy",
            family="visual_argument_transfer_residual_noncode",
            messages=[
                Message(role="system", content="visual_image_ids: img-residual-notice-tile"),
                Message(
                    role="user",
                    content=(
                        "The email preview repeats the renewal warning. Locate the notice tile first."
                    ),
                ),
            ],
            media=["img-residual-notice-tile"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-residual-notice-tile",
                [
                    _region("residual-email-7501", "email preview", "Renewal warning sent", area="email"),
                    _region("residual-notice-tile-7502", "notice tile", "Renewal warning", area="tile"),
                    _region("residual-renewal-7503", "renewal table", "Renewal warning rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["residual-notice-tile-7502"]},
        ),
        ToolDirectiveProbeCase(
            case_id="residual_field_m20_stale_selection_decoy",
            family="visual_tool_routing_transfer_residual",
            messages=[
                Message(role="system", content="visual_image_ids: img-residual-field-m20"),
                Message(
                    role="user",
                    content=(
                        "Ignore old selection_id sel-m20-old from the previous panel. On this screen, locate field m20, "
                        "not the saved owner chip."
                    ),
                ),
            ],
            media=["img-residual-field-m20"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-residual-field-m20",
                [
                    _region("residual-owner-7601", "owner chip", "Owner saved", state="saved"),
                    _region("residual-field-m20-7602", "field m20", "Department is required", field="department"),
                    _region("residual-form-7603", "form table", "Department mappings", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["residual-field-m20-7602"]},
        ),
        ToolDirectiveProbeCase(
            case_id="residual_alert_h73_toggle_decoy",
            family="visual_tool_routing_transfer_residual",
            messages=[
                Message(role="system", content="visual_image_ids: img-residual-alert-h73"),
                Message(
                    role="user",
                    content=(
                        "The compliance toggle is enabled, but locate alert h73 before reading the toggle."
                    ),
                ),
            ],
            media=["img-residual-alert-h73"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-residual-alert-h73",
                [
                    _region("residual-toggle-7701", "compliance toggle", "Compliance enabled", checked=True),
                    _region("residual-alert-h73-7702", "alert h73", "Compliance review overdue", tone="alert"),
                    _region("residual-compliance-7703", "compliance table", "Compliance review rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["residual-alert-h73-7702"]},
        ),
    ]


def _component_value_cases_v9() -> list[ToolDirectiveProbeCase]:
    return [
        ToolDirectiveProbeCase(
            case_id="component_value_state_pill_note_decoy",
            family="visual_argument_transfer_component_value_pill",
            messages=[
                Message(role="system", content="visual_image_ids: img-component-state-pill"),
                Message(
                    role="user",
                    content=(
                        "The note repeats on hold. Locate the state pill itself, not the note."
                    ),
                ),
            ],
            media=["img-component-state-pill"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-component-state-pill",
                [
                    _region("component-note-8001", "review note", "On hold until owner replies", area="note"),
                    _region("component-state-pill-8002", "state pill", "On hold", area="pill"),
                    _region("component-work-8003", "work table", "On hold tasks", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["component-state-pill-8002"]},
        ),
        ToolDirectiveProbeCase(
            case_id="component_value_status_badge_email_decoy",
            family="visual_argument_transfer_component_value_nonpill",
            messages=[
                Message(role="system", content="visual_image_ids: img-component-status-badge"),
                Message(
                    role="user",
                    content=(
                        "The email preview says blocked too. Locate the status badge itself, not the email."
                    ),
                ),
            ],
            media=["img-component-status-badge"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-component-status-badge",
                [
                    _region("component-email-8101", "email preview", "Blocked account warning sent", area="email"),
                    _region("component-status-badge-8102", "status badge", "Blocked", area="badge"),
                    _region("component-account-8103", "account table", "Blocked account rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["component-status-badge-8102"]},
        ),
        ToolDirectiveProbeCase(
            case_id="component_value_phase_tile_ticket_decoy",
            family="visual_argument_transfer_component_value_nonpill",
            messages=[
                Message(role="system", content="visual_image_ids: img-component-phase-tile"),
                Message(
                    role="user",
                    content=(
                        "The ticket summary repeats review. Locate the phase tile itself before reading the ticket."
                    ),
                ),
            ],
            media=["img-component-phase-tile"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-component-phase-tile",
                [
                    _region("component-ticket-8201", "ticket summary", "Review requested by support", area="ticket"),
                    _region("component-phase-tile-8202", "phase tile", "Review", area="tile"),
                    _region("component-review-8203", "review table", "Review queue", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["component-phase-tile-8202"]},
        ),
        ToolDirectiveProbeCase(
            case_id="component_value_priority_chip_table_decoy",
            family="visual_argument_transfer_component_value_nonpill",
            messages=[
                Message(role="system", content="visual_image_ids: img-component-priority-chip"),
                Message(
                    role="user",
                    content=(
                        "The table repeats high priority. Locate the priority chip itself, not the table row."
                    ),
                ),
            ],
            media=["img-component-priority-chip"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-component-priority-chip",
                [
                    _region("component-row-8301", "priority table", "High priority incidents", area="table"),
                    _region("component-priority-chip-8302", "priority chip", "High", area="chip"),
                    _region("component-note-8303", "priority note", "High priority owner listed", area="note"),
                ],
            ),
            expected_execution={"region_ids": ["component-priority-chip-8302"]},
        ),
        ToolDirectiveProbeCase(
            case_id="component_value_severity_pill_chart_decoy",
            family="visual_argument_transfer_component_value_pill",
            messages=[
                Message(role="system", content="visual_image_ids: img-component-severity-pill"),
                Message(
                    role="user",
                    content=(
                        "The chart repeats critical. Locate the severity pill itself before reading the chart."
                    ),
                ),
            ],
            media=["img-component-severity-pill"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-component-severity-pill",
                [
                    _region("component-chart-8401", "incident chart", "Critical incidents rising", area="chart"),
                    _region("component-severity-pill-8402", "severity pill", "Critical", area="pill"),
                    _region("component-incident-8403", "incident table", "Critical incident rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["component-severity-pill-8402"]},
        ),
        ToolDirectiveProbeCase(
            case_id="component_value_result_pill_log_decoy",
            family="visual_argument_transfer_component_value_pill",
            messages=[
                Message(role="system", content="visual_image_ids: img-component-result-pill"),
                Message(
                    role="user",
                    content=(
                        "The log repeats approved. Locate the result pill itself, not the log entry."
                    ),
                ),
            ],
            media=["img-component-result-pill"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-component-result-pill",
                [
                    _region("component-log-8501", "approval log", "Approved by Mina at 09:14", area="log"),
                    _region("component-result-pill-8502", "result pill", "Approved", area="pill"),
                    _region("component-review-8503", "approval table", "Approved request rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["component-result-pill-8502"]},
        ),
        ToolDirectiveProbeCase(
            case_id="component_value_risk_badge_stale_selection_decoy",
            family="visual_tool_routing_component_value",
            messages=[
                Message(role="system", content="visual_image_ids: img-component-risk-badge"),
                Message(
                    role="user",
                    content=(
                        "Ignore old selection_id sel-risk-old from the previous view. On this screen, locate the "
                        "risk badge itself, not the overdue summary."
                    ),
                ),
            ],
            media=["img-component-risk-badge"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-component-risk-badge",
                [
                    _region("component-summary-8601", "overdue summary", "Overdue vendors listed", area="summary"),
                    _region("component-risk-badge-8602", "risk badge", "Overdue", area="badge"),
                    _region("component-vendor-8603", "vendor table", "Overdue vendor rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["component-risk-badge-8602"]},
        ),
        ToolDirectiveProbeCase(
            case_id="component_value_owner_field_stale_selection_decoy",
            family="visual_tool_routing_component_value",
            messages=[
                Message(role="system", content="visual_image_ids: img-component-owner-field"),
                Message(
                    role="user",
                    content=(
                        "Ignore old selection_id sel-owner-previous from the prior panel. Locate the owner field "
                        "itself, not the note that repeats Mina."
                    ),
                ),
            ],
            media=["img-component-owner-field"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-component-owner-field",
                [
                    _region("component-note-8701", "owner note", "Mina owns renewal handoff", area="note"),
                    _region("component-owner-field-8702", "owner field", "Mina", area="field"),
                    _region("component-handoff-8703", "handoff table", "Mina handoff rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["component-owner-field-8702"]},
        ),
    ]


def _h1o_control_factorial_cases_v10() -> list[ToolDirectiveProbeCase]:
    return [
        ToolDirectiveProbeCase(
            case_id="h1o_activation_status_badge_email_decoy",
            family="h1o_activation_no_call",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1o-activation-status-badge"),
                Message(
                    role="user",
                    content=(
                        "The email repeats review twice, but on the current screenshot locate the status badge "
                        "itself before reading the email body."
                    ),
                ),
            ],
            media=["img-h1o-activation-status-badge"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1o-activation-status-badge",
                [
                    _region("h1o-email-9001", "email body", "Review requested by ops", area="email"),
                    _region("h1o-status-badge-9002", "status badge", "Review", area="badge"),
                    _region("h1o-thread-9003", "thread list", "Review queue", area="list"),
                ],
            ),
            expected_execution={"region_ids": ["h1o-status-badge-9002"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1o_activation_owner_field_stale_selection_decoy",
            family="h1o_activation_no_call",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1o-activation-owner-field"),
                Message(
                    role="user",
                    content=(
                        "Do not refine stale selection_id sel-owner-note from an older capture. Locate the owner "
                        "field on the current image."
                    ),
                ),
            ],
            media=["img-h1o-activation-owner-field"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1o-activation-owner-field",
                [
                    _region("h1o-owner-note-9011", "owner note", "Ari owns renewal follow-up", area="note"),
                    _region("h1o-owner-field-9012", "owner field", "Ari", area="field"),
                    _region("h1o-owner-table-9013", "owner table", "Ari rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1o-owner-field-9012"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1o_activation_error_banner_previous_region_decoy",
            family="h1o_activation_no_call",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1o-activation-error-banner"),
                Message(
                    role="user",
                    content=(
                        "Previous region_id region-export-note is not current. Locate the visible error banner "
                        "on this image before using any prior region."
                    ),
                ),
            ],
            media=["img-h1o-activation-error-banner"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1o-activation-error-banner",
                [
                    _region("h1o-export-note-9021", "export note", "Export failed in audit log", area="note"),
                    _region("h1o-error-banner-9022", "error banner", "Export failed: approval missing", area="banner"),
                    _region("h1o-approver-table-9023", "approver table", "Approval routing", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1o-error-banner-9022"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1o_activation_warning_tile_no_call_decoy",
            family="h1o_activation_no_call",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1o-activation-warning-tile"),
                Message(
                    role="user",
                    content=(
                        "The sidebar note says overdue. Locate the warning tile on the current image, not the note."
                    ),
                ),
            ],
            media=["img-h1o-activation-warning-tile"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1o-activation-warning-tile",
                [
                    _region("h1o-warning-note-9031", "sidebar note", "Overdue items require owner", area="note"),
                    _region("h1o-warning-tile-9032", "warning tile", "Overdue review required", area="tile"),
                    _region("h1o-warning-table-9033", "review table", "Overdue queue", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1o-warning-tile-9032"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1o_code_node_j44_table_decoy",
            family="h1o_code_negation_preservation",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1o-code-node-j44"),
                Message(
                    role="user",
                    content=(
                        "The table references J44 in text, but locate node j44 itself before reading the table."
                    ),
                ),
            ],
            media=["img-h1o-code-node-j44"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1o-code-node-j44",
                [
                    _region("h1o-related-table-9101", "related table", "J44 impacted rows", area="table"),
                    _region("h1o-node-j44-9102", "node j44", "Retry queue", area="node"),
                    _region("h1o-node-note-9103", "ops note", "Retry owner pending", area="note"),
                ],
            ),
            expected_execution={"region_ids": ["h1o-node-j44-9102"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1o_code_field_u17_old_selection_decoy",
            family="h1o_code_negation_preservation",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1o-code-field-u17"),
                Message(
                    role="user",
                    content=(
                        "Ignore old selection_id sel-field-u17-archive. On the current image locate field u17, "
                        "not the archived note."
                    ),
                ),
            ],
            media=["img-h1o-code-field-u17"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1o-code-field-u17",
                [
                    _region("h1o-archive-note-9111", "archived note", "u17 was edited yesterday", area="note"),
                    _region("h1o-field-u17-9112", "field u17", "Missing checksum", area="field"),
                    _region("h1o-field-table-9113", "field table", "Validation rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1o-field-u17-9112"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1o_code_alert_s92_negated_toggle_decoy",
            family="h1o_code_negation_preservation",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1o-code-alert-s92"),
                Message(
                    role="user",
                    content=(
                        "Before reading the consent toggle, locate alert s92. Do not target the toggle."
                    ),
                ),
            ],
            media=["img-h1o-code-alert-s92"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1o-code-alert-s92",
                [
                    _region("h1o-toggle-9121", "consent toggle", "Enabled", area="toggle"),
                    _region("h1o-alert-s92-9122", "alert s92", "Consent mismatch", area="alert"),
                    _region("h1o-retry-log-9123", "retry log", "s92 retry failed", area="log"),
                ],
            ),
            expected_execution={"region_ids": ["h1o-alert-s92-9122"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1o_code_badge_c08_note_decoy",
            family="h1o_code_negation_preservation",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1o-code-badge-c08"),
                Message(
                    role="user",
                    content=(
                        "The note says C08 was escalated. Locate badge c08 itself before using the note."
                    ),
                ),
            ],
            media=["img-h1o-code-badge-c08"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1o-code-badge-c08",
                [
                    _region("h1o-case-note-9131", "case note", "C08 escalated by Sana", area="note"),
                    _region("h1o-badge-c08-9132", "badge c08", "Escalated", area="badge"),
                    _region("h1o-badge-table-9133", "case table", "Escalation rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1o-badge-c08-9132"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1o_component_state_pill_value_decoy",
            family="h1o_component_value_boundary",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1o-component-state-pill"),
                Message(
                    role="user",
                    content=(
                        "The note repeats On hold, but locate the state pill component, not the note value."
                    ),
                ),
            ],
            media=["img-h1o-component-state-pill"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1o-component-state-pill",
                [
                    _region("h1o-state-note-9201", "state note", "On hold until QA replies", area="note"),
                    _region("h1o-state-pill-9202", "state pill", "On hold", area="pill"),
                    _region("h1o-state-table-9203", "state table", "Hold rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1o-state-pill-9202"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1o_component_priority_chip_value_decoy",
            family="h1o_component_value_boundary",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1o-component-priority-chip"),
                Message(
                    role="user",
                    content=(
                        "The table heading says High priority. Locate the priority chip component itself."
                    ),
                ),
            ],
            media=["img-h1o-component-priority-chip"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1o-component-priority-chip",
                [
                    _region("h1o-priority-table-9211", "priority table", "High priority cases", area="table"),
                    _region("h1o-priority-chip-9212", "priority chip", "High", area="chip"),
                    _region("h1o-priority-note-9213", "priority note", "Escalation owner listed", area="note"),
                ],
            ),
            expected_execution={"region_ids": ["h1o-priority-chip-9212"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1o_component_result_badge_value_decoy",
            family="h1o_component_value_boundary",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1o-component-result-badge"),
                Message(
                    role="user",
                    content=(
                        "Approved appears in the audit log. Locate the result badge component, not the audit log."
                    ),
                ),
            ],
            media=["img-h1o-component-result-badge"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1o-component-result-badge",
                [
                    _region("h1o-result-log-9221", "audit log", "Approved by Lena", area="log"),
                    _region("h1o-result-badge-9222", "result badge", "Approved", area="badge"),
                    _region("h1o-result-table-9223", "result table", "Approval history", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1o-result-badge-9222"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1o_component_phase_tile_value_decoy",
            family="h1o_component_value_boundary",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1o-component-phase-tile"),
                Message(
                    role="user",
                    content=(
                        "Review appears in the ticket summary. Locate the phase tile component instead."
                    ),
                ),
            ],
            media=["img-h1o-component-phase-tile"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1o-component-phase-tile",
                [
                    _region("h1o-phase-ticket-9231", "ticket summary", "Review is waiting on data", area="ticket"),
                    _region("h1o-phase-tile-9232", "phase tile", "Review", area="tile"),
                    _region("h1o-phase-board-9233", "phase board", "Review lane", area="board"),
                ],
            ),
            expected_execution={"region_ids": ["h1o-phase-tile-9232"]},
        ),
    ]


def _h1p_component_value_holdout_cases_v11() -> list[ToolDirectiveProbeCase]:
    return [
        ToolDirectiveProbeCase(
            case_id="h1p_compact_status_pill_summary_value_decoy",
            family="h1p_component_value_compact",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1p-compact-status-pill"),
                Message(
                    role="user",
                    content=(
                        "The summary repeats Pending approval. Locate the status pill component itself, not "
                        "the summary text."
                    ),
                ),
            ],
            media=["img-h1p-compact-status-pill"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1p-compact-status-pill",
                [
                    _region("h1p-summary-9301", "approval summary", "Pending approval from legal", area="summary"),
                    _region("h1p-status-pill-9302", "status pill", "Pending", area="pill"),
                    _region("h1p-approval-table-9303", "approval table", "Pending requests", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1p-status-pill-9302"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1p_compact_queue_badge_table_value_decoy",
            family="h1p_component_value_compact",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1p-compact-queue-badge"),
                Message(
                    role="user",
                    content=(
                        "Escalated appears in the table heading. Locate the queue badge component, not the table."
                    ),
                ),
            ],
            media=["img-h1p-compact-queue-badge"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1p-compact-queue-badge",
                [
                    _region("h1p-queue-table-9311", "queue table", "Escalated vendor tickets", area="table"),
                    _region("h1p-queue-badge-9312", "queue badge", "Escalated", area="badge"),
                    _region("h1p-queue-note-9313", "queue note", "Escalation owner pending", area="note"),
                ],
            ),
            expected_execution={"region_ids": ["h1p-queue-badge-9312"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1p_compact_stage_chip_email_value_decoy",
            family="h1p_component_value_compact",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1p-compact-stage-chip"),
                Message(
                    role="user",
                    content=(
                        "The email preview says Review twice. Locate the stage chip component instead of the email."
                    ),
                ),
            ],
            media=["img-h1p-compact-stage-chip"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1p-compact-stage-chip",
                [
                    _region("h1p-email-9321", "email preview", "Review requested by finance", area="email"),
                    _region("h1p-stage-chip-9322", "stage chip", "Review", area="chip"),
                    _region("h1p-stage-board-9323", "stage board", "Review lane", area="board"),
                ],
            ),
            expected_execution={"region_ids": ["h1p-stage-chip-9322"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1p_compact_state_tag_log_value_decoy",
            family="h1p_component_value_compact",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1p-compact-state-tag"),
                Message(
                    role="user",
                    content=(
                        "Closed appears in the audit log. Locate the state tag component, not the log row."
                    ),
                ),
            ],
            media=["img-h1p-compact-state-tag"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1p-compact-state-tag",
                [
                    _region("h1p-state-log-9331", "audit log", "Closed by Omar at 13:20", area="log"),
                    _region("h1p-state-tag-9332", "state tag", "Closed", area="tag"),
                    _region("h1p-state-table-9333", "state table", "Closed accounts", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1p-state-tag-9332"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1p_surface_owner_field_note_value_decoy",
            family="h1p_component_value_surface",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1p-surface-owner-field"),
                Message(
                    role="user",
                    content=(
                        "Nia appears in the note. Locate the owner field component, not the note."
                    ),
                ),
            ],
            media=["img-h1p-surface-owner-field"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1p-surface-owner-field",
                [
                    _region("h1p-owner-note-9341", "owner note", "Nia owns the renewal handoff", area="note"),
                    _region("h1p-owner-field-9342", "owner field", "Nia", area="field"),
                    _region("h1p-owner-table-9343", "handoff table", "Nia renewal rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1p-owner-field-9342"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1p_surface_mode_toggle_note_value_decoy",
            family="h1p_component_value_surface",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1p-surface-mode-toggle"),
                Message(
                    role="user",
                    content=(
                        "Manual is written in the note. Locate the mode toggle component itself."
                    ),
                ),
            ],
            media=["img-h1p-surface-mode-toggle"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1p-surface-mode-toggle",
                [
                    _region("h1p-mode-note-9351", "settings note", "Manual override is active", area="note"),
                    _region("h1p-mode-toggle-9352", "mode toggle", "Manual", area="toggle"),
                    _region("h1p-mode-table-9353", "settings table", "Manual exception rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1p-mode-toggle-9352"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1p_surface_lane_tile_board_value_decoy",
            family="h1p_component_value_surface",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1p-surface-lane-tile"),
                Message(
                    role="user",
                    content=(
                        "Blocked appears on the board header. Locate the lane tile component itself."
                    ),
                ),
            ],
            media=["img-h1p-surface-lane-tile"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1p-surface-lane-tile",
                [
                    _region("h1p-lane-board-9361", "board header", "Blocked renewal lane", area="board"),
                    _region("h1p-lane-tile-9362", "lane tile", "Blocked", area="tile"),
                    _region("h1p-lane-table-9363", "blocked table", "Blocked tickets", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1p-lane-tile-9362"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1p_surface_result_badge_comment_value_decoy",
            family="h1p_component_value_surface",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1p-surface-result-badge"),
                Message(
                    role="user",
                    content=(
                        "Approved appears in the reviewer comment. Locate the result badge component."
                    ),
                ),
            ],
            media=["img-h1p-surface-result-badge"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1p-surface-result-badge",
                [
                    _region("h1p-result-comment-9371", "reviewer comment", "Approved after policy review", area="comment"),
                    _region("h1p-result-badge-9372", "result badge", "Approved", area="badge"),
                    _region("h1p-result-table-9373", "result table", "Approved entries", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1p-result-badge-9372"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1p_stale_priority_chip_old_selection_decoy",
            family="h1p_component_value_stale_selection",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1p-stale-priority-chip"),
                Message(
                    role="user",
                    content=(
                        "Ignore stale selection_id sel-priority-row. Locate the priority chip component, not the row."
                    ),
                ),
            ],
            media=["img-h1p-stale-priority-chip"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1p-stale-priority-chip",
                [
                    _region("h1p-priority-row-9381", "priority row", "Urgent accounts", area="row"),
                    _region("h1p-priority-chip-9382", "priority chip", "Urgent", area="chip"),
                    _region("h1p-priority-note-9383", "priority note", "Urgent review owner missing", area="note"),
                ],
            ),
            expected_execution={"region_ids": ["h1p-priority-chip-9382"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1p_stale_severity_pill_previous_region_decoy",
            family="h1p_component_value_stale_selection",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1p-stale-severity-pill"),
                Message(
                    role="user",
                    content=(
                        "Previous region_id region-severity-chart is stale. Locate the severity pill component."
                    ),
                ),
            ],
            media=["img-h1p-stale-severity-pill"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1p-stale-severity-pill",
                [
                    _region("h1p-severity-chart-9391", "severity chart", "Critical incidents rising", area="chart"),
                    _region("h1p-severity-pill-9392", "severity pill", "Critical", area="pill"),
                    _region("h1p-severity-table-9393", "severity table", "Critical rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1p-severity-pill-9392"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1p_stale_phase_tile_archive_decoy",
            family="h1p_component_value_stale_selection",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1p-stale-phase-tile"),
                Message(
                    role="user",
                    content=(
                        "Do not use selection_id sel-phase-archive. Locate the phase tile component, not the archive note."
                    ),
                ),
            ],
            media=["img-h1p-stale-phase-tile"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1p-stale-phase-tile",
                [
                    _region("h1p-phase-archive-9401", "archive note", "Review happened last week", area="note"),
                    _region("h1p-phase-tile-9402", "phase tile", "Review", area="tile"),
                    _region("h1p-phase-board-9403", "phase board", "Review queue", area="board"),
                ],
            ),
            expected_execution={"region_ids": ["h1p-phase-tile-9402"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1p_stale_risk_badge_old_selection_decoy",
            family="h1p_component_value_stale_selection",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1p-stale-risk-badge"),
                Message(
                    role="user",
                    content=(
                        "Ignore old selection_id sel-risk-summary. Locate the risk badge component itself."
                    ),
                ),
            ],
            media=["img-h1p-stale-risk-badge"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1p-stale-risk-badge",
                [
                    _region("h1p-risk-summary-9411", "risk summary", "Overdue vendor checks", area="summary"),
                    _region("h1p-risk-badge-9412", "risk badge", "Overdue", area="badge"),
                    _region("h1p-risk-table-9413", "risk table", "Overdue vendor rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1p-risk-badge-9412"]},
        ),
    ]


def _h1r_component_label_residual_cases_v12() -> list[ToolDirectiveProbeCase]:
    return [
        ToolDirectiveProbeCase(
            case_id="h1r_owner_field_stale_selection_note_decoy",
            family="h1r_stale_selection_component_label",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1r-owner-field"),
                Message(
                    role="user",
                    content=(
                        "Ignore old selection_id sel-owner-archive. Locate the owner field component itself, "
                        "not the note that repeats Mina."
                    ),
                ),
            ],
            media=["img-h1r-owner-field"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1r-owner-field",
                [
                    _region("h1r-owner-note-9501", "owner note", "Mina owns renewal handoff", area="note"),
                    _region("h1r-owner-field-9502", "owner field", "Mina", area="field"),
                    _region("h1r-handoff-table-9503", "handoff table", "Mina handoff rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1r-owner-field-9502"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1r_assignee_field_previous_selection_summary_decoy",
            family="h1r_stale_selection_component_label",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1r-assignee-field"),
                Message(
                    role="user",
                    content=(
                        "Previous selection_id sel-assignee-summary is stale. Locate the assignee field component, "
                        "not the summary row that repeats Theo."
                    ),
                ),
            ],
            media=["img-h1r-assignee-field"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1r-assignee-field",
                [
                    _region("h1r-assignee-summary-9511", "assignment summary", "Theo owns the exception", area="summary"),
                    _region("h1r-assignee-field-9512", "assignee field", "Theo", area="field"),
                    _region("h1r-assignee-table-9513", "assignment table", "Theo exception rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1r-assignee-field-9512"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1r_state_tag_log_value_decoy",
            family="h1r_nonstandard_component_class",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1r-state-tag"),
                Message(
                    role="user",
                    content="Closed appears in the audit log. Locate the state tag component, not the log row.",
                ),
            ],
            media=["img-h1r-state-tag"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1r-state-tag",
                [
                    _region("h1r-state-log-9521", "audit log", "Closed by Omar at 13:20", area="log"),
                    _region("h1r-state-tag-9522", "state tag", "Closed", area="tag"),
                    _region("h1r-state-table-9523", "state table", "Closed accounts", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1r-state-tag-9522"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1r_mode_toggle_note_value_decoy",
            family="h1r_nonstandard_component_class",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1r-mode-toggle"),
                Message(
                    role="user",
                    content="Manual is written in the settings note. Locate the mode toggle component itself.",
                ),
            ],
            media=["img-h1r-mode-toggle"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1r-mode-toggle",
                [
                    _region("h1r-mode-note-9531", "settings note", "Manual override is active", area="note"),
                    _region("h1r-mode-toggle-9532", "mode toggle", "Manual", area="toggle"),
                    _region("h1r-mode-table-9533", "settings table", "Manual exception rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1r-mode-toggle-9532"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1r_alert_s92_toggle_negation_decoy",
            family="h1r_code_label_exactness",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1r-alert-s92"),
                Message(
                    role="user",
                    content=(
                        "The consent toggle is not the target. Locate alert s92 before reading the retry log."
                    ),
                ),
            ],
            media=["img-h1r-alert-s92"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1r-alert-s92",
                [
                    _region("h1r-consent-toggle-9541", "consent toggle", "Enabled", area="toggle"),
                    _region("h1r-alert-s92-9542", "alert s92", "Consent mismatch", area="alert"),
                    _region("h1r-retry-log-9543", "retry log", "s92 retry failed", area="log"),
                ],
            ),
            expected_execution={"region_ids": ["h1r-alert-s92-9542"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1r_badge_c08_note_decoy",
            family="h1r_code_label_exactness",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1r-badge-c08"),
                Message(
                    role="user",
                    content="The case note repeats C08. Locate badge c08, not the note or case table.",
                ),
            ],
            media=["img-h1r-badge-c08"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1r-badge-c08",
                [
                    _region("h1r-case-note-9551", "case note", "C08 escalated by Sana", area="note"),
                    _region("h1r-badge-c08-9552", "badge c08", "Escalated", area="badge"),
                    _region("h1r-case-table-9553", "case table", "Escalation rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1r-badge-c08-9552"]},
        ),
    ]


def _h1w_residual_overlap_cases_v13() -> list[ToolDirectiveProbeCase]:
    return [
        ToolDirectiveProbeCase(
            case_id="h1w_owner_field_memo_stale_selection_decoy",
            family="h1w_stale_field_routing",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1w-owner-field"),
                Message(
                    role="user",
                    content=(
                        "Ignore stale selection_id sel-owner-memo. Locate the owner field component itself, "
                        "not the memo that repeats Jae."
                    ),
                ),
            ],
            media=["img-h1w-owner-field"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1w-owner-field",
                [
                    _region("h1w-owner-memo-9601", "owner memo", "Jae owns the vendor review", area="memo"),
                    _region("h1w-owner-field-9602", "owner field", "Jae", area="field"),
                    _region("h1w-owner-table-9603", "owner table", "Jae vendor rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1w-owner-field-9602"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1w_assignee_field_archive_summary_decoy",
            family="h1w_stale_field_routing",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1w-assignee-field"),
                Message(
                    role="user",
                    content=(
                        "Previous selection_id sel-assignee-archive is stale. Locate the assignee field component, "
                        "not the archived summary that repeats Noor."
                    ),
                ),
            ],
            media=["img-h1w-assignee-field"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1w-assignee-field",
                [
                    _region("h1w-assignee-archive-9611", "assignee archive summary", "Noor handled the old case", area="summary"),
                    _region("h1w-assignee-field-9612", "assignee field", "Noor", area="field"),
                    _region("h1w-assignee-table-9613", "assignee table", "Noor current cases", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1w-assignee-field-9612"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1w_state_tag_audit_log_value_decoy",
            family="h1w_nonstandard_component_class",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1w-state-tag"),
                Message(
                    role="user",
                    content="Closed appears in the audit log. Locate the state tag component, not the log entry.",
                ),
            ],
            media=["img-h1w-state-tag"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1w-state-tag",
                [
                    _region("h1w-state-log-9621", "audit log", "Closed by Luis at 09:14", area="log"),
                    _region("h1w-state-tag-9622", "state tag", "Closed", area="tag"),
                    _region("h1w-state-table-9623", "state table", "Closed accounts", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1w-state-tag-9622"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1w_mode_toggle_settings_note_decoy",
            family="h1w_nonstandard_component_class",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1w-mode-toggle"),
                Message(
                    role="user",
                    content="Manual appears in the settings note. Locate the mode toggle component itself.",
                ),
            ],
            media=["img-h1w-mode-toggle"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1w-mode-toggle",
                [
                    _region("h1w-mode-note-9631", "settings note", "Manual override remains active", area="note"),
                    _region("h1w-mode-toggle-9632", "mode toggle", "Manual", area="toggle"),
                    _region("h1w-mode-table-9633", "settings table", "Manual exception rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1w-mode-toggle-9632"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1w_result_badge_comment_value_decoy",
            family="h1w_surface_component_value",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1w-result-badge"),
                Message(
                    role="user",
                    content="Blocked is written in the comment. Locate the result badge component, not the comment.",
                ),
            ],
            media=["img-h1w-result-badge"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1w-result-badge",
                [
                    _region("h1w-result-comment-9641", "result comment", "Blocked until compliance replies", area="comment"),
                    _region("h1w-result-badge-9642", "result badge", "Blocked", area="badge"),
                    _region("h1w-result-board-9643", "result board", "Blocked review queue", area="board"),
                ],
            ),
            expected_execution={"region_ids": ["h1w-result-badge-9642"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1w_status_pill_summary_value_decoy",
            family="h1w_surface_component_value",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1w-status-pill"),
                Message(
                    role="user",
                    content="Pending appears in the summary. Locate the status pill component itself.",
                ),
            ],
            media=["img-h1w-status-pill"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1w-status-pill",
                [
                    _region("h1w-status-summary-9651", "status summary", "Pending renewal package", area="summary"),
                    _region("h1w-status-pill-9652", "status pill", "Pending", area="pill"),
                    _region("h1w-status-table-9653", "status table", "Pending renewal rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1w-status-pill-9652"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1w_warning_tile_no_call_note_decoy",
            family="h1w_activation_no_call",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1w-warning-tile"),
                Message(
                    role="user",
                    content="Do not summarize the incident note. Locate the warning tile component before reading details.",
                ),
            ],
            media=["img-h1w-warning-tile"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1w-warning-tile",
                [
                    _region("h1w-warning-note-9661", "incident note", "Warning acknowledged by Sam", area="note"),
                    _region("h1w-warning-tile-9662", "warning tile", "Warning", area="tile"),
                    _region("h1w-warning-table-9663", "warning table", "Warning event rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1w-warning-tile-9662"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1w_error_banner_no_call_history_decoy",
            family="h1w_activation_no_call",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1w-error-banner"),
                Message(
                    role="user",
                    content="Do not open the history row. Locate the error banner component itself.",
                ),
            ],
            media=["img-h1w-error-banner"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1w-error-banner",
                [
                    _region("h1w-error-history-9671", "history row", "Error resolved yesterday", area="history"),
                    _region("h1w-error-banner-9672", "error banner", "Error", area="banner"),
                    _region("h1w-error-log-9673", "error log", "Error trace rows", area="log"),
                ],
            ),
            expected_execution={"region_ids": ["h1w-error-banner-9672"]},
        ),
    ]


def _h1x_v11_breaker_cases_v14() -> list[ToolDirectiveProbeCase]:
    return [
        ToolDirectiveProbeCase(
            case_id="h1x_responsible_party_field_old_owner_memo_decoy",
            family="h1x_oblique_stale_field",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1x-owner-field"),
                Message(
                    role="user",
                    content=(
                        "Old selection_id sel-owner-memo points at the memo. The target is the responsible-party "
                        "entry showing Iris; locate the owner field component, not the memo or table."
                    ),
                ),
            ],
            media=["img-h1x-owner-field"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1x-owner-field",
                [
                    _region("h1x-owner-memo-9701", "owner memo", "Iris owned the prior renewal", area="memo"),
                    _region("h1x-owner-field-9702", "owner field", "Iris", area="field"),
                    _region("h1x-owner-table-9703", "responsibility table", "Iris renewal rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1x-owner-field-9702"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1x_workstream_owner_field_previous_summary_decoy",
            family="h1x_oblique_stale_field",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1x-workstream-owner"),
                Message(
                    role="user",
                    content=(
                        "Previous selection_id sel-workstream-summary is stale. Find the accountable workstream "
                        "person control for Remy; the actual component is the owner field."
                    ),
                ),
            ],
            media=["img-h1x-workstream-owner"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1x-workstream-owner",
                [
                    _region("h1x-workstream-summary-9711", "workstream summary", "Remy handled onboarding", area="summary"),
                    _region("h1x-workstream-owner-field-9712", "owner field", "Remy", area="field"),
                    _region("h1x-workstream-grid-9713", "workstream grid", "Remy onboarding rows", area="grid"),
                ],
            ),
            expected_execution={"region_ids": ["h1x-workstream-owner-field-9712"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1x_resolution_chip_comment_result_decoy",
            family="h1x_oblique_surface_value",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1x-resolution-chip"),
                Message(
                    role="user",
                    content=(
                        "The comment also says Blocked. Select the compact resolution indicator for Blocked; "
                        "the visible component is the result chip, not the comment."
                    ),
                ),
            ],
            media=["img-h1x-resolution-chip"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1x-resolution-chip",
                [
                    _region("h1x-resolution-comment-9721", "resolution comment", "Blocked by compliance", area="comment"),
                    _region("h1x-result-chip-9722", "result chip", "Blocked", area="chip"),
                    _region("h1x-resolution-board-9723", "resolution board", "Blocked queue", area="board"),
                ],
            ),
            expected_execution={"region_ids": ["h1x-result-chip-9722"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1x_progress_marker_summary_status_decoy",
            family="h1x_oblique_surface_value",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1x-progress-marker"),
                Message(
                    role="user",
                    content=(
                        "Pending is repeated in the summary. Locate the compact progress marker for the row; "
                        "the layout label is status pill."
                    ),
                ),
            ],
            media=["img-h1x-progress-marker"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1x-progress-marker",
                [
                    _region("h1x-progress-summary-9731", "progress summary", "Pending for legal review", area="summary"),
                    _region("h1x-status-pill-9732", "status pill", "Pending", area="pill"),
                    _region("h1x-progress-table-9733", "progress table", "Pending review rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1x-status-pill-9732"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1x_lifecycle_marker_log_state_tag_decoy",
            family="h1x_oblique_nonstandard_class",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1x-lifecycle-marker"),
                Message(
                    role="user",
                    content=(
                        "Closed is in the audit trail. Locate the lifecycle marker component; the layout label is "
                        "state tag, not audit log."
                    ),
                ),
            ],
            media=["img-h1x-lifecycle-marker"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1x-lifecycle-marker",
                [
                    _region("h1x-lifecycle-log-9741", "audit log", "Closed by Nia at 16:05", area="log"),
                    _region("h1x-state-tag-9742", "state tag", "Closed", area="tag"),
                    _region("h1x-lifecycle-table-9743", "lifecycle table", "Closed rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1x-state-tag-9742"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1x_operation_mode_control_note_toggle_decoy",
            family="h1x_oblique_nonstandard_class",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1x-operation-mode"),
                Message(
                    role="user",
                    content=(
                        "Manual appears in the note. Locate the operation mode control; the layout label is mode toggle."
                    ),
                ),
            ],
            media=["img-h1x-operation-mode"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1x-operation-mode",
                [
                    _region("h1x-mode-note-9751", "mode note", "Manual override was approved", area="note"),
                    _region("h1x-mode-toggle-9752", "mode toggle", "Manual", area="toggle"),
                    _region("h1x-mode-table-9753", "mode table", "Manual exception rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1x-mode-toggle-9752"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1x_warning_panel_note_activation_decoy",
            family="h1x_oblique_activation_no_call",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1x-warning-panel"),
                Message(
                    role="user",
                    content=(
                        "Do not summarize the note. Locate the warning panel for review; the layout label is warning tile."
                    ),
                ),
            ],
            media=["img-h1x-warning-panel"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1x-warning-panel",
                [
                    _region("h1x-warning-note-9761", "warning note", "Warning acknowledged yesterday", area="note"),
                    _region("h1x-warning-tile-9762", "warning tile", "Warning", area="tile"),
                    _region("h1x-warning-table-9763", "warning table", "Warning history rows", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["h1x-warning-tile-9762"]},
        ),
        ToolDirectiveProbeCase(
            case_id="h1x_error_notice_history_activation_decoy",
            family="h1x_oblique_activation_no_call",
            messages=[
                Message(role="system", content="visual_image_ids: img-h1x-error-notice"),
                Message(
                    role="user",
                    content=(
                        "The history row is not the target. Locate the error notice component; the layout label is "
                        "error banner."
                    ),
                ),
            ],
            media=["img-h1x-error-notice"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-h1x-error-notice",
                [
                    _region("h1x-error-history-9771", "error history", "Error cleared last week", area="history"),
                    _region("h1x-error-banner-9772", "error banner", "Error", area="banner"),
                    _region("h1x-error-log-9773", "error log", "Error trace rows", area="log"),
                ],
            ),
            expected_execution={"region_ids": ["h1x-error-banner-9772"]},
        ),
    ]


def _visual_state(image_id: str, local_layouts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "visual_executor_mode": "local",
        "images": {
            image_id: {
                "entities": [],
                "layouts": [],
                "local_layouts": local_layouts,
            }
        },
    }


def _region(region_id: str, label: str, text: str, **attributes: Any) -> dict[str, Any]:
    return {
        "region_id": region_id,
        "label": label,
        "text": text,
        "attributes": attributes,
    }


def _stress_failure_mode(family: str) -> str:
    if family in {
        "visual_tool_routing_stress",
        "visual_tool_routing_transfer",
        "visual_tool_routing_transfer_repeat",
        "visual_tool_routing_transfer_oblique",
        "visual_tool_routing_transfer_post_repair",
        "visual_tool_routing_transfer_residual",
        "visual_tool_routing_component_value",
        "h1o_activation_no_call",
        "h1p_component_value_stale_selection",
        "h1r_stale_selection_component_label",
        "h1w_stale_field_routing",
        "h1w_activation_no_call",
        "h1x_oblique_stale_field",
        "h1x_oblique_activation_no_call",
    }:
        return "wrong_tool_or_stale_selection_risk"
    return "argument_alias_or_decoy_risk"


def _live_replay_command(*, packet_dir: Path, registry_path: Path, replay_system_id: str, case_id: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "gemma4_capability_map.runtime.cli",
        "replay-live",
        "--packet-dir",
        str(packet_dir.resolve()),
        "--system-id",
        replay_system_id,
        "--registry",
        str(registry_path.resolve()),
        "--case-id",
        case_id,
        "--execute",
    ]


def _count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, ""))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
