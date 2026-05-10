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
    parser.add_argument("--suite", choices=["v1", "alias_repeat_v2", "alias_transfer_v3"], default="v1")
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
    raise ValueError(f"Unknown visual live stress suite: {suite}")


def _expected_call_payloads(
    *,
    case: ToolDirectiveProbeCase,
    tool_specs: list[Any],
    suite: str,
) -> list[dict[str, Any]]:
    if suite == "alias_transfer_v3":
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
    if family in {"visual_tool_routing_stress", "visual_tool_routing_transfer"}:
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
