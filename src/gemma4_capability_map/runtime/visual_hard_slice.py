from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from gemma4_capability_map.runtime.tool_directive_probe import ToolDirectiveProbeCase
from gemma4_capability_map.schemas import Message


@dataclass(frozen=True)
class VisualCaseDesign:
    case_id: str
    family: str
    primary_discriminator: str
    expected_tool: str
    expected_argument_focus: str
    failure_pressure: str
    publishable_reason: str


VISUAL_HARD_SLICE_DESIGNS: tuple[VisualCaseDesign, ...] = (
    VisualCaseDesign(
        case_id="visual_form_error_vs_message_author",
        family="visual_argument_copying",
        primary_discriminator="target_query_region_class_vs_business_subject",
        expected_tool="extract_layout",
        expected_argument_focus="target_query should name the visible error or warning region, not message author/source.",
        failure_pressure="v2/v4 tend to select recruiter/note/phone/source concepts instead of executable visual regions.",
        publishable_reason="Separates executable visual targeting from task-story nouns.",
    ),
    VisualCaseDesign(
        case_id="visual_form_error_with_prior_selection_decoy",
        family="visual_tool_routing",
        primary_discriminator="extract_layout_vs_refine_selection_when_no_real_selection_id",
        expected_tool="extract_layout",
        expected_argument_focus="image_id is copied from visual state; target_query stays on visible form error class.",
        failure_pressure="v4 over-preferred refine_selection with selection_id=latest on the form-target case.",
        publishable_reason="Tests whether schema hints cause false selection carryover.",
    ),
    VisualCaseDesign(
        case_id="visual_latest_filter_existing_selection",
        family="visual_referent_carryover",
        primary_discriminator="compact_filter_query_after_selection_id",
        expected_tool="refine_selection",
        expected_argument_focus="selection_id copied exactly; filter_query remains the literal token latest.",
        failure_pressure="v1 expanded latest into latest issue; v2/v4 fixed it.",
        publishable_reason="Preserves the current positive result as a regression guard.",
    ),
    VisualCaseDesign(
        case_id="visual_remaining_filter_existing_selection",
        family="visual_referent_carryover",
        primary_discriminator="compact_filter_query_non_latest_token",
        expected_tool="refine_selection",
        expected_argument_focus="filter_query remains remaining without surrounding nouns.",
        failure_pressure="Tests whether the latest-only fix generalizes to other compact selector tokens.",
        publishable_reason="Checks generality rather than overfitting to one literal.",
    ),
    VisualCaseDesign(
        case_id="visual_region_readback_after_layout_result",
        family="visual_region_readback",
        primary_discriminator="read_region_text_json_shape",
        expected_tool="read_region_text",
        expected_argument_focus="top-level call key remains name and region_id is copied as an opaque id.",
        failure_pressure="v3 emitted tool_name instead of name on readback.",
        publishable_reason="Guards protocol shape separately from visual selection semantics.",
    ),
    VisualCaseDesign(
        case_id="visual_metric_panel_vs_table_selector",
        family="visual_argument_copying",
        primary_discriminator="target_query_specific_visible_region_class",
        expected_tool="extract_layout",
        expected_argument_focus="target_query distinguishes metric panel from table without copying business prose.",
        failure_pressure="Tests target_query specificity without relying on validation-error wording.",
        publishable_reason="Adds fresh visual region classes beyond current replay cases.",
    ),
    VisualCaseDesign(
        case_id="visual_callout_warning_with_user_decoy",
        family="visual_argument_copying",
        primary_discriminator="target_query_visible_warning_vs_user_decoy",
        expected_tool="extract_layout",
        expected_argument_focus="target_query uses warning/callout region even when the user mentions a person or ticket.",
        failure_pressure="Targets the same semantic drift as recruiter note without reusing that surface.",
        publishable_reason="Fresh decoy case for form-target executability.",
    ),
    VisualCaseDesign(
        case_id="visual_selection_id_opaque_copy_with_filter",
        family="visual_referent_carryover",
        primary_discriminator="opaque_selection_id_copy",
        expected_tool="refine_selection",
        expected_argument_focus="selection_id is copied exactly from prior tool result and not replaced with latest/open/etc.",
        failure_pressure="v4 produced selection_id=latest on a case without a valid selection id.",
        publishable_reason="Separates selector token copying from opaque id copying.",
    ),
)


def build_visual_hard_slice_cases() -> list[ToolDirectiveProbeCase]:
    return [
        ToolDirectiveProbeCase(
            case_id="visual_form_error_vs_message_author",
            family="visual_argument_copying",
            messages=[
                Message(role="system", content="visual_image_ids: img-hard-form-author"),
                Message(
                    role="user",
                    content=(
                        "The recruiter note mentions Morgan, but use the local visual executor path to locate the "
                        "visible validation error on the form first and read that message."
                    ),
                ),
            ],
            media=["img-hard-form-author"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-hard-form-author",
                [
                    _region("hard-form-author-501", "recruiter note", "Morgan asked for a phone-field check", author="Morgan"),
                    _region("hard-form-author-502", "validation error", "Phone number format invalid", field="phone"),
                ],
            ),
            expected_execution={"region_ids": ["hard-form-author-502"]},
        ),
        ToolDirectiveProbeCase(
            case_id="visual_form_error_with_prior_selection_decoy",
            family="visual_tool_routing",
            messages=[
                Message(role="system", content="visual_image_ids: img-hard-form-decoy"),
                Message(
                    role="user",
                    content=(
                        "Ignore stale selection_id sel-stale from the earlier screenshot. On the current form, locate "
                        "the visible validation error first and read it back."
                    ),
                ),
            ],
            media=["img-hard-form-decoy"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-hard-form-decoy",
                [
                    _region("hard-form-decoy-601", "status message", "Profile saved by reviewer note", source="reviewer"),
                    _region("hard-form-decoy-602", "validation error", "Email address is blocked", field="email"),
                ],
            ),
            expected_execution={"region_ids": ["hard-form-decoy-602"]},
        ),
        ToolDirectiveProbeCase(
            case_id="visual_latest_filter_existing_selection",
            family="visual_referent_carryover",
            messages=[
                Message(role="system", content="visual_image_ids: img-hard-form-latest"),
                Message(
                    role="user",
                    content="Inspect the form errors, keep only the latest issue first, then read the remaining message.",
                ),
                _tool_message(
                    "extract_layout",
                    {"image_id": "img-hard-form-latest", "target_query": "validation error"},
                    {
                        "selection_id": "sel-hard-errors",
                        "image_id": "img-hard-form-latest",
                        "selection_kind": "regions",
                        "count": 2,
                        "region_ids": ["hard-form-priority-701", "hard-form-priority-702"],
                        "region_id": "hard-form-priority-701",
                    },
                ),
            ],
            media=["img-hard-form-latest"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-hard-form-latest",
                [
                    _region("hard-form-priority-701", "validation error", "Work authorization required", priority="earlier"),
                    _region("hard-form-priority-702", "validation error", "Phone number format invalid", priority="latest"),
                ],
                selections={
                    "sel-hard-errors": {
                        "image_id": "img-hard-form-latest",
                        "selection_kind": "regions",
                        "items": [
                            _region("hard-form-priority-701", "validation error", "Work authorization required", priority="earlier"),
                            _region("hard-form-priority-702", "validation error", "Phone number format invalid", priority="latest"),
                        ],
                        "query": "validation error",
                        "parent_selection_id": None,
                    }
                },
            ),
            expected_execution={"region_ids": ["hard-form-priority-702"]},
        ),
        ToolDirectiveProbeCase(
            case_id="visual_remaining_filter_existing_selection",
            family="visual_referent_carryover",
            messages=[
                Message(role="system", content="visual_image_ids: img-hard-form-remaining"),
                Message(
                    role="user",
                    content="Inspect the issue list, keep only the remaining items, then read what remains.",
                ),
                _tool_message(
                    "extract_layout",
                    {"image_id": "img-hard-form-remaining", "target_query": "validation error"},
                    {
                        "selection_id": "sel-hard-remaining",
                        "image_id": "img-hard-form-remaining",
                        "selection_kind": "regions",
                        "count": 2,
                        "region_ids": ["hard-form-queue-801", "hard-form-queue-802"],
                    },
                ),
            ],
            media=["img-hard-form-remaining"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-hard-form-remaining",
                [
                    _region("hard-form-queue-801", "validation error", "Address issue already cleared", status="cleared"),
                    _region("hard-form-queue-802", "validation error", "Email issue still remaining", status="remaining"),
                ],
                selections={
                    "sel-hard-remaining": {
                        "image_id": "img-hard-form-remaining",
                        "selection_kind": "regions",
                        "items": [
                            _region("hard-form-queue-801", "validation error", "Address issue already cleared", status="cleared"),
                            _region("hard-form-queue-802", "validation error", "Email issue still remaining", status="remaining"),
                        ],
                        "query": "validation error",
                        "parent_selection_id": None,
                    }
                },
            ),
            expected_execution={"region_ids": ["hard-form-queue-802"]},
        ),
        ToolDirectiveProbeCase(
            case_id="visual_region_readback_after_layout_result",
            family="visual_region_readback",
            messages=[
                Message(role="system", content="visual_image_ids: img-hard-callout-readback"),
                Message(role="user", content="Inspect the slide callout, then read the selected warning text."),
                _tool_message(
                    "extract_layout",
                    {"image_id": "img-hard-callout-readback", "target_query": "slide callout"},
                    {
                        "selection_id": "sel-hard-callout",
                        "image_id": "img-hard-callout-readback",
                        "selection_kind": "regions",
                        "count": 1,
                        "region_ids": ["hard-callout-901"],
                        "region_id": "hard-callout-901",
                    },
                ),
            ],
            media=["img-hard-callout-readback"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-hard-callout-readback",
                [_region("hard-callout-901", "slide callout", "Warning: approval is required before release", tone="warning")],
            ),
            expected_execution={"region_id": "hard-callout-901"},
        ),
        ToolDirectiveProbeCase(
            case_id="visual_metric_panel_vs_table_selector",
            family="visual_argument_copying",
            messages=[
                Message(role="system", content="visual_image_ids: img-hard-metric-table"),
                Message(
                    role="user",
                    content=(
                        "The dashboard also has a table, but locate the metric panel that needs review before "
                        "reading any table contents."
                    ),
                ),
            ],
            media=["img-hard-metric-table"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-hard-metric-table",
                [
                    _region("hard-metric-1001", "dashboard metric", "Support backlog below target", area="metric panel"),
                    _region("hard-metric-1002", "invoice totals table", "Q2 invoice totals", area="table"),
                ],
            ),
            expected_execution={"region_ids": ["hard-metric-1001"]},
        ),
        ToolDirectiveProbeCase(
            case_id="visual_callout_warning_with_user_decoy",
            family="visual_argument_copying",
            messages=[
                Message(role="system", content="visual_image_ids: img-hard-callout-decoy"),
                Message(
                    role="user",
                    content=(
                        "Dana is mentioned in the ticket, but use the visual tool to locate the slide callout "
                        "warning first and read it back."
                    ),
                ),
            ],
            media=["img-hard-callout-decoy"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-hard-callout-decoy",
                [
                    _region("hard-callout-decoy-1101", "assignee note", "Dana owns the next review", person="Dana"),
                    _region("hard-callout-decoy-1102", "slide callout", "Warning: customer data is stale", tone="warning"),
                ],
            ),
            expected_execution={"region_ids": ["hard-callout-decoy-1102"]},
        ),
        ToolDirectiveProbeCase(
            case_id="visual_selection_id_opaque_copy_with_filter",
            family="visual_referent_carryover",
            messages=[
                Message(role="system", content="visual_image_ids: img-hard-opaque-selection"),
                Message(
                    role="user",
                    content="From the current selected set, keep only the blocked items and read the selected message.",
                ),
                _tool_message(
                    "extract_layout",
                    {"image_id": "img-hard-opaque-selection", "target_query": "validation error"},
                    {
                        "selection_id": "sel-opaque-77",
                        "image_id": "img-hard-opaque-selection",
                        "selection_kind": "regions",
                        "count": 2,
                        "region_ids": ["hard-opaque-1201", "hard-opaque-1202"],
                    },
                ),
            ],
            media=["img-hard-opaque-selection"],
            tool_names=["extract_layout", "refine_selection", "read_region_text"],
            initial_state=_visual_state(
                "img-hard-opaque-selection",
                [
                    _region("hard-opaque-1201", "validation error", "Name issue open", status="open"),
                    _region("hard-opaque-1202", "validation error", "Email domain is blocked", status="blocked"),
                ],
                selections={
                    "sel-opaque-77": {
                        "image_id": "img-hard-opaque-selection",
                        "selection_kind": "regions",
                        "items": [
                            _region("hard-opaque-1201", "validation error", "Name issue open", status="open"),
                            _region("hard-opaque-1202", "validation error", "Email domain is blocked", status="blocked"),
                        ],
                        "query": "validation error",
                        "parent_selection_id": None,
                    }
                },
            ),
            expected_execution={"region_ids": ["hard-opaque-1202"]},
        ),
    ]


def visual_hard_slice_case_ids() -> list[str]:
    return [case.case_id for case in build_visual_hard_slice_cases()]


def _visual_state(
    image_id: str,
    local_layouts: list[dict[str, Any]],
    *,
    selections: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "visual_executor_mode": "local",
        "images": {
            image_id: {
                "entities": [],
                "layouts": [],
                "local_layouts": local_layouts,
            }
        },
    }
    if selections:
        state["visual_selections"] = selections
        state["visual_selection_counter"] = 100
        state["visual_last_selection_id"] = next(reversed(selections))
    return state


def _region(region_id: str, label: str, text: str, **attributes: str) -> dict[str, Any]:
    return {
        "region_id": region_id,
        "label": label,
        "text": text,
        "attributes": dict(attributes),
    }


def _tool_message(tool_name: str, arguments: dict[str, Any], output: dict[str, Any]) -> Message:
    return Message(
        role="tool",
        content=json.dumps(
            {
                "tool_name": tool_name,
                "status": "pass",
                "arguments": arguments,
                "output": output,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )
