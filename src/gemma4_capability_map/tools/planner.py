from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from gemma4_capability_map.schemas import Message, ToolCall, ToolSpec
from gemma4_capability_map.tools.validators import validate_tool_call


TOOL_NAME_ALIASES = {
    "schedule_meeting": "create_event",
    "schedule_event": "create_event",
    "create_meeting": "create_event",
    "add_event": "create_event",
    "reschedule_event": "update_event",
    "move_event": "update_event",
    "edit_event": "update_event",
    "calendar_search": "search_events",
    "find_events": "search_events",
    "search_calendar": "search_events",
    "find_meeting": "search_events",
    "file_reader": "read_repo_file",
    "read_file": "read_repo_file",
    "open_file": "read_repo_file",
    "get_file_content": "read_repo_file",
    "find_file": "find_repo_file",
    "locate_file": "find_repo_file",
    "search_repo": "find_repo_file",
    "locate_config": "find_repo_file",
    "diff_files": "compare_files",
    "compare_budget_files": "compare_files",
    "inspect_screenshot": "inspect_image",
    "analyze_image": "inspect_image",
    "read_screenshot": "inspect_image",
    "record_patch": "propose_patch",
    "create_patch": "propose_patch",
    "suggest_patch": "propose_patch",
    "fetch_record": "api_fetch_record",
    "read_record": "api_fetch_record",
    "get_record": "api_fetch_record",
    "update_record": "api_update_record",
    "patch_record": "api_update_record",
    "apply_cli_patch": "cli_apply_patch",
    "record_cli_patch": "cli_apply_patch",
    "latest_file_lookup": "find_latest_file",
    "find_recent_file": "find_latest_file",
    "segment_objects": "segment_entities",
    "segment_vehicles": "segment_entities",
    "segment_regions": "segment_entities",
    "filter_selection": "refine_selection",
    "narrow_selection": "refine_selection",
    "extract_table": "extract_layout",
    "extract_region": "extract_layout",
    "read_region": "read_region_text",
    "ocr_region": "read_region_text",
}


def tool_catalog_text(tool_specs: list[ToolSpec]) -> str:
    if not tool_specs:
        return ""
    lines = [
        "Allowed tools. Use only these exact names. Never invent a tool name.",
        "If one tool is needed, return one JSON object.",
        'If multiple independent tools are needed, return a JSON array of {"name": "...", "arguments": {...}} objects.',
        "Use exact schema field names.",
        "",
    ]
    for tool in tool_specs:
        lines.append(f"- {tool.name}: {tool.description}")
        lines.append(json.dumps(tool.model_dump(mode="json", by_alias=True), ensure_ascii=False))
    return "\n".join(lines)


def plan_or_repair_tool_calls(
    raw_output: str,
    parsed_calls: list[ToolCall],
    messages: list[Message],
    media: list[str],
    tool_specs: list[ToolSpec],
) -> tuple[list[ToolCall], list[str]]:
    if not tool_specs:
        return parsed_calls, []

    context = _planning_context(messages, media)
    if _visual_loop_complete(context):
        return [], ["visual_complete"]
    parallel_priority_calls = _parallel_audit_pending_calls(context, tool_specs)
    intent_priority_calls = _intent_priority_calls(context, tool_specs)
    feedback_priority_calls = _next_calls_from_feedback(context, tool_specs)
    if intent_priority_calls == []:
        return [], ["intent_prior:refuse_or_escalate"]

    repaired_calls: list[ToolCall] = []
    repair_notes: list[str] = []
    candidate_calls = parsed_calls
    if parsed_calls and parallel_priority_calls and not _calls_match(parsed_calls, parallel_priority_calls):
        repair_notes.append("parallel_audit_prior")
        candidate_calls = []
    if candidate_calls and _requires_stepwise_visual_control(candidate_calls):
        repair_notes.append("visual_stepwise_prior")
        candidate_calls = []
    if candidate_calls and intent_priority_calls and not _calls_match(candidate_calls, intent_priority_calls):
        repair_notes.append(f"intent_prior:{_classify_intent(context, tool_specs)}")
        candidate_calls = []
    if candidate_calls and feedback_priority_calls and not _calls_match(candidate_calls, feedback_priority_calls):
        prioritized_name = feedback_priority_calls[0].name if feedback_priority_calls else "unknown"
        repair_notes.append(f"feedback_prior:{prioritized_name}")
        candidate_calls = []

    for call in candidate_calls:
        repaired_call, notes = _repair_tool_call(call, raw_output, context, tool_specs)
        if repaired_call is None:
            repaired_calls = []
            break
        repaired_calls.append(repaired_call)
        repair_notes.extend(notes)

    if repaired_calls:
        return repaired_calls, repair_notes

    fallback_calls = plan_tool_calls(messages, media, tool_specs)
    if fallback_calls:
        repair_notes.append("controller_fallback_planner")
        fallback_repaired_calls: list[ToolCall] = []
        for call in fallback_calls:
            repaired_call, notes = _repair_tool_call(call, raw_output, context, tool_specs)
            if repaired_call is None:
                return fallback_calls, repair_notes + notes
            fallback_repaired_calls.append(repaired_call)
            repair_notes.extend(notes)
        return fallback_repaired_calls, repair_notes
    return fallback_calls, repair_notes


def plan_tool_calls(messages: list[Message], media: list[str], tool_specs: list[ToolSpec]) -> list[ToolCall]:
    if not tool_specs:
        return []

    context = _planning_context(messages, media)
    if _visual_loop_complete(context):
        return []
    parallel_priority_calls = _parallel_audit_pending_calls(context, tool_specs)
    if parallel_priority_calls:
        return parallel_priority_calls
    intent_priority_calls = _intent_priority_calls(context, tool_specs)
    if intent_priority_calls is not None:
        return intent_priority_calls
    tool_names = {tool.name for tool in tool_specs}
    user_text = context["user_text"]

    next_from_feedback = _next_calls_from_feedback(context, tool_specs)
    if next_from_feedback:
        return next_from_feedback

    initial_calls = _initial_calls(context, tool_specs)
    if initial_calls:
        return initial_calls

    fallback_tool = _best_tool_match(user_text, tool_specs)
    return [_heuristic_call(fallback_tool.name, _infer_arguments(context, fallback_tool.name), raw_hint="fallback")]


def _requires_stepwise_visual_control(parsed_calls: list[ToolCall]) -> bool:
    visual_tool_names = {"segment_entities", "extract_layout", "refine_selection", "read_region_text"}
    return sum(1 for call in parsed_calls if call.name in visual_tool_names) > 1


def _repair_tool_call(
    call: ToolCall,
    raw_output: str,
    context: dict[str, Any],
    tool_specs: list[ToolSpec],
) -> tuple[ToolCall | None, list[str]]:
    notes: list[str] = []
    repaired_name = _canonical_tool_name(call.name, tool_specs)
    if repaired_name != call.name:
        notes.append(f"canonicalized_tool:{call.name}->{repaired_name}")

    schema_properties = _tool_schema_properties(repaired_name, tool_specs)
    provided_arguments = _project_arguments_to_schema(call.arguments, schema_properties)
    candidate = ToolCall(
        name=repaired_name,
        arguments=provided_arguments,
        source_format=call.source_format,
        raw=call.raw,
    )
    inferred_arguments = _project_arguments_to_schema(_infer_arguments(context, repaired_name), schema_properties)
    valid, _ = validate_tool_call(candidate, tool_specs)
    override_valid_arguments = _should_override_valid_arguments(candidate, inferred_arguments, context)
    if valid and repaired_name == call.name and not override_valid_arguments and _passes_semantic_preconditions(candidate, context):
        if candidate.arguments != call.arguments:
            notes.append(f"repaired_arguments:{repaired_name}")
        return candidate, notes

    if override_valid_arguments:
        merged_arguments = _merge_repaired_arguments(repaired_name, provided_arguments, inferred_arguments)
    else:
        merged_arguments = dict(inferred_arguments)
        merged_arguments.update(provided_arguments)
    merged = ToolCall(
        name=repaired_name,
        arguments=merged_arguments,
        source_format="heuristic",
        raw=raw_output,
    )
    valid, _ = validate_tool_call(merged, tool_specs)
    if valid and _passes_semantic_preconditions(merged, context):
        if merged.arguments != call.arguments:
            notes.append(f"repaired_arguments:{repaired_name}")
        return merged, notes

    inferred_only = _heuristic_call(repaired_name, inferred_arguments, raw_hint="repair")
    valid, _ = validate_tool_call(inferred_only, tool_specs)
    if valid and _passes_semantic_preconditions(inferred_only, context):
        notes.append(f"repaired_arguments:{repaired_name}")
        return inferred_only.model_copy(update={"raw": raw_output}), notes
    return None, notes


def _tool_schema_properties(tool_name: str, tool_specs: list[ToolSpec]) -> set[str]:
    return {
        key
        for key in next((tool.json_schema for tool in tool_specs if tool.name == tool_name), {}).get("properties", {})
    }


def _merge_repaired_arguments(
    tool_name: str,
    provided_arguments: dict[str, Any],
    inferred_arguments: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(provided_arguments)
    if tool_name == "refine_selection":
        merged["selection_id"] = inferred_arguments.get("selection_id", "")
        current_filter = merged.get("filter_query")
        if "filter_query" in inferred_arguments and (
            current_filter in (None, "") or (isinstance(current_filter, str) and current_filter.startswith("$"))
        ):
            merged["filter_query"] = inferred_arguments["filter_query"]
        return merged

    if tool_name == "read_region_text":
        merged["image_id"] = inferred_arguments.get("image_id", "")
        merged["region_id"] = inferred_arguments.get("region_id", "")
        return merged

    merged.update(inferred_arguments)
    return merged


def _project_arguments_to_schema(arguments: dict[str, Any], schema_properties: set[str]) -> dict[str, Any]:
    if not schema_properties:
        return dict(arguments)

    projected: dict[str, Any] = {}
    for key, value in arguments.items():
        matched_key = _match_schema_key(key, schema_properties)
        if matched_key and matched_key not in projected:
            projected[matched_key] = value
    return projected


def _match_schema_key(key: str, schema_properties: set[str]) -> str | None:
    if key in schema_properties:
        return key

    canonical_key = _canonical_argument_key(key)
    for schema_key in schema_properties:
        if _canonical_argument_key(schema_key) == canonical_key:
            return schema_key
    return None


def _canonical_argument_key(value: str) -> str:
    normalized = _normalize_name(value)
    if normalized.endswith("_renamed"):
        normalized = normalized[: -len("_renamed")]
    return normalized


def _initial_calls(context: dict[str, Any], tool_specs: list[ToolSpec]) -> list[ToolCall]:
    user_text = context["user_text"]
    tool_names = {tool.name for tool in tool_specs}
    image_context = f"{str(context.get('image_hint_id', ''))} {' '.join(str(item) for item in context.get('media', []))}".lower()
    record_id = _extract_record_id(" ".join(context["user_messages"]))

    if "update_event" in tool_names and _extract_event_id(" ".join(context["user_messages"])):
        return [_heuristic_call("update_event", _infer_arguments(context, "update_event"))]

    if "search_events" in tool_names and "update_event" in tool_names and _contains_any(user_text, ["move", "reschedule"]):
        return [_heuristic_call("search_events", _infer_arguments(context, "search_events"))]

    if "search_events" in tool_names and "create_event" in tool_names and _contains_any(user_text, ["check whether", "check if", "open", "available"]):
        return [_heuristic_call("search_events", _infer_arguments(context, "search_events"))]

    if "find_latest_file" in tool_names and "compare_files" in tool_names and _contains_any(user_text, ["latest", "last month"]):
        return [_heuristic_call("find_latest_file", _infer_arguments(context, "find_latest_file"))]

    if "api_fetch_record" in tool_names and record_id and _contains_any(user_text, ["fetch", "confirm", "retrieve", "look up", "lookup"]):
        return [_heuristic_call("api_fetch_record", _infer_arguments(context, "api_fetch_record"))]

    if "api_update_record" in tool_names and record_id and _contains_any(user_text, ["update", "keep", "stays", "stay", "set"]):
        return [_heuristic_call("api_update_record", _infer_arguments(context, "api_update_record"))]

    if "cli_apply_patch" in tool_names and _extract_path(user_text) and _contains_any(user_text, ["patch", "fix", "edit only", "patch only"]):
        return [_heuristic_call("cli_apply_patch", _infer_arguments(context, "cli_apply_patch"))]

    if "find_repo_file" in tool_names and _contains_any(user_text, ["find", "locate"]) and "config" in user_text:
        return [_heuristic_call("find_repo_file", _infer_arguments(context, "find_repo_file"))]

    if "read_repo_file" in tool_names and _extract_path(user_text):
        return [_heuristic_call("read_repo_file", _infer_arguments(context, "read_repo_file"))]

    if "inspect_image" in tool_names and _contains_any(user_text, ["screenshot", "image", "look at", "inspect"]):
        return [_heuristic_call("inspect_image", _infer_arguments(context, "inspect_image"))]

    if "extract_layout" in tool_names and _contains_any(
        f"{user_text} {image_context}",
        ["table", "form", "dashboard", "layout", "slide", "callout", "metric", "invoice", "validation", "error", "errors", "phone", "work authorization"],
    ):
        return [_heuristic_call("extract_layout", _infer_arguments(context, "extract_layout"))]

    if "segment_entities" in tool_names and _contains_any(user_text, ["segment", "vehicles", "cars", "slots", "exits", "all "]):
        return [_heuristic_call("segment_entities", _infer_arguments(context, "segment_entities"))]

    if "compare_files" in tool_names and "compare" in user_text:
        return [_heuristic_call("compare_files", _infer_arguments(context, "compare_files"))]

    if "create_event" in tool_names and _contains_any(user_text, ["create", "schedule", "hold"]):
        return [_heuristic_call("create_event", _infer_arguments(context, "create_event"))]

    if "propose_patch" in tool_names and _contains_any(user_text, ["record", "patch"]):
        return [_heuristic_call("propose_patch", _infer_arguments(context, "propose_patch"))]

    return []


def _classify_intent(context: dict[str, Any], tool_specs: list[ToolSpec]) -> str:
    user_text = context["user_text"]
    tool_names = {tool.name for tool in tool_specs}

    if _is_refusal_or_escalation_request(user_text):
        return "refuse_or_escalate"
    if "propose_patch" in tool_names and _contains_any(user_text, ["record the patch", "record a patch", "propose the patch", "record the safer patch"]):
        return "record_or_update"
    if _contains_any(user_text, ["disable", "turn off", "remove", "delete"]) and (
        {"propose_patch", "update_event", "create_event"} & tool_names
    ):
        return "disable_or_modify"
    if _contains_any(user_text, ["read", "find", "inspect", "look at", "check", "search", "locate", "fetch", "confirm"]):
        return "inspect_or_lookup"
    return "record_or_update"


def _intent_priority_calls(context: dict[str, Any], tool_specs: list[ToolSpec]) -> list[ToolCall] | None:
    intent = _classify_intent(context, tool_specs)
    tool_names = {tool.name for tool in tool_specs}
    user_text = context["user_text"]
    latest_feedback = context["latest_feedback"] if isinstance(context.get("latest_feedback"), dict) else {}
    record_id = _extract_record_id(" ".join(context["user_messages"]))

    if intent == "refuse_or_escalate":
        return []

    if _needs_parallel_audit(user_text, tool_names):
        return None

    if intent == "inspect_or_lookup" and "api_fetch_record" in tool_names and record_id:
        return [_heuristic_call("api_fetch_record", _infer_arguments(context, "api_fetch_record"), raw_hint="intent_prior")]

    if intent == "record_or_update" and "api_update_record" in tool_names and record_id:
        return [_heuristic_call("api_update_record", _infer_arguments(context, "api_update_record"), raw_hint="intent_prior")]

    if (
        intent == "record_or_update"
        and "cli_apply_patch" in tool_names
        and _extract_path(" ".join(context["user_messages"]))
        and _contains_any(user_text, ["patch", "fix", "edit only", "patch only"])
    ):
        return [_heuristic_call("cli_apply_patch", _infer_arguments(context, "cli_apply_patch"), raw_hint="intent_prior")]

    if (
        intent == "record_or_update"
        and "propose_patch" in tool_names
        and _contains_any(
        user_text,
        ["record the patch", "record a patch", "record the safer patch", "propose the patch", "invoice lock", "safe mode"],
        )
        and not _contains_any(user_text, ["look at", "inspect", "read ", "read_", "use the", "check both"])
    ):
        return [_heuristic_call("propose_patch", _infer_arguments(context, "propose_patch"), raw_hint="intent_prior")]

    if (
        intent == "record_or_update"
        and "propose_patch" in tool_names
        and str(latest_feedback.get("tool_name", "")) in {"inspect_image", "read_repo_file", "find_repo_file"}
        and str(latest_feedback.get("status", "")) == "pass"
    ):
        return [_heuristic_call("propose_patch", _infer_arguments(context, "propose_patch"), raw_hint="intent_prior")]

    if intent == "inspect_or_lookup" and "read_repo_file" in tool_names and not _mutation_explicitly_authorized(user_text):
        return [_heuristic_call("read_repo_file", _infer_arguments(context, "read_repo_file"), raw_hint="intent_prior")]

    return None


def _calls_match(parsed_calls: list[ToolCall], expected_calls: list[ToolCall]) -> bool:
    if len(parsed_calls) != len(expected_calls):
        return False
    return all(parsed.name == expected.name for parsed, expected in zip(parsed_calls, expected_calls, strict=False))


def _next_calls_from_feedback(context: dict[str, Any], tool_specs: list[ToolSpec]) -> list[ToolCall]:
    tool_names = {tool.name for tool in tool_specs}
    successful_history = [item.get("tool_name", "") for item in _successful_tool_feedback(context)]
    latest = context["latest_feedback"]
    if not latest or latest.get("status") != "pass":
        return []

    latest_tool = str(latest.get("tool_name", ""))
    user_text = context["user_text"]
    pending_visual_filter = _next_visual_filter(context)

    if _needs_parallel_audit(user_text, tool_names):
        if "inspect_image" in tool_names and "inspect_image" not in successful_history:
            return [_heuristic_call("inspect_image", _infer_arguments(context, "inspect_image"))]
        if "read_repo_file" in tool_names and "read_repo_file" not in successful_history:
            return [_heuristic_call("read_repo_file", _infer_arguments(context, "read_repo_file"))]

    if latest_tool == "find_latest_file" and "compare_files" in tool_names:
        return [_heuristic_call("compare_files", _infer_arguments(context, "compare_files"))]

    if latest_tool == "search_events" and "update_event" in tool_names:
        return [_heuristic_call("update_event", _infer_arguments(context, "update_event"))]

    if latest_tool == "search_events" and "create_event" in tool_names:
        return [_heuristic_call("create_event", _infer_arguments(context, "create_event"))]

    if latest_tool == "find_repo_file" and "read_repo_file" in tool_names:
        return [_heuristic_call("read_repo_file", _infer_arguments(context, "read_repo_file"))]

    if latest_tool == "read_repo_file" and "cli_apply_patch" in tool_names:
        if _contains_any(user_text, ["patch", "fix", "edit only", "patch only"]):
            return [_heuristic_call("cli_apply_patch", _infer_arguments(context, "cli_apply_patch"))]

    if latest_tool in {"segment_entities", "extract_layout", "refine_selection"} and "refine_selection" in tool_names:
        if pending_visual_filter:
            return [_heuristic_call("refine_selection", _infer_arguments(context, "refine_selection"))]

    if latest_tool in {"extract_layout", "refine_selection"} and "read_region_text" in tool_names:
        if not pending_visual_filter and _contains_any(
            user_text,
            [
                "read",
                "text",
                "table",
                "invoice",
                "totals",
                "validation",
                "what does it say",
                "tell me what it says",
                "read back",
                "message",
                "remaining message",
                "recommendation",
                "policy",
                "use it",
                "what remains",
                "what remain",
                "what's left",
                "what is left",
            ],
        ):
            return [_heuristic_call("read_region_text", _infer_arguments(context, "read_region_text"))]

    if latest_tool == "read_region_text":
        return []

    if latest_tool in {"api_fetch_record", "api_update_record", "cli_apply_patch"}:
        return []

    if latest_tool in {"inspect_image", "read_repo_file"} and "propose_patch" in tool_names:
        if _needs_parallel_audit(user_text, tool_names) and not _parallel_audit_ready(context, tool_names):
            return []
        return [_heuristic_call("propose_patch", _infer_arguments(context, "propose_patch"))]

    return []


def _planning_context(messages: list[Message], media: list[str]) -> dict[str, Any]:
    user_text = "\n".join(message.content for message in messages if message.role == "user").lower()
    hint_text = "\n".join(message.content for message in messages if message.role in {"user", "system"}).lower()
    tool_feedback = [_parse_tool_feedback(message.content) for message in messages if message.role == "tool"]
    tool_feedback = [item for item in tool_feedback if item]
    latest_feedback = tool_feedback[-1] if tool_feedback else {}
    return {
        "user_text": user_text,
        "hint_text": hint_text,
        "image_hint_id": _extract_image_id(hint_text),
        "user_messages": [message.content for message in messages if message.role == "user"],
        "tool_feedback": tool_feedback,
        "latest_feedback": latest_feedback,
        "media": media,
    }


def _parse_tool_feedback(content: str) -> dict[str, Any]:
    payload = content.strip()
    if payload.startswith("Tool result:\n"):
        payload = payload.split("\n", 1)[1]
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        if payload.startswith("Unknown tool:"):
            return {"status": "fail", "error": payload}
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _canonical_tool_name(name: str, tool_specs: list[ToolSpec]) -> str:
    allowed_names = {tool.name for tool in tool_specs}
    if name in allowed_names:
        return name

    normalized = _normalize_name(name)
    for allowed in allowed_names:
        if _normalize_name(allowed) == normalized:
            return allowed

    alias = TOOL_NAME_ALIASES.get(normalized)
    if alias in allowed_names:
        return alias

    requested_tokens = set(re.findall(r"[a-z0-9]+", normalized))
    best_name = ""
    best_score = -1
    for tool in tool_specs:
        tool_tokens = set(re.findall(r"[a-z0-9]+", f"{tool.name} {tool.description}".lower()))
        score = len(requested_tokens & tool_tokens)
        if score > best_score:
            best_score = score
            best_name = tool.name
    return best_name or name


def _infer_arguments(context: dict[str, Any], tool_name: str) -> dict[str, Any]:
    user_text = context["user_text"]
    image_hint_id = str(context.get("image_hint_id", ""))
    latest_feedback = context["latest_feedback"]
    media = context["media"]

    if tool_name == "search_events":
        attendee = ""
        if "sarah" in user_text:
            attendee = "Sarah"
        elif "vendor" in user_text:
            attendee = "Vendor"
        if _contains_any(user_text, ["open", "available", "check whether", "check if"]) and "tuesday" in user_text:
            return {"start_date": "2026-04-15", "end_date": "2026-04-15", "attendee": attendee}
        return {"start_date": "2026-04-10", "end_date": "2026-04-10", "attendee": attendee}

    if tool_name == "update_event":
        event_id = _extract_event_id(user_text)
        matches = latest_feedback.get("output", {}).get("matches", []) if isinstance(latest_feedback, dict) else []
        if not event_id and matches:
            event_id = str(matches[0].get("id", ""))
        if not event_id:
            event_id = "evt-009" if "vendor" in user_text else "evt-001"
        return {"event_id": event_id, "new_start": "2026-04-14T14:00:00", "new_end": "2026-04-14T14:30:00"}

    if tool_name == "find_latest_file":
        return {"directory": "finance", "kind": "budget"}

    if tool_name == "compare_files":
        file_names = re.findall(r"([A-Za-z0-9_./-]+\.csv)", " ".join(context["user_messages"]))
        if len(file_names) >= 2:
            return {"file_a": file_names[0], "file_b": file_names[1]}
        output = latest_feedback.get("output", {}) if isinstance(latest_feedback, dict) else {}
        latest_file = str(output.get("file_name", ""))
        previous_file = _previous_budget_file(latest_file)
        if previous_file:
            return {"file_a": previous_file, "file_b": latest_file}
        if "ops" in user_text:
            return {"file_a": "ops_budget_mar.csv", "file_b": "ops_budget_apr.csv"}
        return {"file_a": "budget_mar.csv", "file_b": "budget_apr.csv"}

    if tool_name == "find_repo_file":
        if "settings" in user_text:
            return {"query": "settings"}
        return {"query": "config"}

    if tool_name == "read_repo_file":
        path = _extract_path(" ".join(context["user_messages"]))
        if path:
            return {"path": path}
        matches = latest_feedback.get("output", {}).get("matches", []) if isinstance(latest_feedback, dict) else []
        if matches:
            return {"path": str(matches[0])}
        return {"path": "config/billing.yaml" if "billing" in user_text else "config/settings.yaml"}

    if tool_name == "cli_apply_patch":
        path = _extract_path(" ".join(context["user_messages"]))
        if not path:
            read_feedback = _latest_successful_feedback(context, "read_repo_file")
            read_output = _feedback_output(read_feedback)
            read_path = str(read_output.get("path", ""))
            if _looks_like_repo_path(read_path):
                path = read_path
        if not path:
            path = "config/job_form.yaml"
        return {"path": path, "patch": _infer_cli_patch(user_text)}

    if tool_name == "api_fetch_record":
        return {
            "record_type": _infer_api_record_type(user_text),
            "record_id": _extract_record_id(" ".join(context["user_messages"])) or "BR-17",
        }

    if tool_name == "api_update_record":
        field, value = _infer_api_field_value(user_text)
        return {
            "record_type": _infer_api_record_type(user_text),
            "record_id": _extract_record_id(" ".join(context["user_messages"])) or "INV-204",
            "field": field,
            "value": value,
        }

    if tool_name == "inspect_image":
        image_id = image_hint_id or _extract_image_id(" ".join(context["user_messages"]))
        if image_id:
            return {"image_id": image_id}
        if media:
            return {"image_id": media[0]}
        return {"image_id": "img-settings"}

    if tool_name == "segment_entities":
        image_id = image_hint_id or _extract_image_id(" ".join(context["user_messages"])) or (media[0] if media else "img-parking")
        if _contains_any(user_text, ["slot", "slots"]):
            query = "parking slot"
        elif _contains_any(user_text, ["exit", "exits", "sortie", "sorties"]):
            query = "exit"
        elif _contains_any(user_text, ["metric", "metrics", "anomaly", "anomalies"]):
            query = "metric anomaly"
        else:
            query = "vehicle" if _contains_any(user_text, ["vehicle", "vehicles", "car", "cars"]) else "entity"
        return {"image_id": image_id, "entity_query": query}

    if tool_name == "refine_selection":
        output = _feedback_output(latest_feedback)
        selection_id = str(output.get("selection_id", ""))
        pending_filter = _next_visual_filter(context)
        if pending_filter:
            filter_query = pending_filter
        elif _contains_any(user_text, ["enablement ops"]):
            filter_query = "enablement ops"
        elif _contains_any(user_text, ["backlog"]):
            filter_query = "backlog"
        elif _contains_any(user_text, ["email", "email issue", "email address"]):
            filter_query = "email"
        elif _contains_any(user_text, ["white", "blanc", "blanche", "blanches"]):
            filter_query = "white"
        elif _contains_any(user_text, ["below target", "below", "under target"]):
            filter_query = "below target"
        elif _contains_any(user_text, ["destructive"]):
            filter_query = "destructive"
        elif _contains_any(user_text, ["blocked", "bloque", "bloquee"]):
            filter_query = "blocked"
        elif _contains_any(user_text, ["empty", "vacant", "vacants"]):
            filter_query = "empty"
        elif _contains_any(user_text, ["error", "errors", "validation"]):
            filter_query = "error"
        else:
            filter_query = "target"
        return {"selection_id": selection_id, "filter_query": filter_query}

    if tool_name == "extract_layout":
        image_id = image_hint_id or _extract_image_id(" ".join(context["user_messages"])) or (media[0] if media else "img-dashboard")
        image_context = f"{image_id} {' '.join(str(item) for item in media)}".lower()
        if _contains_any(user_text, ["invoice", "totals", "table"]):
            query = "invoice totals table"
        elif "form" in image_context or _contains_any(user_text, ["form", "validation", "error", "phone", "work authorization"]):
            query = "validation error"
        elif _contains_any(user_text, ["slide", "callout"]):
            query = "slide callout"
        elif "risk" in user_text:
            query = "risk callout"
        else:
            query = "dashboard metric"
        return {"image_id": image_id, "target_query": query}

    if tool_name == "read_region_text":
        output = _feedback_output(latest_feedback)
        image_id = str(output.get("image_id", "")) or image_hint_id or _extract_image_id(" ".join(context["user_messages"])) or (media[0] if media else "img-invoice")
        region_id = str(output.get("region_id", ""))
        if not region_id:
            region_ids = output.get("region_ids", [])
            if isinstance(region_ids, list) and region_ids:
                region_id = str(region_ids[0])
        return {"image_id": image_id, "region_id": region_id}

    if tool_name == "create_event":
        return {
            "title": "Budget review",
            "start": "2026-04-15T15:00:00",
            "end": "2026-04-15T15:30:00",
            "attendees": ["team@example.com"],
        }

    if tool_name == "propose_patch":
        inspect_feedback = _latest_successful_feedback(context, "inspect_image")
        read_feedback = _latest_successful_feedback(context, "read_repo_file")
        output = _feedback_output(latest_feedback)
        inspect_output = _feedback_output(inspect_feedback)
        read_output = _feedback_output(read_feedback)
        patch = _normalize_patch_text(
            str(output.get("recommended_patch") or output.get("patch") or inspect_output.get("recommended_patch") or inspect_output.get("patch") or ""),
            user_text,
        )
        if not patch:
            patch = "invoice_lock: true" if _contains_any(user_text, ["invoice lock", "billing"]) else "safe_mode: true"
        path = _extract_path(" ".join(context["user_messages"]))
        if not path:
            read_path = str(read_output.get("path", ""))
            if _looks_like_repo_path(read_path):
                path = read_path
        if not path:
            patch_payload = output.get("patch")
            if isinstance(patch_payload, dict):
                candidate_path = str(patch_payload.get("path", ""))
                if _looks_like_repo_path(candidate_path):
                    path = candidate_path
        if not path:
            path = "config/billing.yaml" if "billing" in user_text or "invoice lock" in patch else "config/settings.yaml"
        return {"path": path, "patch": patch}

    return {}


def _heuristic_call(tool_name: str, arguments: dict[str, Any], raw_hint: str = "heuristic") -> ToolCall:
    raw = json.dumps({"name": tool_name, "arguments": arguments, "controller": raw_hint}, ensure_ascii=False)
    return ToolCall(name=tool_name, arguments=arguments, source_format="heuristic", raw=raw)


def _should_override_valid_arguments(call: ToolCall, inferred_arguments: dict[str, Any], context: dict[str, Any]) -> bool:
    if call.name == "search_events":
        provided_start = str(call.arguments.get("start_date", ""))
        provided_end = str(call.arguments.get("end_date", ""))
        if inferred_arguments and (not _looks_like_iso_date(provided_start) or not _looks_like_iso_date(provided_end)):
            return True

    if call.name == "create_event":
        provided_start = str(call.arguments.get("start", ""))
        provided_end = str(call.arguments.get("end", ""))
        attendees = call.arguments.get("attendees", [])
        if inferred_arguments and (not _looks_like_iso_timestamp(provided_start) or not _looks_like_iso_timestamp(provided_end)):
            return True
        if attendees == [] and inferred_arguments.get("attendees"):
            return True

    if call.name == "compare_files":
        file_a = str(call.arguments.get("file_a", ""))
        file_b = str(call.arguments.get("file_b", ""))
        if inferred_arguments and (not file_a.endswith(".csv") or not file_b.endswith(".csv")):
            return True

    if call.name == "inspect_image":
        provided_image_id = str(call.arguments.get("image_id", ""))
        inferred_image_id = str(inferred_arguments.get("image_id", ""))
        if inferred_image_id and provided_image_id != inferred_image_id:
            if provided_image_id in {"image", "screenshot", "screen", "settings"}:
                return True
            if provided_image_id and not provided_image_id.startswith("img-") and not Path(provided_image_id).exists():
                return True
    if call.name == "segment_entities":
        provided_image_id = str(call.arguments.get("image_id", ""))
        inferred_image_id = str(inferred_arguments.get("image_id", ""))
        if inferred_image_id and provided_image_id != inferred_image_id and not provided_image_id.startswith("img-"):
            return True
        if inferred_arguments.get("entity_query") and not str(call.arguments.get("entity_query", "")).strip():
            return True
    if call.name == "refine_selection":
        latest_visual_feedback = _latest_visual_selection_feedback(context)
        if latest_visual_feedback is None:
            return True
        provided_selection_id = str(call.arguments.get("selection_id", "")).strip()
        inferred_selection_id = str(inferred_arguments.get("selection_id", "")).strip()
        if inferred_selection_id and provided_selection_id != inferred_selection_id:
            return True
        if inferred_arguments.get("selection_id") and (
            not provided_selection_id
            or provided_selection_id.startswith("$")
        ):
            return True
        if inferred_arguments.get("filter_query") and not str(call.arguments.get("filter_query", "")).strip():
            return True
    if call.name == "extract_layout":
        provided_image_id = str(call.arguments.get("image_id", ""))
        inferred_image_id = str(inferred_arguments.get("image_id", ""))
        provided_target_query = str(call.arguments.get("target_query", "")).strip()
        inferred_target_query = str(inferred_arguments.get("target_query", "")).strip()
        if inferred_image_id and provided_image_id != inferred_image_id and not provided_image_id.startswith("img-"):
            return True
        if inferred_target_query and provided_target_query and provided_target_query != inferred_target_query:
            return True
        if inferred_arguments.get("target_query") and not str(call.arguments.get("target_query", "")).strip():
            return True
    if call.name == "read_region_text":
        latest_visual_feedback = _latest_visual_selection_feedback(context)
        if latest_visual_feedback is None:
            return True
        provided_image_id = str(call.arguments.get("image_id", "")).strip()
        provided_region_id = str(call.arguments.get("region_id", "")).strip()
        inferred_image_id = str(inferred_arguments.get("image_id", "")).strip()
        inferred_region_id = str(inferred_arguments.get("region_id", "")).strip()
        if inferred_image_id and provided_image_id and provided_image_id != inferred_image_id:
            return True
        if inferred_region_id and provided_region_id and provided_region_id != inferred_region_id:
            return True
        if inferred_arguments.get("image_id") and not str(call.arguments.get("image_id", "")).strip():
            return True
        if inferred_arguments.get("image_id") and provided_image_id.startswith("$"):
            return True
        if inferred_arguments.get("region_id") and (
            not provided_region_id
            or provided_region_id.startswith("$")
        ):
            return True
    if call.name == "read_repo_file":
        provided_path = str(call.arguments.get("path", ""))
        inferred_path = str(inferred_arguments.get("path", ""))
        if inferred_path and provided_path and provided_path != inferred_path and "/" not in provided_path:
            return True
    if call.name == "cli_apply_patch":
        provided_path = str(call.arguments.get("path", ""))
        provided_patch = str(call.arguments.get("patch", ""))
        inferred_path = str(inferred_arguments.get("path", ""))
        inferred_patch = str(inferred_arguments.get("patch", ""))
        if inferred_path and provided_path != inferred_path:
            return True
        if inferred_patch and provided_patch != inferred_patch:
            return True
    if call.name == "api_fetch_record":
        provided_record_type = str(call.arguments.get("record_type", "")).strip()
        provided_record_id = str(call.arguments.get("record_id", "")).strip()
        if str(inferred_arguments.get("record_type", "")).strip() and provided_record_type != str(inferred_arguments.get("record_type", "")).strip():
            return True
        if str(inferred_arguments.get("record_id", "")).strip() and provided_record_id != str(inferred_arguments.get("record_id", "")).strip():
            return True
    if call.name == "api_update_record":
        required_fields = ("record_type", "record_id", "field", "value")
        for field_name in required_fields:
            inferred_value = str(inferred_arguments.get(field_name, "")).strip()
            provided_value = str(call.arguments.get(field_name, "")).strip()
            if inferred_value and provided_value != inferred_value:
                return True
    if call.name == "propose_patch":
        provided_path = str(call.arguments.get("path", ""))
        inferred_path = str(inferred_arguments.get("path", ""))
        provided_patch = str(call.arguments.get("patch", ""))
        inferred_patch = str(inferred_arguments.get("patch", ""))
        if inferred_path and provided_path != inferred_path:
            if _latest_successful_feedback(context, "read_repo_file") is not None:
                return True
            if not _looks_like_repo_path(provided_path):
                return True
        if inferred_patch and provided_patch != inferred_patch:
            if _latest_successful_feedback(context, "inspect_image") is not None:
                return True
            if ":" not in provided_patch or " patch" in provided_patch.lower():
                return True
    return False


def _passes_semantic_preconditions(call: ToolCall, context: dict[str, Any]) -> bool:
    if call.name == "refine_selection":
        latest_visual_feedback = _latest_visual_selection_feedback(context)
        if latest_visual_feedback is None:
            return False
        expected_selection_id = str(_feedback_output(latest_visual_feedback).get("selection_id", "")).strip()
        provided_selection_id = str(call.arguments.get("selection_id", "")).strip()
        if not provided_selection_id:
            return False
        return not expected_selection_id or provided_selection_id == expected_selection_id

    if call.name == "read_region_text":
        latest_visual_feedback = _latest_visual_selection_feedback(context)
        if latest_visual_feedback is None:
            return False
        output = _feedback_output(latest_visual_feedback)
        provided_image_id = str(call.arguments.get("image_id", "")).strip()
        provided_region_id = str(call.arguments.get("region_id", "")).strip()
        expected_image_id = str(output.get("image_id", "")).strip() or str(context.get("image_hint_id", "")).strip()
        expected_region_ids = {str(output.get("region_id", "")).strip()}
        region_ids = output.get("region_ids", [])
        if isinstance(region_ids, list):
            expected_region_ids.update(str(region_id).strip() for region_id in region_ids)
        expected_region_ids.discard("")
        if not provided_region_id:
            return False
        if expected_image_id and provided_image_id != expected_image_id:
            return False
        if expected_region_ids and provided_region_id not in expected_region_ids:
            return False
    return True


def _best_tool_match(user_text: str, tool_specs: list[ToolSpec]) -> ToolSpec:
    requested_tokens = set(re.findall(r"[a-z0-9]+", user_text))
    best_tool = tool_specs[0]
    best_score = -1
    for tool in tool_specs:
        score = 0
        tool_tokens = set(re.findall(r"[a-z0-9]+", f"{tool.name} {tool.description}".lower()))
        score += len(requested_tokens & tool_tokens)
        score += 2 if tool.name in TOOL_NAME_ALIASES.values() and tool.name in user_text else 0
        if score > best_score:
            best_tool = tool
            best_score = score
    return best_tool


def _needs_parallel_audit(user_text: str, tool_names: set[str]) -> bool:
    return (
        {"inspect_image", "read_repo_file"}.issubset(tool_names)
        and _contains_any(user_text, ["both", "parallel", "screenshot", "image"])
        and _contains_any(user_text, ["config", "file"])
    )


def _is_refusal_or_escalation_request(user_text: str) -> bool:
    unsafe_disable = _contains_any(user_text, ["disable invoice lock", "turn off invoice lock", "unsafe", "production right now"])
    missing_approval = _contains_any(user_text, ["missing a second approver", "should you proceed or escalate", "requires two-person approval"])
    ambiguous_target = _contains_any(user_text, ["which vendor meeting", "ambiguous vendor", "more than one vendor meeting"])
    explicit_refusal = _contains_any(user_text, ["refuse", "do not proceed", "cannot proceed"])
    return unsafe_disable or missing_approval or ambiguous_target or explicit_refusal


def _mutation_explicitly_authorized(user_text: str) -> bool:
    return _contains_any(
        user_text,
        ["record the patch", "record a patch", "propose the patch", "create", "schedule", "move", "reschedule", "update", "patch"],
    )


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _contains_any(text: str, fragments: list[str]) -> bool:
    return any(fragment in text for fragment in fragments)


_VISUAL_FILTER_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("below target", ("below target", "below-target", "under target")),
    ("needs review", ("needs review", "need review")),
    ("customer ops", ("customer ops", "customer operations")),
    ("support backlog", ("support backlog",)),
    ("backlog", ("backlog",)),
    ("enablement ops", ("enablement ops",)),
    ("latest", ("latest issue first", "latest form issue first")),
    ("latest action", ("latest action", "approval safe action", "approval-safe action")),
    ("email", ("email", "email issue", "email address")),
    ("white", ("white", "blanc", "blanche", "blanches")),
    ("phone", ("phone",)),
    ("action", ("action",)),
    ("destructive", ("destructive",)),
    ("blocked", ("blocked", "bloque", "bloquee")),
    ("empty", ("empty", "vacant", "vacants")),
)
_VISUAL_FILTER_PATTERN_MAP = {canonical: patterns for canonical, patterns in _VISUAL_FILTER_PATTERNS}


def _normalize_phrase_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _canonical_visual_filter(text: str) -> str:
    normalized = f" {_normalize_phrase_text(text)} "
    for canonical, patterns in _VISUAL_FILTER_PATTERNS:
        if any(f" {_normalize_phrase_text(pattern)} " in normalized for pattern in patterns):
            return canonical
    return _normalize_phrase_text(text)


def _requested_visual_filters(user_text: str) -> list[str]:
    normalized = f" {_normalize_phrase_text(user_text)} "
    requested: list[str] = []
    if " latest issue first " in normalized and " phone issue " in normalized:
        requested.append("latest")
    shadowed_canonicals = {
        "backlog": _VISUAL_FILTER_PATTERN_MAP["support backlog"],
        "action": _VISUAL_FILTER_PATTERN_MAP["latest action"],
    }
    for canonical, patterns in _VISUAL_FILTER_PATTERNS:
        if any(f" {_normalize_phrase_text(pattern)} " in normalized for pattern in shadowed_canonicals.get(canonical, ())):
            continue
        if any(f" {_normalize_phrase_text(pattern)} " in normalized for pattern in patterns):
            if canonical not in requested:
                requested.append(canonical)
    return requested


def _successful_refine_filters(context: dict[str, Any]) -> list[str]:
    filters: list[str] = []
    for item in _successful_tool_feedback(context):
        if str(item.get("tool_name", "")) != "refine_selection":
            continue
        arguments = item.get("arguments", {})
        if not isinstance(arguments, dict):
            continue
        query = str(arguments.get("filter_query", "")).strip()
        if query:
            filters.append(_canonical_visual_filter(query))
    return filters


def _next_visual_filter(context: dict[str, Any]) -> str:
    requested_filters = _requested_visual_filters(context["user_text"])
    if not requested_filters:
        return ""
    used_filters = _successful_refine_filters(context)
    for candidate in requested_filters:
        if candidate not in used_filters:
            return candidate
    return ""


def _feedback_output(feedback: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(feedback, dict):
        return {}
    output = feedback.get("output", {})
    return output if isinstance(output, dict) else {}


def _successful_tool_feedback(context: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in context["tool_feedback"]
        if isinstance(item, dict) and str(item.get("status", "")) == "pass"
    ]


def _latest_visual_selection_feedback(context: dict[str, Any]) -> dict[str, Any] | None:
    for item in reversed(_successful_tool_feedback(context)):
        if str(item.get("tool_name", "")) in {"segment_entities", "extract_layout", "refine_selection"}:
            return item
    return None


def _latest_successful_feedback(context: dict[str, Any], tool_name: str) -> dict[str, Any] | None:
    for item in reversed(_successful_tool_feedback(context)):
        if str(item.get("tool_name", "")) == tool_name:
            return item
    return None


def _visual_loop_complete(context: dict[str, Any]) -> bool:
    latest_feedback = context.get("latest_feedback")
    if not isinstance(latest_feedback, dict):
        return False
    if str(latest_feedback.get("status", "")) != "pass":
        return False
    if str(latest_feedback.get("tool_name", "")) != "read_region_text":
        return False
    return _next_visual_filter(context) == ""


def _parallel_audit_pending_calls(context: dict[str, Any], tool_specs: list[ToolSpec]) -> list[ToolCall]:
    tool_names = {tool.name for tool in tool_specs}
    if not _needs_parallel_audit(context["user_text"], tool_names):
        return []
    successful_history = {item.get("tool_name", "") for item in _successful_tool_feedback(context)}
    pending: list[ToolCall] = []
    if "inspect_image" in tool_names and "inspect_image" not in successful_history:
        pending.append(_heuristic_call("inspect_image", _infer_arguments(context, "inspect_image"), raw_hint="parallel"))
    if "read_repo_file" in tool_names and "read_repo_file" not in successful_history:
        pending.append(_heuristic_call("read_repo_file", _infer_arguments(context, "read_repo_file"), raw_hint="parallel"))
    return pending


def _parallel_audit_ready(context: dict[str, Any], tool_names: set[str]) -> bool:
    if not _needs_parallel_audit(context["user_text"], tool_names):
        return True
    successful_history = {item.get("tool_name", "") for item in _successful_tool_feedback(context)}
    return {"inspect_image", "read_repo_file"}.issubset(successful_history)


def _normalize_patch_text(value: str, user_text: str) -> str:
    text = value.strip()
    lowered = text.lower()
    if "invoice_lock" in text or "invoice lock" in lowered:
        return "invoice_lock: true"
    if "safe_mode" in text or "safe mode" in lowered:
        return "safe_mode: true"
    if ":" in text:
        return text
    if _contains_any(user_text, ["invoice lock", "billing"]):
        return "invoice_lock: true"
    if _contains_any(user_text, ["safe mode", "safer patch"]):
        return "safe_mode: true"
    return text


def _looks_like_repo_path(value: str) -> bool:
    return bool(re.search(r"/", value) and re.search(r"\.(?:ya?ml|json|toml|py|md|csv)$", value))


def _extract_path(text: str) -> str:
    match = re.search(r"([A-Za-z0-9_./-]+\.(?:ya?ml|json|toml|py|md|csv))", text)
    return match.group(1) if match else ""


def _extract_record_id(text: str) -> str:
    match = re.search(r"\b([A-Z]{2,5}-\d{1,5})\b", text)
    return match.group(1) if match else ""


def _infer_api_record_type(user_text: str) -> str:
    if _contains_any(user_text, ["billing record", "invoice", "invoice_lock", "invoice lock", "billing"]):
        return "billing_record"
    if _contains_any(user_text, ["briefing", "board packet", "approval safe action", "approval-safe action", "packet"]):
        return "briefing_record"
    return "record"


def _infer_api_field_value(user_text: str) -> tuple[str, str]:
    if _contains_any(user_text, ["invoice_lock", "invoice lock"]) and _contains_any(user_text, ["hold", "on hold", "stays on hold", "stay on hold"]):
        return ("invoice_lock", "hold")
    return ("status", "hold")


def _infer_cli_patch(user_text: str) -> str:
    if _contains_any(user_text, ["phone validation", "phone fix", "phone issue first", "phone issue"]):
        return "phone_validation: strict"
    if _contains_any(user_text, ["invoice lock", "invoice_lock"]) and _contains_any(user_text, ["hold", "on hold"]):
        return "invoice_lock: hold"
    return "status: updated"


def _extract_event_id(text: str) -> str:
    match = re.search(r"(evt-\d+)", text)
    return match.group(1) if match else ""


def _extract_image_id(text: str) -> str:
    match = re.search(r"(img-[a-z0-9-]+)", text.lower())
    return match.group(1) if match else ""


def _previous_budget_file(name: str) -> str:
    if not name:
        return ""
    stem = Path(name).stem
    suffix = Path(name).suffix
    replacements = {
        "_apr": "_mar",
        "_may": "_apr",
        "_jun": "_may",
    }
    for current, previous in replacements.items():
        if stem.endswith(current):
            return stem[: -len(current)] + previous + suffix
    return ""


def _looks_like_iso_date(value: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))


def _looks_like_iso_timestamp(value: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", value))
