from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

from gemma4_capability_map.schemas import ToolSpec
from gemma4_capability_map.tools.visual_executor import VisualExecutor, build_visual_executor

ToolHandler = Callable[[dict[str, Any], dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]]


@dataclass
class ToolRegistry:
    handlers: dict[str, ToolHandler] = field(default_factory=dict)
    specs: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        self.handlers[spec.name] = handler
        self.specs[spec.name] = spec

    def execute(self, state: dict[str, Any], name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if name not in self.handlers:
            raise KeyError(f"No handler registered for tool: {name}")
        return self.handlers[name](deepcopy(state), arguments)

    def list_specs(self) -> list[ToolSpec]:
        return list(self.specs.values())


def build_default_registry(visual_executor: VisualExecutor | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    visual_executor = visual_executor or build_visual_executor()
    registry.register(
        ToolSpec(
            name="find_latest_file",
            description="Find the latest file in a logical directory by kind.",
            tool_family="function_call",
            tool_intent="search",
            schema={
                "type": "object",
                "properties": {
                    "directory": {"type": "string"},
                    "kind": {"type": "string"},
                },
                "required": ["directory", "kind"],
            },
        ),
        _find_latest_file,
    )
    registry.register(
        ToolSpec(
            name="compare_files",
            description="Compare two budget files and compute deltas.",
            tool_family="function_call",
            tool_intent="inspect",
            schema={
                "type": "object",
                "properties": {
                    "file_a": {"type": "string"},
                    "file_b": {"type": "string"},
                },
                "required": ["file_a", "file_b"],
            },
        ),
        _compare_files,
    )
    registry.register(
        ToolSpec(
            name="search_events",
            description="Search calendar events by date range and attendee.",
            tool_family="function_call",
            tool_intent="search",
            schema={
                "type": "object",
                "properties": {
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "attendee": {"type": "string"},
                },
                "required": ["start_date", "end_date"],
            },
        ),
        _search_events,
    )
    registry.register(
        ToolSpec(
            name="update_event",
            description="Update an existing calendar event.",
            tool_family="function_call",
            tool_intent="write",
            schema={
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "new_start": {"type": "string"},
                    "new_end": {"type": "string"},
                },
                "required": ["event_id", "new_start", "new_end"],
            },
        ),
        _update_event,
    )
    registry.register(
        ToolSpec(
            name="create_event",
            description="Create a new calendar event.",
            tool_family="function_call",
            tool_intent="write",
            schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "attendees": {"type": "array"},
                },
                "required": ["title", "start", "end"],
            },
        ),
        _create_event,
    )
    registry.register(
        ToolSpec(
            name="find_repo_file",
            description="Search the repo for a likely config file.",
            tool_family="function_call",
            tool_intent="search",
            schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
        _find_repo_file,
    )
    registry.register(
        ToolSpec(
            name="read_repo_file",
            description="Read a repository file by path.",
            tool_family="function_call",
            tool_intent="read",
            schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        _read_repo_file,
    )
    registry.register(
        ToolSpec(
            name="propose_patch",
            description="Record a proposed patch for a repository file.",
            tool_family="function_call",
            tool_intent="patch",
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "patch": {"type": "string"},
                },
                "required": ["path", "patch"],
            },
        ),
        _propose_patch,
    )
    registry.register(
        ToolSpec(
            name="inspect_image",
            description="Inspect a screenshot or document image and extract the relevant action.",
            tool_family="function_call",
            tool_intent="inspect",
            schema={
                "type": "object",
                "properties": {"image_id": {"type": "string"}},
                "required": ["image_id"],
            },
        ),
        _inspect_image,
    )
    registry.register(
        ToolSpec(
            name="segment_entities",
            description="Segment or select entities in an image by query.",
            tool_family="function_call",
            tool_intent="inspect",
            schema={
                "type": "object",
                "properties": {
                    "image_id": {"type": "string"},
                    "entity_query": {"type": "string"},
                },
                "required": ["image_id", "entity_query"],
            },
        ),
        _segment_entities(visual_executor),
    )
    registry.register(
        ToolSpec(
            name="refine_selection",
            description="Refine a prior visual selection using a follow-up filter query.",
            tool_family="function_call",
            tool_intent="revise",
            schema={
                "type": "object",
                "properties": {
                    "selection_id": {"type": "string"},
                    "filter_query": {"type": "string"},
                },
                "required": ["selection_id", "filter_query"],
            },
        ),
        _refine_selection(visual_executor),
    )
    registry.register(
        ToolSpec(
            name="extract_layout",
            description="Extract a layout region such as a table, callout, validation error, or metric panel from an image.",
            tool_family="function_call",
            tool_intent="inspect",
            schema={
                "type": "object",
                "properties": {
                    "image_id": {"type": "string"},
                    "target_query": {"type": "string"},
                },
                "required": ["image_id", "target_query"],
            },
        ),
        _extract_layout(visual_executor),
    )
    registry.register(
        ToolSpec(
            name="read_region_text",
            description="Read the text inside a previously selected region.",
            tool_family="function_call",
            tool_intent="read",
            schema={
                "type": "object",
                "properties": {
                    "image_id": {"type": "string"},
                    "region_id": {"type": "string"},
                },
                "required": ["image_id", "region_id"],
            },
        ),
        _read_region_text(visual_executor),
    )
    registry.register(
        ToolSpec(
            name="cli_search_logs",
            description="Search a CLI-accessible log file for the latest matching line.",
            tool_family="cli",
            tool_intent="search",
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["path", "query"],
            },
        ),
        _cli_search_logs,
    )
    registry.register(
        ToolSpec(
            name="cli_apply_patch",
            description="Record a CLI patch action against a local config or source file.",
            tool_family="cli",
            tool_intent="patch",
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "patch": {"type": "string"},
                },
                "required": ["path", "patch"],
            },
        ),
        _cli_apply_patch,
    )
    registry.register(
        ToolSpec(
            name="cli_inspect_diff",
            description="Inspect a prepared CLI diff or patch excerpt by id.",
            tool_family="cli",
            tool_intent="inspect",
            schema={
                "type": "object",
                "properties": {
                    "diff_id": {"type": "string"},
                },
                "required": ["diff_id"],
            },
        ),
        _cli_inspect_diff,
    )
    registry.register(
        ToolSpec(
            name="api_fetch_record",
            description="Fetch a record from a simulated API by type and id.",
            tool_family="api",
            tool_intent="read",
            schema={
                "type": "object",
                "properties": {
                    "record_type": {"type": "string"},
                    "record_id": {"type": "string"},
                },
                "required": ["record_type", "record_id"],
            },
        ),
        _api_fetch_record,
    )
    registry.register(
        ToolSpec(
            name="api_update_record",
            description="Update a single field on a simulated API record.",
            tool_family="api",
            tool_intent="write",
            schema={
                "type": "object",
                "properties": {
                    "record_type": {"type": "string"},
                    "record_id": {"type": "string"},
                    "field": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["record_type", "record_id", "field", "value"],
            },
        ),
        _api_update_record,
    )
    return registry


def _find_latest_file(state: dict[str, Any], arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = arguments["directory"]
    kind = arguments["kind"].lower()
    files = [item for item in state.get("files", []) if item.get("directory") == directory and kind in item.get("name", "").lower()]
    latest = sorted(files, key=lambda item: item.get("timestamp", ""), reverse=True)[0]
    return state, {"file_id": latest["id"], "file_name": latest["name"], "content": latest["content"]}


def _compare_files(state: dict[str, Any], arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    files = {item["name"]: item for item in state.get("files", [])}
    file_a = files[arguments["file_a"]]
    file_b = files[arguments["file_b"]]
    delta = file_b.get("amount", 0) - file_a.get("amount", 0)
    return state, {"file_a": file_a["name"], "file_b": file_b["name"], "delta": delta}


def _search_events(state: dict[str, Any], arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    attendee = arguments.get("attendee", "").lower()
    matches = [
        event
        for event in state.get("calendar_events", [])
        if not attendee or attendee in " ".join(event.get("attendees", [])).lower()
    ]
    return state, {"matches": matches}


def _update_event(state: dict[str, Any], arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    for event in state.get("calendar_events", []):
        if event["id"] == arguments["event_id"]:
            event["start"] = arguments["new_start"]
            event["end"] = arguments["new_end"]
            return state, {"updated_event": event}
    raise KeyError(f"Event not found: {arguments['event_id']}")


def _create_event(state: dict[str, Any], arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    new_event = {
        "id": f"evt-{len(state.get('calendar_events', [])) + 1:03d}",
        "title": arguments["title"],
        "start": arguments["start"],
        "end": arguments["end"],
        "attendees": arguments.get("attendees", []),
    }
    state.setdefault("calendar_events", []).append(new_event)
    return state, {"created_event": new_event}


def _find_repo_file(state: dict[str, Any], arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    query = arguments["query"].lower()
    matches = [path for path in state.get("repo_files", {}) if query in path.lower()]
    return state, {"matches": matches}


def _read_repo_file(state: dict[str, Any], arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = arguments["path"]
    content = state.get("repo_files", {}).get(path)
    if content is None:
        raise KeyError(f"Repo file not found: {path}")
    return state, {"path": path, "content": content}


def _propose_patch(state: dict[str, Any], arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    patch = {"path": arguments["path"], "patch": arguments["patch"]}
    state.setdefault("proposed_patches", []).append(patch)
    return state, {"patch": patch}


def _inspect_image(state: dict[str, Any], arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    image = state.get("images", {}).get(arguments["image_id"])
    if image is None:
        raise KeyError(f"Image not found: {arguments['image_id']}")
    return state, image


def _cli_search_logs(state: dict[str, Any], arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = arguments["path"]
    query = arguments["query"].lower()
    raw = state.get("cli_logs", {}).get(path)
    if raw is None:
        raise KeyError(f"CLI log not found: {path}")
    lines = [line for line in str(raw).splitlines() if query in line.lower()]
    latest_match = lines[-1] if lines else ""
    return state, {"path": path, "query": arguments["query"], "matches": lines, "latest_match": latest_match}


def _cli_apply_patch(state: dict[str, Any], arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    patch = {"path": arguments["path"], "patch": arguments["patch"], "source": "cli"}
    state.setdefault("cli_patches", []).append(patch)
    return state, {"patch": patch}


def _cli_inspect_diff(state: dict[str, Any], arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    diff_id = arguments["diff_id"]
    diff = state.get("cli_diffs", {}).get(diff_id)
    if diff is None:
        raise KeyError(f"CLI diff not found: {diff_id}")
    return state, {"diff_id": diff_id, "diff": diff}


def _api_fetch_record(state: dict[str, Any], arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    record_type = arguments["record_type"]
    record_id = arguments["record_id"]
    record = deepcopy(state.get("api_records", {}).get(record_type, {}).get(record_id))
    if record is None:
        raise KeyError(f"API record not found: {record_type}:{record_id}")
    return state, {"record_type": record_type, "record_id": record_id, "record": record}


def _api_update_record(state: dict[str, Any], arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    record_type = arguments["record_type"]
    record_id = arguments["record_id"]
    field = arguments["field"]
    value = arguments["value"]
    records = state.setdefault("api_records", {}).setdefault(record_type, {})
    record = deepcopy(records.get(record_id))
    if record is None:
        raise KeyError(f"API record not found: {record_type}:{record_id}")
    record[field] = value
    records[record_id] = record
    return state, {"record_type": record_type, "record_id": record_id, "field": field, "value": value, "record": deepcopy(record)}


def _segment_entities(visual_executor: VisualExecutor) -> ToolHandler:
    def handler(state: dict[str, Any], arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return visual_executor.segment_entities(state, arguments["image_id"], arguments["entity_query"])

    return handler


def _refine_selection(visual_executor: VisualExecutor) -> ToolHandler:
    def handler(state: dict[str, Any], arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return visual_executor.refine_selection(state, arguments["selection_id"], arguments["filter_query"])

    return handler


def _extract_layout(visual_executor: VisualExecutor) -> ToolHandler:
    def handler(state: dict[str, Any], arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return visual_executor.extract_layout(state, arguments["image_id"], arguments["target_query"])

    return handler


def _read_region_text(visual_executor: VisualExecutor) -> ToolHandler:
    def handler(state: dict[str, Any], arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return visual_executor.read_region_text(state, arguments["image_id"], arguments["region_id"])

    return handler
