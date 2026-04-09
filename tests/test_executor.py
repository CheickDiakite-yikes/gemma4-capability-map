from __future__ import annotations

from copy import deepcopy

from gemma4_capability_map.schemas import ToolCall
from gemma4_capability_map.tools.executor import DeterministicExecutor
from gemma4_capability_map.tools.registry import build_default_registry


def test_executor_updates_event_state() -> None:
    executor = DeterministicExecutor(registry=build_default_registry())
    state = {
        "calendar_events": [
            {"id": "evt-001", "title": "Sarah sync", "start": "2026-04-10T09:00:00", "end": "2026-04-10T09:30:00", "attendees": ["Sarah"]},
        ]
    }
    result = executor.step(
        state=state,
        tool_call=ToolCall(
            name="update_event",
            arguments={"event_id": "evt-001", "new_start": "2026-04-14T14:00:00", "new_end": "2026-04-14T14:30:00"},
            source_format="json",
            raw="{}",
        ),
    )
    assert result.validator_result == "pass"
    assert result.state_after["calendar_events"][0]["start"] == "2026-04-14T14:00:00"


def test_executor_is_deterministic_for_image_lookup() -> None:
    executor = DeterministicExecutor(registry=build_default_registry())
    state = {"images": {"img-settings": {"summary": "Security page", "recommended_patch": "safe_mode: true"}}}
    call = ToolCall(name="inspect_image", arguments={"image_id": "img-settings"}, source_format="json", raw="{}")
    result_a = executor.step(state=state, tool_call=call)
    result_b = executor.step(state=state, tool_call=call)
    assert result_a.output == result_b.output


def test_executor_returns_failed_tool_result_on_runtime_error() -> None:
    executor = DeterministicExecutor(registry=build_default_registry())
    state = {"images": {}}
    call = ToolCall(name="inspect_image", arguments={"image_id": "missing"}, source_format="json", raw="{}")
    result = executor.step(state=state, tool_call=call)
    assert result.validator_result == "fail"
    assert "Image not found" in (result.error or "")


def test_executor_maps_renamed_schema_fields_back_to_runtime_keys() -> None:
    registry = build_default_registry()
    executor = DeterministicExecutor(registry=registry)

    renamed_create_event = registry.specs["create_event"].model_copy(deep=True)
    create_schema = deepcopy(renamed_create_event.json_schema)
    create_schema["properties"]["title_renamed"] = create_schema["properties"].pop("title")
    create_schema["required"] = ["title_renamed" if field == "title" else field for field in create_schema.get("required", [])]
    renamed_create_event.json_schema = create_schema

    renamed_read_repo_file = registry.specs["read_repo_file"].model_copy(deep=True)
    read_schema = deepcopy(renamed_read_repo_file.json_schema)
    read_schema["properties"]["path_renamed"] = read_schema["properties"].pop("path")
    read_schema["required"] = ["path_renamed" if field == "path" else field for field in read_schema.get("required", [])]
    renamed_read_repo_file.json_schema = read_schema

    adapted_executor = executor.with_tool_specs([renamed_create_event, renamed_read_repo_file])

    create_result = adapted_executor.step(
        state={"calendar_events": []},
        tool_call=ToolCall(
            name="create_event",
            arguments={
                "title_renamed": "Budget review",
                "start": "2026-04-15T15:00:00",
                "end": "2026-04-15T15:30:00",
                "attendees": ["team@example.com"],
            },
            source_format="json",
            raw="{}",
        ),
    )
    assert create_result.validator_result == "pass"
    assert create_result.arguments["title_renamed"] == "Budget review"
    assert create_result.output["created_event"]["title"] == "Budget review"

    read_result = adapted_executor.step(
        state={"repo_files": {"config/settings.yaml": "safe_mode: false"}},
        tool_call=ToolCall(
            name="read_repo_file",
            arguments={"path_renamed": "config/settings.yaml"},
            source_format="json",
            raw="{}",
        ),
    )
    assert read_result.validator_result == "pass"
    assert read_result.arguments["path_renamed"] == "config/settings.yaml"
    assert read_result.output["path"] == "config/settings.yaml"
