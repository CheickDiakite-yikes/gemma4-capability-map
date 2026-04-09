from __future__ import annotations

from collections import OrderedDict

from gemma4_capability_map.schemas import ExpectedEvent, StressorKind, Task, ToolSpec, Variant, VariantOverrides


def apply_schema_variant(task: Task, flavor: str) -> Variant:
    updated_tools = [tool.model_copy(deep=True) for tool in task.tool_specs]
    updated_events = [event.model_copy(deep=True) for event in task.expected_events]
    if flavor == "renamed_fields":
        rename_map: dict[str, str] = {}
        for tool in updated_tools:
            properties = tool.json_schema.get("properties", {})
            if not properties:
                continue
            first_key = next(iter(properties))
            renamed_key = f"{first_key}_renamed"
            properties[renamed_key] = properties.pop(first_key)
            required = tool.json_schema.get("required", [])
            tool.json_schema["required"] = [renamed_key if field == first_key else field for field in required]
            rename_map[first_key] = renamed_key
        for event in updated_events:
            event.arguments = {rename_map.get(key, key): value for key, value in event.arguments.items()}
    elif flavor == "reordered_schema":
        for tool in updated_tools:
            properties = tool.json_schema.get("properties", {})
            tool.json_schema["properties"] = OrderedDict(reversed(list(properties.items())))
    elif flavor == "enum_trap":
        for tool in updated_tools:
            if tool.name == "create_event":
                tool.json_schema["properties"]["priority"] = {"type": "string", "enum": ["low", "normal", "urgent"]}
    elif flavor == "validator_feedback":
        pass
    return Variant(
        variant_id=f"{task.task_id}_schema_{flavor}",
        base_task_id=task.task_id,
        primary_stressor=StressorKind.SCHEMA,
        stressors={"language": None, "schema": flavor, "context": None, "efficiency": None},
        overrides=VariantOverrides(
            tool_specs=updated_tools,
            expected_events=updated_events,
            initial_state_patch={"validator_feedback_enabled": flavor == "validator_feedback"},
        ),
    )
