from __future__ import annotations

from gemma4_capability_map.schemas import Message, StressorKind, Task, Variant, VariantOverrides


def apply_context_variant(task: Task, flavor: str) -> Variant:
    prefix_messages: list[Message] = []
    if flavor == "long_history":
        prefix_messages = [
            Message(role="user", content="Reminder: I once preferred morning meetings, but that may be stale."),
            Message(role="assistant", content="Noted. I will use the latest constraint when relevant."),
        ]
    elif flavor == "stale_preference":
        prefix_messages = [Message(role="user", content="Old preference: always schedule for Friday mornings.")]
    elif flavor == "irrelevant_tool_output":
        prefix_messages = [Message(role="tool", content='{"noise": "previous tool run unrelated to this task"}')]
    elif flavor == "changed_constraint":
        prefix_messages = [Message(role="user", content="Update: use the newest instruction, not the earlier plan.")]
    return Variant(
        variant_id=f"{task.task_id}_context_{flavor}",
        base_task_id=task.task_id,
        primary_stressor=StressorKind.CONTEXT,
        stressors={"language": None, "schema": None, "context": flavor, "efficiency": None},
        overrides=VariantOverrides(messages_prefix=prefix_messages),
    )

