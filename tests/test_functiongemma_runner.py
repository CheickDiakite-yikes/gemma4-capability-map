from __future__ import annotations

from gemma4_capability_map.models.functiongemma_runner import FunctionGemmaRunner
from gemma4_capability_map.schemas import Message, ToolSpec


def test_functiongemma_prompt_uses_catalog_specific_format_hint() -> None:
    runner = FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle")
    tool = ToolSpec(
        name="api_fetch_record",
        description="Fetch a record.",
        schema={"properties": {"record_type": {"type": "string"}, "record_id": {"type": "string"}}},
        tool_family="api",
        tool_intent="read",
    )

    prompt_messages = runner._build_prompt_messages(  # noqa: SLF001 - prompt contract regression.
        [Message(role="user", content="Read billing record INV-204.")],
        media=[],
        tool_specs=[tool],
    )
    system_prompt = prompt_messages[0]["content"]

    assert "call:api_fetch_record{record_type:" in system_prompt
    assert "call:tool_name" not in system_prompt
    assert "{arg:" not in system_prompt
    assert "never emit placeholder names" in system_prompt.lower()
