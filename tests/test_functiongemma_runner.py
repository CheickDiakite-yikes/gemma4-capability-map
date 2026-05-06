from __future__ import annotations

from gemma4_capability_map.models.functiongemma_runner import FunctionGemmaRunner
from gemma4_capability_map.schemas import Message, ToolSpec


VISUAL_SPECS = [
    ToolSpec(
        name="extract_layout",
        description="Extract a visual layout.",
        schema={"properties": {"image_id": {"type": "string"}, "target_query": {"type": "string"}}},
        tool_family="function_call",
        tool_intent="read",
    ),
    ToolSpec(
        name="refine_selection",
        description="Refine a visual selection.",
        schema={"properties": {"selection_id": {"type": "string"}, "filter_query": {"type": "string"}}},
        tool_family="function_call",
        tool_intent="read",
    ),
    ToolSpec(
        name="read_region_text",
        description="Read text from a visual region.",
        schema={"properties": {"image_id": {"type": "string"}, "region_id": {"type": "string"}}},
        tool_family="function_call",
        tool_intent="read",
    ),
]


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

    assert "call:api_fetch_record{record_type:<escape>billing_record<escape>,record_id:<escape>INV-204<escape>}" in system_prompt
    assert "call:tool_name" not in system_prompt
    assert "{arg:" not in system_prompt
    assert "<escape>value<escape>" not in system_prompt
    assert "never emit placeholder names or values" in system_prompt.lower()
    assert "Router directive for this turn" not in prompt_messages[-1]["content"]


def test_functiongemma_prompt_marks_next_visual_refinement_after_progress() -> None:
    runner = FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle")

    prompt_messages = runner._build_prompt_messages(  # noqa: SLF001 - prompt contract regression.
        [
            Message(role="system", content="visual_image_ids: img-dashboard-followup"),
            Message(
                role="user",
                content="Inspect the dashboard metrics, keep only the needs review panels, then the customer ops panel, and tell me what it says.",
            ),
            Message(
                role="tool",
                content='{"tool_name":"extract_layout","status":"pass","arguments":{"image_id":"img-dashboard-followup","target_query":"dashboard metric"},"output":{"selection_id":"sel-001","region_ids":["metric-101","metric-102","metric-103"]}}',
            ),
            Message(
                role="tool",
                content='{"tool_name":"refine_selection","status":"pass","arguments":{"selection_id":"sel-001","filter_query":"needs review"},"output":{"image_id":"img-dashboard-followup","selection_id":"sel-002","region_ids":["metric-101","metric-102"]}}',
            ),
        ],
        media=["img-dashboard-followup"],
        tool_specs=VISUAL_SPECS,
    )
    system_prompt = prompt_messages[0]["content"]
    turn_directive = prompt_messages[-1]["content"]

    assert "Visual sequencing rules" in system_prompt
    assert "For this turn, return exactly the next visual call below" in system_prompt
    assert "Completed successful refine_selection filters: needs review. Do not repeat completed filters." in system_prompt
    assert "Next visual action: use the latest selection_id and apply the next unfinished filter exactly as shown." in system_prompt
    assert (
        "Next visual call for this request: "
        "<start_function_call>call:refine_selection{selection_id:<escape>sel-002<escape>,filter_query:<escape>customer ops<escape>}"
    ) in system_prompt
    assert turn_directive.startswith("Router directive for this turn:")
    assert "Return only this exact function call and no other output." in turn_directive
    assert (
        "<start_function_call>call:refine_selection{selection_id:<escape>sel-002<escape>,filter_query:<escape>customer ops<escape>}"
    ) in turn_directive


def test_functiongemma_prompt_marks_visual_readback_after_final_refinement() -> None:
    runner = FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle")

    prompt_messages = runner._build_prompt_messages(  # noqa: SLF001 - prompt contract regression.
        [
            Message(role="system", content="visual_image_ids: img-dashboard-followup"),
            Message(
                role="user",
                content="Inspect the dashboard metrics, keep only the needs review panels, then the customer ops panel, and tell me what it says.",
            ),
            Message(
                role="tool",
                content='{"tool_name":"extract_layout","status":"pass","arguments":{"image_id":"img-dashboard-followup","target_query":"dashboard metric"},"output":{"selection_id":"sel-001","region_ids":["metric-101","metric-102","metric-103"]}}',
            ),
            Message(
                role="tool",
                content='{"tool_name":"refine_selection","status":"pass","arguments":{"selection_id":"sel-001","filter_query":"needs review"},"output":{"image_id":"img-dashboard-followup","selection_id":"sel-002","region_ids":["metric-101","metric-102"]}}',
            ),
            Message(
                role="tool",
                content='{"tool_name":"refine_selection","status":"pass","arguments":{"selection_id":"sel-002","filter_query":"customer ops"},"output":{"image_id":"img-dashboard-followup","selection_id":"sel-003","region_ids":["metric-102"]}}',
            ),
        ],
        media=["img-dashboard-followup"],
        tool_specs=VISUAL_SPECS,
    )
    system_prompt = prompt_messages[0]["content"]
    turn_directive = prompt_messages[-1]["content"]

    assert "Completed successful refine_selection filters: needs review, customer ops. Do not repeat completed filters." in system_prompt
    assert "Next visual action: requested filtering is complete; read the latest region exactly as shown instead of extracting or refining again." in system_prompt
    assert (
        "Next visual call for this request: "
        "<start_function_call>call:read_region_text{image_id:<escape>img-dashboard-followup<escape>,region_id:<escape>metric-102<escape>}"
    ) in system_prompt
    assert "Return only this exact function call and no other output." in turn_directive
    assert (
        "<start_function_call>call:read_region_text{image_id:<escape>img-dashboard-followup<escape>,region_id:<escape>metric-102<escape>}"
    ) in turn_directive
