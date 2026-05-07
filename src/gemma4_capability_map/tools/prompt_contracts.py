from __future__ import annotations

from dataclasses import dataclass

from gemma4_capability_map.schemas import Message, ToolSpec


DEFAULT_TOOL_PROMPT_CONTRACT_ID = ""


@dataclass(frozen=True)
class ToolPromptContract:
    contract_id: str
    label: str
    description: str
    hypothesis: str
    tags: tuple[str, ...]
    instructions: tuple[str, ...]

    def render(self, messages: list[Message], media: list[str], tool_specs: list[ToolSpec]) -> str:
        if not tool_specs:
            return ""
        lines = [
            f"Tool prompt contract candidate: {self.contract_id}",
            self.description,
            "This is a generic contract only; it does not reveal the expected tool call for this turn.",
            "",
            "Contract rules:",
        ]
        lines.extend(f"- {instruction}" for instruction in self.instructions)
        tool_names = ", ".join(tool.name for tool in tool_specs)
        lines.extend(
            [
                "",
                f"Allowed tool names for this turn: {tool_names}.",
                "Use only these names and the exact schema fields shown in the tool catalog.",
            ]
        )
        if any(message.role == "tool" for message in messages):
            lines.append("Resolve the next call from the latest passing tool result, not from stale earlier calls.")
        if media or any(_is_visual_tool(tool.name) for tool in tool_specs):
            lines.append("For visual tools, preserve selection_id, region_id, image_id, target_query, and filter_query literally.")
        return "\n".join(lines)


TOOL_PROMPT_CONTRACTS: dict[str, ToolPromptContract] = {
    "schema_anchor_v1": ToolPromptContract(
        contract_id="schema_anchor_v1",
        label="Schema Anchor v1",
        description="Strengthens generic JSON and schema obedience without providing the planned call.",
        hypothesis="No-directive CLI/API misses may improve if the model is reminded that tool names and fields are literal interface tokens.",
        tags=("schema", "json", "cli", "api"),
        instructions=(
            "When tools are available and the user asks for inspect, read, update, patch, search, or visual work, return a JSON tool call instead of prose.",
            'Return one object as {"name":"tool_name","arguments":{...}} or an array of those objects for independent parallel checks.',
            "Do not rename schema fields, invent fields, wrap the object in markdown, or add explanation around the JSON.",
            "Copy path, query, record_type, record_id, field, value, image_id, target_query, filter_query, selection_id, and region_id strings exactly when they appear in the prompt or latest tool result.",
        ),
    ),
    "literal_argument_guard_v1": ToolPromptContract(
        contract_id="literal_argument_guard_v1",
        label="Literal Argument Guard v1",
        description="Targets canonical argument drift while keeping the contract generic.",
        hypothesis="No-directive rows often choose the right tool but drift on arguments; stronger literal-copy rules may reduce repair burden.",
        tags=("arguments", "canonicalization", "cli", "api", "visual"),
        instructions=(
            "Treat argument values as interface tokens, not paraphrasable summaries.",
            "Prefer canonical ids and labels from system or tool-result messages over wording inferred from the user request.",
            "For API records, keep record_type and record_id as separate fields; do not merge ids into record_type or use latest as record_id unless the schema asks for it.",
            "For CLI work, keep path and query separate; do not broaden or rewrite the search query.",
            "For visual work, copy target_query and filter_query labels exactly even when the user used a more natural phrase.",
        ),
    ),
    "tool_required_parallel_v1": ToolPromptContract(
        contract_id="tool_required_parallel_v1",
        label="Tool Required Parallel v1",
        description="Targets no-tool-call and parallel-tool collapses without exposing a planned call.",
        hypothesis="No-directive visual and parallel cases may fail because the model exits the tool protocol; stronger tool-required wording should reduce no-call failures.",
        tags=("no_tool_call", "parallel", "visual"),
        instructions=(
            "If tools are listed, do not answer in natural language until after the needed tool result is available.",
            "If the request says to check two independent sources, return a JSON array with one call per source in the same turn.",
            "If the request is a visual workflow and a visual tool result is already present, make exactly the next visual call needed from that latest result.",
            "If no prior visual result is present, start with the visual inspection or layout-extraction tool before any readback.",
        ),
    ),
}


def known_tool_prompt_contract_ids() -> list[str]:
    return sorted(TOOL_PROMPT_CONTRACTS)


def get_tool_prompt_contract(contract_id: str) -> ToolPromptContract | None:
    normalized = contract_id.strip()
    if not normalized:
        return None
    return TOOL_PROMPT_CONTRACTS.get(normalized)


def render_tool_prompt_contract(
    contract_id: str,
    messages: list[Message],
    media: list[str],
    tool_specs: list[ToolSpec],
) -> str:
    contract = get_tool_prompt_contract(contract_id)
    if contract is None:
        if contract_id.strip():
            known = ", ".join(known_tool_prompt_contract_ids())
            raise ValueError(f"Unknown tool prompt contract `{contract_id}`. Known contracts: {known}")
        return ""
    return contract.render(messages, media, tool_specs)


def _is_visual_tool(tool_name: str) -> bool:
    return tool_name in {
        "inspect_image",
        "extract_layout",
        "refine_selection",
        "read_region_text",
        "segment_entities",
    }
