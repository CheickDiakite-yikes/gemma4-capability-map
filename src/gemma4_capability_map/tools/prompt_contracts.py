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
    "schema_literal_tool_required_v2": ToolPromptContract(
        contract_id="schema_literal_tool_required_v2",
        label="Schema Literal Tool-Required v2",
        description="Combines schema anchoring, literal argument copying, and tool-required behavior in one generic contract.",
        hypothesis="The first wave split exact-copy and executable visual gains; a combined contract may preserve schema obedience while reducing no-call failures.",
        tags=("schema", "arguments", "no_tool_call", "json", "cli", "api", "visual"),
        instructions=(
            "When a listed tool can inspect, read, update, patch, search, or handle a visual request, output tool-call JSON only; do not answer in prose.",
            'Return exactly {"name":"tool_name","arguments":{...}} for one call or a JSON array of those objects for independent calls.',
            "Use only listed tool names and required schema fields; never rename fields, invent aliases, or wrap the JSON in markdown.",
            "Copy path, query, record_type, record_id, field, value, image_id, target_query, filter_query, selection_id, and region_id literally from the prompt or latest tool result.",
            "If you are unsure which literal to use, prefer the latest tool-result id or label over a paraphrase from the user request.",
        ),
    ),
    "visual_next_call_state_v2": ToolPromptContract(
        contract_id="visual_next_call_state_v2",
        label="Visual Next-Call State v2",
        description="Forces visual workflows to continue with the next stateful visual tool call instead of dropping into prose.",
        hypothesis="No-directive visual failures are concentrated in no-call behavior after a visual referent exists; explicit state-transition wording may reduce that collapse.",
        tags=("visual", "no_tool_call", "state_machine", "readback"),
        instructions=(
            "For visual workflows, make the next visual tool call required by the latest visual state before answering in prose.",
            "If the latest visual result has selection_id and the user asks to narrow or filter, call refine_selection with that exact selection_id and filter_query.",
            "If the latest visual result has region_id and the user asks to read or report the remaining text, call read_region_text with that exact region_id and image_id.",
            "Do not skip a visual tool call because the answer seems inferable from context; use the tool protocol first.",
            "Resolve selection_id, region_id, image_id, target_query, and filter_query from the latest passing tool result before older context.",
        ),
    ),
    "parallel_array_required_v2": ToolPromptContract(
        contract_id="parallel_array_required_v2",
        label="Parallel Array Required v2",
        description="Forces independent multi-source checks to remain inside JSON-array tool calling.",
        hypothesis="The parallel probe collapsed to no tool call under no-directive prompting; explicit array-shape rules may recover parallel calls.",
        tags=("parallel", "no_tool_call", "json_array", "multi_source"),
        instructions=(
            "If the request asks for two independent checks, sources, files, images, records, or evidence streams, output a JSON array with one tool-call object per independent check.",
            "Do not summarize, defer, or ask for confirmation before the independent tool calls when the needed tools are listed.",
            "Each array element must use one listed tool name and that tool's exact argument field names.",
            "Keep independent visual, repo, API, CLI, or document checks as separate array elements instead of merging them into one call.",
            "After parallel tool results are available, use the latest results to decide whether another tool call is needed.",
        ),
    ),
    "canonical_json_copy_v3": ToolPromptContract(
        contract_id="canonical_json_copy_v3",
        label="Canonical JSON Copy v3",
        description="Targets exact CLI/API argument fidelity with concise JSON-only and literal-copy rules.",
        hypothesis="Live replay shows no-directive MLX often enters the tool protocol but drifts on canonical CLI/API arguments; tighter token-copy rules may reduce argument repair without leaking the planned call.",
        tags=("schema", "arguments", "canonicalization", "json", "cli", "api"),
        instructions=(
            "When the user asks to inspect, search, read, update, or patch with listed tools, output only valid tool-call JSON.",
            'Use exactly {"name":"tool_name","arguments":{...}} for one call; do not add markdown, prose, comments, or alternate keys.',
            "Choose argument values by literal copy from the current prompt or latest tool result; do not paraphrase, normalize, translate, or abbreviate them.",
            "For CLI/API calls, keep path, query, record_type, record_id, field, and value in their separate schema fields exactly as named.",
            "If a literal path, record id, or query appears with punctuation, hyphens, spaces, or quotes, preserve that spelling inside the JSON string.",
        ),
    ),
    "visual_tool_initiation_v3": ToolPromptContract(
        contract_id="visual_tool_initiation_v3",
        label="Visual Tool Initiation v3",
        description="Targets visual no-call failures by making visual state transitions tool-first and id-preserving.",
        hypothesis="CLI-live visual replay shows no-directive MLX often answers or defers instead of initiating the next visual tool call; a compact state-transition contract may recover tool entry before exact selector tuning.",
        tags=("visual", "no_tool_call", "state_machine", "readback", "arguments"),
        instructions=(
            "When visual tools are listed and the user asks about visible state, make a visual tool call before prose.",
            "Use the latest visual state, not a guessed description, to choose the next visual tool.",
            "Carry image_id, selection_id, region_id, target_query, and filter_query literally from the latest user or tool message.",
            "If the task asks to locate a target, call the locating or refinement tool with the literal target_query or filter_query.",
            "If the task asks to read remaining text or verify a region, call read_region_text with the latest region_id and image_id.",
        ),
    ),
    "parallel_two_call_array_v3": ToolPromptContract(
        contract_id="parallel_two_call_array_v3",
        label="Parallel Two-Call Array v3",
        description="Targets independent evidence-gathering by requiring one JSON-array element per available source.",
        hypothesis="CLI-live parallel replay shows no-directive MLX asks the operator for inputs already present; explicit source-count and array-shape rules may preserve the two-call contract.",
        tags=("parallel", "no_tool_call", "json_array", "multi_source", "arguments"),
        instructions=(
            "When the request asks for independent checks across two available sources, output a JSON array with exactly one tool-call object per source.",
            "Do not ask the user to provide a file, screenshot, id, or path that is already present in the prompt, media, or latest tool result.",
            "Each array element must use one listed tool name, the exact schema field names, and literal source identifiers.",
            "Keep visual inspection and file/repo/API lookup as separate array elements when both are needed.",
            "Use the same turn for the independent calls; do not serialize them into prose instructions or defer the second call.",
        ),
    ),
    "visual_state_tool_selection_v4": ToolPromptContract(
        contract_id="visual_state_tool_selection_v4",
        label="Visual State Tool Selection v4",
        description="Targets visual wrong-tool failures by pairing tool-first visual behavior with state-specific tool selection rules.",
        hypothesis="Wave three recovered visual tool initiation but still chose the wrong visual tool for a filter/refinement case; state-specific selection rules may preserve tool entry while improving exact visual replay.",
        tags=("visual", "state_machine", "tool_selection", "no_tool_call", "arguments"),
        instructions=(
            "When visual tools are listed and the user asks about visible state, make a visual tool call before prose.",
            "Select the visual tool from the latest visual state: use an existing selection_id for narrowing, filtering, constraining, latest-only, or refinement requests.",
            "Use an existing region_id for readback, transcription, text reporting, or verification requests.",
            "If no selection_id or region_id is available and the task asks to locate a target, start with the locating or inspection visual tool.",
            "Carry image_id, selection_id, region_id, target_query, and filter_query literally; do not replace ids with descriptive phrases.",
            "Do not ask the user to provide visual state that is already present in the prompt, media, or latest tool result.",
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
