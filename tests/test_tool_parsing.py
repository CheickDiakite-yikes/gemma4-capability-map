from __future__ import annotations

from gemma4_capability_map.tools.validators import normalize_tool_output


def test_tool_parsers_normalize_to_same_shape() -> None:
    json_call = '{"name":"search_events","arguments":{"start_date":"2026-04-10","end_date":"2026-04-10","attendee":"Sarah"}}'
    python_call = 'search_events(start_date="2026-04-10", end_date="2026-04-10", attendee="Sarah")'
    functiongemma_call = "<start_function_call>call:search_events{start_date:<escape>2026-04-10<escape>,end_date:<escape>2026-04-10<escape>,attendee:<escape>Sarah<escape>}<end_function_call>"

    normalized = [normalize_tool_output(payload)[0] for payload in [json_call, python_call, functiongemma_call]]
    assert {call.name for call in normalized} == {"search_events"}
    assert all(call.arguments == normalized[0].arguments for call in normalized)

