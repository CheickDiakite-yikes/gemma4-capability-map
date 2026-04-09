from __future__ import annotations

import ast
import json
import re
from typing import Any

from gemma4_capability_map.schemas import ToolCall, ToolSpec


FUNCTIONGEMMA_RE = re.compile(r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>", re.DOTALL)
FUNCTIONGEMMA_ARG_RE = re.compile(r"(\w+):(?:<escape>(.*?)<escape>|([^,}]*))")


def normalize_tool_output(raw_output: str) -> list[ToolCall]:
    text = raw_output.strip()
    if not text:
        return []
    if "<start_function_call>" in text:
        return _parse_functiongemma(text)
    if text.startswith("{") or text.startswith("["):
        parsed = _parse_json(text)
        if parsed:
            return parsed
    python_call = _parse_python_call(text)
    if python_call:
        return python_call
    return []


def validate_tool_call(tool_call: ToolCall, tool_specs: list[ToolSpec]) -> tuple[bool, str | None]:
    tool_map = {tool.name: tool for tool in tool_specs}
    if tool_call.name not in tool_map:
        return False, f"Unknown tool: {tool_call.name}"
    schema = tool_map[tool_call.name].json_schema
    required = schema.get("required", [])
    missing = [field for field in required if field not in tool_call.arguments]
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"
    properties = schema.get("properties", {})
    enums = {name: definition.get("enum") for name, definition in properties.items() if isinstance(definition, dict) and "enum" in definition}
    for key, valid_values in enums.items():
        if valid_values and key in tool_call.arguments and tool_call.arguments[key] not in valid_values:
            return False, f"Invalid enum value for {key}: {tool_call.arguments[key]}"
    return True, None


def _parse_json(text: str) -> list[ToolCall]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    candidates = payload if isinstance(payload, list) else payload.get("tool_calls", [payload])
    calls: list[ToolCall] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        name = candidate.get("name") or candidate.get("tool") or candidate.get("function")
        arguments = candidate.get("arguments") or candidate.get("args") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if name:
            calls.append(ToolCall(name=name, arguments=arguments, source_format="json", raw=text))
    return calls


def _parse_python_call(text: str) -> list[ToolCall]:
    try:
        expression = ast.parse(text, mode="eval").body
    except SyntaxError:
        return []
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        return []
    arguments: dict[str, Any] = {}
    for keyword in expression.keywords:
        arguments[keyword.arg or "arg"] = ast.literal_eval(keyword.value)
    return [
        ToolCall(
            name=expression.func.id,
            arguments=arguments,
            source_format="python",
            raw=text,
        )
    ]


def _parse_functiongemma(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for name, args_blob in FUNCTIONGEMMA_RE.findall(text):
        arguments: dict[str, Any] = {}
        for key, escaped_value, raw_value in FUNCTIONGEMMA_ARG_RE.findall(args_blob):
            value = escaped_value if escaped_value != "" else raw_value
            arguments[key] = _cast_value(value)
        calls.append(ToolCall(name=name, arguments=arguments, source_format="functiongemma", raw=text))
    return calls


def _cast_value(value: str) -> Any:
    normalized = value.strip()
    if normalized.lower() in {"true", "false"}:
        return normalized.lower() == "true"
    try:
        return ast.literal_eval(normalized)
    except (ValueError, SyntaxError):
        return normalized
