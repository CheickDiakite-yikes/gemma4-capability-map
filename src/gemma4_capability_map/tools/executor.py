from __future__ import annotations

from copy import deepcopy
from typing import Any

from gemma4_capability_map.schemas import ToolCall, ToolResult, ToolSpec
from gemma4_capability_map.tools.registry import ToolRegistry, build_default_registry
from gemma4_capability_map.tools.validators import validate_tool_call


class DeterministicExecutor:
    def __init__(self, registry: ToolRegistry | None = None, tool_specs: list[ToolSpec] | None = None) -> None:
        self.registry = registry or build_default_registry()
        self.tool_specs = tool_specs or self.registry.list_specs()

    def with_tool_specs(self, tool_specs: list[ToolSpec]) -> "DeterministicExecutor":
        return DeterministicExecutor(registry=self.registry, tool_specs=tool_specs)

    def step(self, state: dict[str, Any], tool_call: ToolCall, step: int = 1) -> ToolResult:
        spec = next((candidate for candidate in self.tool_specs if candidate.name == tool_call.name), None)
        valid, error = validate_tool_call(tool_call, self.tool_specs)
        if not valid:
            return ToolResult(
                step=step,
                selected_tool=tool_call.name,
                tool_family=spec.tool_family if spec is not None else "",
                tool_intent=spec.tool_intent if spec is not None else "",
                arguments=tool_call.arguments,
                validator_result="fail",
                output={},
                error=error,
                state_after=deepcopy(state),
            )
        runtime_arguments = _canonicalize_runtime_arguments(
            tool_name=tool_call.name,
            provided_arguments=tool_call.arguments,
            active_specs=self.tool_specs,
            registry_specs=self.registry.specs,
        )
        try:
            new_state, output = self.registry.execute(state=state, name=tool_call.name, arguments=runtime_arguments)
        except Exception as exc:
            return ToolResult(
                step=step,
                selected_tool=tool_call.name,
                tool_family=spec.tool_family if spec is not None else "",
                tool_intent=spec.tool_intent if spec is not None else "",
                arguments=tool_call.arguments,
                validator_result="fail",
                output={},
                error=str(exc),
                state_after=deepcopy(state),
            )
        return ToolResult(
            step=step,
            selected_tool=tool_call.name,
            tool_family=spec.tool_family if spec is not None else "",
            tool_intent=spec.tool_intent if spec is not None else "",
            arguments=tool_call.arguments,
            validator_result="pass",
            output=output,
            state_after=new_state,
        )


def diff_state(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    all_keys = set(before) | set(after)
    for key in all_keys:
        if before.get(key) != after.get(key):
            diff[key] = {"before": before.get(key), "after": after.get(key)}
    return diff


def _canonicalize_runtime_arguments(
    tool_name: str,
    provided_arguments: dict[str, Any],
    active_specs: list[ToolSpec],
    registry_specs: dict[str, ToolSpec],
) -> dict[str, Any]:
    canonical_spec = registry_specs.get(tool_name)
    if canonical_spec is None:
        return dict(provided_arguments)

    canonical_properties = tuple(canonical_spec.json_schema.get("properties", {}).keys())
    if not canonical_properties:
        return dict(provided_arguments)

    active_spec = next((spec for spec in active_specs if spec.name == tool_name), None)
    active_properties = tuple(active_spec.json_schema.get("properties", {}).keys()) if active_spec else ()

    normalized_arguments: dict[str, Any] = {}
    for key, value in provided_arguments.items():
        runtime_key = _match_runtime_key(key, canonical_properties, active_properties)
        normalized_arguments[runtime_key] = value
    return normalized_arguments


def _match_runtime_key(
    key: str,
    canonical_properties: tuple[str, ...],
    active_properties: tuple[str, ...],
) -> str:
    if key in canonical_properties:
        return key

    if key in active_properties:
        requested = _canonicalize_argument_name(key)
        for candidate in canonical_properties:
            if _canonicalize_argument_name(candidate) == requested:
                return candidate

    requested = _canonicalize_argument_name(key)
    for candidate in canonical_properties:
        if _canonicalize_argument_name(candidate) == requested:
            return candidate
    return key


def _canonicalize_argument_name(value: str) -> str:
    if value.endswith("_renamed"):
        return value[: -len("_renamed")]
    return value
