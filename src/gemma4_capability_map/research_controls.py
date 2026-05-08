from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ResearchControls:
    disable_controller_repair: bool = False
    disable_controller_fallback: bool = False
    disable_visual_rescue: bool = False
    disable_intent_priority: bool = False
    disable_argument_repair: bool = False
    disable_deterministic_visual_follow_on: bool = False
    disable_tool_turn_directive: bool = False
    tool_prompt_contract_id: str = ""
    tool_catalog_profile_id: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ResearchControls":
        payload = payload or {}
        return cls(
            disable_controller_repair=bool(payload.get("disable_controller_repair", False)),
            disable_controller_fallback=bool(payload.get("disable_controller_fallback", False)),
            disable_visual_rescue=bool(payload.get("disable_visual_rescue", False)),
            disable_intent_priority=bool(payload.get("disable_intent_priority", False)),
            disable_argument_repair=bool(payload.get("disable_argument_repair", False)),
            disable_deterministic_visual_follow_on=bool(payload.get("disable_deterministic_visual_follow_on", False)),
            disable_tool_turn_directive=bool(payload.get("disable_tool_turn_directive", False)),
            tool_prompt_contract_id=str(payload.get("tool_prompt_contract_id", "") or ""),
            tool_catalog_profile_id=str(payload.get("tool_catalog_profile_id", "") or ""),
        )

    def enabled(self) -> bool:
        return any(asdict(self).values())

    def manifest_payload(self) -> dict[str, bool | str]:
        return {key: value for key, value in asdict(self).items() if value}
