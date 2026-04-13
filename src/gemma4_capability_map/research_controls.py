from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ResearchControls:
    disable_controller_repair: bool = False
    disable_controller_fallback: bool = False
    disable_visual_rescue: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ResearchControls":
        payload = payload or {}
        return cls(
            disable_controller_repair=bool(payload.get("disable_controller_repair", False)),
            disable_controller_fallback=bool(payload.get("disable_controller_fallback", False)),
            disable_visual_rescue=bool(payload.get("disable_visual_rescue", False)),
        )

    def enabled(self) -> bool:
        return any(asdict(self).values())

    def manifest_payload(self) -> dict[str, bool]:
        return {key: value for key, value in asdict(self).items() if value}
