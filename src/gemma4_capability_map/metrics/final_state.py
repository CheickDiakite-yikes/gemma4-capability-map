from __future__ import annotations

from typing import Any


def final_state_match(expected: dict[str, Any], actual: dict[str, Any]) -> float:
    for key, value in expected.items():
        if actual.get(key) != value:
            return 0.0
    return 1.0

