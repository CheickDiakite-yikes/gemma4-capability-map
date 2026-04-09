from __future__ import annotations


def exact_match(expected: object, actual: object) -> float:
    return 1.0 if expected == actual else 0.0

