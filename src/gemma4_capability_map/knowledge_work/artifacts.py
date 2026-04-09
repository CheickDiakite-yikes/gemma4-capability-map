from __future__ import annotations

import re

from gemma4_capability_map.knowledge_work.schemas import ArtifactSpec, ArtifactVersion


def grade_artifact(version: ArtifactVersion | None, spec: ArtifactSpec) -> float:
    if version is None:
        return 0.0
    content = version.content
    normalized = content.lower()
    contract = spec.scoring_contract
    checks: list[float] = []
    checks.extend(float(fragment.lower() in normalized) for fragment in contract.required_fragments)
    checks.extend(float(section.lower() in normalized) for section in contract.required_sections)
    checks.extend(float(fragment.lower() not in normalized) for fragment in contract.forbidden_fragments)
    checks.extend(float(_table_row_present(content, row)) for row in contract.required_table_rows)
    field_map = _parse_field_pairs(content)
    checks.extend(
        float(field.lower() in field_map and expected.lower() in field_map[field.lower()])
        for field, expected in contract.required_field_pairs.items()
    )
    slide_titles = _slide_titles(content)
    checks.extend(float(title.lower() in slide_titles) for title in contract.required_slide_titles)
    checks.extend(float(_bullet_present(content, bullet)) for bullet in contract.required_bullets)
    if contract.minimum_citations:
        checks.append(float(_citation_count(content) >= contract.minimum_citations))
    if contract.expected_format:
        checks.append(float(contract.expected_format.lower() in normalized))
    return sum(checks) / len(checks) if checks else 1.0


def _table_row_present(content: str, row: list[str]) -> bool:
    rows = [line.lower() for line in content.splitlines() if line.strip().startswith("|")]
    expected_cells = [cell.lower() for cell in row]
    return any(all(cell in line for cell in expected_cells) for line in rows)


def _parse_field_pairs(content: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for line in content.splitlines():
        stripped = line.strip().lstrip("-").strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        pairs[key.strip().lower()] = value.strip().lower()
    return pairs


def _slide_titles(content: str) -> set[str]:
    titles: set[str] = set()
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("## slide:"):
            titles.add(stripped.split(":", 1)[1].strip().lower())
    return titles


def _bullet_present(content: str, bullet: str) -> bool:
    bullet_lower = bullet.lower()
    for line in content.splitlines():
        stripped = line.strip().lstrip("-").strip().lower()
        if bullet_lower in stripped:
            return True
    return False


def _citation_count(content: str) -> int:
    count = 0
    for line in content.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("source:") or stripped.startswith("- source:"):
            count += 1
    count += len(re.findall(r"\[[0-9]+\]", content))
    return count
