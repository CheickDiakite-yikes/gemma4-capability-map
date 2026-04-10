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
    formula_map = _parse_formula_pairs(content)
    checks.extend(
        float(label.lower() in formula_map and _normalize_formula(expected) == _normalize_formula(formula_map[label.lower()]))
        for label, expected in contract.required_formulas.items()
    )
    slide_titles = _slide_titles(content)
    checks.extend(float(title.lower() in slide_titles) for title in contract.required_slide_titles)
    slide_sections = _slide_sections(content)
    for title, sections in contract.required_slide_sections.items():
        actual = slide_sections.get(title.lower(), set())
        checks.extend(float(section.lower() in actual) for section in sections)
    checks.extend(float(_bullet_present(content, bullet)) for bullet in contract.required_bullets)
    checks.extend(float(_field_value_echoed(content, field_map, field)) for field in contract.consistency_fields)
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


def _parse_formula_pairs(content: str) -> dict[str, str]:
    formulas: dict[str, str] = {}
    capture = False
    for line in content.splitlines():
        stripped = line.strip().lstrip("-").strip()
        if stripped == "## Formulas":
            capture = True
            continue
        if capture and stripped.startswith("## "):
            break
        if capture and ":" in stripped:
            key, value = stripped.split(":", 1)
            formulas[key.strip().lower()] = value.strip()
    return formulas


def _slide_titles(content: str) -> set[str]:
    titles: set[str] = set()
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("## slide:"):
            titles.add(stripped.split(":", 1)[1].strip().lower())
    return titles


def _slide_sections(content: str) -> dict[str, set[str]]:
    sections: dict[str, set[str]] = {}
    current_slide: str | None = None
    for line in content.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("## slide:"):
            current_slide = stripped.split(":", 1)[1].strip().lower()
            sections.setdefault(current_slide, set())
            continue
        if current_slide and lowered.startswith("### section:"):
            sections[current_slide].add(stripped.split(":", 1)[1].strip().lower())
    return sections


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


def _normalize_formula(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _field_value_echoed(content: str, field_map: dict[str, str], field: str) -> bool:
    key = field.lower()
    value = field_map.get(key, "")
    if not value:
        return False
    summary = _section_text(content, "## Response Summary").lower()
    if value in summary:
        return True
    return content.lower().count(value) > 1


def _section_text(content: str, title: str) -> str:
    lines = content.splitlines()
    capture = False
    collected: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == title:
            capture = True
            continue
        if capture and stripped.startswith("## "):
            break
        if capture:
            collected.append(line)
    return "\n".join(collected)
