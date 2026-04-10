from __future__ import annotations

import re
from pathlib import Path

from gemma4_capability_map.knowledge_work.native_artifacts import inspect_artifact
from gemma4_capability_map.knowledge_work.schemas import ArtifactSpec, ArtifactVersion


ROOT = Path(__file__).resolve().parents[3]
GOLDEN_ARTIFACT_ROOT = ROOT / "data" / "knowledge_work" / "artifact_goldens"


def grade_artifact(version: ArtifactVersion | None, spec: ArtifactSpec, episode_id: str | None = None) -> float:
    if version is None:
        return 0.0
    content = version.content
    normalized = content.lower()
    contract = spec.scoring_contract
    native = inspect_artifact(version.file_path) if version.file_path else {}
    golden = _load_golden_artifact(episode_id, spec) if episode_id else {}
    checks: list[float] = []
    checks.extend(float(fragment.lower() in normalized) for fragment in contract.required_fragments)
    checks.extend(float(section.lower() in normalized) for section in contract.required_sections)
    checks.extend(float(fragment.lower() not in normalized) for fragment in contract.forbidden_fragments)
    heading_order = native.get("headings") if native else None
    if contract.required_heading_order:
        checks.append(float(_ordered_subsequence(contract.required_heading_order, heading_order or _markdown_headings(content))))
    checks.extend(float(_table_row_present(content, row, native_rows=native.get("table_rows") if native else None)) for row in contract.required_table_rows)
    field_map = _parse_field_pairs(content)
    field_map.update({key.lower(): value.lower() for key, value in _native_field_map(native).items()})
    checks.extend(
        float(field.lower() in field_map and expected.lower() in field_map[field.lower()])
        for field, expected in contract.required_field_pairs.items()
    )
    formula_map = _parse_formula_pairs(content)
    formula_map.update({key.lower(): value for key, value in _native_formula_map(native).items()})
    checks.extend(
        float(label.lower() in formula_map and _normalize_formula(expected) == _normalize_formula(formula_map[label.lower()]))
        for label, expected in contract.required_formulas.items()
    )
    formula_cells = _native_formula_cells(native)
    checks.extend(
        float(cell.lower() in formula_cells and _normalize_formula(expected) == _normalize_formula(formula_cells[cell.lower()]))
        for cell, expected in contract.required_formula_cells.items()
    )
    slide_titles = _slide_titles(content, native_titles=native.get("slide_titles") if native else None)
    checks.extend(float(title.lower() in slide_titles) for title in contract.required_slide_titles)
    slide_sections = _slide_sections(content, native_sections=native.get("slide_sections") if native else None)
    for title, sections in contract.required_slide_sections.items():
        actual = slide_sections.get(title.lower(), set())
        checks.extend(float(section.lower() in actual) for section in sections)
    slide_bullets = _slide_bullets(content, native_bullets=native.get("slide_bullets") if native else None)
    for title, bullets in contract.required_slide_bullets_by_title.items():
        actual = slide_bullets.get(title.lower(), [])
        checks.extend(float(any(expected.lower() in bullet for bullet in actual)) for expected in bullets)
    checks.extend(float(_bullet_present(content, bullet, native_bullets=native.get("slide_bullets") if native else None)) for bullet in contract.required_bullets)
    checks.extend(float(_field_value_echoed(content, field_map, field)) for field in contract.consistency_fields)
    if contract.minimum_citations:
        checks.append(float(_citation_count(content) >= contract.minimum_citations))
    if contract.expected_format:
        checks.append(float(contract.expected_format.lower() in normalized))
    checks.extend(_native_alignment_checks(native, golden, spec))
    return sum(checks) / len(checks) if checks else 1.0


def _table_row_present(content: str, row: list[str], native_rows: object | None = None) -> bool:
    if isinstance(native_rows, list):
        expected = [cell.lower() for cell in row]
        for native_row in native_rows:
            if not isinstance(native_row, list):
                continue
            lowered = [str(cell).lower() for cell in native_row]
            if all(any(expected_cell in native_cell for native_cell in lowered) for expected_cell in expected):
                return True
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


def _markdown_headings(content: str) -> list[str]:
    headings: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            headings.append(stripped.lstrip("#").strip())
    return headings


def _slide_titles(content: str, native_titles: object | None = None) -> set[str]:
    if isinstance(native_titles, list):
        return {str(title).strip().lower() for title in native_titles if str(title).strip()}
    titles: set[str] = set()
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("## slide:"):
            titles.add(stripped.split(":", 1)[1].strip().lower())
    return titles


def _slide_sections(content: str, native_sections: object | None = None) -> dict[str, set[str]]:
    if isinstance(native_sections, dict):
        return {
            str(title).strip().lower(): {str(section).strip().lower() for section in sections}
            for title, sections in native_sections.items()
        }
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


def _slide_bullets(content: str, native_bullets: object | None = None) -> dict[str, list[str]]:
    if isinstance(native_bullets, dict):
        return {
            str(title).strip().lower(): [str(item).strip().lower() for item in bullets]
            for title, bullets in native_bullets.items()
        }
    bullets: dict[str, list[str]] = {}
    current_slide: str | None = None
    for line in content.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("## slide:"):
            current_slide = stripped.split(":", 1)[1].strip().lower()
            bullets.setdefault(current_slide, [])
            continue
        if current_slide and stripped.startswith("- "):
            bullets[current_slide].append(stripped[2:].strip().lower())
    return bullets


def _bullet_present(content: str, bullet: str, native_bullets: object | None = None) -> bool:
    bullet_lower = bullet.lower()
    if isinstance(native_bullets, dict):
        for bullets in native_bullets.values():
            if any(bullet_lower in str(item).strip().lower() for item in bullets):
                return True
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


def _native_field_map(native: dict[str, object]) -> dict[str, str]:
    field_pairs = native.get("field_pairs")
    if isinstance(field_pairs, dict):
        return {str(key): str(value) for key, value in field_pairs.items()}
    return {}


def _native_formula_map(native: dict[str, object]) -> dict[str, str]:
    formulas = native.get("formulas")
    if isinstance(formulas, dict):
        return {str(key): str(value) for key, value in formulas.items()}
    return {}


def _native_formula_cells(native: dict[str, object]) -> dict[str, str]:
    formulas = native.get("formula_cells")
    if isinstance(formulas, dict):
        return {str(key).lower(): str(value) for key, value in formulas.items()}
    return {}


def _load_golden_artifact(episode_id: str | None, spec: ArtifactSpec) -> dict[str, object]:
    if not episode_id:
        return {}
    suffix = Path(spec.path_or_target).suffix
    path = GOLDEN_ARTIFACT_ROOT / f"{episode_id}__{spec.artifact_id}{suffix}"
    if not path.exists():
        return {}
    return inspect_artifact(path)


def _native_alignment_checks(native: dict[str, object], golden: dict[str, object], spec: ArtifactSpec) -> list[float]:
    if not native:
        return []
    checks: list[float] = []
    kind = spec.kind.value
    if kind in {"spreadsheet", "model"}:
        native_rows = native.get("table_rows")
        golden_rows = golden.get("table_rows")
        if isinstance(native_rows, list) and isinstance(golden_rows, list) and golden_rows:
            checks.append(float(_rows_share_header(native_rows, golden_rows)))
            checks.append(float(len(native_rows) >= len(golden_rows)))
        native_formulas = _native_formula_map(native)
        golden_formulas = _native_formula_map(golden)
        if golden_formulas:
            checks.append(float(set(golden_formulas).issubset(native_formulas)))
    if kind in {"memo", "email", "research_note", "form_submission", "schedule"}:
        native_fields = _native_field_map(native)
        golden_fields = _native_field_map(golden)
        if golden_fields:
            checks.append(float(set(golden_fields).issubset(native_fields)))
        native_headings = native.get("headings")
        if isinstance(native_headings, list):
            required_headings = {
                section.replace("#", "").strip().lower()
                for section in spec.scoring_contract.required_sections
                if section.strip()
            }
            if required_headings:
                checks.append(float(required_headings.issubset({str(item).lower() for item in native_headings})))
            if spec.scoring_contract.required_heading_order:
                checks.append(float(_ordered_subsequence(spec.scoring_contract.required_heading_order, native_headings)))
    if kind == "deck":
        native_titles = native.get("slide_titles")
        golden_titles = golden.get("slide_titles")
        if isinstance(native_titles, list) and isinstance(golden_titles, list) and golden_titles:
            checks.append(float(_ordered_subsequence(golden_titles, native_titles)))
        native_sections = native.get("slide_sections")
        golden_sections = golden.get("slide_sections")
        if isinstance(native_sections, dict) and isinstance(golden_sections, dict) and golden_sections:
            checks.append(float(_slide_sections_align(native_sections, golden_sections)))
        native_bullets = native.get("slide_bullets")
        golden_bullets = golden.get("slide_bullets")
        if isinstance(native_bullets, dict) and isinstance(golden_bullets, dict) and golden_bullets:
            checks.append(float(_slide_bullets_align(native_bullets, golden_bullets)))
    return checks


def _rows_share_header(native_rows: list, golden_rows: list) -> bool:
    if not native_rows or not golden_rows:
        return False
    native_header = [str(cell).strip().lower() for cell in native_rows[0]]
    golden_header = [str(cell).strip().lower() for cell in golden_rows[0]]
    return native_header == golden_header


def _slide_sections_align(native_sections: dict, golden_sections: dict) -> bool:
    for title, golden_values in golden_sections.items():
        native_values = native_sections.get(title)
        if native_values is None:
            return False
        lowered_native = {str(value).strip().lower() for value in native_values}
        lowered_golden = {str(value).strip().lower() for value in golden_values}
        if not lowered_golden.issubset(lowered_native):
            return False
    return True


def _slide_bullets_align(native_bullets: dict, golden_bullets: dict) -> bool:
    for title, golden_values in golden_bullets.items():
        if str(title).strip().lower() in {"brief", "sources", "revision diff"}:
            continue
        native_values = native_bullets.get(title)
        if native_values is None:
            return False
        lowered_native = [str(value).strip().lower() for value in native_values]
        lowered_golden = [str(value).strip().lower() for value in golden_values]
        if not all(any(expected in actual for actual in lowered_native) for expected in lowered_golden):
            return False
    return True


def _ordered_subsequence(expected: list, actual: list) -> bool:
    if not expected:
        return True
    actual_iter = iter(actual)
    for expected_item in expected:
        expected_value = str(expected_item).strip().lower()
        for actual_item in actual_iter:
            if str(actual_item).strip().lower() == expected_value:
                break
        else:
            return False
    return True


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
