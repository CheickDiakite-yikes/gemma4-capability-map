from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_capability_map.runtime.visual_hard_slice import VISUAL_HARD_SLICE_DESIGNS, VisualCaseDesign


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "visual_hard_slice_design"
CASE_DESIGNS: tuple[VisualCaseDesign, ...] = VISUAL_HARD_SLICE_DESIGNS


def build_design(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    output.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = [_case_row(case) for case in CASE_DESIGNS]
    family_counts: dict[str, int] = {}
    for row in rows:
        family_counts[row["family"]] = family_counts.get(row["family"], 0) + 1
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output.resolve()),
        "case_count": len(rows),
        "family_counts": family_counts,
        "purpose": "Design a fresh visual hard slice after v2/v3/v4 catalog-profile results; this is a design packet, not model evidence.",
    }
    _write_csv(tables_dir / "visual_hard_slice_case_designs.csv", rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "design.json").write_text(
        json.dumps({"manifest": manifest, "case_designs": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "design.md").write_text(_markdown(manifest, rows), encoding="utf-8")
    return {"manifest": manifest, "case_designs": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fresh visual hard-slice design packet.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_design(output_dir=args.output_dir)
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


def _case_row(case: VisualCaseDesign) -> dict[str, str]:
    return {
        "case_id": case.case_id,
        "family": case.family,
        "primary_discriminator": case.primary_discriminator,
        "expected_tool": case.expected_tool,
        "expected_argument_focus": case.expected_argument_focus,
        "failure_pressure": case.failure_pressure,
        "publishable_reason": case.publishable_reason,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(manifest: dict[str, Any], rows: list[dict[str, str]]) -> str:
    lines = [
        "# Visual Hard Slice Design",
        "",
        "This packet designs the next fresh visual hard slice. It is not model evidence yet.",
        "",
        f"- generated_at: `{manifest['generated_at']}`",
        f"- case_count: `{manifest['case_count']}`",
        f"- purpose: {manifest['purpose']}",
        "",
        "| Case ID | Family | Discriminator | Expected Tool | Argument Focus | Failure Pressure |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["case_id"],
                    row["family"],
                    row["primary_discriminator"],
                    row["expected_tool"],
                    row["expected_argument_focus"],
                    row["failure_pressure"],
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
