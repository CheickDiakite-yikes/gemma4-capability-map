from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = ROOT / "results" / "reports" / "mlx_tool_contract_harnessing"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "packaged_replay_gap_diagnostic"


@dataclass(frozen=True)
class Surface:
    surface_id: str
    replay_table: Path
    packaged_table: Path
    replay_signal: str
    packaged_surface: str


DEFAULT_SURFACES: tuple[Surface, ...] = (
    Surface(
        surface_id="h1l_visual_executor_equivalence",
        replay_table=DEFAULT_REPORT_DIR / "tables" / "visual_hard_slice_live_replay_summary.csv",
        packaged_table=DEFAULT_REPORT_DIR / "tables" / "h1l_visual_executor_equivalence_candidate_metrics.csv",
        replay_signal="preserved two-case visual hard-slice replay",
        packaged_surface="five packaged visual workflows",
    ),
    Surface(
        surface_id="h1m_visual_alias_repeat",
        replay_table=DEFAULT_REPORT_DIR / "tables" / "visual_hard_slice_alias_repeat_live_replay_summary.csv",
        packaged_table=DEFAULT_REPORT_DIR / "tables" / "h1m_visual_alias_repeat_candidate_metrics.csv",
        replay_signal="eight-case visual alias-repeat replay",
        packaged_surface="three packaged alias-repeat visual workflows",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose where replay-shaped visual signals vanish in packaged H1 workflows.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = analyze_packaged_replay_gap(output_dir=Path(args.output_dir))
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


def analyze_packaged_replay_gap(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    surfaces: tuple[Surface, ...] = DEFAULT_SURFACES,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = [_surface_row(surface) for surface in surfaces]
    recommendations = _recommendations(rows)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output_dir.resolve()),
        "surface_count": len(rows),
        "saturated_packaged_surface_count": sum(
            1 for row in rows if row["classification"] == "positive_replay_saturated_packaged_surface"
        ),
        "purpose": "Compare replay-shaped visual gains against packaged H1 visual workflow saturation.",
    }
    payload = {
        "manifest": manifest,
        "surface_rows": rows,
        "recommendations": recommendations,
    }

    _write_csv(tables_dir / "packaged_replay_gap_surfaces.csv", rows)
    _write_csv(tables_dir / "packaged_replay_gap_recommendations.csv", recommendations)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "diagnostic.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "diagnostic.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _surface_row(surface: Surface) -> dict[str, Any]:
    replay_rows = _read_csv(surface.replay_table)
    packaged_rows = _read_csv(surface.packaged_table)
    replay_max_delta_exact = max(_float(row, "delta_exact_rate") for row in replay_rows)
    replay_max_delta_executor = max(_float(row, "delta_executor_equivalence_rate") for row in replay_rows)
    packaged_readiness_span = _span(packaged_rows, "real_world_readiness_avg")
    packaged_strict_span = _span(packaged_rows, "strict_interface_avg")
    packaged_recovered_span = _span(packaged_rows, "recovered_execution_avg")
    packaged_controller_burden_max = max(
        max(
            _float(row, "controller_repair_avg"),
            _float(row, "controller_fallback_avg"),
            _float(row, "argument_repair_avg"),
        )
        for row in packaged_rows
    )
    classification = "needs_review"
    if replay_max_delta_executor > 0 and _is_zero(packaged_readiness_span) and _is_zero(packaged_strict_span):
        classification = "positive_replay_saturated_packaged_surface"
    elif replay_max_delta_executor > 0:
        classification = "positive_replay_packaged_surface_separates"
    return {
        "surface_id": surface.surface_id,
        "replay_signal": surface.replay_signal,
        "packaged_surface": surface.packaged_surface,
        "replay_comparison_count": len(replay_rows),
        "replay_max_delta_exact_rate": replay_max_delta_exact,
        "replay_max_delta_executor_equivalence_rate": replay_max_delta_executor,
        "replay_positive_executor_delta_count": sum(
            1 for row in replay_rows if _float(row, "delta_executor_equivalence_rate") > 0
        ),
        "packaged_system_count": len(packaged_rows),
        "packaged_readiness_span": packaged_readiness_span,
        "packaged_strict_interface_span": packaged_strict_span,
        "packaged_recovered_execution_span": packaged_recovered_span,
        "packaged_controller_burden_max": packaged_controller_burden_max,
        "classification": classification,
    }


def _recommendations(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    saturated = [row for row in rows if row["classification"] == "positive_replay_saturated_packaged_surface"]
    return [
        {
            "recommendation_id": "do_not_spend_helper_budget_on_saturated_packaged_visual_surfaces",
            "status": "active",
            "evidence": f"{len(saturated)} visual surfaces have positive replay gains but zero packaged strict/readiness span.",
        },
        {
            "recommendation_id": "preserve_replay_shape_before_packaging",
            "status": "active",
            "evidence": "The next visual task should keep alias/decoy pressure in the live prompt instead of decomposing it into staged packaged steps.",
        },
        {
            "recommendation_id": "report_strict_and_executor_equivalence_separately",
            "status": "active",
            "evidence": "Replay gains appear first as executor-equivalence gains, while packaged readiness can saturate.",
        },
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else 0.0


def _span(rows: list[dict[str, str]], key: str) -> float:
    values = [_float(row, key) for row in rows]
    return max(values) - min(values)


def _is_zero(value: float, *, tolerance: float = 1e-12) -> bool:
    return abs(value) <= tolerance


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    surface_rows = payload["surface_rows"]
    recommendations = payload["recommendations"]
    lines = [
        "# Packaged Replay Gap Diagnostic",
        "",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Surface count: `{manifest['surface_count']}`",
        f"- Saturated packaged surface count: `{manifest['saturated_packaged_surface_count']}`",
        "",
        "## Surface Summary",
        "",
        _markdown_table(surface_rows),
        "",
        "## Recommendations",
        "",
        _markdown_table(recommendations),
        "",
        "## Interpretation",
        "",
        "H1l and H1m now tell the same methodological story: replay-shaped visual packets can separate strict fidelity from executor-equivalent target success, while current packaged visual workflows erase row-level differences. That makes packaged workflow saturation an experimental-design finding, not a model-quality win.",
        "",
    ]
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
