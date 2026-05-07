from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_capability_map.reporting.knowledge_work_board import DEFAULT_REGISTRY_PATH, load_model_registry
from gemma4_capability_map.research_controls import ResearchControls
from gemma4_capability_map.runtime.tool_directive_probe import run_tool_directive_probe, write_tool_directive_probe_comparison


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "tool_prompt_contract_probe_packets"
DEFAULT_BASELINE_PROBE = ROOT / "results" / "tool_directive_probe" / "20260506T_mlx_tool_directive_probe_v4"
DEFAULT_CANDIDATE_SYSTEM_IDS = [
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_anchor",
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_literal_guard",
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_tool_required",
]


def build_tool_prompt_contract_probe_packet(
    *,
    baseline_probe_dir: str | Path = DEFAULT_BASELINE_PROBE,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    run_group_id: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    system_ids: list[str] | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    registry_path = Path(registry_path)
    registry = load_model_registry(registry_path)
    systems = registry.get("systems", {})
    candidate_ids = system_ids or list(DEFAULT_CANDIDATE_SYSTEM_IDS)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    packet_run_id = run_group_id or f"{timestamp}_mlx_prompt_contract_probe_packet"
    packet_dir = Path(output_root) / packet_run_id
    packet_dir.mkdir(parents=True, exist_ok=True)

    baseline_probe_dir = Path(baseline_probe_dir)
    manifest = {
        "packet_run_id": packet_run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "baseline_probe_dir": str(baseline_probe_dir.resolve()),
        "registry_path": str(registry_path.resolve()),
        "execute": execute,
        "system_ids": candidate_ids,
    }
    rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for system_id in candidate_ids:
        meta = systems.get(system_id)
        if meta is None:
            raise ValueError(f"Unknown system profile `{system_id}`.")
        controls = ResearchControls.from_mapping(meta.get("research_controls"))
        if not controls.disable_tool_turn_directive:
            raise ValueError(f"Candidate `{system_id}` must disable the exact tool-turn directive.")
        if not controls.tool_prompt_contract_id:
            raise ValueError(f"Candidate `{system_id}` must set tool_prompt_contract_id.")

        output_dir = packet_dir / system_id
        command = _probe_command(system_id=system_id, output_dir=output_dir, registry_path=registry_path)
        commands.append(
            {
                "system_id": system_id,
                "tool_prompt_contract_id": controls.tool_prompt_contract_id,
                "output_dir": str(output_dir.resolve()),
                "command": command,
            }
        )
        row = {
            "system_id": system_id,
            "tool_prompt_contract_id": controls.tool_prompt_contract_id,
            "execute": execute,
            "output_dir": str(output_dir.resolve()),
            "comparison_path": "",
            "exact_match_rate": "",
            "executable_match_rate": "",
        }
        if execute:
            result = run_tool_directive_probe(system_id=system_id, output_dir=output_dir, registry_path=registry_path)
            comparison_outputs = write_tool_directive_probe_comparison(baseline_probe_dir, output_dir)
            row["comparison_path"] = comparison_outputs["summary"]
            row["exact_match_rate"] = result["summary"]["exact_match_rate"]
            row["executable_match_rate"] = result["summary"]["executable_match_rate"]
            results.append(
                {
                    "system_id": system_id,
                    "probe_output_dir": result["output_dir"],
                    "summary": result["summary"],
                    "comparison_outputs": comparison_outputs,
                }
            )
        rows.append(row)

    summary = {
        "packet_dir": str(packet_dir.resolve()),
        "manifest": manifest,
        "candidate_count": len(candidate_ids),
        "executed_count": sum(1 for row in rows if row["execute"]),
        "dry_run_count": sum(1 for row in rows if not row["execute"]),
        "rows": rows,
        "commands": commands,
        "results": results,
    }
    (packet_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (packet_dir / "commands.json").write_text(json.dumps(commands, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (packet_dir / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(packet_dir / "candidate_summary.csv", rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or run MLX prompt-contract candidate probes.")
    parser.add_argument("--baseline-probe-dir", default=str(DEFAULT_BASELINE_PROBE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-group-id", default=None)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--system-id", action="append", dest="system_ids", default=[])
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_tool_prompt_contract_probe_packet(
        baseline_probe_dir=args.baseline_probe_dir,
        output_root=args.output_root,
        run_group_id=args.run_group_id,
        registry_path=args.registry,
        system_ids=args.system_ids or None,
        execute=args.execute,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _probe_command(*, system_id: str, output_dir: Path, registry_path: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_tool_directive_probe.py"),
        "--system-id",
        system_id,
        "--registry",
        str(registry_path),
        "--output-dir",
        str(output_dir),
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
