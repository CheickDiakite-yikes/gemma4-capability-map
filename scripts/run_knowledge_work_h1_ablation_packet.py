from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_capability_map.knowledge_work.h1 import DEFAULT_H1_SLICE_PATH, load_h1_slice, validate_h1_slice
from gemma4_capability_map.reporting.knowledge_work_board import DEFAULT_REGISTRY_PATH, load_model_registry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "knowledge_work_h1_slice"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the H1 HF Gemma ablation wave with one shared runtime bundle.")
    parser.add_argument("--config", default=str(DEFAULT_H1_SLICE_PATH))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-group-id", default=None)
    parser.add_argument("--lane", choices=["replayable_core", "live_web_stress"], default="replayable_core")
    parser.add_argument("--system-id", action="append", dest="system_ids", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_h1_slice(args.config)
    validation_errors = validate_h1_slice(config)
    if validation_errors:
        raise SystemExit("Invalid H1 slice config:\n" + "\n".join(f"- {error}" for error in validation_errors))

    registry = load_model_registry(args.registry)
    selected_system_ids = args.system_ids or list(config.ablation_system_ids)
    missing_system_ids = [system_id for system_id in selected_system_ids if system_id not in registry.get("systems", {})]
    if missing_system_ids:
        raise SystemExit(f"Unknown system ids: {', '.join(missing_system_ids)}")

    run_group_id = args.run_group_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ_h1_ablation")
    lane_config = config.lanes[args.lane]
    command = h1_ablation_packet_command(
        run_group_id=run_group_id,
        lane=args.lane,
        bundle_system_id=config.ablation_bundle_system_id,
        system_ids=selected_system_ids,
        episode_ids=lane_config.episode_ids,
        output_root=Path(args.output_root),
    )
    output_dir = Path(args.output_root) / f"{run_group_id}_knowledge_work_ablation_packet"
    wrapper_manifest = {
        "run_group_id": run_group_id,
        "created_at": datetime.now(UTC).isoformat(),
        "h1_slice": config.model_dump(mode="json"),
        "lane": args.lane,
        "bundle_system_id": config.ablation_bundle_system_id,
        "system_ids": selected_system_ids,
        "episode_ids": lane_config.episode_ids,
        "command": command,
        "output_dir": str(output_dir.resolve()),
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        print(json.dumps(wrapper_manifest, indent=2, ensure_ascii=False))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "h1_ablation_packet_manifest.json").write_text(
        json.dumps(wrapper_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    result: dict[str, Any] = {
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "output_dir": str(output_dir.resolve()),
    }
    child_results = _read_json(output_dir / "results.json")
    child_manifest = _read_json(output_dir / "manifest.json")
    if child_results is not None:
        result["results"] = child_results
    if child_manifest is not None:
        result["manifest"] = child_manifest
    (output_dir / "h1_ablation_packet_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if process.returncode:
        raise SystemExit(process.returncode)


def h1_ablation_packet_command(
    *,
    run_group_id: str,
    lane: str,
    bundle_system_id: str,
    system_ids: list[str],
    episode_ids: list[str],
    output_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_knowledge_work_ablation_packet.py"),
        "--lane",
        lane,
        "--bundle-system-id",
        bundle_system_id,
        "--output-root",
        str(output_root),
        "--run-group-id",
        run_group_id,
        "--run-intent",
        "exploratory",
    ]
    for system_id in system_ids:
        command.extend(["--system-id", system_id])
    for episode_id in episode_ids:
        command.extend(["--episode-id", episode_id])
    return command


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
