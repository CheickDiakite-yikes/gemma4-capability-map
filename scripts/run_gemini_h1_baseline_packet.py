from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_capability_map.knowledge_work.h1 import H1SliceConfig, h1_packet_selection, load_h1_slice
from gemma4_capability_map.runtime.gemini_cli import GeminiCliBaselineResult, run_gemini_cli_baseline
from gemma4_capability_map.runtime.workflows import DEFAULT_WORKFLOWS_PATH, PackagedWorkflow, load_packaged_workflows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_H1H_CONFIG_PATH = ROOT / "configs" / "knowledge_work_h1h_slice.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "gemini_cli"


def build_h1_gemini_workflows(
    *,
    config: H1SliceConfig,
    packet_id: str,
    workflows_path: str | Path = DEFAULT_WORKFLOWS_PATH,
) -> list[dict[str, Any]]:
    packet = h1_packet_selection(config, packet_id)
    packaged_by_id = {workflow.workflow_id: workflow for workflow in load_packaged_workflows(workflows_path)}
    family_by_episode = {
        family.live_episode_id if packet.lane == "live_web_stress" else family.replayable_episode_id: family
        for family in config.workflow_families
    }

    workflows: list[dict[str, Any]] = []
    for episode_id in packet.episode_ids:
        family = family_by_episode.get(episode_id)
        if family is None:
            raise ValueError(f"Packet `{packet_id}` episode `{episode_id}` is not mapped to an H1 workflow family.")
        packaged = packaged_by_id.get(family.workflow_id)
        if packaged is None:
            raise ValueError(f"Unknown packaged workflow `{family.workflow_id}`.")
        workflows.append(_gemini_workflow_payload(config, packet_id, packet.lane, episode_id, family, packaged))
    return workflows


def run_gemini_h1_baseline_packet(
    *,
    config_path: str | Path = DEFAULT_H1H_CONFIG_PATH,
    packet_id: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    run_group_id: str | None = None,
    binary: str | None = None,
    execute: bool = False,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_h1_slice(config_path)
    workflows = build_h1_gemini_workflows(config=config, packet_id=packet_id)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    packet_run_id = run_group_id or f"{timestamp}_{config.name}_{packet_id}_gemini_cli_baseline"
    packet_dir = Path(output_root) / packet_run_id
    workflow_root = packet_dir / "workflows"
    workflow_root.mkdir(parents=True, exist_ok=True)

    results: list[GeminiCliBaselineResult] = []
    for workflow in workflows:
        result = run_gemini_cli_baseline(
            workflow=workflow,
            output_dir=workflow_root / str(workflow["workflow_id"]),
            binary=binary,
            dry_run=not execute,
            timeout_s=timeout_s,
        )
        results.append(result)

    manifest = {
        "packet_run_id": packet_run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "config_path": str(config_path.resolve()),
        "h1_slice": f"{config.name}:{config.version}",
        "packet_id": packet_id,
        "lane": h1_packet_selection(config, packet_id).lane,
        "dry_run": not execute,
        "binary": binary,
        "workflow_count": len(workflows),
        "workflow_ids": [workflow["workflow_id"] for workflow in workflows],
    }
    payload_results = [result.as_payload() for result in results]
    summary = {
        "packet_dir": str(packet_dir.resolve()),
        "manifest": manifest,
        "workflow_count": len(results),
        "dry_run_count": sum(1 for result in results if result.dry_run),
        "available_count": sum(1 for result in results if result.availability.available),
        "unavailable_count": sum(1 for result in results if not result.availability.available),
        "nonzero_returncode_count": sum(1 for result in results if result.returncode not in (None, 0)),
        "results": payload_results,
    }

    (packet_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (packet_dir / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or run a Gemini CLI baseline packet over an H1 workflow slice.")
    parser.add_argument("--config", default=str(DEFAULT_H1H_CONFIG_PATH))
    parser.add_argument("--packet-id", default="mlx_full_tool_contract_breaker")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-group-id", default=None)
    parser.add_argument("--binary", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_gemini_h1_baseline_packet(
        config_path=args.config,
        packet_id=args.packet_id,
        output_root=args.output_root,
        run_group_id=args.run_group_id,
        binary=args.binary,
        execute=args.execute,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _gemini_workflow_payload(
    config: H1SliceConfig,
    packet_id: str,
    lane: str,
    episode_id: str,
    family: Any,
    packaged: PackagedWorkflow,
) -> dict[str, Any]:
    return {
        "workflow_id": family.workflow_id,
        "title": packaged.title,
        "description": packaged.description,
        "lane": lane,
        "episode_id": episode_id,
        "role_family": family.role_family,
        "purpose": family.purpose,
        "h1_stressors": list(family.h1_stressors),
        "packet_id": packet_id,
        "h1_slice": f"{config.name}:{config.version}",
        "live_entrypoint": config.live_entrypoint,
        "supports_approval": packaged.supports_approval,
        "recommended_system_id": packaged.recommended_system_id,
        "tags": list(packaged.tags),
        "moonies_evaluation_contract": {
            "external_baseline_only": True,
            "packaged_workflow_only": True,
            "no_public_side_effects": True,
            "attribute_findings_to_workflow_family": True,
        },
    }


if __name__ == "__main__":
    main()
