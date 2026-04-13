from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_capability_map.benchmark import build_runtime_bundle, load_tasks, runtime_bundle_snapshot, warm_runtime_bundle
from gemma4_capability_map.io import dump_jsonl
from gemma4_capability_map.knowledge_work.exporters import export_episode_leaderboard_csv
from gemma4_capability_map.knowledge_work.loader import load_episodes
from gemma4_capability_map.knowledge_work.replay import summarize_episode_traces
from gemma4_capability_map.knowledge_work.runner import EpisodeRunner
from gemma4_capability_map.research_controls import ResearchControls
from gemma4_capability_map.reporting.knowledge_work_board import DEFAULT_REGISTRY_PATH, load_model_registry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "knowledge_work_matrix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a focused KnowledgeWorkArena ablation packet with one shared runtime bundle.")
    parser.add_argument("--lane", choices=["replayable_core", "live_web_stress"], default="replayable_core")
    parser.add_argument("--bundle-system-id", required=True)
    parser.add_argument("--system-id", action="append", dest="system_ids", default=[])
    parser.add_argument("--episode-id", action="append", dest="episode_ids", default=[])
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-group-id", default=None)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--run-intent", choices=["canonical", "exploratory"], default="exploratory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.system_ids:
        raise SystemExit("Provide at least one --system-id to evaluate.")
    if not args.episode_ids:
        raise SystemExit("Provide at least one --episode-id for the focused packet.")

    registry = load_model_registry(args.registry)
    systems = registry.get("systems", {})
    bundle_system = systems.get(args.bundle_system_id)
    if bundle_system is None:
        raise SystemExit(f"Unknown bundle system `{args.bundle_system_id}`.")

    run_group_id = args.run_group_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_root) / f"{run_group_id}_knowledge_work_ablation_packet"
    output_root.mkdir(parents=True, exist_ok=True)

    lane_dir = ROOT / "data" / "knowledge_work" / args.lane
    episodes = load_episodes(lane_dir / "episodes.jsonl")
    allowed_ids = set(args.episode_ids)
    episodes = [episode for episode in episodes if episode.episode_id in allowed_ids]
    missing_ids = [episode_id for episode_id in args.episode_ids if episode_id not in {episode.episode_id for episode in episodes}]
    if missing_ids:
        raise SystemExit(f"Unknown episode ids: {', '.join(missing_ids)}")

    tasks = load_tasks(track=None)
    bundle = build_runtime_bundle(
        tasks=tasks,
        pipeline_name="modular",
        backend=str(bundle_system.get("backend", "")),
        reasoner_backend=str(bundle_system.get("backend", "")),
        router_backend=_router_backend(bundle_system),
        retriever_backend=_retriever_backend(bundle_system),
        reasoner_id=str(bundle_system.get("reasoner", "")),
        router_id=str(bundle_system.get("router", "") or ""),
        retriever_id=str(bundle_system.get("retriever", "") or ""),
        reasoner_device="auto",
        router_device=None,
        retriever_device=None,
        reasoner_max_new_tokens=int(bundle_system.get("reasoner_max_new_tokens", 96) or 96),
        request_timeout_seconds=float(bundle_system.get("request_timeout_seconds", 600.0) or 600.0),
    )
    warmup = warm_runtime_bundle(bundle, tasks)
    bundle_snapshot = runtime_bundle_snapshot(bundle)

    batch_manifest = {
        "run_group_id": run_group_id,
        "created_at": datetime.now(UTC).isoformat(),
        "lane": args.lane,
        "bundle_system_id": args.bundle_system_id,
        "system_ids": args.system_ids,
        "episode_ids": args.episode_ids,
        "run_intent": args.run_intent,
        "warmup": warmup,
        "runtime_bundle": bundle_snapshot,
    }
    (output_root / "manifest.json").write_text(json.dumps(batch_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    results: list[dict[str, Any]] = []
    for system_id in args.system_ids:
        meta = systems.get(system_id)
        if meta is None:
            raise SystemExit(f"Unknown system `{system_id}`.")

        research_controls = ResearchControls.from_mapping(meta.get("research_controls"))
        runner = EpisodeRunner(
            tasks=tasks,
            bundle=bundle,
            thinking_enabled=bool(meta.get("thinking", False)),
            planning_max_new_tokens=int(meta.get("reasoner_max_new_tokens", 96) or 96),
            final_max_new_tokens=int(meta.get("reasoner_max_new_tokens", 96) or 96),
            research_controls=research_controls,
        )

        created_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = output_root / f"{system_id}__{args.lane}"
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_group_id": f"{run_group_id}_{args.lane}",
            "created_at": created_at,
            "lane": args.lane,
            "system_id": system_id,
            "backend": bundle_system.get("backend", ""),
            "reasoner_backend": bundle_system.get("backend", ""),
            "router_backend": _router_backend(bundle_system),
            "retriever_backend": _retriever_backend(bundle_system),
            "reasoner": bundle_system.get("reasoner", ""),
            "router": bundle_system.get("router", ""),
            "retriever": bundle_system.get("retriever", ""),
            "reasoner_device": "auto",
            "router_device": None,
            "retriever_device": None,
            "reasoner_max_new_tokens": int(bundle_system.get("reasoner_max_new_tokens", 96) or 96),
            "request_timeout_seconds": float(bundle_system.get("request_timeout_seconds", 600.0) or 600.0),
            "thinking": bool(meta.get("thinking", False)),
            "limit": None,
            "episode_count": len(episodes),
            "episodes_path": str((lane_dir / "episodes.jsonl").resolve()),
            "runtime_bundle": bundle_snapshot,
            "run_intent": args.run_intent,
            "research_controls": research_controls.manifest_payload(),
            "warmup": warmup,
            "ablation_packet_episode_ids": args.episode_ids,
        }
        _write_progress(
            output_dir,
            {
                "status": "running",
                "lane": args.lane,
                "created_at": created_at,
                "completed_runs": 0,
                "planned_runs": len(episodes),
                "completed_episode_ids": [],
                "warmup": warmup,
                "runtime_bundle": bundle_snapshot,
            },
        )
        traces = []
        completed_episode_ids: list[str] = []
        for index, episode in enumerate(episodes, start=1):
            trace = runner.run(episode)
            traces.append(trace)
            completed_episode_ids.append(episode.episode_id)
            summary = summarize_episode_traces(traces)
            _write_outputs(output_dir, traces, {**summary, "lane": args.lane, "created_at": created_at, "output_dir": str(output_dir.resolve())}, manifest)
            _write_progress(
                output_dir,
                {
                    "status": "running" if index < len(episodes) else "completed",
                    "lane": args.lane,
                    "created_at": created_at,
                    "completed_runs": len(traces),
                    "planned_runs": len(episodes),
                    "last_completed_episode_id": episode.episode_id,
                    "completed_episode_ids": completed_episode_ids,
                    "latest_summary": summary,
                },
            )

        summary = summarize_episode_traces(traces)
        summary_payload = {
            **summary,
            "lane": args.lane,
            "created_at": created_at,
            "output_dir": str(output_dir.resolve()),
            "episode_ids": args.episode_ids,
        }
        _write_outputs(output_dir, traces, summary_payload, manifest)
        results.append({"system_id": system_id, "output_dir": str(output_dir.resolve()), "summary": summary_payload})

    (output_root / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"run_group_id": run_group_id, "systems": len(args.system_ids), "episodes": len(args.episode_ids), "output_dir": str(output_root.resolve())}, indent=2, ensure_ascii=False))


def _router_backend(system_meta: dict[str, Any]) -> str:
    if str(system_meta.get("executor_mode", "")) == "local_specialists":
        return "hf_service" if str(system_meta.get("backend", "")) == "hf_service" else "hf"
    return "heuristic"


def _retriever_backend(system_meta: dict[str, Any]) -> str:
    if str(system_meta.get("executor_mode", "")) == "local_specialists":
        return "hf_service" if str(system_meta.get("backend", "")) == "hf_service" else "hf"
    return "heuristic"


def _write_outputs(output_dir: Path, traces: list[Any], summary_payload: dict[str, Any], manifest: dict[str, Any]) -> None:
    dump_jsonl(output_dir / "episode_traces.jsonl", traces)
    export_episode_leaderboard_csv(traces, output_dir / "episode_leaderboard.csv")
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_progress(output_dir: Path, progress_payload: dict[str, Any]) -> None:
    (output_dir / "progress.json").write_text(json.dumps(progress_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
