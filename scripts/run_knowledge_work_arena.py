from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from gemma4_capability_map.benchmark import build_runtime_bundle, load_tasks, runtime_bundle_snapshot, warm_runtime_bundle
from gemma4_capability_map.io import dump_jsonl
from gemma4_capability_map.knowledge_work.exporters import export_episode_leaderboard_csv
from gemma4_capability_map.knowledge_work.loader import load_episodes
from gemma4_capability_map.knowledge_work.replay import summarize_episode_traces
from gemma4_capability_map.knowledge_work.runner import EpisodeRunner


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KnowledgeWorkArena episodes.")
    parser.add_argument("--lane", choices=["replayable_core", "live_web_stress"], default="replayable_core")
    parser.add_argument("--episodes-path", default=None)
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-update-latest", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", default="oracle")
    parser.add_argument("--reasoner-backend", default=None)
    parser.add_argument("--router-backend", default=None)
    parser.add_argument("--retriever-backend", default=None)
    parser.add_argument("--reasoner", default="google/gemma-4-E2B-it")
    parser.add_argument("--router", default="google/functiongemma-270m-it")
    parser.add_argument("--retriever", default="google/embeddinggemma-300m")
    parser.add_argument("--reasoner-device", default="auto")
    parser.add_argument("--router-device", default=None)
    parser.add_argument("--retriever-device", default=None)
    parser.add_argument("--reasoner-max-new-tokens", type=int, default=96)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--run-intent", choices=["canonical", "exploratory"], default=None)
    parser.add_argument("--system-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lane_dir = ROOT / "data" / "knowledge_work" / args.lane
    episodes_path = Path(args.episodes_path) if args.episodes_path else lane_dir / "episodes.jsonl"
    created_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "results" / "knowledge_work" / f"{created_at}_{args.lane}"
    )
    latest_dir = ROOT / "results" / "knowledge_work" / args.lane
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)
    target_dirs = (output_dir,) if args.no_update_latest else (output_dir, latest_dir)
    run_intent = args.run_intent or ("exploratory" if args.no_update_latest else "canonical")

    episodes = load_episodes(episodes_path)
    if args.episode_id:
        allowed = set(args.episode_id)
        episodes = [episode for episode in episodes if episode.episode_id in allowed]
    if args.limit is not None:
        episodes = episodes[: args.limit]
    tasks = load_tasks(track=None)
    bundle = build_runtime_bundle(
        tasks=tasks,
        pipeline_name="modular",
        backend=args.backend,
        reasoner_backend=args.reasoner_backend,
        router_backend=args.router_backend,
        retriever_backend=args.retriever_backend,
        reasoner_id=args.reasoner,
        router_id=args.router,
        retriever_id=args.retriever,
        reasoner_device=args.reasoner_device,
        router_device=args.router_device,
        retriever_device=args.retriever_device,
        reasoner_max_new_tokens=args.reasoner_max_new_tokens,
    )
    runner = EpisodeRunner(tasks=tasks, bundle=bundle, thinking_enabled=args.thinking)
    manifest = {
        "run_group_id": f"{created_at}_{args.lane}",
        "created_at": created_at,
        "lane": args.lane,
        "system_id": _infer_system_id(args),
        "backend": args.backend,
        "reasoner_backend": args.reasoner_backend or args.backend,
        "router_backend": args.router_backend or "",
        "retriever_backend": args.retriever_backend or "",
        "reasoner": args.reasoner,
        "router": args.router,
        "retriever": args.retriever,
        "reasoner_device": args.reasoner_device,
        "router_device": args.router_device,
        "retriever_device": args.retriever_device,
        "thinking": args.thinking,
        "limit": args.limit,
        "episode_count": len(episodes),
        "episodes_path": str(episodes_path.resolve()),
        "runtime_bundle": runtime_bundle_snapshot(bundle),
        "run_intent": run_intent,
    }
    _write_manifest(target_dirs, manifest)
    _write_progress(
        target_dirs,
        {
            "status": "starting",
            "lane": args.lane,
            "created_at": created_at,
            "completed_runs": 0,
            "planned_runs": len(episodes),
            "completed_episode_ids": [],
        },
    )
    _write_progress(
        target_dirs,
        {
            "status": "warming_runtime",
            "lane": args.lane,
            "created_at": created_at,
            "completed_runs": 0,
            "planned_runs": len(episodes),
            "completed_episode_ids": [],
        },
    )
    warmup = warm_runtime_bundle(bundle, tasks)
    manifest["warmup"] = warmup
    manifest["runtime_bundle"] = runtime_bundle_snapshot(bundle)
    _write_manifest(target_dirs, manifest)
    _write_progress(
        target_dirs,
        {
            "status": "running",
            "lane": args.lane,
            "created_at": created_at,
            "completed_runs": 0,
            "planned_runs": len(episodes),
            "completed_episode_ids": [],
            "warmup": warmup,
            "runtime_bundle": manifest["runtime_bundle"],
        },
    )

    traces = []
    completed_episode_ids: list[str] = []
    try:
        for index, episode in enumerate(episodes, start=1):
            trace = runner.run(episode)
            traces.append(trace)
            completed_episode_ids.append(episode.episode_id)
            summary = summarize_episode_traces(traces)
            summary_payload = {
                **summary,
                "lane": args.lane,
                "created_at": created_at,
                "episodes_path": str(episodes_path.resolve()),
                "output_dir": str(output_dir.resolve()),
            }
            _write_outputs(target_dirs, traces, summary_payload, manifest)
            _write_progress(
                target_dirs,
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
    except Exception as exc:
        partial_summary = summarize_episode_traces(traces)
        _write_outputs(
            target_dirs,
            traces,
            {
                **partial_summary,
                "lane": args.lane,
                "created_at": created_at,
                "episodes_path": str(episodes_path.resolve()),
                "output_dir": str(output_dir.resolve()),
            },
            manifest,
        )
        _write_progress(
            target_dirs,
            {
                "status": "failed",
                "lane": args.lane,
                "created_at": created_at,
                "completed_runs": len(traces),
                "planned_runs": len(episodes),
                "completed_episode_ids": completed_episode_ids,
                "error": f"{type(exc).__name__}: {exc}",
                "latest_summary": partial_summary,
            },
        )
        raise

    summary = summarize_episode_traces(traces)
    summary_payload = {
        **summary,
        "lane": args.lane,
        "created_at": created_at,
        "episodes_path": str(episodes_path.resolve()),
        "output_dir": str(output_dir.resolve()),
    }
    _write_outputs(target_dirs, traces, summary_payload, manifest)
    _append_history_record(summary_payload, manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _infer_system_id(args: argparse.Namespace) -> str | None:
    if args.system_id:
        return args.system_id

    backend = str(args.backend or "")
    reasoner = str(args.reasoner or "")
    router = str(args.router or "")
    retriever = str(args.retriever or "")
    reasoner_backend = str(args.reasoner_backend or args.backend or "").lower()
    router_backend = str(args.router_backend or "").lower()
    retriever_backend = str(args.retriever_backend or "").lower()

    if backend == "oracle":
        return "oracle_gemma4_e2b"

    if (
        backend == "hf_service"
        and reasoner == "google/gemma-4-E2B-it"
        and reasoner_backend == "hf_service"
        and router_backend in {"", "heuristic"}
        and retriever_backend in {"", "heuristic"}
    ):
        return "hf_service_gemma4_reasoner_only"

    if (
        backend == "hf"
        and reasoner == "google/gemma-4-E2B-it"
        and reasoner_backend == "hf"
        and router_backend in {"", "heuristic"}
        and retriever_backend in {"", "heuristic"}
    ):
        return "hf_gemma4_e2b_reasoner_only"

    if (
        backend == "hf_service"
        and reasoner == "google/gemma-4-E2B-it"
        and router == "google/functiongemma-270m-it"
        and retriever == "google/embeddinggemma-300m"
        and reasoner_backend == "hf_service"
        and router_backend in {"hf", "hf_service"}
        and retriever_backend in {"hf", "hf_service"}
    ):
        return "hf_service_gemma4_specialists_cpu"

    if (
        backend == "hf"
        and reasoner == "google/gemma-4-E2B-it"
        and router == "google/functiongemma-270m-it"
        and retriever == "google/embeddinggemma-300m"
        and reasoner_backend == "hf"
        and router_backend == "hf"
        and retriever_backend == "hf"
    ):
        return "hf_gemma4_e2b_specialists_cpu"

    return None


def _append_history_record(summary: dict, manifest: dict) -> None:
    history_dir = ROOT / "results" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "recorded_at": manifest["created_at"],
        "manifest": manifest,
        "summary": summary,
    }
    with (history_dir / "knowledge_work_runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_outputs(
    directories: tuple[Path, ...],
    traces,
    summary_payload: dict,
    manifest: dict,
) -> None:
    for directory in directories:
        dump_jsonl(directory / "episode_traces.jsonl", traces)
        export_episode_leaderboard_csv(traces, directory / "episode_leaderboard.csv")
        (directory / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_manifest(directories: tuple[Path, ...], manifest: dict) -> None:
    for directory in directories:
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_progress(directories: tuple[Path, ...], progress_payload: dict) -> None:
    for directory in directories:
        (directory / "progress.json").write_text(json.dumps(progress_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
