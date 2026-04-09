from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from gemma4_capability_map.benchmark import build_runtime_bundle, load_tasks
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
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--backend", default="oracle")
    parser.add_argument("--reasoner-backend", default=None)
    parser.add_argument("--router-backend", default=None)
    parser.add_argument("--retriever-backend", default=None)
    parser.add_argument("--reasoner", default="google/gemma-4-E2B-it")
    parser.add_argument("--router", default="google/functiongemma-270m-it")
    parser.add_argument("--retriever", default="google/embeddinggemma-300m")
    parser.add_argument("--reasoner-device", default="auto")
    parser.add_argument("--reasoner-max-new-tokens", type=int, default=96)
    parser.add_argument("--thinking", action="store_true")
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

    episodes = load_episodes(episodes_path)[: args.limit]
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
        reasoner_max_new_tokens=args.reasoner_max_new_tokens,
    )
    runner = EpisodeRunner(tasks=tasks, bundle=bundle, thinking_enabled=args.thinking)
    traces = [runner.run(episode) for episode in episodes]
    dump_jsonl(output_dir / "episode_traces.jsonl", traces)
    dump_jsonl(latest_dir / "episode_traces.jsonl", traces)
    export_episode_leaderboard_csv(traces, output_dir / "episode_leaderboard.csv")
    export_episode_leaderboard_csv(traces, latest_dir / "episode_leaderboard.csv")
    summary = summarize_episode_traces(traces)
    summary_payload = {
        **summary,
        "lane": args.lane,
        "created_at": created_at,
        "episodes_path": str(episodes_path.resolve()),
        "output_dir": str(output_dir.resolve()),
    }
    manifest = {
        "run_group_id": f"{created_at}_{args.lane}",
        "created_at": created_at,
        "lane": args.lane,
        "backend": args.backend,
        "reasoner_backend": args.reasoner_backend or args.backend,
        "router_backend": args.router_backend or "",
        "retriever_backend": args.retriever_backend or "",
        "reasoner": args.reasoner,
        "router": args.router,
        "retriever": args.retriever,
        "thinking": args.thinking,
        "limit": args.limit,
        "episode_count": len(episodes),
        "episodes_path": str(episodes_path.resolve()),
    }
    for directory in (output_dir, latest_dir):
        (directory / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _append_history_record(summary_payload, manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


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


if __name__ == "__main__":
    main()
