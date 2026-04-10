from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma4_capability_map.io import dump_jsonl
from gemma4_capability_map.knowledge_work.exporters import export_episode_leaderboard_csv
from gemma4_capability_map.knowledge_work.loader import load_episodes
from gemma4_capability_map.knowledge_work.replay import load_episode_traces, summarize_episode_traces
from gemma4_capability_map.knowledge_work.scoring import score_episode


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODE_PATHS = (
    ROOT / "data" / "knowledge_work" / "replayable_core" / "episodes.jsonl",
    ROOT / "data" / "knowledge_work" / "live_web_stress" / "episodes.jsonl",
)


def _episode_index() -> dict[str, object]:
    episodes = {}
    for path in DEFAULT_EPISODE_PATHS:
        for episode in load_episodes(path):
            episodes[episode.episode_id] = episode
    return episodes


def rescore_run_dir(run_dir: Path, episode_index: dict[str, object]) -> dict[str, float]:
    traces_path = run_dir / "episode_traces.jsonl"
    if not traces_path.exists():
        raise FileNotFoundError(f"missing episode traces: {traces_path}")

    traces = load_episode_traces(traces_path)
    for trace in traces:
        episode = episode_index.get(trace.episode_id)
        if episode is None:
            raise KeyError(f"unknown episode_id for rescore: {trace.episode_id}")
        trace.scorecard = score_episode(episode, trace)

    dump_jsonl(traces_path, traces)
    export_episode_leaderboard_csv(traces, run_dir / "episode_leaderboard.csv")
    summary = summarize_episode_traces(traces)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore saved KnowledgeWorkArena runs from episode traces.")
    parser.add_argument("run_dirs", nargs="+", help="One or more KnowledgeWorkArena run directories to rescore.")
    args = parser.parse_args()

    episode_index = _episode_index()
    for raw_dir in args.run_dirs:
        run_dir = Path(raw_dir)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        summary = rescore_run_dir(run_dir, episode_index)
        print(f"{run_dir}: {json.dumps(summary, sort_keys=True)}")


if __name__ == "__main__":
    main()
