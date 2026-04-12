from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma4_capability_map.evals.agent_eval import score_full_stack_trace
from gemma4_capability_map.evals.retrieval_eval import score_retrieval_trace
from gemma4_capability_map.evals.thinking_eval import score_thinking_trace
from gemma4_capability_map.evals.tool_eval import score_tool_trace
from gemma4_capability_map.evals.visual_eval import score_visual_trace
from gemma4_capability_map.io import dump_jsonl
from gemma4_capability_map.benchmark import load_tasks
from gemma4_capability_map.knowledge_work.exporters import export_episode_leaderboard_csv
from gemma4_capability_map.knowledge_work.loader import load_episodes
from gemma4_capability_map.knowledge_work.replay import load_episode_traces, summarize_episode_traces
from gemma4_capability_map.knowledge_work.scoring import score_episode
from gemma4_capability_map.schemas import Track


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


def _task_index() -> dict[str, object]:
    return {task.task_id: task for task in load_tasks(track=None)}


def _score_task_trace(task, trace) -> dict[str, float | int | bool]:  # noqa: ANN001
    if task.track == Track.THINKING:
        return score_thinking_trace(task, trace)
    if task.track == Track.TOOL_ROUTING:
        return score_tool_trace(task, trace)
    if task.track == Track.RETRIEVAL:
        return score_retrieval_trace(task, trace)
    if task.track == Track.VISUAL_TOOL_ORCHESTRATION:
        return score_visual_trace(task, trace)
    return score_full_stack_trace(task, trace)


def rescore_run_dir(run_dir: Path, episode_index: dict[str, object], task_index: dict[str, object]) -> dict[str, float]:
    traces_path = run_dir / "episode_traces.jsonl"
    if not traces_path.exists():
        raise FileNotFoundError(f"missing episode traces: {traces_path}")

    traces = load_episode_traces(traces_path)
    for trace in traces:
        episode = episode_index.get(trace.episode_id)
        if episode is None:
            raise KeyError(f"unknown episode_id for rescore: {trace.episode_id}")
        for stage_trace in trace.stage_traces:
            for task_trace in stage_trace.task_traces:
                task = task_index.get(task_trace.task_id)
                if task is None:
                    raise KeyError(f"unknown task_id for rescore: {task_trace.task_id}")
                task_trace.metrics = _score_task_trace(task, task_trace)
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
    task_index = _task_index()
    for raw_dir in args.run_dirs:
        run_dir = Path(raw_dir)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        summary = rescore_run_dir(run_dir, episode_index, task_index)
        print(f"{run_dir}: {json.dumps(summary, sort_keys=True)}")


if __name__ == "__main__":
    main()
