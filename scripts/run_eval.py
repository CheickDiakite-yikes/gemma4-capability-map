from __future__ import annotations

import argparse

from pathlib import Path

from gemma4_capability_map.benchmark import ROOT, load_tasks, load_variants, run_benchmark
from gemma4_capability_map.traces.exporters import export_leaderboard_csv
from gemma4_capability_map.traces.recorder import TraceRecorder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["monolith", "hybrid", "modular"], default="monolith")
    parser.add_argument("--backend", choices=["oracle", "heuristic", "hf", "hf_service", "mlx"], default="oracle")
    parser.add_argument("--reasoner-backend", choices=["oracle", "heuristic", "hf", "hf_service", "mlx"], default=None)
    parser.add_argument("--router-backend", choices=["oracle", "heuristic", "hf"], default=None)
    parser.add_argument("--retriever-backend", choices=["heuristic", "hf"], default=None)
    parser.add_argument("--reasoner", default="google/gemma-4-E4B-it")
    parser.add_argument("--router", default="google/functiongemma-270m-it")
    parser.add_argument("--retriever", default="google/embeddinggemma-300m")
    parser.add_argument("--reasoner-device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--planning-max-new-tokens", type=int, default=None)
    parser.add_argument("--final-max-new-tokens", type=int, default=None)
    parser.add_argument("--track", choices=["thinking", "tool_routing", "retrieval", "full_stack", "visual_tool_orchestration"], default=None)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--variants", action="store_true", help="Run generated variants instead of only clean runs.")
    parser.add_argument("--trace-path", default=str(ROOT / "results" / "raw" / "traces.jsonl"))
    parser.add_argument("--leaderboard-path", default=str(ROOT / "results" / "leaderboard.csv"))
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()

    tasks = load_tasks(track=args.track, task_ids=args.task_id)
    variants = load_variants(tasks, include_generated=args.variants)
    runs = run_benchmark(
        tasks=tasks,
        variants=variants,
        pipeline_name=args.pipeline,
        backend=args.backend,
        reasoner_backend=args.reasoner_backend,
        router_backend=args.router_backend,
        retriever_backend=args.retriever_backend,
        reasoner_id=args.reasoner,
        router_id=args.router,
        retriever_id=args.retriever,
        reasoner_device=args.reasoner_device,
        reasoner_max_new_tokens=args.max_new_tokens,
        planning_max_new_tokens=args.planning_max_new_tokens,
        final_max_new_tokens=args.final_max_new_tokens,
        limit=args.limit,
        thinking_enabled=args.thinking,
    )
    recorder = TraceRecorder()
    for trace in runs:
        recorder.add(trace)
    recorder.write(args.trace_path)
    export_leaderboard_csv(runs, args.leaderboard_path)
    print(f"Wrote {len(runs)} traces to {args.trace_path}")
    print(f"Wrote leaderboard to {args.leaderboard_path}")


if __name__ == "__main__":
    main()
