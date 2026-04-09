from __future__ import annotations

from pathlib import Path

from run_eval import load_tasks, load_variants, run_benchmark


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    tasks = load_tasks(track=None)
    variants = load_variants(tasks, include_generated=True)
    for pipeline in ["monolith", "hybrid", "modular"]:
        traces = run_benchmark(
            tasks=tasks,
            variants=variants,
            pipeline_name=pipeline,
            backend="oracle",
            reasoner_backend=None,
            router_backend=None,
            retriever_backend=None,
            reasoner_id="google/gemma-4-E4B-it",
            router_id="google/functiongemma-270m-it",
            retriever_id="google/embeddinggemma-300m",
            limit=24,
            thinking_enabled=pipeline == "monolith",
        )
        print(f"{pipeline}: {len(traces)} traces")


if __name__ == "__main__":
    main()
