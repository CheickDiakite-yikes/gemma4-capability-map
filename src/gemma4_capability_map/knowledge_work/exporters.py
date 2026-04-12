from __future__ import annotations

import csv
from pathlib import Path

from gemma4_capability_map.knowledge_work.schemas import EpisodeTrace


def export_episode_leaderboard_csv(traces: list[EpisodeTrace], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "episode_id",
                "role_family",
                "lane",
                "workspace_id",
                "benchmark_tags",
                "artifact_quality_score",
                "browser_workflow_score",
                "strict_interface_score",
                "recovered_execution_score",
                "revision_responsiveness",
                "memory_retention_score",
                "escalation_correctness",
                "collateral_damage_free",
                "human_time_ratio",
                "controller_repair_count",
                "argument_repair_count",
                "controller_fallback_count",
                "intent_override_count",
                "raw_planning_clean_rate",
                "role_readiness_score",
            ],
        )
        writer.writeheader()
        for trace in traces:
            writer.writerow(
                {
                    "run_id": trace.run_id,
                    "episode_id": trace.episode_id,
                    "role_family": trace.role_family.value,
                    "lane": trace.lane.value,
                    "workspace_id": trace.workspace_id,
                    "benchmark_tags": ",".join(trace.benchmark_tags),
                    **trace.scorecard.model_dump(mode="json"),
                }
            )
