from __future__ import annotations

import csv
from pathlib import Path

from gemma4_capability_map.metrics.failure_taxonomy import failure_tags
from gemma4_capability_map.schemas import RunTrace


def export_leaderboard_csv(traces: list[RunTrace], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metric_names = sorted({name for trace in traces for name in trace.metrics})
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "task_id",
                "variant_id",
                "track",
                "architecture",
                "backend",
                "thinking_enabled",
                "benchmark_tags",
                "autonomy_level",
                "risk_tier",
                "language_stressor",
                "schema_stressor",
                "context_stressor",
                "efficiency_stressor",
                "failure_summary",
            ]
            + metric_names,
        )
        writer.writeheader()
        for trace in traces:
            tags = failure_tags(trace)
            writer.writerow(
                {
                    "run_id": trace.run_id,
                    "task_id": trace.task_id,
                    "variant_id": trace.variant_id,
                    "track": trace.track.value,
                    "architecture": trace.architecture,
                    "backend": trace.backend,
                    "thinking_enabled": trace.thinking_enabled,
                    "benchmark_tags": ",".join(trace.benchmark_tags),
                    "autonomy_level": trace.real_world_profile.autonomy_level if trace.real_world_profile else "",
                    "risk_tier": trace.real_world_profile.risk_tier if trace.real_world_profile else "",
                    "language_stressor": trace.stressors.get("language"),
                    "schema_stressor": trace.stressors.get("schema"),
                    "context_stressor": trace.stressors.get("context"),
                    "efficiency_stressor": trace.stressors.get("efficiency"),
                    "failure_summary": ", ".join(tags) if tags else "none",
                    **trace.metrics,
                }
            )
