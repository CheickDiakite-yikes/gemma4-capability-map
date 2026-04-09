from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-dir", default=str(ROOT / "results" / "history"))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "history"))
    parser.add_argument("--matrix-root", default=str(ROOT / "results" / "alpha_matrix"))
    args = parser.parse_args()

    history_dir = Path(args.history_dir)
    output_dir = Path(args.output_dir)
    matrix_root = Path(args.matrix_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows = _load_jsonl(history_dir / "experiment_runs.jsonl")
    improvement_rows = _load_jsonl(history_dir / "improvements.jsonl")
    run_group_snapshots = _load_run_group_snapshots(matrix_root)

    latest_completed = _latest_completed_by_experiment(run_rows)
    best_completed = _best_completed_by_experiment(run_rows)
    run_groups = _summarize_run_groups(run_rows)
    latest_complete_run_groups = _latest_complete_run_groups_by_matrix(run_group_snapshots)
    latest_run_groups = _latest_run_groups_by_matrix(run_group_snapshots)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "history_dir": str(history_dir.resolve()),
        "matrix_root": str(matrix_root.resolve()),
        "totals": {
            "logged_experiments": len(run_rows),
            "completed_experiments": sum(1 for row in run_rows if row.get("summary", {}).get("status") == "completed"),
            "blocked_experiments": sum(1 for row in run_rows if row.get("summary", {}).get("status") == "blocked"),
            "failed_experiments": sum(1 for row in run_rows if row.get("summary", {}).get("status") == "failed"),
            "run_groups": len(run_groups),
            "experiments_with_completed_runs": len(latest_completed),
            "matrix_snapshots": len(run_group_snapshots),
        },
        "latest_completed_by_experiment": latest_completed,
        "best_completed_by_experiment": best_completed,
        "run_groups": run_groups,
        "run_group_snapshots": run_group_snapshots,
        "latest_complete_run_groups_by_matrix": latest_complete_run_groups,
        "latest_run_groups_by_matrix": latest_run_groups,
        "recent_improvements": improvement_rows[-20:],
    }

    _write_json(output_dir / "history_report.json", report)
    (output_dir / "history_report.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_csv(output_dir / "latest_completed_experiments.csv", latest_completed)
    _write_csv(output_dir / "run_groups.csv", run_groups)
    _write_csv(output_dir / "canonical_run_groups.csv", latest_complete_run_groups)
    print(f"Wrote history report to {output_dir}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _latest_completed_by_experiment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_experiment: dict[str, dict[str, Any]] = {}
    for row in rows:
        summary = row.get("summary", {})
        if summary.get("status") != "completed":
            continue
        experiment_id = summary.get("experiment_id")
        if experiment_id:
            by_experiment[experiment_id] = _experiment_snapshot(row)
    return sorted(by_experiment.values(), key=lambda item: item["experiment_id"])


def _best_completed_by_experiment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        summary = row.get("summary", {})
        if summary.get("status") == "completed" and summary.get("experiment_id"):
            grouped[summary["experiment_id"]].append(row)

    snapshots: list[dict[str, Any]] = []
    for experiment_id, candidates in grouped.items():
        best = max(
            candidates,
            key=lambda row: (
                float(row.get("summary", {}).get("success_rate", 0.0)),
                -float(row.get("summary", {}).get("avg_latency_ms", 0.0)),
                row.get("created_at", ""),
            ),
        )
        snapshots.append(_experiment_snapshot(best))
    return sorted(snapshots, key=lambda item: item["experiment_id"])


def _experiment_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    summary = row.get("summary", {})
    experiment = row.get("experiment", {})
    return {
        "experiment_id": summary.get("experiment_id"),
        "run_group_id": row.get("run_group_id"),
        "matrix_name": row.get("matrix_name"),
        "created_at": row.get("created_at"),
        "track": summary.get("track"),
        "pipeline": summary.get("pipeline"),
        "backend": summary.get("backend"),
        "reasoner": summary.get("reasoner"),
        "variants": summary.get("variants"),
        "status": summary.get("status"),
        "success_rate": float(summary.get("success_rate", 0.0)),
        "strict_interface_rate": float(summary.get("strict_interface_rate", 0.0)),
        "recovered_execution_rate": float(summary.get("recovered_execution_rate", 0.0)),
        "real_world_readiness_avg": float(summary.get("real_world_readiness_avg", 0.0)),
        "avg_latency_ms": float(summary.get("avg_latency_ms", 0.0)),
        "runs": float(summary.get("runs", 0.0)),
        "failure_breakdown": summary.get("failure_breakdown", {}),
        "notes": experiment.get("notes", []),
        "output_dir": row.get("output_dir"),
    }


def _summarize_run_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("run_group_id", ""), row.get("matrix_name", ""))].append(row)

    summaries: list[dict[str, Any]] = []
    for (run_group_id, matrix_name), entries in grouped.items():
        latest_created_at = max((row.get("created_at", "") for row in entries), default="")
        completed = [row for row in entries if row.get("summary", {}).get("status") == "completed"]
        blocked = [row for row in entries if row.get("summary", {}).get("status") == "blocked"]
        failed = [row for row in entries if row.get("summary", {}).get("status") == "failed"]
        success_values = [float(row.get("summary", {}).get("success_rate", 0.0)) for row in completed]
        strict_values = [float(row.get("summary", {}).get("strict_interface_rate", 0.0)) for row in completed]
        recovered_values = [float(row.get("summary", {}).get("recovered_execution_rate", 0.0)) for row in completed]
        readiness_values = [float(row.get("summary", {}).get("real_world_readiness_avg", 0.0)) for row in completed if "real_world_readiness_avg" in row.get("summary", {})]
        latency_values = [float(row.get("summary", {}).get("avg_latency_ms", 0.0)) for row in completed]
        failure_counter: Counter[str] = Counter()
        for row in completed:
            failure_counter.update(row.get("summary", {}).get("failure_breakdown", {}))
        summaries.append(
            {
                "run_group_id": run_group_id,
                "matrix_name": matrix_name,
                "created_at": latest_created_at,
                "experiments_logged": len(entries),
                "completed_experiments": len(completed),
                "blocked_experiments": len(blocked),
                "failed_experiments": len(failed),
                "avg_success_rate": round(sum(success_values) / len(success_values), 4) if success_values else 0.0,
                "avg_strict_interface_rate": round(sum(strict_values) / len(strict_values), 4) if strict_values else 0.0,
                "avg_recovered_execution_rate": round(sum(recovered_values) / len(recovered_values), 4) if recovered_values else 0.0,
                "avg_real_world_readiness": round(sum(readiness_values) / len(readiness_values), 4) if readiness_values else 0.0,
                "avg_latency_ms": round(sum(latency_values) / len(latency_values), 2) if latency_values else 0.0,
                "failure_breakdown": dict(sorted(failure_counter.items())),
            }
        )
    return sorted(summaries, key=lambda item: item["run_group_id"], reverse=True)


def _load_run_group_snapshots(matrix_root: Path) -> list[dict[str, Any]]:
    if not matrix_root.exists():
        return []

    snapshots: list[dict[str, Any]] = []
    for path in sorted(matrix_root.iterdir(), reverse=True):
        if not path.is_dir():
            continue
        manifest_path = path / "manifest.json"
        summary_path = path / "summary.json"
        if not manifest_path.exists() or not summary_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_experiments = len(manifest.get("experiments", []))
        experiment_rows = summary.get("experiments", [])
        completed_experiments = int(
            summary.get(
                "completed_experiments",
                sum(1 for item in experiment_rows if item.get("status") == "completed"),
            )
        )
        blocked_experiments = int(
            summary.get(
                "blocked_experiments",
                sum(1 for item in experiment_rows if item.get("status") == "blocked"),
            )
        )
        failed_experiments = int(
            summary.get(
                "failed_experiments",
                sum(1 for item in experiment_rows if item.get("status") == "failed"),
            )
        )
        matrix_complete = bool(summary.get("matrix_complete")) or (
            expected_experiments > 0
            and completed_experiments + blocked_experiments == expected_experiments
            and failed_experiments == 0
        )
        snapshots.append(
            {
                "run_group_id": manifest.get("run_group_id"),
                "matrix_name": manifest.get("matrix_name"),
                "created_at": manifest.get("created_at"),
                "description": manifest.get("description"),
                "expected_experiments": expected_experiments,
                "completed_experiments": completed_experiments,
                "blocked_experiments": blocked_experiments,
                "failed_experiments": failed_experiments,
                "matrix_complete": matrix_complete,
                "avg_success_rate": round(
                    sum(float(item.get("success_rate", 0.0)) for item in experiment_rows)
                    / max(len(experiment_rows), 1),
                    4,
                ),
                "avg_strict_interface_rate": round(
                    sum(float(item.get("strict_interface_rate", 0.0)) for item in experiment_rows)
                    / max(len(experiment_rows), 1),
                    4,
                ),
                "avg_recovered_execution_rate": round(
                    sum(float(item.get("recovered_execution_rate", 0.0)) for item in experiment_rows)
                    / max(len(experiment_rows), 1),
                    4,
                ),
                "avg_real_world_readiness": round(
                    sum(float(item.get("real_world_readiness_avg", 0.0)) for item in experiment_rows if "real_world_readiness_avg" in item)
                    / max(sum(1 for item in experiment_rows if "real_world_readiness_avg" in item), 1),
                    4,
                ),
                "failure_breakdown": summary.get("failure_breakdown", {}),
                "output_dir": str(path.resolve()),
            }
        )
    return snapshots


def _latest_complete_run_groups_by_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("matrix_complete"):
            matrix_name = str(row.get("matrix_name", ""))
            if matrix_name:
                grouped[matrix_name].append(row)

    latest: list[dict[str, Any]] = []
    for matrix_name, candidates in grouped.items():
        max_expected = max(int(candidate.get("expected_experiments", 0)) for candidate in candidates)
        full_scope = [
            candidate
            for candidate in candidates
            if int(candidate.get("expected_experiments", 0)) == max_expected
        ]
        latest.append(
            max(
                full_scope,
                key=lambda item: (
                    int(item.get("expected_experiments", 0)),
                    str(item.get("run_group_id", "")),
                ),
            )
        )
    return sorted(latest, key=lambda item: item["matrix_name"])


def _latest_run_groups_by_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        matrix_name = str(row.get("matrix_name", ""))
        if matrix_name:
            latest[matrix_name] = row
    return sorted(latest.values(), key=lambda item: item["matrix_name"])


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# History Report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Logged experiments: `{report['totals']['logged_experiments']}`",
        f"- Completed experiments: `{report['totals']['completed_experiments']}`",
        f"- Blocked experiments: `{report['totals']['blocked_experiments']}`",
        f"- Failed experiments: `{report['totals']['failed_experiments']}`",
        f"- Run groups: `{report['totals']['run_groups']}`",
        f"- Matrix snapshots: `{report['totals']['matrix_snapshots']}`",
        "",
        "## Canonical Matrix Snapshots",
        "",
        "| Matrix | Run group | Complete | Completed / Expected | Avg success | Strict | Recovered | Readiness |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["latest_complete_run_groups_by_matrix"]:
        lines.append(
            f"| {row['matrix_name']} | {row['run_group_id']} | {row['matrix_complete']} | "
            f"{row['completed_experiments']}/{row['expected_experiments']} | {row['avg_success_rate']} | "
            f"{row.get('avg_strict_interface_rate', 0.0)} | {row.get('avg_recovered_execution_rate', 0.0)} | "
            f"{row.get('avg_real_world_readiness', 0.0)} |"
        )
    lines.extend(
        [
            "",
            "## Latest Run Groups By Matrix",
            "",
            "| Matrix | Run group | Complete | Completed / Expected | Blocked | Failed |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in report["latest_run_groups_by_matrix"]:
        lines.append(
            f"| {row['matrix_name']} | {row['run_group_id']} | {row['matrix_complete']} | "
            f"{row['completed_experiments']}/{row['expected_experiments']} | {row.get('blocked_experiments', 0)} | {row['failed_experiments']} |"
        )
    lines.extend(
        [
            "",
        "## Latest Completed Experiments",
        "",
        "| Experiment | Run group | Track | Backend | Success | Strict | Recovered | Readiness | Avg latency ms | Runs |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["latest_completed_by_experiment"]:
        lines.append(
            f"| {row['experiment_id']} | {row['run_group_id']} | {row['track']} | {row['backend']} | "
            f"{row['success_rate']} | {row.get('strict_interface_rate', 0.0)} | "
            f"{row.get('recovered_execution_rate', 0.0)} | {row.get('real_world_readiness_avg', 0.0)} | "
            f"{row['avg_latency_ms']} | {row['runs']} |"
        )
    lines.extend(
        [
            "",
            "## Best Completed Experiments",
            "",
            "| Experiment | Run group | Track | Backend | Success | Strict | Recovered | Readiness | Avg latency ms |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["best_completed_by_experiment"]:
        lines.append(
            f"| {row['experiment_id']} | {row['run_group_id']} | {row['track']} | {row['backend']} | "
            f"{row['success_rate']} | {row.get('strict_interface_rate', 0.0)} | "
            f"{row.get('recovered_execution_rate', 0.0)} | {row.get('real_world_readiness_avg', 0.0)} | "
            f"{row['avg_latency_ms']} |"
        )
    lines.extend(
        [
            "",
            "## Recent Run Groups",
            "",
            "| Run group | Matrix | Completed | Blocked | Failed | Avg success | Strict | Recovered | Readiness | Avg latency ms | Complete snapshot |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in report["run_groups"][:12]:
        snapshot = next(
            (
                item
                for item in report["run_group_snapshots"]
                if item["run_group_id"] == row["run_group_id"] and item["matrix_name"] == row["matrix_name"]
            ),
            None,
        )
        lines.append(
            f"| {row['run_group_id']} | {row['matrix_name']} | {row['completed_experiments']} | "
            f"{row.get('blocked_experiments', 0)} | {row['failed_experiments']} | {row['avg_success_rate']} | "
            f"{row.get('avg_strict_interface_rate', 0.0)} | {row.get('avg_recovered_execution_rate', 0.0)} | "
            f"{row.get('avg_real_world_readiness', 0.0)} | {row['avg_latency_ms']} | "
            f"{snapshot.get('matrix_complete') if snapshot else 'unknown'} |"
        )
    if report["recent_improvements"]:
        lines.extend(
            [
                "",
                "## Recent Improvements",
                "",
                "| Experiment | Current run | Previous run | Success delta | Latency improvement ms |",
                "| --- | --- | --- | ---: | ---: |",
            ]
        )
        for row in report["recent_improvements"][-12:]:
            lines.append(
                f"| {row['experiment_id']} | {row['current_run_group_id']} | {row['previous_run_group_id']} | "
                f"{row['success_rate_delta']} | {row['latency_improvement_ms']} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
