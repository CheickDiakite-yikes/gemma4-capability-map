from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from gemma4_capability_map.reporting.knowledge_work_board import build_board_rows, write_board_exports


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    runs = _load_runs(ROOT / "results" / "history" / "knowledge_work_runs.jsonl")
    snapshots = _load_snapshot_dirs(ROOT / "results" / "knowledge_work")
    latest_canonical_by_lane = _latest_by_lane(snapshots, intent="canonical")
    latest_exploratory_by_lane = _latest_by_lane(snapshots, intent="exploratory")
    best_by_lane = _best_by_lane(snapshots)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_runs": len(runs),
        "snapshot_count": len(snapshots),
        "latest_by_lane": latest_canonical_by_lane,
        "latest_canonical_by_lane": latest_canonical_by_lane,
        "latest_exploratory_by_lane": latest_exploratory_by_lane,
        "best_by_lane": best_by_lane,
        "recent_runs": runs[-20:],
    }

    history_dir = ROOT / "results" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    (history_dir / "knowledge_work_history.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (history_dir / "knowledge_work_history.md").write_text(_markdown_report(report), encoding="utf-8")
    _write_csv(history_dir / "knowledge_work_canonical.csv", latest_canonical_by_lane)
    _write_csv(history_dir / "knowledge_work_exploratory.csv", latest_exploratory_by_lane)
    write_board_exports(build_board_rows(ROOT / "results" / "knowledge_work"), history_dir)
    print(f"Wrote KnowledgeWorkArena history report to {history_dir}")


def _load_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _load_snapshot_dirs(root: Path) -> list[dict]:
    snapshots: list[dict] = []
    if not root.exists():
        return snapshots
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        manifest_path = path / "manifest.json"
        summary_path = path / "summary.json"
        leaderboard_path = path / "episode_leaderboard.csv"
        if not manifest_path.exists() or not summary_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "run_group_id": manifest.get("run_group_id"),
            "created_at": manifest.get("created_at"),
            "lane": manifest.get("lane"),
            "run_intent": _infer_run_intent(path, manifest),
            "backend": manifest.get("backend"),
            "reasoner": manifest.get("reasoner"),
            "episode_count": manifest.get("episode_count"),
            "runs": float(summary.get("runs", 0.0)),
            "artifact_quality_avg": float(summary.get("artifact_quality_avg", 0.0)),
            "browser_workflow_avg": float(summary.get("browser_workflow_avg", 0.0)),
            "strict_interface_avg": float(summary.get("strict_interface_avg", 0.0)),
            "recovered_execution_avg": float(summary.get("recovered_execution_avg", 0.0)),
            "real_world_readiness_avg": float(summary.get("real_world_readiness_avg", 0.0)),
            "escalation_correctness_avg": float(summary.get("escalation_correctness_avg", 0.0)),
            "output_dir": str(path.resolve()),
        }
        if leaderboard_path.exists():
            row["role_breakdown"] = _role_breakdown(leaderboard_path)
        snapshots.append(row)
    return snapshots


def _role_breakdown(path: Path) -> dict[str, dict[str, float]]:
    by_role: dict[str, list[dict]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_role.setdefault(row["role_family"], []).append(row)
    summary: dict[str, dict[str, float]] = {}
    for role, rows in by_role.items():
        summary[role] = {
            "artifact_quality_avg": round(sum(float(row["artifact_quality_score"]) for row in rows) / len(rows), 4),
            "browser_workflow_avg": round(sum(float(row.get("browser_workflow_score", 0.0) or 0.0) for row in rows) / len(rows), 4),
            "strict_interface_avg": round(sum(float(row["strict_interface_score"]) for row in rows) / len(rows), 4),
            "role_readiness_avg": round(sum(float(row["role_readiness_score"]) for row in rows) / len(rows), 4),
        }
    return summary


def _latest_by_lane(rows: list[dict], intent: str) -> list[dict]:
    by_lane: dict[str, dict] = {}
    for row in rows:
        lane = str(row.get("lane", ""))
        if lane and row.get("run_intent") == intent:
            by_lane[lane] = row
    return sorted(by_lane.values(), key=lambda item: item["lane"])


def _best_by_lane(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("lane", "")), []).append(row)
    best: list[dict] = []
    for lane, candidates in grouped.items():
        winner = max(
            candidates,
            key=lambda row: (
                _bounded_score(row.get("real_world_readiness_avg", 0.0)),
                float(row.get("browser_workflow_avg", 0.0)),
                float(row.get("strict_interface_avg", 0.0)),
                row.get("created_at", ""),
            ),
        )
        best.append(winner)
    return sorted(best, key=lambda item: item["lane"])


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys() if key != "role_breakdown"})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload.pop("role_breakdown", None)
            writer.writerow(payload)


def _markdown_report(report: dict) -> str:
    lines = [
        "# KnowledgeWorkArena History",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Total logged runs: `{report['total_runs']}`",
        f"- Snapshot count: `{report['snapshot_count']}`",
        "",
        "## Latest Canonical by Lane",
        "",
    ]
    for row in report["latest_canonical_by_lane"]:
        lines.append(
            f"- `{row['lane']}`: readiness `{row['real_world_readiness_avg']}`, browser `{row['browser_workflow_avg']}`, "
            f"strict `{row['strict_interface_avg']}`, recovered `{row['recovered_execution_avg']}`, output `{row['output_dir']}`"
        )
    lines.extend(["", "## Latest Exploratory by Lane", ""])
    for row in report["latest_exploratory_by_lane"]:
        lines.append(
            f"- `{row['lane']}`: readiness `{row['real_world_readiness_avg']}`, browser `{row['browser_workflow_avg']}`, "
            f"strict `{row['strict_interface_avg']}`, recovered `{row['recovered_execution_avg']}`, output `{row['output_dir']}`"
        )
    lines.extend(["", "## Best by Lane", ""])
    for row in report["best_by_lane"]:
        lines.append(
            f"- `{row['lane']}`: readiness `{row['real_world_readiness_avg']}`, browser `{row['browser_workflow_avg']}`, "
            f"strict `{row['strict_interface_avg']}`, recovered `{row['recovered_execution_avg']}`, output `{row['output_dir']}`"
        )
    return "\n".join(lines) + "\n"


def _bounded_score(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _infer_run_intent(path: Path, manifest: dict) -> str:
    manifest_intent = str(manifest.get("run_intent", "")).strip().lower()
    if manifest_intent in {"canonical", "exploratory"}:
        return manifest_intent
    lane = str(manifest.get("lane", "")).strip()
    if lane and path.name == lane:
        return "canonical"
    return "exploratory"


if __name__ == "__main__":
    main()
