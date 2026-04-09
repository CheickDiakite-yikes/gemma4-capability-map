from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_history_report.py"
SPEC = importlib.util.spec_from_file_location("build_history_report_script", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

_best_completed_by_experiment = MODULE._best_completed_by_experiment
_latest_completed_by_experiment = MODULE._latest_completed_by_experiment
_latest_complete_run_groups_by_matrix = MODULE._latest_complete_run_groups_by_matrix
_load_run_group_snapshots = MODULE._load_run_group_snapshots
_summarize_run_groups = MODULE._summarize_run_groups


def _row(run_group_id: str, experiment_id: str, success: float, latency: float, created_at: str) -> dict:
    return {
        "run_group_id": run_group_id,
        "matrix_name": "alpha",
        "created_at": created_at,
        "experiment": {"id": experiment_id, "notes": []},
        "summary": {
            "experiment_id": experiment_id,
            "track": "thinking",
            "pipeline": "monolith",
            "backend": "mlx",
            "reasoner": "google/gemma-4-E2B-it",
            "variants": False,
            "status": "completed",
            "success_rate": success,
            "avg_latency_ms": latency,
            "runs": 10.0,
            "failure_breakdown": {},
        },
        "output_dir": f"/tmp/{run_group_id}/{experiment_id}",
    }


def test_latest_completed_by_experiment_prefers_latest_row() -> None:
    rows = [
        _row("group-a", "exp-1", 0.4, 1000.0, "2026-04-09T00:00:00+00:00"),
        _row("group-b", "exp-1", 0.8, 900.0, "2026-04-09T01:00:00+00:00"),
    ]

    latest = _latest_completed_by_experiment(rows)

    assert len(latest) == 1
    assert latest[0]["run_group_id"] == "group-b"
    assert latest[0]["success_rate"] == 0.8


def test_best_completed_by_experiment_prefers_success_then_latency() -> None:
    rows = [
        _row("group-a", "exp-1", 0.8, 1200.0, "2026-04-09T00:00:00+00:00"),
        _row("group-b", "exp-1", 0.8, 900.0, "2026-04-09T01:00:00+00:00"),
        _row("group-c", "exp-1", 0.7, 800.0, "2026-04-09T02:00:00+00:00"),
    ]

    best = _best_completed_by_experiment(rows)

    assert len(best) == 1
    assert best[0]["run_group_id"] == "group-b"


def test_summarize_run_groups_aggregates_failures() -> None:
    rows = [
        _row("group-a", "exp-1", 1.0, 1000.0, "2026-04-09T00:00:00+00:00"),
        {
            "run_group_id": "group-a",
            "matrix_name": "alpha",
            "created_at": "2026-04-09T00:05:00+00:00",
            "experiment": {"id": "exp-2", "notes": []},
            "summary": {
                "experiment_id": "exp-2",
                "track": "tool_routing",
                "pipeline": "monolith",
                "backend": "mlx",
                "reasoner": "google/gemma-4-E2B-it",
                "variants": True,
                "status": "completed",
                "success_rate": 0.5,
                "avg_latency_ms": 2000.0,
                "runs": 8.0,
                "failure_breakdown": {"arg_mismatch": 2},
            },
            "output_dir": "/tmp/group-a/exp-2",
        },
    ]

    groups = _summarize_run_groups(rows)

    assert len(groups) == 1
    assert groups[0]["run_group_id"] == "group-a"
    assert groups[0]["avg_success_rate"] == 0.75
    assert groups[0]["failure_breakdown"] == {"arg_mismatch": 2}


def test_latest_complete_run_groups_by_matrix_filters_partial_snapshots() -> None:
    snapshots = [
        {
            "run_group_id": "group-a",
            "matrix_name": "alpha",
            "matrix_complete": False,
            "completed_experiments": 3,
            "expected_experiments": 6,
            "failed_experiments": 0,
            "avg_success_rate": 1.0,
        },
        {
            "run_group_id": "group-b",
            "matrix_name": "alpha",
            "matrix_complete": True,
            "completed_experiments": 6,
            "expected_experiments": 6,
            "failed_experiments": 0,
            "avg_success_rate": 0.95,
        },
        {
            "run_group_id": "group-c",
            "matrix_name": "drift",
            "matrix_complete": True,
            "completed_experiments": 4,
            "expected_experiments": 4,
            "failed_experiments": 0,
            "avg_success_rate": 0.91,
        },
    ]

    latest = _latest_complete_run_groups_by_matrix(snapshots)

    assert [row["run_group_id"] for row in latest] == ["group-b", "group-c"]


def test_latest_complete_run_groups_by_matrix_prefers_broadest_scope_before_recency() -> None:
    snapshots = [
        {
            "run_group_id": "group-a",
            "matrix_name": "alpha",
            "matrix_complete": True,
            "completed_experiments": 1,
            "expected_experiments": 1,
            "failed_experiments": 0,
            "avg_success_rate": 1.0,
        },
        {
            "run_group_id": "group-b",
            "matrix_name": "alpha",
            "matrix_complete": True,
            "completed_experiments": 6,
            "expected_experiments": 6,
            "failed_experiments": 0,
            "avg_success_rate": 0.95,
        },
        {
            "run_group_id": "group-c",
            "matrix_name": "alpha",
            "matrix_complete": True,
            "completed_experiments": 6,
            "expected_experiments": 6,
            "failed_experiments": 0,
            "avg_success_rate": 0.97,
        },
    ]

    latest = _latest_complete_run_groups_by_matrix(snapshots)

    assert len(latest) == 1
    assert latest[0]["run_group_id"] == "group-c"


def test_load_run_group_snapshots_derives_completion_from_experiment_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260409T090608Z_alpha_mlx_default"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        """
        {
          "run_group_id": "20260409T090608Z",
          "matrix_name": "alpha_mlx_default",
          "description": "clean baseline",
          "experiments": [{"id": "exp-1"}, {"id": "exp-2"}]
        }
        """.strip(),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        """
        {
          "experiments": [
            {"experiment_id": "exp-1", "status": "completed", "success_rate": 1.0},
            {"experiment_id": "exp-2", "status": "completed", "success_rate": 0.9}
          ],
          "failure_breakdown": {}
        }
        """.strip(),
        encoding="utf-8",
    )

    snapshots = _load_run_group_snapshots(tmp_path)

    assert len(snapshots) == 1
    assert snapshots[0]["matrix_complete"] is True
    assert snapshots[0]["completed_experiments"] == 2
    assert snapshots[0]["expected_experiments"] == 2
