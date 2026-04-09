from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_alpha_matrix.py"
SPEC = importlib.util.spec_from_file_location("run_alpha_matrix_script", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

_has_logged_run = MODULE._has_logged_run
_latest_matching_run = MODULE._latest_matching_run


def test_has_logged_run_matches_group_and_experiment() -> None:
    rows = [
        {
            "run_group_id": "group-a",
            "summary": {"experiment_id": "exp-1", "status": "completed"},
        }
    ]

    assert _has_logged_run(rows, run_group_id="group-a", experiment_id="exp-1") is True
    assert _has_logged_run(rows, run_group_id="group-b", experiment_id="exp-1") is False


def test_latest_matching_run_can_exclude_current_run_group() -> None:
    rows = [
        {
            "run_group_id": "group-a",
            "summary": {"experiment_id": "exp-1", "status": "completed", "success_rate": 0.4},
        },
        {
            "run_group_id": "group-b",
            "summary": {"experiment_id": "exp-1", "status": "completed", "success_rate": 0.8},
        },
    ]

    previous = _latest_matching_run(rows, experiment_id="exp-1", exclude_run_group_id="group-b")
    assert previous is not None
    assert previous["run_group_id"] == "group-a"
