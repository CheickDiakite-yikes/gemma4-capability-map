from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_tool_probe_replay_packets.py"
SPEC = importlib.util.spec_from_file_location("compare_tool_probe_replay_packets_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_compare_tool_probe_replay_packets_writes_case_and_family_deltas(tmp_path: Path) -> None:
    baseline = _write_packet(
        tmp_path / "baseline",
        system_id="contracted",
        rows=[
            {"case_id": "cli_case", "family": "cli_canonicalization", "replay_exact_match": True, "replay_failure_mode": "exact", "replay_actual_call_count": 1},
            {"case_id": "visual_case", "family": "visual_argument_copying", "replay_exact_match": False, "replay_failure_mode": "executable_paraphrase", "replay_actual_call_count": 1},
        ],
        exact_rate=0.5,
    )
    candidate = _write_packet(
        tmp_path / "candidate",
        system_id="no_directive",
        rows=[
            {"case_id": "cli_case", "family": "cli_canonicalization", "replay_exact_match": False, "replay_failure_mode": "argument_mismatch", "replay_actual_call_count": 1},
            {"case_id": "visual_case", "family": "visual_argument_copying", "replay_exact_match": False, "replay_failure_mode": "no_tool_call", "replay_actual_call_count": 0},
        ],
        exact_rate=0.0,
    )

    comparison = SCRIPT.compare_tool_probe_replay_packets(
        baseline_packet=baseline,
        candidate_packet=candidate,
        output_dir=tmp_path / "comparison",
    )

    assert comparison["summary"]["shared_case_count"] == 2
    assert comparison["summary"]["delta_exact_match_rate"] == -0.5
    assert comparison["case_deltas"][0]["delta_exact_match"] == -1
    visual = next(row for row in comparison["case_deltas"] if row["case_id"] == "visual_case")
    assert visual["delta_actual_call_count"] == -1
    family = next(row for row in comparison["family_deltas"] if row["family"] == "cli_canonicalization")
    assert family["delta_exact_rate"] == -1.0
    assert (tmp_path / "comparison" / "replay_comparison.json").exists()
    assert (tmp_path / "comparison" / "replay_case_deltas.csv").exists()
    assert (tmp_path / "comparison" / "replay_family_deltas.csv").exists()


def _write_packet(path: Path, *, system_id: str, rows: list[dict[str, object]], exact_rate: float) -> Path:
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps({"replay_system_id": system_id, "replay_exact_match_rate": exact_rate}, indent=2) + "\n",
        encoding="utf-8",
    )
    (path / "replay_results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return path
