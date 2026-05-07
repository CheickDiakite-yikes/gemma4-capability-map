from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_tool_probe_replay_live_packets.py"
SPEC = importlib.util.spec_from_file_location("compare_tool_probe_replay_live_packets_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_compare_tool_probe_replay_live_packets_writes_case_delta(tmp_path: Path) -> None:
    baseline = _write_packet(
        tmp_path / "baseline",
        system_id="contracted",
        exact=True,
        failure_mode="exact",
        actual_call_count=2,
    )
    candidate = _write_packet(
        tmp_path / "candidate",
        system_id="no_directive",
        exact=False,
        failure_mode="no_tool_call",
        actual_call_count=0,
    )

    comparison = SCRIPT.compare_tool_probe_replay_live_packets(
        baseline_packet=baseline,
        candidate_packet=candidate,
        output_dir=tmp_path / "comparison",
    )

    assert comparison["summary"]["baseline_exact_rate"] == 1.0
    assert comparison["summary"]["candidate_exact_rate"] == 0.0
    assert comparison["summary"]["delta_exact_rate"] == -1.0
    assert comparison["case_deltas"][0]["delta_actual_call_count"] == -2
    assert comparison["case_deltas"][0]["candidate_replay_failure_mode"] == "no_tool_call"
    assert (tmp_path / "comparison" / "live_replay_comparison.json").exists()
    assert (tmp_path / "comparison" / "live_replay_case_deltas.csv").exists()
    assert (tmp_path / "comparison" / "live_replay_summary.md").exists()


def _write_packet(path: Path, *, system_id: str, exact: bool, failure_mode: str, actual_call_count: int) -> Path:
    path.mkdir(parents=True)
    path.joinpath("manifest.json").write_text(
        json.dumps({"packet_run_id": path.name, "system_id": system_id}) + "\n",
        encoding="utf-8",
    )
    path.joinpath("summary.json").write_text(
        json.dumps({"exact_rate": 1.0 if exact else 0.0}) + "\n",
        encoding="utf-8",
    )
    path.joinpath("live_replay_results.json").write_text(
        json.dumps(
            [
                {
                    "case_id": "parallel_audit_array_literal",
                    "family": "parallel_tool_calling",
                    "source_failure_mode": "no_tool_call",
                    "replay_failure_mode": failure_mode,
                    "replay_exact_match": exact,
                    "replay_actual_call_count": actual_call_count,
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path
