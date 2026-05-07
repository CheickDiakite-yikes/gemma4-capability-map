from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from gemma4_capability_map.knowledge_work.h1 import load_h1_slice


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_gemini_h1_baseline_packet.py"
SPEC = importlib.util.spec_from_file_location("run_gemini_h1_baseline_packet_script", MODULE_PATH)
assert SPEC and SPEC.loader
gemini_h1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gemini_h1)


def test_h1_gemini_workflows_preserve_packet_order_and_attribution() -> None:
    config = load_h1_slice(gemini_h1.DEFAULT_H1H_CONFIG_PATH)

    workflows = gemini_h1.build_h1_gemini_workflows(config=config, packet_id="mlx_full_tool_contract_breaker")

    assert len(workflows) == 10
    assert workflows[0]["workflow_id"] == "executive_stale_brief_packet"
    assert workflows[0]["episode_id"] == "kwa_exec_live_backlog_resume_hold_v5"
    assert workflows[0]["packet_id"] == "mlx_full_tool_contract_breaker"
    assert workflows[0]["live_entrypoint"] == "packaged_workflows_only"
    assert workflows[0]["moonies_evaluation_contract"]["external_baseline_only"] is True
    assert "approval_safe_stop" in workflows[0]["h1_stressors"]


def test_run_gemini_h1_baseline_packet_writes_dry_run_outputs(tmp_path: Path) -> None:
    summary = gemini_h1.run_gemini_h1_baseline_packet(
        config_path=gemini_h1.DEFAULT_H1H_CONFIG_PATH,
        packet_id="mlx_full_tool_contract_breaker",
        output_root=tmp_path,
        run_group_id="test_gemini_h1h_packet",
        binary="definitely-missing-gemini-cli",
        execute=False,
        timeout_s=5.0,
    )

    packet_dir = Path(summary["packet_dir"])
    assert summary["workflow_count"] == 10
    assert summary["dry_run_count"] == 10
    assert summary["unavailable_count"] == 10
    assert (packet_dir / "manifest.json").exists()
    assert (packet_dir / "results.json").exists()
    first_output = packet_dir / "workflows" / "executive_stale_brief_packet" / "gemini_cli_baseline.json"
    assert first_output.exists()
    payload = json.loads(first_output.read_text(encoding="utf-8"))
    assert payload["workflow_id"] == "executive_stale_brief_packet"
    assert payload["dry_run"] is True
    assert "external baseline for Moonie" in payload["prompt"]
