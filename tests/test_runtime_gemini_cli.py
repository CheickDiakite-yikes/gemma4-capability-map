from __future__ import annotations

from pathlib import Path

from gemma4_capability_map.runtime.gemini_cli import build_gemini_cli_prompt, resolve_gemini_cli, run_gemini_cli_baseline


WORKFLOW = {
    "workflow_id": "executive_visual_dashboard_review",
    "title": "Dashboard Visual Review",
    "description": "Inspect a dashboard and produce an action brief.",
    "lane": "replayable_core",
    "episode_id": "kwa_exec_visual_dashboard_brief",
}


def test_gemini_cli_prompt_keeps_moonie_as_controller() -> None:
    prompt = build_gemini_cli_prompt(WORKFLOW)

    assert "external baseline for Moonie" in prompt
    assert "Do not perform public side effects" in prompt
    assert "controller_help_needed" in prompt
    assert "executive_visual_dashboard_review" in prompt


def test_gemini_cli_dry_run_writes_baseline_packet(tmp_path: Path) -> None:
    result = run_gemini_cli_baseline(
        workflow=WORKFLOW,
        output_dir=tmp_path,
        binary="definitely-missing-gemini-cli",
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.availability.available is False
    assert result.command[-2] == "-p"
    assert Path(result.output_path).exists()


def test_gemini_cli_execute_uses_configured_binary(tmp_path: Path) -> None:
    fake_gemini = tmp_path / "gemini"
    fake_gemini.write_text("#!/bin/sh\nprintf 'fake gemini baseline\\n'\n", encoding="utf-8")
    fake_gemini.chmod(0o755)

    result = run_gemini_cli_baseline(
        workflow=WORKFLOW,
        output_dir=tmp_path / "out",
        binary=str(fake_gemini),
        dry_run=False,
        timeout_s=5.0,
    )

    assert result.dry_run is False
    assert result.availability.available is True
    assert result.returncode == 0
    assert result.stdout == "fake gemini baseline\n"
