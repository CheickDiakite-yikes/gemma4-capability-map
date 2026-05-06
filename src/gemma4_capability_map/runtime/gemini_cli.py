from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_GEMINI_CLI_BIN = "gemini"


@dataclass(frozen=True)
class GeminiCliAvailability:
    available: bool
    binary: str = ""
    reason: str = ""

    def as_payload(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "binary": self.binary,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GeminiCliBaselineResult:
    workflow_id: str
    lane: str
    episode_id: str
    dry_run: bool
    availability: GeminiCliAvailability
    command: list[str]
    prompt: str
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    output_path: str = ""

    def as_payload(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "lane": self.lane,
            "episode_id": self.episode_id,
            "dry_run": self.dry_run,
            "availability": self.availability.as_payload(),
            "command": self.command,
            "prompt": self.prompt,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "output_path": self.output_path,
        }


def resolve_gemini_cli(binary: str | None = None) -> GeminiCliAvailability:
    requested = binary or os.environ.get("GEMINI_CLI_BIN") or DEFAULT_GEMINI_CLI_BIN
    resolved = shutil.which(requested)
    if not resolved:
        return GeminiCliAvailability(available=False, binary=requested, reason="Gemini CLI binary was not found on PATH.")
    return GeminiCliAvailability(available=True, binary=resolved)


def build_gemini_cli_prompt(workflow: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are running as an external baseline for Moonie, not as Moonie's controller.",
            "Follow these constraints exactly:",
            "- Treat this as a dry-run benchmark workflow.",
            "- Do not perform public side effects, submissions, sends, purchases, or account changes.",
            "- If a step would require external action, describe the sandbox-only action and stop for approval.",
            "- Return concise JSON with keys: plan, expected_artifacts, approval_gates, controller_help_needed, risks.",
            "",
            "Workflow:",
            json.dumps(workflow, indent=2, ensure_ascii=False),
        ]
    )


def run_gemini_cli_baseline(
    *,
    workflow: dict[str, Any],
    output_dir: str | Path,
    binary: str | None = None,
    dry_run: bool = True,
    timeout_s: float = 120.0,
) -> GeminiCliBaselineResult:
    availability = resolve_gemini_cli(binary)
    prompt = build_gemini_cli_prompt(workflow)
    command = [availability.binary or (binary or DEFAULT_GEMINI_CLI_BIN), "-p", prompt]
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "gemini_cli_baseline.json"

    if dry_run or not availability.available:
        result = GeminiCliBaselineResult(
            workflow_id=str(workflow.get("workflow_id", "")),
            lane=str(workflow.get("lane", "")),
            episode_id=str(workflow.get("episode_id", "")),
            dry_run=True,
            availability=availability,
            command=command,
            prompt=prompt,
            output_path=str(output_path.resolve()),
        )
        output_path.write_text(json.dumps(result.as_payload(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return result

    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_s)  # noqa: S603
    result = GeminiCliBaselineResult(
        workflow_id=str(workflow.get("workflow_id", "")),
        lane=str(workflow.get("lane", "")),
        episode_id=str(workflow.get("episode_id", "")),
        dry_run=False,
        availability=availability,
        command=command,
        prompt=prompt,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        output_path=str(output_path.resolve()),
    )
    output_path.write_text(json.dumps(result.as_payload(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
