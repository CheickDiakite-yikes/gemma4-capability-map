from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from gemma4_capability_map.knowledge_work.schemas import BenchmarkLane


DEFAULT_SANDBOX_POLICY_ID = "packaged_workflow_ephemeral_v1"


class SandboxViolation(ValueError):
    pass


@dataclass(frozen=True)
class PreparedSandbox:
    mode: Literal["ephemeral_copy", "disabled"]
    root: Path
    source: str
    policy_id: str
    manifest_path: Path
    input_dir: Path
    output_dir: Path
    artifact_dir: Path

    def session_update(self) -> dict[str, str]:
        return {
            "sandbox_mode": self.mode,
            "sandbox_root": str(self.root.resolve()),
            "sandbox_source": self.source,
            "sandbox_policy_id": self.policy_id,
            "sandbox_manifest_path": str(self.manifest_path.resolve()),
        }


@dataclass(frozen=True)
class SandboxPolicyBlock:
    block_id: str
    policy_id: str
    reason: str
    severity: Literal["info", "warning", "error"]
    stage_id: str = ""
    action: str = ""
    target: str = ""
    submission_gate: str = ""
    sandbox_endpoint: str = ""

    def as_payload(self) -> dict[str, str]:
        return {
            "block_id": self.block_id,
            "policy_id": self.policy_id,
            "reason": self.reason,
            "severity": self.severity,
            "stage_id": self.stage_id,
            "action": self.action,
            "target": self.target,
            "submission_gate": self.submission_gate,
            "sandbox_endpoint": self.sandbox_endpoint,
        }


def prepare_packaged_workflow_sandbox(
    *,
    session_id: str,
    session_dir: Path,
    workflow_id: str,
    workflow_payload: dict[str, Any],
    episode_id: str,
    episode_payload: dict[str, Any],
    episode_source_path: Path,
    mode: Literal["ephemeral_copy", "disabled"] = "ephemeral_copy",
    policy_id: str = DEFAULT_SANDBOX_POLICY_ID,
) -> PreparedSandbox:
    if mode == "disabled":
        root = session_dir
        input_dir = session_dir
        output_dir = session_dir
        artifact_dir = session_dir / "artifacts"
    else:
        root = session_dir / "sandbox"
        input_dir = root / "input"
        output_dir = root / "output"
        artifact_dir = output_dir / "artifacts"

    for path in (input_dir, output_dir, artifact_dir):
        assert_path_inside(path, root)
        path.mkdir(parents=True, exist_ok=True)

    source = f"{episode_source_path.resolve()}#{episode_id}"
    workflow_copy_path = input_dir / "workflow.json"
    episode_copy_path = input_dir / "episode.json"
    manifest_path = root / "sandbox_manifest.json"

    workflow_copy_path.write_text(json.dumps(workflow_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    episode_copy_path.write_text(json.dumps(episode_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "session_id": session_id,
        "mode": mode,
        "policy_id": policy_id,
        "entrypoint": "packaged_workflow",
        "workflow_id": workflow_id,
        "episode_id": episode_id,
        "source": source,
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "allowed_write_roots": [str(output_dir.resolve())],
        "live_web_policy": {
            "dry_run_required": True,
            "public_side_effects_allowed": False,
            "packaged_workflows_only": True,
        },
        "copied_inputs": [
            str(workflow_copy_path.resolve()),
            str(episode_copy_path.resolve()),
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return PreparedSandbox(
        mode=mode,
        root=root.resolve(),
        source=source,
        policy_id=policy_id,
        manifest_path=manifest_path.resolve(),
        input_dir=input_dir.resolve(),
        output_dir=output_dir.resolve(),
        artifact_dir=artifact_dir.resolve(),
    )


def assert_path_inside(path: str | Path, root: str | Path) -> Path:
    resolved_path = Path(path).expanduser().resolve()
    resolved_root = Path(root).expanduser().resolve()
    if resolved_path == resolved_root or resolved_root in resolved_path.parents:
        return resolved_path
    raise SandboxViolation(f"Path `{resolved_path}` escapes sandbox root `{resolved_root}`.")


def sandbox_policy_blocks_for_trace(
    *,
    trace: Any,
    lane: Any,
    policy_id: str = DEFAULT_SANDBOX_POLICY_ID,
) -> list[SandboxPolicyBlock]:
    lane_value = lane.value if isinstance(lane, BenchmarkLane) else str(lane)
    if lane_value != BenchmarkLane.LIVE_WEB_STRESS.value:
        return []

    blocks: list[SandboxPolicyBlock] = []
    for index, action in enumerate(getattr(trace, "browser_actions", []), start=1):
        gate = str(getattr(action, "submission_gate", "") or "")
        gate_result = str(getattr(action, "gate_result", "") or "")
        status = str(getattr(action, "status", "") or "")
        sandbox_endpoint = str(getattr(action, "sandbox_endpoint", "") or "")
        if status != "dry_run":
            blocks.append(
                SandboxPolicyBlock(
                    block_id=f"{trace.episode_id}:sandbox_policy:{index}:dry_run_required",
                    policy_id=policy_id,
                    severity="error",
                    reason="Live-web action was not marked dry_run.",
                    stage_id=str(getattr(action, "stage_id", "") or ""),
                    action=str(getattr(action, "action", "") or ""),
                    target=str(getattr(action, "target", "") or ""),
                    submission_gate=gate,
                    sandbox_endpoint=sandbox_endpoint,
                )
            )
            continue
        if gate in {"sandbox_only", "approval_required", "blocked"} or gate_result in {"approval_required", "blocked"}:
            reason = "Live-web side effect held inside sandbox."
            severity: Literal["info", "warning", "error"] = "info"
            if gate == "approval_required" or gate_result == "approval_required":
                reason = "Live-web action requires human approval before any external side effect."
                severity = "warning"
            elif gate == "blocked" or gate_result == "blocked":
                reason = "Live-web action is blocked by policy."
                severity = "warning"
            blocks.append(
                SandboxPolicyBlock(
                    block_id=f"{trace.episode_id}:sandbox_policy:{index}:{gate or gate_result or 'dry_run'}",
                    policy_id=policy_id,
                    severity=severity,
                    reason=reason,
                    stage_id=str(getattr(action, "stage_id", "") or ""),
                    action=str(getattr(action, "action", "") or ""),
                    target=str(getattr(action, "target", "") or ""),
                    submission_gate=gate,
                    sandbox_endpoint=sandbox_endpoint,
                )
            )
    return blocks
