from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


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
