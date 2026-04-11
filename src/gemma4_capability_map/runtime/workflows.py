from __future__ import annotations

from pathlib import Path

from pydantic import Field

from gemma4_capability_map.io import load_yaml
from gemma4_capability_map.schemas import StrictModel


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORKFLOWS_PATH = ROOT / "configs" / "packaged_workflows.yaml"


class PackagedWorkflow(StrictModel):
    workflow_id: str
    title: str
    subtitle: str = ""
    description: str
    role_family: str
    category: str
    preview_asset: str | None = None
    recommended_system_id: str
    default_lane: str = "replayable_core"
    supports_approval: bool = False
    tags: list[str] = Field(default_factory=list)
    lane_episode_map: dict[str, str] = Field(default_factory=dict)

    def episode_id_for_lane(self, lane: str | None = None) -> str:
        selected_lane = lane or self.default_lane
        episode_id = self.lane_episode_map.get(selected_lane)
        if not episode_id:
            raise KeyError(f"Workflow `{self.workflow_id}` does not define an episode for lane `{selected_lane}`.")
        return episode_id


def load_packaged_workflows(path: str | Path = DEFAULT_WORKFLOWS_PATH) -> list[PackagedWorkflow]:
    payload = load_yaml(path) or {}
    workflows = payload.get("workflows", {})
    return [
        PackagedWorkflow.model_validate({"workflow_id": workflow_id, **config})
        for workflow_id, config in workflows.items()
    ]
