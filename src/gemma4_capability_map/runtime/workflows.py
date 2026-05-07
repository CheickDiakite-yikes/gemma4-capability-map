from __future__ import annotations

from pathlib import Path

from pydantic import Field

from gemma4_capability_map.io import load_yaml
from gemma4_capability_map.knowledge_work.loader import load_episodes
from gemma4_capability_map.schemas import StrictModel


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORKFLOWS_PATH = ROOT / "configs" / "packaged_workflows.yaml"
DEFAULT_EPISODE_ROOT = ROOT / "data" / "knowledge_work"


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


def validate_packaged_workflows(
    path: str | Path = DEFAULT_WORKFLOWS_PATH,
    *,
    episode_root: str | Path = DEFAULT_EPISODE_ROOT,
    required_lanes: tuple[str, ...] = ("replayable_core", "live_web_stress"),
) -> list[str]:
    workflows = load_packaged_workflows(path)
    root = Path(episode_root)
    episodes_by_lane: dict[str, dict[str, object]] = {}
    errors: list[str] = []
    for lane in required_lanes:
        episode_path = root / lane / "episodes.jsonl"
        if not episode_path.exists():
            errors.append(f"Lane `{lane}` has no episode file at `{episode_path}`.")
            episodes_by_lane[lane] = {}
            continue
        episodes_by_lane[lane] = {episode.episode_id: episode for episode in load_episodes(episode_path)}

    for workflow in workflows:
        if workflow.default_lane not in workflow.lane_episode_map:
            errors.append(f"Workflow `{workflow.workflow_id}` default lane `{workflow.default_lane}` is missing from lane_episode_map.")
        for lane in required_lanes:
            episode_id = workflow.lane_episode_map.get(lane)
            if not episode_id:
                errors.append(f"Workflow `{workflow.workflow_id}` is missing lane `{lane}`.")
                continue
            episode = episodes_by_lane.get(lane, {}).get(episode_id)
            if episode is None:
                errors.append(f"Workflow `{workflow.workflow_id}` lane `{lane}` points at missing episode `{episode_id}`.")
                continue
            if getattr(getattr(episode, "lane", None), "value", None) != lane:
                errors.append(f"Episode `{episode_id}` is not declared in lane `{lane}`.")
            episode_role = getattr(getattr(episode, "role_family", None), "value", None)
            if episode_role != workflow.role_family:
                errors.append(
                    f"Workflow `{workflow.workflow_id}` role `{workflow.role_family}` does not match episode `{episode_id}` role `{episode_role}`."
                )
    return errors
