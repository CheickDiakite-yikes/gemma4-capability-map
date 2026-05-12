from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from gemma4_capability_map.io import load_yaml
from gemma4_capability_map.knowledge_work.loader import load_episodes
from gemma4_capability_map.research_controls import ResearchControls
from gemma4_capability_map.runtime.workflows import DEFAULT_WORKFLOWS_PATH, load_packaged_workflows
from gemma4_capability_map.schemas import StrictModel


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_H1_SLICE_PATH = ROOT / "configs" / "knowledge_work_h1_slice.yaml"
DEFAULT_EPISODE_ROOT = ROOT / "data" / "knowledge_work"

H1RunSet = Literal["primary", "comparison", "ablation", "all"]


class H1LaneConfig(StrictModel):
    episode_ids: list[str]


class H1WorkflowFamily(StrictModel):
    workflow_id: str
    role_family: str
    purpose: str
    replayable_episode_id: str
    live_episode_id: str
    h1_stressors: list[str] = Field(default_factory=list)


class H1PacketConfig(StrictModel):
    lane: Literal["replayable_core", "live_web_stress"]
    purpose: str
    system_ids: list[str]
    episode_ids: list[str]
    failure_modes: list[str] = Field(default_factory=list)


class H1SliceConfig(StrictModel):
    name: str
    version: str
    description: str
    run_intent: Literal["canonical", "exploratory"] = "exploratory"
    update_latest: bool = False
    live_entrypoint: Literal["packaged_workflows_only"]
    primary_system_id: str
    baseline_system_ids: list[str] = Field(default_factory=list)
    ablation_bundle_system_id: str
    ablation_system_ids: list[str] = Field(default_factory=list)
    lanes: dict[Literal["replayable_core", "live_web_stress"], H1LaneConfig]
    workflow_families: list[H1WorkflowFamily]
    packets: dict[str, H1PacketConfig] = Field(default_factory=dict)
    saturation_breaker_metrics: list[str] = Field(default_factory=list)
    attribution_tags: list[str] = Field(default_factory=list)


def load_h1_slice(path: str | Path = DEFAULT_H1_SLICE_PATH) -> H1SliceConfig:
    payload = load_yaml(path) or {}
    return H1SliceConfig.model_validate(payload.get("h1_slice", {}))


def validate_h1_slice(
    config: H1SliceConfig,
    *,
    episode_root: str | Path = DEFAULT_EPISODE_ROOT,
    workflows_path: str | Path = DEFAULT_WORKFLOWS_PATH,
) -> list[str]:
    errors: list[str] = []
    workflow_index = {workflow.workflow_id: workflow for workflow in load_packaged_workflows(workflows_path)}
    episode_root_path = Path(episode_root)
    lane_episode_ids: dict[str, set[str]] = {}
    for lane, lane_config in config.lanes.items():
        episodes_path = episode_root_path / lane / "episodes.jsonl"
        episodes = load_episodes(episodes_path)
        available_ids = {episode.episode_id for episode in episodes}
        lane_episode_ids[lane] = set(lane_config.episode_ids)
        missing = [episode_id for episode_id in lane_config.episode_ids if episode_id not in available_ids]
        errors.extend(f"{lane}: unknown episode `{episode_id}`" for episode_id in missing)

    family_replayable_ids: set[str] = set()
    family_live_ids: set[str] = set()
    for family in config.workflow_families:
        workflow = workflow_index.get(family.workflow_id)
        if workflow is None:
            errors.append(f"unknown packaged workflow `{family.workflow_id}`")
            continue
        expected_replayable = workflow.lane_episode_map.get("replayable_core")
        expected_live = workflow.lane_episode_map.get("live_web_stress")
        if expected_replayable != family.replayable_episode_id:
            errors.append(
                f"{family.workflow_id}: replayable episode mismatch "
                f"({family.replayable_episode_id} != {expected_replayable})"
            )
        if expected_live != family.live_episode_id:
            errors.append(f"{family.workflow_id}: live episode mismatch ({family.live_episode_id} != {expected_live})")
        family_replayable_ids.add(family.replayable_episode_id)
        family_live_ids.add(family.live_episode_id)

    configured_replayable = lane_episode_ids.get("replayable_core", set())
    configured_live = lane_episode_ids.get("live_web_stress", set())
    extra_replayable = configured_replayable - family_replayable_ids
    extra_live = configured_live - family_live_ids
    missing_replayable = family_replayable_ids - configured_replayable
    missing_live = family_live_ids - configured_live
    errors.extend(f"replayable_core: episode not mapped to workflow `{episode_id}`" for episode_id in sorted(extra_replayable))
    errors.extend(f"live_web_stress: episode not mapped to workflow `{episode_id}`" for episode_id in sorted(extra_live))
    errors.extend(f"workflow episode missing from replayable_core `{episode_id}`" for episode_id in sorted(missing_replayable))
    errors.extend(f"workflow episode missing from live_web_stress `{episode_id}`" for episode_id in sorted(missing_live))
    for packet_id, packet in config.packets.items():
        lane_ids = lane_episode_ids.get(packet.lane, set())
        missing_packet_episodes = [episode_id for episode_id in packet.episode_ids if episode_id not in lane_ids]
        errors.extend(f"{packet_id}: packet episode not in {packet.lane} `{episode_id}`" for episode_id in missing_packet_episodes)
    return errors


def h1_packet_selection(config: H1SliceConfig, packet_id: str) -> H1PacketConfig:
    packet = config.packets.get(packet_id)
    if packet is None:
        available = ", ".join(sorted(config.packets)) or "none"
        raise ValueError(f"Unknown H1 packet `{packet_id}`. Available packets: {available}.")
    return packet


def h1_system_ids(config: H1SliceConfig, run_set: H1RunSet = "primary", explicit_system_ids: list[str] | None = None) -> list[str]:
    if explicit_system_ids:
        return _dedupe(explicit_system_ids)
    if run_set == "primary":
        return [config.primary_system_id]
    if run_set == "comparison":
        return _dedupe([config.primary_system_id, *config.baseline_system_ids])
    if run_set == "ablation":
        return _dedupe(config.ablation_system_ids)
    return _dedupe([config.primary_system_id, *config.baseline_system_ids, *config.ablation_system_ids])


def build_h1_run_specs(
    config: H1SliceConfig,
    registry: dict[str, Any],
    *,
    lanes: list[str] | None = None,
    run_set: H1RunSet = "primary",
    system_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected_lanes = lanes or list(config.lanes.keys())
    selected_system_ids = h1_system_ids(config, run_set=run_set, explicit_system_ids=system_ids)
    systems = registry.get("systems", {})
    specs: list[dict[str, Any]] = []
    for system_id in selected_system_ids:
        meta = systems.get(system_id)
        if meta is None:
            raise ValueError(f"Unknown system `{system_id}`.")
        for lane in selected_lanes:
            lane_config = config.lanes.get(lane)  # type: ignore[arg-type]
            if lane_config is None:
                raise ValueError(f"H1 slice does not define lane `{lane}`.")
            specs.append(
                {
                    "run_id": f"{system_id}__{lane}",
                    "system_id": system_id,
                    "lane": lane,
                    "episode_ids": list(lane_config.episode_ids),
                    "run_intent": config.run_intent,
                    "update_latest": config.update_latest,
                    "h1_slice": f"{config.name}:{config.version}",
                    **_system_run_args(system_id, meta),
                }
            )
    return specs


def _system_run_args(system_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    backend = str(meta.get("backend", "") or "")
    executor_mode = str(meta.get("executor_mode", "") or "")
    router = str(meta.get("router", "") or "")
    retriever = str(meta.get("retriever", "") or "")
    pipeline_name = "modular"
    router_backend = ""
    retriever_backend = ""
    if executor_mode == "local_specialists":
        pipeline_name = "modular"
        router_backend = "hf"
        retriever_backend = "hf"
    elif executor_mode == "local_reasoner":
        pipeline_name = "monolith"
        router_backend = "heuristic"
        retriever_backend = "heuristic"
        router = ""
        retriever = ""
    controls = ResearchControls.from_mapping(meta.get("research_controls"))
    return {
        "backend": backend,
        "pipeline_name": pipeline_name,
        "reasoner_backend": backend,
        "router_backend": router_backend,
        "retriever_backend": retriever_backend,
        "reasoner": str(meta.get("reasoner", "") or ""),
        "router": router,
        "retriever": retriever,
        "thinking": bool(meta.get("thinking", False)),
        "reasoner_max_new_tokens": int(meta.get("reasoner_max_new_tokens", 96) or 96),
        "request_timeout_seconds": float(meta.get("request_timeout_seconds", 600.0) or 600.0),
        "run_timeout_seconds": float(meta.get("run_timeout_seconds", 0.0) or 0.0),
        "research_controls": controls.manifest_payload(),
        "disable_controller_repair": controls.disable_controller_repair,
        "disable_controller_fallback": controls.disable_controller_fallback,
        "disable_visual_rescue": controls.disable_visual_rescue,
        "disable_intent_priority": controls.disable_intent_priority,
        "disable_argument_repair": controls.disable_argument_repair,
        "disable_deterministic_visual_follow_on": controls.disable_deterministic_visual_follow_on,
        "disable_tool_turn_directive": controls.disable_tool_turn_directive,
        "enable_visual_stale_selection_gate": controls.enable_visual_stale_selection_gate,
        "enable_visual_target_query_normalization": controls.enable_visual_target_query_normalization,
        "tool_prompt_contract_id": controls.tool_prompt_contract_id,
        "tool_catalog_profile_id": controls.tool_catalog_profile_id,
    }


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
