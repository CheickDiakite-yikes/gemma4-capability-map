from __future__ import annotations

import importlib.util
from pathlib import Path

from gemma4_capability_map.knowledge_work.h1 import build_h1_run_specs, h1_packet_selection, load_h1_slice, validate_h1_slice
from gemma4_capability_map.reporting.knowledge_work_board import load_model_registry


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_knowledge_work_h1_slice.py"
SPEC = importlib.util.spec_from_file_location("run_knowledge_work_h1_slice_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)

PACKET_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_knowledge_work_h1_ablation_packet.py"
PACKET_SPEC = importlib.util.spec_from_file_location("run_knowledge_work_h1_ablation_packet_script", PACKET_MODULE_PATH)
assert PACKET_SPEC and PACKET_SPEC.loader
PACKET_SCRIPT = importlib.util.module_from_spec(PACKET_SPEC)
PACKET_SPEC.loader.exec_module(PACKET_SCRIPT)
H1B_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "knowledge_work_h1b_slice.yaml"
H1C_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "knowledge_work_h1c_slice.yaml"


def test_h1_slice_config_maps_to_existing_packaged_workflows_and_episodes() -> None:
    config = load_h1_slice()

    errors = validate_h1_slice(config)

    assert errors == []
    assert config.live_entrypoint == "packaged_workflows_only"
    assert config.update_latest is False
    assert len(config.workflow_families) == 5
    assert config.lanes["replayable_core"].episode_ids == [
        "kwa_exec_visual_dashboard_brief",
        "kwa_exec_backlog_resume_hold_v5",
        "kwa_jobs_email_block_resume_hold_v5",
        "kwa_finance_diff_review_hold_v5",
        "kwa_finance_invoice_lock_direction_hold_v4",
    ]
    assert config.lanes["live_web_stress"].episode_ids == [
        "kwa_exec_live_visual_dashboard_brief",
        "kwa_exec_live_backlog_resume_hold_v5",
        "kwa_jobs_live_email_block_resume_hold_v5",
        "kwa_finance_live_diff_review_hold_v5",
        "kwa_finance_live_invoice_lock_direction_hold_v4",
    ]
    packet = h1_packet_selection(config, "visual_semantics_no_controller_repair")
    assert packet.lane == "replayable_core"
    assert packet.system_ids == [
        "hf_service_gemma4_specialists_cpu",
        "hf_service_gemma4_specialists_cpu_no_controller_repair",
        "hf_service_gemma4_specialists_cpu_no_deterministic_visual_follow_on",
    ]
    assert packet.episode_ids == [
        "kwa_exec_backlog_resume_hold_v5",
        "kwa_jobs_email_block_resume_hold_v5",
        "kwa_finance_invoice_lock_direction_hold_v4",
    ]


def test_h1b_slice_config_maps_to_existing_packaged_workflows_and_episodes() -> None:
    config = load_h1_slice(H1B_CONFIG_PATH)

    errors = validate_h1_slice(config)

    assert errors == []
    assert config.name == "knowledge_work_h1b_visual_policy_controller_dependence"
    assert config.live_entrypoint == "packaged_workflows_only"
    assert config.lanes["replayable_core"].episode_ids == [
        "kwa_exec_visual_dashboard_referent_hold_v3",
        "kwa_exec_latest_action_resume_hold_v4",
        "kwa_jobs_visual_constraint_override_hold_v2",
        "kwa_jobs_phone_patch_resume_hold_v4",
        "kwa_finance_visual_invoice_revision_hold_v2",
    ]
    assert config.lanes["live_web_stress"].episode_ids == [
        "kwa_exec_live_visual_dashboard_referent_hold_v3",
        "kwa_exec_live_latest_action_resume_hold_v4",
        "kwa_jobs_live_visual_constraint_override_hold_v2",
        "kwa_jobs_live_phone_patch_resume_hold_v4",
        "kwa_finance_live_visual_invoice_revision_hold_v2",
    ]
    packet = h1_packet_selection(config, "visual_policy_no_controller_repair")
    assert packet.system_ids == [
        "hf_service_gemma4_specialists_cpu",
        "hf_service_gemma4_specialists_cpu_no_controller_repair",
        "hf_service_gemma4_specialists_cpu_no_deterministic_visual_follow_on",
    ]
    assert packet.episode_ids == [
        "kwa_exec_visual_dashboard_referent_hold_v3",
        "kwa_jobs_visual_constraint_override_hold_v2",
        "kwa_finance_visual_invoice_revision_hold_v2",
    ]


def test_h1c_slice_config_maps_to_live_policy_packet() -> None:
    config = load_h1_slice(H1C_CONFIG_PATH)

    errors = validate_h1_slice(config)

    assert errors == []
    assert config.name == "knowledge_work_h1c_live_policy_controller_dependence"
    assert config.lanes["live_web_stress"].episode_ids == [
        "kwa_exec_live_visual_dashboard_brief",
        "kwa_finance_live_invoice_lock_direction_hold_v4",
        "kwa_jobs_live_email_block_resume_hold_v5",
        "kwa_jobs_live_phone_patch_resume_hold_v4",
        "kwa_finance_live_diff_review_hold_v5",
    ]
    packet = h1_packet_selection(config, "live_policy_controller_helpers")
    assert packet.lane == "live_web_stress"
    assert packet.system_ids == [
        "hf_service_gemma4_specialists_cpu",
        "hf_service_gemma4_specialists_cpu_no_controller_repair",
        "hf_service_gemma4_specialists_cpu_no_controller_fallback",
        "hf_service_gemma4_specialists_cpu_no_argument_repair",
    ]
    assert packet.episode_ids == [
        "kwa_jobs_live_email_block_resume_hold_v5",
        "kwa_finance_live_invoice_lock_direction_hold_v4",
        "kwa_jobs_live_phone_patch_resume_hold_v4",
    ]


def test_h1_primary_run_specs_default_to_mlx_gemma_reasoner_only() -> None:
    config = load_h1_slice()
    registry = load_model_registry()

    specs = build_h1_run_specs(config, registry, lanes=["replayable_core"])

    assert len(specs) == 1
    spec = specs[0]
    assert spec["system_id"] == "mlx_gemma4_e2b_reasoner_only"
    assert spec["lane"] == "replayable_core"
    assert spec["backend"] == "mlx"
    assert spec["router_backend"] == "heuristic"
    assert spec["retriever_backend"] == "heuristic"
    assert spec["episode_ids"] == config.lanes["replayable_core"].episode_ids
    assert spec["update_latest"] is False


def test_h1_ablation_specs_preserve_controller_flags() -> None:
    config = load_h1_slice()
    registry = load_model_registry()

    specs = build_h1_run_specs(config, registry, lanes=["replayable_core"], run_set="ablation")
    by_system = {spec["system_id"]: spec for spec in specs}

    assert by_system["hf_service_gemma4_specialists_cpu"]["research_controls"] == {}
    assert by_system["hf_service_gemma4_specialists_cpu_no_controller_repair"]["disable_controller_repair"] is True
    assert by_system["hf_service_gemma4_specialists_cpu_no_controller_fallback"]["disable_controller_fallback"] is True
    assert by_system["hf_service_gemma4_specialists_cpu_no_visual_rescue"]["disable_visual_rescue"] is True
    assert by_system["hf_service_gemma4_specialists_cpu_no_intent_priority"]["disable_intent_priority"] is True
    assert by_system["hf_service_gemma4_specialists_cpu_no_argument_repair"]["disable_argument_repair"] is True
    assert by_system["hf_service_gemma4_specialists_cpu_no_deterministic_visual_follow_on"]["disable_deterministic_visual_follow_on"] is True


def test_h1_arena_command_is_episode_filtered_and_exploratory(tmp_path: Path) -> None:
    config = load_h1_slice()
    registry = load_model_registry()
    spec = build_h1_run_specs(
        config,
        registry,
        lanes=["live_web_stress"],
        system_ids=["hf_gemma4_e2b_specialists_cpu_no_controller_fallback"],
    )[0]

    command = SCRIPT._arena_command(spec, tmp_path / spec["run_id"])

    assert "--no-update-latest" in command
    assert command[command.index("--run-intent") + 1] == "exploratory"
    assert "--disable-controller-fallback" in command
    assert command.count("--episode-id") == len(config.lanes["live_web_stress"].episode_ids)
    assert "kwa_jobs_live_email_block_resume_hold_v5" in command


def test_h1_ablation_packet_command_uses_shared_bundle_and_episode_filters(tmp_path: Path) -> None:
    config = load_h1_slice()

    command = PACKET_SCRIPT.h1_ablation_packet_command(
        run_group_id="h1_packet_test",
        lane="replayable_core",
        bundle_system_id=config.ablation_bundle_system_id,
        system_ids=config.ablation_system_ids,
        episode_ids=config.lanes["replayable_core"].episode_ids,
        output_root=tmp_path,
    )

    assert "run_knowledge_work_ablation_packet.py" in command[1]
    assert command[command.index("--bundle-system-id") + 1] == "hf_service_gemma4_specialists_cpu"
    assert command.count("--system-id") == len(config.ablation_system_ids)
    assert command.count("--episode-id") == len(config.lanes["replayable_core"].episode_ids)
    assert command[command.index("--run-intent") + 1] == "exploratory"


def test_h1_ablation_packet_command_can_use_named_visual_semantics_packet(tmp_path: Path) -> None:
    config = load_h1_slice()
    packet = h1_packet_selection(config, "visual_semantics_no_controller_repair")

    command = PACKET_SCRIPT.h1_ablation_packet_command(
        run_group_id="h1_visual_semantics_packet_test",
        lane=packet.lane,
        bundle_system_id=config.ablation_bundle_system_id,
        system_ids=packet.system_ids,
        episode_ids=packet.episode_ids,
        output_root=tmp_path,
    )

    assert command[command.index("--bundle-system-id") + 1] == "hf_service_gemma4_specialists_cpu"
    assert command.count("--system-id") == 3
    assert command.count("--episode-id") == 3
    assert "kwa_exec_visual_dashboard_brief" not in command
    assert "hf_service_gemma4_specialists_cpu_no_controller_repair" in command
