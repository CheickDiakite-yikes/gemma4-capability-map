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
H1D_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "knowledge_work_h1d_slice.yaml"
H1E_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "knowledge_work_h1e_slice.yaml"
H1F_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "knowledge_work_h1f_slice.yaml"
H1G_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "knowledge_work_h1g_slice.yaml"
H1H_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "knowledge_work_h1h_slice.yaml"
H1I_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "knowledge_work_h1i_slice.yaml"
H1J_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "knowledge_work_h1j_slice.yaml"
H1K_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "knowledge_work_h1k_slice.yaml"
H1L_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "knowledge_work_h1l_slice.yaml"
H1M_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "knowledge_work_h1m_slice.yaml"


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


def test_h1d_slice_config_maps_to_mlx_monolith_packet() -> None:
    config = load_h1_slice(H1D_CONFIG_PATH)

    errors = validate_h1_slice(config)

    assert errors == []
    assert config.name == "knowledge_work_h1d_mlx_monolith_controller_stress"
    packet = h1_packet_selection(config, "mlx_monolith_controller_helpers")
    assert packet.lane == "live_web_stress"
    assert packet.system_ids == [
        "mlx_gemma4_e2b_reasoner_only",
        "mlx_gemma4_e2b_reasoner_only_no_controller_repair",
        "mlx_gemma4_e2b_reasoner_only_no_controller_fallback",
        "mlx_gemma4_e2b_reasoner_only_no_argument_repair",
    ]
    assert packet.episode_ids == config.lanes["live_web_stress"].episode_ids
    assert "visual_stepwise_control" in packet.failure_modes
    assert "api_canonicalization" in packet.failure_modes


def test_h1e_slice_config_maps_to_full_live_packaged_workflow_packet() -> None:
    config = load_h1_slice(H1E_CONFIG_PATH)

    errors = validate_h1_slice(config)

    assert errors == []
    assert config.name == "knowledge_work_h1e_mlx_full_live_packaged_workflows"
    assert len(config.workflow_families) == 10
    assert config.lanes["live_web_stress"].episode_ids == [
        "kwa_exec_live_backlog_resume_hold_v5",
        "kwa_exec_live_visual_dashboard_brief",
        "kwa_jobs_live_email_block_resume_hold_v5",
        "kwa_finance_live_diff_review_hold_v5",
        "kwa_finance_live_invoice_lock_direction_hold_v4",
        "kwa_exec_live_visual_dashboard_referent_hold_v3",
        "kwa_exec_live_latest_action_resume_hold_v4",
        "kwa_jobs_live_visual_constraint_override_hold_v2",
        "kwa_jobs_live_phone_patch_resume_hold_v4",
        "kwa_finance_live_visual_invoice_revision_hold_v2",
    ]
    packet = h1_packet_selection(config, "mlx_monolith_full_live_workflows")
    assert packet.lane == "live_web_stress"
    assert packet.system_ids == [
        "mlx_gemma4_e2b_reasoner_only",
        "mlx_gemma4_e2b_reasoner_only_no_controller_repair",
        "mlx_gemma4_e2b_reasoner_only_no_controller_fallback",
        "mlx_gemma4_e2b_reasoner_only_no_argument_repair",
    ]
    assert packet.episode_ids == config.lanes["live_web_stress"].episode_ids
    assert "saturation_breaker" in config.attribution_tags


def test_h1f_slice_config_maps_to_tool_contract_ablation_packet() -> None:
    config = load_h1_slice(H1F_CONFIG_PATH)

    errors = validate_h1_slice(config)

    assert errors == []
    assert config.name == "knowledge_work_h1f_mlx_tool_contract_ablation"
    packet = h1_packet_selection(config, "mlx_tool_contract_breaker")
    assert packet.lane == "live_web_stress"
    assert packet.system_ids == [
        "mlx_gemma4_e2b_reasoner_only",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_repair",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_fallback",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_argument_repair",
    ]
    assert packet.episode_ids == config.lanes["live_web_stress"].episode_ids
    assert "prompt_contract_ablation" in config.attribution_tags


def test_h1g_slice_config_maps_to_remaining_helper_packet() -> None:
    config = load_h1_slice(H1G_CONFIG_PATH)

    errors = validate_h1_slice(config)

    assert errors == []
    assert config.name == "knowledge_work_h1g_mlx_remaining_helper_ablation"
    packet = h1_packet_selection(config, "mlx_remaining_helpers")
    assert packet.lane == "live_web_stress"
    assert packet.system_ids == [
        "mlx_gemma4_e2b_reasoner_only",
        "mlx_gemma4_e2b_reasoner_only_no_visual_rescue",
        "mlx_gemma4_e2b_reasoner_only_no_intent_priority",
        "mlx_gemma4_e2b_reasoner_only_no_deterministic_visual_follow_on",
    ]
    assert packet.episode_ids == config.lanes["live_web_stress"].episode_ids
    assert "second_wave_helper_ablation" in config.attribution_tags


def test_h1h_slice_config_maps_to_full_tool_contract_packet() -> None:
    config = load_h1_slice(H1H_CONFIG_PATH)

    errors = validate_h1_slice(config)

    assert errors == []
    assert config.name == "knowledge_work_h1h_mlx_full_tool_contract_ablation"
    assert len(config.workflow_families) == 10
    packet = h1_packet_selection(config, "mlx_full_tool_contract_breaker")
    assert packet.lane == "live_web_stress"
    assert packet.system_ids == [
        "mlx_gemma4_e2b_reasoner_only",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_repair",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_fallback",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_argument_repair",
    ]
    assert packet.episode_ids == config.lanes["live_web_stress"].episode_ids
    assert "full_live_surface" in config.attribution_tags


def test_h1i_slice_config_maps_to_worst_h1h_workflow_families() -> None:
    config = load_h1_slice(H1I_CONFIG_PATH)

    errors = validate_h1_slice(config)

    assert errors == []
    assert config.name == "knowledge_work_h1i_mlx_worst_family_tool_contract"
    assert len(config.workflow_families) == 4
    assert [family.workflow_id for family in config.workflow_families] == [
        "executive_latest_action_resume",
        "jobs_phone_patch_resume",
        "jobs_visual_form_hold",
        "executive_stale_brief_packet",
    ]
    packet = h1_packet_selection(config, "mlx_worst_family_tool_contract_breaker")
    assert packet.lane == "live_web_stress"
    assert packet.system_ids == [
        "mlx_gemma4_e2b_reasoner_only",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_repair",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_fallback",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_argument_repair",
    ]
    assert packet.episode_ids == [
        "kwa_exec_live_latest_action_resume_hold_v4",
        "kwa_jobs_live_phone_patch_resume_hold_v4",
        "kwa_jobs_live_email_block_resume_hold_v5",
        "kwa_exec_live_backlog_resume_hold_v5",
    ]
    assert "worst_workflow_families" in config.attribution_tags


def test_h1i_slice_config_maps_to_prompt_contract_candidate_packet() -> None:
    config = load_h1_slice(H1I_CONFIG_PATH)

    errors = validate_h1_slice(config)

    assert errors == []
    packet = h1_packet_selection(config, "mlx_prompt_contract_candidates")
    assert packet.lane == "live_web_stress"
    assert packet.system_ids == [
        "mlx_gemma4_e2b_reasoner_only",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_anchor",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_literal_guard",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_tool_required",
    ]
    assert packet.episode_ids == [
        "kwa_exec_live_latest_action_resume_hold_v4",
        "kwa_jobs_live_phone_patch_resume_hold_v4",
        "kwa_jobs_live_email_block_resume_hold_v5",
        "kwa_exec_live_backlog_resume_hold_v5",
    ]
    assert packet.failure_modes == [
        "no_directive_argument_drift",
        "no_tool_call",
        "cli_canonicalization",
        "api_canonicalization",
        "visual_stepwise_control",
        "parallel_tool_protocol",
    ]


def test_h1j_slice_config_maps_to_probe_derived_live_packet() -> None:
    config = load_h1_slice(H1J_CONFIG_PATH)

    errors = validate_h1_slice(config)

    assert errors == []
    assert config.name == "knowledge_work_h1j_probe_derived_tool_contract_live"
    assert len(config.workflow_families) == 6
    assert [family.workflow_id for family in config.workflow_families] == [
        "executive_visual_dashboard_review",
        "executive_visual_referent_review",
        "jobs_visual_constraint_override",
        "jobs_phone_patch_resume",
        "finance_visual_invoice_review",
        "finance_billing_patch_hold",
    ]
    packet = h1_packet_selection(config, "mlx_probe_derived_tool_contract_candidates")
    assert packet.lane == "live_web_stress"
    assert packet.system_ids == [
        "mlx_gemma4_e2b_reasoner_only",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_anchor",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_literal_guard",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_tool_required",
    ]
    assert packet.episode_ids == [
        "kwa_exec_live_visual_dashboard_brief",
        "kwa_exec_live_visual_dashboard_referent_hold_v3",
        "kwa_jobs_live_visual_constraint_override_hold_v2",
        "kwa_jobs_live_phone_patch_resume_hold_v4",
        "kwa_finance_live_invoice_lock_direction_hold_v4",
        "kwa_finance_live_diff_review_hold_v5",
    ]
    assert "no_tool_call" in packet.failure_modes
    assert "argument_mismatch" in packet.failure_modes
    assert "parallel_tool_protocol_deferred" in packet.failure_modes


def test_h1k_slice_config_maps_to_parallel_audit_live_packet() -> None:
    config = load_h1_slice(H1K_CONFIG_PATH)

    errors = validate_h1_slice(config)

    assert errors == []
    assert config.name == "knowledge_work_h1k_parallel_audit_tool_contract_live"
    assert config.lanes["replayable_core"].episode_ids == ["kwa_ops_parallel_audit_review_v1"]
    assert config.lanes["live_web_stress"].episode_ids == ["kwa_ops_live_parallel_audit_review_v1"]
    assert [family.workflow_id for family in config.workflow_families] == ["ops_parallel_audit_review"]
    assert config.workflow_families[0].h1_stressors == [
        "parallel_audit_array_literal",
        "parallel_tool_calling",
        "two_source_evidence",
        "inspect_image",
        "read_repo_file",
    ]
    packet = h1_packet_selection(config, "mlx_parallel_audit_tool_contract_candidates")
    assert packet.lane == "live_web_stress"
    assert packet.system_ids == [
        "mlx_gemma4_e2b_reasoner_only",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_literal_tool_required",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_parallel_array_required",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_tool_required",
    ]
    assert packet.episode_ids == ["kwa_ops_live_parallel_audit_review_v1"]
    assert "skipped_evidence_source" in packet.failure_modes
    assert "parallel_audit_array_literal" in config.attribution_tags


def test_h1l_slice_config_maps_to_visual_executor_equivalence_packet() -> None:
    config = load_h1_slice(H1L_CONFIG_PATH)

    errors = validate_h1_slice(config)

    assert errors == []
    assert config.name == "knowledge_work_h1l_visual_executor_equivalence_live"
    assert [family.workflow_id for family in config.workflow_families] == [
        "executive_visual_dashboard_review",
        "executive_visual_referent_review",
        "jobs_visual_constraint_override",
        "finance_visual_invoice_review",
        "finance_visual_invoice_revision",
    ]
    packet = h1_packet_selection(config, "mlx_visual_executor_equivalence_candidates")
    assert packet.lane == "live_web_stress"
    assert packet.system_ids == [
        "mlx_gemma4_e2b_reasoner_only",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_literal_targets",
    ]
    assert packet.episode_ids == [
        "kwa_exec_live_visual_dashboard_brief",
        "kwa_exec_live_visual_dashboard_referent_hold_v3",
        "kwa_jobs_live_visual_constraint_override_hold_v2",
        "kwa_finance_live_invoice_lock_direction_hold_v4",
        "kwa_finance_live_visual_invoice_revision_hold_v2",
    ]
    assert "executor_equivalence" in packet.failure_modes
    helper_packet = h1_packet_selection(config, "mlx_visual_executor_equivalence_helper_ablation")
    assert helper_packet.system_ids == [
        "mlx_gemma4_e2b_reasoner_only",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_repair",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_fallback",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_argument_repair",
    ]
    assert helper_packet.episode_ids == packet.episode_ids
    assert "visual_executor_equivalence" in config.attribution_tags


def test_h1m_slice_config_maps_to_visual_alias_repeat_packet() -> None:
    config = load_h1_slice(H1M_CONFIG_PATH)

    errors = validate_h1_slice(config)

    assert errors == []
    assert config.name == "knowledge_work_h1m_visual_alias_repeat_packaged_live"
    assert [family.workflow_id for family in config.workflow_families] == [
        "executive_visual_dashboard_revision",
        "jobs_visual_latest_issue_review",
        "finance_visual_invoice_hold_review",
    ]
    packet = h1_packet_selection(config, "mlx_visual_alias_repeat_packaged_candidates")
    assert packet.lane == "live_web_stress"
    assert packet.system_ids == [
        "mlx_gemma4_e2b_reasoner_only",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_literal_targets",
    ]
    assert packet.episode_ids == [
        "kwa_exec_live_visual_dashboard_revision_hold_v2",
        "kwa_jobs_live_visual_latest_issue_hold_v3",
        "kwa_finance_live_visual_invoice_hold",
    ]
    assert "visual_alias_repeat" in packet.failure_modes
    helper_packet = h1_packet_selection(config, "mlx_visual_alias_repeat_helper_ablation")
    assert helper_packet.system_ids == [
        "mlx_gemma4_e2b_reasoner_only",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_repair",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_fallback",
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_argument_repair",
    ]
    assert helper_packet.episode_ids == packet.episode_ids
    assert "visual_alias_repeat" in config.attribution_tags


def test_h1_primary_run_specs_default_to_mlx_gemma_reasoner_only() -> None:
    config = load_h1_slice()
    registry = load_model_registry()

    specs = build_h1_run_specs(config, registry, lanes=["replayable_core"])

    assert len(specs) == 1
    spec = specs[0]
    assert spec["system_id"] == "mlx_gemma4_e2b_reasoner_only"
    assert spec["lane"] == "replayable_core"
    assert spec["pipeline_name"] == "monolith"
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


def test_h1_mlx_reasoner_only_ablation_specs_are_monolith() -> None:
    config = load_h1_slice(H1C_CONFIG_PATH)
    registry = load_model_registry()

    specs = build_h1_run_specs(
        config,
        registry,
        lanes=["live_web_stress"],
        system_ids=[
            "mlx_gemma4_e2b_reasoner_only_no_controller_repair",
            "mlx_gemma4_e2b_reasoner_only_no_controller_fallback",
            "mlx_gemma4_e2b_reasoner_only_no_argument_repair",
        ],
    )
    by_system = {spec["system_id"]: spec for spec in specs}

    for spec in by_system.values():
        assert spec["pipeline_name"] == "monolith"
        assert spec["backend"] == "mlx"
        assert spec["router"] == ""
        assert spec["retriever"] == ""
    assert by_system["mlx_gemma4_e2b_reasoner_only_no_controller_repair"]["disable_controller_repair"] is True
    assert by_system["mlx_gemma4_e2b_reasoner_only_no_controller_fallback"]["disable_controller_fallback"] is True
    assert by_system["mlx_gemma4_e2b_reasoner_only_no_argument_repair"]["disable_argument_repair"] is True


def test_h1n_argument_hints_helper_ablation_registry_rows_preserve_catalog_profile() -> None:
    config = load_h1_slice(H1M_CONFIG_PATH)
    registry = load_model_registry()

    specs = build_h1_run_specs(
        config,
        registry,
        lanes=["replayable_core"],
        system_ids=[
            "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints_no_controller_repair",
            "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints_no_controller_fallback",
            "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints_no_argument_repair",
        ],
    )
    by_system = {spec["system_id"]: spec for spec in specs}

    for spec in by_system.values():
        assert spec["pipeline_name"] == "monolith"
        assert spec["disable_tool_turn_directive"] is True
        assert spec["tool_catalog_profile_id"] == "visual_role_catalog_argument_hints_v2"
    assert (
        by_system[
            "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints_no_controller_repair"
        ]["disable_controller_repair"]
        is True
    )
    assert (
        by_system[
            "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints_no_controller_fallback"
        ]["disable_controller_fallback"]
        is True
    )
    assert (
        by_system[
            "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints_no_argument_repair"
        ]["disable_argument_repair"]
        is True
    )


def test_h1n_oblique_code_hints_registry_row_preserves_catalog_profile() -> None:
    config = load_h1_slice(H1M_CONFIG_PATH)
    registry = load_model_registry()

    specs = build_h1_run_specs(
        config,
        registry,
        lanes=["replayable_core"],
        system_ids=[
            "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_hints",
        ],
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec["pipeline_name"] == "monolith"
    assert spec["disable_tool_turn_directive"] is True
    assert spec["tool_catalog_profile_id"] == "visual_role_catalog_oblique_code_hints_v6"


def test_h1n_oblique_code_guard_registry_row_preserves_catalog_profile() -> None:
    config = load_h1_slice(H1M_CONFIG_PATH)
    registry = load_model_registry()

    specs = build_h1_run_specs(
        config,
        registry,
        lanes=["replayable_core"],
        system_ids=[
            "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_guard",
        ],
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec["pipeline_name"] == "monolith"
    assert spec["disable_tool_turn_directive"] is True
    assert spec["tool_catalog_profile_id"] == "visual_role_catalog_oblique_code_guard_v7"


def test_h1n_hybrid_label_guard_registry_row_preserves_catalog_profile() -> None:
    config = load_h1_slice(H1M_CONFIG_PATH)
    registry = load_model_registry()

    specs = build_h1_run_specs(
        config,
        registry,
        lanes=["replayable_core"],
        system_ids=[
            "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_hybrid_label_guard",
        ],
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec["pipeline_name"] == "monolith"
    assert spec["disable_tool_turn_directive"] is True
    assert spec["tool_catalog_profile_id"] == "visual_role_catalog_hybrid_label_guard_v8"


def test_h1f_ablation_specs_preserve_tool_directive_flags() -> None:
    config = load_h1_slice(H1F_CONFIG_PATH)
    registry = load_model_registry()

    specs = build_h1_run_specs(config, registry, lanes=["live_web_stress"], run_set="ablation")
    by_system = {spec["system_id"]: spec for spec in specs}

    assert by_system["mlx_gemma4_e2b_reasoner_only"]["disable_tool_turn_directive"] is False
    assert by_system["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive"]["disable_tool_turn_directive"] is True
    assert by_system["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_repair"]["disable_tool_turn_directive"] is True
    assert by_system["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_repair"]["disable_controller_repair"] is True
    assert by_system["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_fallback"]["disable_controller_fallback"] is True
    assert by_system["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_argument_repair"]["disable_argument_repair"] is True


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
    assert command[command.index("--pipeline-name") + 1] == spec["pipeline_name"]
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
    assert command[command.index("--repeat") + 1] == "1"


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


def test_h1_ablation_packet_command_can_repeat_named_packet(tmp_path: Path) -> None:
    config = load_h1_slice(H1I_CONFIG_PATH)
    packet = h1_packet_selection(config, "mlx_prompt_contract_candidates")

    command = PACKET_SCRIPT.h1_ablation_packet_command(
        run_group_id="h1i_repeat_packet_test",
        lane=packet.lane,
        bundle_system_id=config.ablation_bundle_system_id,
        system_ids=packet.system_ids,
        episode_ids=packet.episode_ids,
        output_root=tmp_path,
        repeat_count=3,
    )

    assert command[command.index("--repeat") + 1] == "3"
    assert command.count("--episode-id") == 4
    assert command.count("--system-id") == 5
