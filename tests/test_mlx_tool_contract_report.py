from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_mlx_tool_contract_report.py"
SPEC = importlib.util.spec_from_file_location("build_mlx_tool_contract_report_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_build_mlx_tool_contract_report_writes_tables_figures_and_payload(tmp_path: Path) -> None:
    payload = SCRIPT.build_report(output_dir=tmp_path)

    packet_rows = {row["packet"]: row for row in payload["packet_summary"]}
    assert packet_rows["H1h full"]["contracted_readiness"] == 0.96891
    assert packet_rows["H1h full"]["no_directive_controller_repair"] == 0.7
    assert packet_rows["H1i worst-family"]["no_repair_readiness"] == 0.64697
    assert packet_rows["H1i worst-family"]["failure_candidates"] == 12

    h1i_rows = {row["label"]: row for row in payload["h1i_system_metrics"]}
    assert h1i_rows["contracted"]["strict_interface_avg"] == 1.0
    assert h1i_rows["no directive"]["raw_planning_clean_rate_avg"] == 0.0
    assert h1i_rows["no directive + no repair"]["recovered_execution_avg"] == 0.0

    failure_modes = {
        (row["side"], row["failure_mode"]): row["case_count"] for row in payload["probe_failure_modes"]
    }
    assert failure_modes[("candidate", "argument_mismatch")] == 4
    assert failure_modes[("candidate", "no_tool_call")] == 4
    assert failure_modes[("baseline_non_exact", "executable_paraphrase")] == 1

    assert payload["gemini"]["dry_run"] is True
    assert payload["gemini"]["workflow_count"] == 10
    assert payload["manifest"]["table_count"] == 97
    assert payload["manifest"]["figure_count"] == 42

    candidate_ids = {row["tool_prompt_contract_id"] for row in payload["prompt_contract_candidates"]}
    assert candidate_ids == {
        "schema_anchor_v1",
        "literal_argument_guard_v1",
        "tool_required_parallel_v1",
        "schema_literal_tool_required_v2",
        "visual_next_call_state_v2",
        "parallel_array_required_v2",
        "canonical_json_copy_v3",
        "visual_tool_initiation_v3",
        "parallel_two_call_array_v3",
        "visual_state_tool_selection_v4",
        "visual_refine_selection_v5",
    }
    candidates = {row["tool_prompt_contract_id"]: row for row in payload["prompt_contract_candidates"]}
    combined_candidate = next(
        row
        for row in payload["prompt_contract_candidates"]
        if row["system_id"]
        == "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_literal_guard"
    )
    assert candidates["schema_anchor_v1"]["disable_tool_turn_directive"] is True
    assert candidates["canonical_json_copy_v3"]["label"] == "Canonical JSON Copy v3"
    assert candidates["parallel_two_call_array_v3"]["disable_tool_turn_directive"] is True
    assert candidates["visual_state_tool_selection_v4"]["label"] == "Visual State Tool Selection v4"
    assert candidates["visual_refine_selection_v5"]["label"] == "Visual Refine Selection v5"
    assert combined_candidate["tool_catalog_profile_id"] == "visual_role_catalog_v1"
    gates = {row["tool_prompt_contract_id"]: row for row in payload["prompt_contract_probe_gates"]}
    assert gates["schema_anchor_v1"]["recommendation"] == "weak_exact_gain"
    assert gates["literal_argument_guard_v1"]["recommendation"] == "visual_executable_gain_only"
    wave2_gates = {row["tool_prompt_contract_id"]: row for row in payload["prompt_contract_wave2_probe_gates"]}
    assert wave2_gates["schema_literal_tool_required_v2"]["recommendation"] == "weak_exact_gain"
    assert wave2_gates["visual_next_call_state_v2"]["executable_match_rate"] == "1.0"
    assert wave2_gates["parallel_array_required_v2"]["probe_gate"] == "no_probe_improvement_vs_no_directive"
    wave3_gates = {row["tool_prompt_contract_id"]: row for row in payload["prompt_contract_wave3_probe_gates"]}
    assert wave3_gates["canonical_json_copy_v3"]["recommendation"] == "weak_exact_gain"
    assert wave3_gates["visual_tool_initiation_v3"]["executable_match_rate"] == "1.0"
    assert wave3_gates["parallel_two_call_array_v3"]["probe_gate"] == "no_probe_improvement_vs_no_directive"
    wave4_gates = {row["tool_prompt_contract_id"]: row for row in payload["prompt_contract_wave4_probe_gates"]}
    assert wave4_gates["visual_state_tool_selection_v4"]["recommendation"] == "weak_exact_gain"
    assert wave4_gates["visual_state_tool_selection_v4"]["dominant_failure_mode"] == "no_tool_call"
    wave5_gates = {row["tool_prompt_contract_id"]: row for row in payload["prompt_contract_wave5_probe_gates"]}
    assert wave5_gates["visual_refine_selection_v5"]["recommendation"] == "no_probe_gain"
    assert wave5_gates["visual_refine_selection_v5"]["probe_gate"] == "no_probe_improvement_vs_no_directive"
    catalog_gates = {row["tool_catalog_profile_id"]: row for row in payload["tool_catalog_profile_probe_gates"]}
    assert catalog_gates["visual_role_catalog_v1"]["exact_match_rate"] == "0.125"
    assert catalog_gates["visual_role_catalog_v1"]["executable_match_rate"] == "1.0"
    assert catalog_gates["visual_role_catalog_argument_hints_v2"]["exact_match_rate"] == "0.25"
    assert catalog_gates["visual_role_catalog_argument_hints_v2"]["executable_match_rate"] == "0.0"
    assert catalog_gates["visual_role_catalog_split_selector_hints_v3"]["exact_match_rate"] == "0.125"
    assert catalog_gates["visual_role_catalog_split_selector_hints_v3"]["executable_match_rate"] == "0.0"
    assert catalog_gates["visual_role_catalog_schema_field_hints_v4"]["exact_match_rate"] == "0.25"
    assert catalog_gates["visual_role_catalog_schema_field_hints_v4"]["executable_match_rate"] == "0.0"
    argument_hint_probe_cases = {
        row["case_id"]: row for row in payload["tool_catalog_argument_hints_vs_role_catalog_case_deltas"]
    }
    assert argument_hint_probe_cases["visual_latest_filter_literal"]["delta_exact_match"] == "1"
    assert argument_hint_probe_cases["visual_form_target_literal"]["delta_executable_match"] == "-1"
    split_selector_cases = {
        row["case_id"]: row for row in payload["tool_catalog_split_selector_vs_argument_hints_case_deltas"]
    }
    assert split_selector_cases["visual_readback_region_literal"]["delta_exact_match"] == "-1"
    split_selector_decision = payload["tool_catalog_split_selector_live_replay_decision"][0]
    assert split_selector_decision["decision"] == "skip_live_replay"
    assert split_selector_decision["best_current_exact_candidate"] == "visual_role_catalog_argument_hints_v2"
    schema_field_cases = {
        row["case_id"]: row for row in payload["tool_catalog_schema_field_hints_vs_argument_hints_case_deltas"]
    }
    assert schema_field_cases["visual_form_target_literal"]["candidate_failure_mode"] == "wrong_tool"
    assert schema_field_cases["visual_readback_region_literal"]["delta_exact_match"] == "0"
    schema_field_decision = payload["tool_catalog_schema_field_hints_live_replay_decision"][0]
    assert schema_field_decision["decision"] == "skip_live_replay"
    assert schema_field_decision["candidate_exact_match_rate"] == 0.25
    wave6_gates = {row["tool_catalog_profile_id"]: row for row in payload["prompt_contract_wave6_probe_gates"]}
    assert wave6_gates["visual_role_catalog_v1"]["tool_prompt_contract_id"] == "literal_argument_guard_v1"
    assert wave6_gates["visual_role_catalog_v1"]["executable_match_rate"] == "0.0"
    hard_slice_gates = {row["system_id"]: row for row in payload["visual_hard_slice_probe_gates"]}
    assert hard_slice_gates["mlx_gemma4_e2b_reasoner_only"]["exact_match_rate"] == "1.0"
    assert hard_slice_gates["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive"]["dominant_failure_mode"] == "no_tool_call"
    schema_hard_slice = hard_slice_gates[
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints"
    ]
    assert schema_hard_slice["exact_match_rate"] == "0.75"
    assert schema_hard_slice["executable_match_rate"] == "1.0"
    assert schema_hard_slice["executor_equivalence_match_rate"] == "1.0"
    assert schema_hard_slice["label"] == "catalog schema fields"
    v5_hard_slice = hard_slice_gates[
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_literal_targets"
    ]
    assert v5_hard_slice["exact_match_rate"] == "0.625"
    assert v5_hard_slice["executable_match_rate"] == "0.875"
    assert v5_hard_slice["executor_equivalence_match_rate"] == "0.875"
    assert v5_hard_slice["label"] == "catalog schema target literals"
    hard_slice_families = {
        (row["system_id"], row["family"]): row for row in payload["visual_hard_slice_family_summary"]
    }
    assert hard_slice_families[
        (
            "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints",
            "visual_argument_copying",
        )
    ]["executable_rate"] == "1.0"
    assert hard_slice_families[
        (
            "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints",
            "visual_argument_copying",
        )
    ]["executor_equivalence_rate"] == "1.0"
    assert hard_slice_families[
        (
            "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_literal_targets",
            "visual_tool_routing",
        )
    ]["executable_rate"] == "0.0"
    assert hard_slice_families[
        (
            "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_literal_targets",
            "visual_tool_routing",
        )
    ]["executor_equivalence_rate"] == "0.0"
    exactness_summary = {row["system_label"]: row for row in payload["visual_hard_slice_exactness_summary"]}
    assert exactness_summary["catalog schema fields"]["benchmark_label_artifact_candidate_count"] == "2"
    assert exactness_summary["catalog schema fields"]["true_harness_failure_count"] == "0"
    assert exactness_summary["catalog schema target literals"]["true_harness_failure_count"] == "1"
    exactness_gaps = {
        (row["system_label"], row["case_id"]): row for row in payload["visual_hard_slice_exactness_gaps"]
    }
    assert exactness_gaps[
        ("catalog schema fields", "visual_metric_panel_vs_table_selector")
    ]["research_interpretation"] == "benchmark_label_artifact_candidate"
    assert exactness_gaps[
        ("catalog schema target literals", "visual_form_error_with_prior_selection_decoy")
    ]["research_interpretation"] == "true_harness_failure"
    promotion = {row["tool_prompt_contract_id"]: row for row in payload["prompt_contract_promotion_decisions"]}
    assert promotion["schema_anchor_v1"]["promotion_decision"] == "hold_for_exact_probe_replay"
    assert promotion["visual_next_call_state_v2"]["promotion_reason"].startswith("executable recovery exists")
    assert promotion["parallel_array_required_v2"]["promotion_decision"] == "reject_for_h1_promotion"
    assert promotion["visual_tool_initiation_v3"]["promotion_decision"] == "hold_for_exact_probe_replay"
    assert promotion["parallel_two_call_array_v3"]["promotion_decision"] == "reject_for_h1_promotion"
    assert promotion["visual_state_tool_selection_v4"]["promotion_decision"] == "hold_for_exact_probe_replay"
    assert promotion["visual_refine_selection_v5"]["promotion_decision"] == "reject_for_h1_promotion"
    replay_summary = payload["exact_probe_replay_comparison"]["summary"]
    assert replay_summary["baseline_exact_match_rate"] == 0.875
    assert replay_summary["candidate_exact_match_rate"] == 0.0
    assert replay_summary["delta_exact_match_rate"] == -0.875
    replay_cases = {row["case_id"]: row for row in payload["exact_probe_replay_case_deltas"]}
    assert replay_cases["parallel_audit_array_literal"]["delta_actual_call_count"] == "-2"
    focus = {row["slice"]: row for row in payload["exact_probe_replay_focus_summary"]}
    assert focus["all failures"]["delta_exact_match_rate"] == -0.875
    assert focus["canonical arguments"]["delta_exact_match_rate"] == -1.0
    assert focus["canonical arguments"]["shared_case_count"] == 4
    assert focus["visual no-call"]["candidate_exact_match_rate"] == 0.0
    assert focus["parallel array"]["baseline_exact_match_rate"] == 1.0
    live_parallel = payload["live_parallel_replay_comparison"]["summary"]
    assert live_parallel["baseline_exact_rate"] == 1.0
    assert live_parallel["candidate_exact_rate"] == 0.0
    assert live_parallel["delta_exact_rate"] == -1.0
    live_parallel_cases = {row["case_id"]: row for row in payload["live_parallel_replay_case_deltas"]}
    assert live_parallel_cases["parallel_audit_array_literal"]["delta_actual_call_count"] == "-2"
    live_visual = payload["live_visual_replay_comparison"]["summary"]
    assert live_visual["baseline_exact_rate"] == 2 / 3
    assert live_visual["candidate_exact_rate"] == 0.0
    live_visual_cases = {row["case_id"]: row for row in payload["live_visual_replay_case_deltas"]}
    assert live_visual_cases["visual_latest_filter_literal"]["delta_actual_call_count"] == "-1"
    live_canonical = payload["live_canonical_replay_comparison"]["summary"]
    assert live_canonical["baseline_exact_rate"] == 1.0
    assert live_canonical["candidate_exact_rate"] == 0.0
    live_canonical_cases = {row["case_id"]: row for row in payload["live_canonical_replay_case_deltas"]}
    assert live_canonical_cases["cli_invoice_lock_hyphen_query"]["delta_actual_call_count"] == "0"
    live_focus = {row["slice"]: row for row in payload["live_replay_focus_summary"]}
    assert live_focus["canonical arguments"]["delta_exact_rate"] == -1.0
    assert live_focus["parallel array"]["delta_exact_rate"] == -1.0
    assert live_focus["visual no-call"]["shared_case_count"] == 3
    wave3_live = {row["comparison"]: row for row in payload["wave3_live_candidate_replay_summary"]}
    assert wave3_live["canonical JSON vs no directive"]["candidate_exact_rate"] == 0.0
    assert wave3_live["visual initiation vs no directive"]["delta_exact_rate"] == 1 / 3
    assert wave3_live["visual initiation vs no directive"]["candidate_executable_rate"] == 1.0
    wave3_live_cases = {
        (row["comparison"], row["case_id"]): row for row in payload["wave3_live_candidate_case_deltas"]
    }
    assert wave3_live_cases[("visual initiation vs no directive", "visual_readback_region_literal")][
        "candidate_replay_exact_match"
    ]
    assert wave3_live_cases[("canonical JSON vs no directive", "api_invoice_lock_hold_update")][
        "delta_actual_call_count"
    ] == -1
    wave4_live = {row["comparison"]: row for row in payload["wave4_live_candidate_replay_summary"]}
    assert wave4_live["visual state tool selection vs no directive"]["delta_exact_rate"] == 1 / 3
    assert wave4_live["visual state tool selection vs contracted"]["delta_executable_rate"] == -1.0
    wave4_live_cases = {
        (row["comparison"], row["case_id"]): row for row in payload["wave4_live_candidate_case_deltas"]
    }
    assert wave4_live_cases[("visual state tool selection vs no directive", "visual_readback_region_literal")][
        "candidate_replay_exact_match"
    ]
    assert wave4_live_cases[("visual state tool selection vs contracted", "visual_latest_filter_literal")][
        "candidate_replay_failure_mode"
    ] == "wrong_tool"
    catalog_live = {row["comparison"]: row for row in payload["visual_catalog_live_candidate_replay_summary"]}
    assert catalog_live["visual role catalog vs no directive"]["delta_exact_rate"] == 1 / 3
    assert catalog_live["visual role catalog vs visual state tool"]["delta_executable_rate"] == 1.0
    catalog_live_cases = {
        (row["comparison"], row["case_id"]): row for row in payload["visual_catalog_live_candidate_case_deltas"]
    }
    assert catalog_live_cases[("visual role catalog vs visual state tool", "visual_latest_filter_literal")][
        "candidate_replay_failure_mode"
    ] == "argument_mismatch"
    argument_hints_live = {
        row["comparison"]: row for row in payload["visual_catalog_argument_hints_live_candidate_replay_summary"]
    }
    assert argument_hints_live["visual argument hints vs no directive"]["delta_exact_rate"] == 2 / 3
    assert argument_hints_live["visual argument hints vs contracted"]["delta_exact_rate"] == 0.0
    assert argument_hints_live["visual argument hints vs role catalog"]["delta_exact_rate"] == 1 / 3
    assert argument_hints_live["visual argument hints vs role catalog"]["delta_executable_rate"] == -1.0
    argument_hints_live_cases = {
        (row["comparison"], row["case_id"]): row
        for row in payload["visual_catalog_argument_hints_live_candidate_case_deltas"]
    }
    assert argument_hints_live_cases[("visual argument hints vs role catalog", "visual_latest_filter_literal")][
        "candidate_replay_failure_mode"
    ] == "exact"
    assert argument_hints_live_cases[("visual argument hints vs contracted", "visual_form_target_literal")][
        "delta_executable_match"
    ] == -1
    visual_hard_slice_live = {row["comparison"]: row for row in payload["visual_hard_slice_live_replay_summary"]}
    assert set(visual_hard_slice_live) == {
        "contracted vs no directive",
        "role catalog vs no directive",
        "argument hints vs no directive",
        "schema-field hints vs no directive",
        "schema literal targets vs no directive",
    }
    assert visual_hard_slice_live["contracted vs no directive"]["delta_exact_rate"] == 1.0
    assert visual_hard_slice_live["contracted vs no directive"]["delta_executor_equivalence_rate"] == 1.0
    assert visual_hard_slice_live["role catalog vs no directive"]["delta_exact_rate"] == 0.5
    assert visual_hard_slice_live["role catalog vs no directive"]["delta_executor_equivalence_rate"] == 0.5
    assert visual_hard_slice_live["argument hints vs no directive"]["delta_exact_rate"] == 0.5
    assert visual_hard_slice_live["argument hints vs no directive"]["delta_executor_equivalence_rate"] == 0.5
    assert visual_hard_slice_live["schema-field hints vs no directive"]["delta_exact_rate"] == 0.5
    assert visual_hard_slice_live["schema-field hints vs no directive"]["delta_executor_equivalence_rate"] == 1.0
    assert visual_hard_slice_live["schema literal targets vs no directive"]["delta_exact_rate"] == 0.0
    assert visual_hard_slice_live["schema literal targets vs no directive"]["delta_executor_equivalence_rate"] == 0.5
    visual_hard_slice_live_cases = {
        (row["comparison"], row["case_id"]): row for row in payload["visual_hard_slice_live_replay_case_deltas"]
    }
    assert visual_hard_slice_live_cases[
        ("schema-field hints vs no directive", "visual_metric_panel_vs_table_selector")
    ][
        "candidate_replay_executor_equivalence_match"
    ] is True
    assert visual_hard_slice_live_cases[
        ("schema literal targets vs no directive", "visual_form_error_with_prior_selection_decoy")
    ]["candidate_replay_failure_mode"] == "wrong_tool"
    visual_hard_slice_stress_live = {
        row["comparison"]: row for row in payload["visual_hard_slice_stress_live_replay_summary"]
    }
    assert visual_hard_slice_stress_live["stress contracted vs no directive"]["delta_exact_rate"] == 0.5
    assert (
        visual_hard_slice_stress_live["stress contracted vs no directive"]["delta_executor_equivalence_rate"] == 0.25
    )
    assert visual_hard_slice_stress_live["stress role catalog vs no directive"]["delta_exact_rate"] == -0.25
    assert (
        visual_hard_slice_stress_live["stress role catalog vs no directive"]["delta_executor_equivalence_rate"]
        == -0.25
    )
    assert visual_hard_slice_stress_live["stress argument hints vs no directive"]["delta_exact_rate"] == 0.0
    assert (
        visual_hard_slice_stress_live["stress schema-field hints vs no directive"][
            "delta_executor_equivalence_rate"
        ]
        == 0.25
    )
    assert (
        visual_hard_slice_stress_live["stress schema literal targets vs no directive"][
            "delta_executor_equivalence_rate"
        ]
        == 0.25
    )
    visual_hard_slice_stress_cases = {
        (row["comparison"], row["case_id"]): row
        for row in payload["visual_hard_slice_stress_live_replay_case_deltas"]
    }
    assert visual_hard_slice_stress_cases[
        ("stress schema-field hints vs no directive", "stress_metric_panel_with_chart_table_decoys")
    ]["delta_executor_equivalence_match"] == 1
    assert visual_hard_slice_stress_cases[
        ("stress role catalog vs no directive", "stress_form_error_stale_selection_warning_decoy")
    ]["candidate_replay_failure_mode"] == "no_tool_call"
    visual_hard_slice_alias_repeat = {
        row["comparison"]: row for row in payload["visual_hard_slice_alias_repeat_live_replay_summary"]
    }
    assert (
        visual_hard_slice_alias_repeat["alias-repeat contracted vs no directive"]["delta_exact_rate"]
        == 0.625
    )
    assert (
        visual_hard_slice_alias_repeat["alias-repeat contracted vs no directive"][
            "delta_executor_equivalence_rate"
        ]
        == 0.375
    )
    assert (
        visual_hard_slice_alias_repeat["alias-repeat role catalog vs no directive"]["delta_exact_rate"]
        == -0.125
    )
    assert (
        visual_hard_slice_alias_repeat["alias-repeat argument hints vs no directive"][
            "delta_executor_equivalence_rate"
        ]
        == 0.125
    )
    assert (
        visual_hard_slice_alias_repeat["alias-repeat schema-field hints vs no directive"][
            "delta_exact_rate"
        ]
        == 0.0
    )
    assert (
        visual_hard_slice_alias_repeat["alias-repeat schema-field hints vs no directive"][
            "delta_executor_equivalence_rate"
        ]
        == 0.25
    )
    assert (
        visual_hard_slice_alias_repeat["alias-repeat schema literal targets vs no directive"][
            "delta_exact_rate"
        ]
        == 0.125
    )
    assert (
        visual_hard_slice_alias_repeat["alias-repeat schema literal targets vs no directive"][
            "delta_executor_equivalence_rate"
        ]
        == 0.375
    )
    visual_hard_slice_alias_repeat_cases = {
        (row["comparison"], row["case_id"]): row
        for row in payload["visual_hard_slice_alias_repeat_live_replay_case_deltas"]
    }
    assert visual_hard_slice_alias_repeat_cases[
        ("alias-repeat schema literal targets vs no directive", "stress_callout_warning_risk_note_decoy")
    ]["delta_exact_match"] == 1
    assert visual_hard_slice_alias_repeat_cases[
        ("alias-repeat schema-field hints vs no directive", "stress_callout_warning_person_table_decoy")
    ]["delta_executor_equivalence_match"] == 1
    visual_hard_slice_alias_transfer = {
        row["comparison"]: row for row in payload["visual_hard_slice_alias_transfer_live_replay_summary"]
    }
    assert (
        visual_hard_slice_alias_transfer["alias-transfer argument hints vs no directive"][
            "delta_executor_equivalence_rate"
        ]
        == 0.6666666666666667
    )
    assert (
        visual_hard_slice_alias_transfer["alias-transfer contracted vs no directive"]["delta_exact_rate"]
        == 0.8333333333333334
    )
    assert (
        visual_hard_slice_alias_transfer["alias-transfer contracted vs no directive"][
            "delta_executor_equivalence_rate"
        ]
        == -0.16666666666666666
    )
    assert (
        visual_hard_slice_alias_transfer["alias-transfer schema literal targets vs no directive"][
            "delta_executor_equivalence_rate"
        ]
        == 0.3333333333333333
    )
    visual_hard_slice_alias_transfer_cases = {
        (row["comparison"], row["case_id"]): row
        for row in payload["visual_hard_slice_alias_transfer_live_replay_case_deltas"]
    }
    assert visual_hard_slice_alias_transfer_cases[
        ("alias-transfer argument hints vs no directive", "transfer_review_tile_notice_table_decoy")
    ]["delta_executor_equivalence_match"] == 1
    visual_hard_slice_alias_transfer_oracle = {
        row["comparison"]: row for row in payload["visual_hard_slice_alias_transfer_oracle_live_replay_summary"]
    }
    assert (
        visual_hard_slice_alias_transfer_oracle["alias-transfer oracle contracted vs no directive"][
            "delta_exact_rate"
        ]
        == -0.16666666666666666
    )
    assert (
        visual_hard_slice_alias_transfer_oracle["alias-transfer oracle argument hints vs no directive"][
            "candidate_exact_rate"
        ]
        == 0.8333333333333334
    )
    assert (
        visual_hard_slice_alias_transfer_oracle["alias-transfer oracle argument hints vs no directive"][
            "candidate_executor_equivalence_rate"
        ]
        == 1.0
    )
    assert (
        visual_hard_slice_alias_transfer_oracle["alias-transfer oracle schema literal targets vs no directive"][
            "candidate_executor_equivalence_rate"
        ]
        == 0.6666666666666666
    )
    visual_hard_slice_alias_transfer_oracle_cases = {
        (row["comparison"], row["case_id"]): row
        for row in payload["visual_hard_slice_alias_transfer_oracle_live_replay_case_deltas"]
    }
    assert visual_hard_slice_alias_transfer_oracle_cases[
        ("alias-transfer oracle argument hints vs no directive", "transfer_status_pill_chart_decoy")
    ]["delta_executor_equivalence_match"] == 1
    visual_hard_slice_post_repair = {
        row["comparison"]: row for row in payload["visual_hard_slice_post_repair_live_replay_summary"]
    }
    assert (
        visual_hard_slice_post_repair["post-repair contracted vs no directive"]["candidate_exact_rate"]
        == 0.375
    )
    assert (
        visual_hard_slice_post_repair["post-repair argument hints vs no directive"]["candidate_exact_rate"]
        == 0.625
    )
    assert (
        visual_hard_slice_post_repair["post-repair oblique code hints vs no directive"]["candidate_exact_rate"]
        == 0.625
    )
    assert (
        visual_hard_slice_post_repair["post-repair oblique code guard vs no directive"]["candidate_exact_rate"]
        == 0.75
    )
    assert (
        visual_hard_slice_post_repair["post-repair oblique code guard vs no directive"][
            "delta_executor_equivalence_rate"
        ]
        == 0.5
    )
    visual_hard_slice_post_repair_cases = {
        (row["comparison"], row["case_id"]): row
        for row in payload["visual_hard_slice_post_repair_live_replay_case_deltas"]
    }
    assert visual_hard_slice_post_repair_cases[
        ("post-repair oblique code guard vs no directive", "post_repair_badge_t64_notice_decoy")
    ]["delta_exact_match"] == 1
    assert visual_hard_slice_post_repair_cases[
        ("post-repair oblique code hints vs no directive", "post_repair_review_tile_table_decoy")
    ]["delta_exact_match"] == -1
    visual_hard_slice_residual = {
        row["comparison"]: row for row in payload["visual_hard_slice_residual_live_replay_summary"]
    }
    assert (
        visual_hard_slice_residual["residual contracted vs no directive"]["candidate_exact_rate"]
        == 0.25
    )
    assert (
        visual_hard_slice_residual["residual argument hints vs no directive"]["candidate_exact_rate"]
        == 0.625
    )
    assert (
        visual_hard_slice_residual["residual oblique code guard vs no directive"][
            "candidate_executor_equivalence_rate"
        ]
        == 0.875
    )
    assert (
        visual_hard_slice_residual["residual hybrid label guard vs no directive"]["candidate_exact_rate"]
        == 0.875
    )
    visual_hard_slice_residual_cases = {
        (row["comparison"], row["case_id"]): row
        for row in payload["visual_hard_slice_residual_live_replay_case_deltas"]
    }
    assert visual_hard_slice_residual_cases[
        ("residual hybrid label guard vs no directive", "residual_field_m20_stale_selection_decoy")
    ]["delta_exact_match"] == 1
    assert visual_hard_slice_residual_cases[
        ("residual hybrid label guard vs no directive", "residual_state_pill_note_decoy")
    ]["delta_exact_match"] == 0
    visual_hard_slice_component_value = {
        row["comparison"]: row for row in payload["visual_hard_slice_component_value_live_replay_summary"]
    }
    assert (
        visual_hard_slice_component_value["component-value contracted vs no directive"][
            "candidate_exact_rate"
        ]
        == 0.125
    )
    assert (
        visual_hard_slice_component_value["component-value argument hints vs no directive"][
            "candidate_executor_equivalence_rate"
        ]
        == 0.875
    )
    assert (
        visual_hard_slice_component_value["component-value hybrid label guard vs no directive"][
            "candidate_exact_rate"
        ]
        == 0.75
    )
    assert (
        visual_hard_slice_component_value["component-value component value guard vs no directive"][
            "delta_executor_equivalence_rate"
        ]
        == -0.25
    )
    assert (
        visual_hard_slice_component_value["component-value no-call control rescue vs no directive"][
            "candidate_exact_rate"
        ]
        == 0.875
    )
    assert (
        visual_hard_slice_component_value["component-value no-call control rescue vs no directive"][
            "candidate_executor_equivalence_rate"
        ]
        == 1.0
    )
    visual_hard_slice_component_value_cases = {
        (row["comparison"], row["case_id"]): row
        for row in payload["visual_hard_slice_component_value_live_replay_case_deltas"]
    }
    assert visual_hard_slice_component_value_cases[
        (
            "component-value component value guard vs no directive",
            "component_value_state_pill_note_decoy",
        )
    ]["delta_exact_match"] == -1
    assert visual_hard_slice_component_value_cases[
        (
            "component-value argument hints vs no directive",
            "component_value_owner_field_stale_selection_decoy",
        )
    ]["delta_exact_match"] == 1
    assert visual_hard_slice_component_value_cases[
        (
            "component-value no-call control rescue vs no directive",
            "component_value_status_badge_email_decoy",
        )
    ]["delta_executor_equivalence_match"] == 1
    visual_hard_slice_h1o = {
        row["comparison"]: row for row in payload["visual_hard_slice_h1o_live_replay_summary"]
    }
    assert (
        visual_hard_slice_h1o["h1o argument hints vs no directive"]["candidate_exact_rate"]
        == 0.75
    )
    assert (
        visual_hard_slice_h1o["h1o component-value guard vs no directive"]["candidate_exact_rate"]
        == 0.75
    )
    assert (
        visual_hard_slice_h1o["h1o hybrid label guard vs no directive"][
            "candidate_executor_equivalence_rate"
        ]
        == 0.8333333333333334
    )
    assert (
        visual_hard_slice_h1o["h1o no-call control rescue vs no directive"]["candidate_exact_rate"]
        == 0.5833333333333334
    )
    visual_hard_slice_h1o_cases = {
        (row["comparison"], row["case_id"]): row
        for row in payload["visual_hard_slice_h1o_live_replay_case_deltas"]
    }
    assert visual_hard_slice_h1o_cases[
        (
            "h1o no-call control rescue vs no directive",
            "h1o_activation_error_banner_previous_region_decoy",
        )
    ]["delta_exact_match"] == -1
    assert visual_hard_slice_h1o_cases[
        (
            "h1o argument hints vs no directive",
            "h1o_code_field_u17_old_selection_decoy",
        )
    ]["delta_exact_match"] == 1
    assert visual_hard_slice_h1o_cases[
        (
            "h1o component-value guard vs no directive",
            "h1o_component_phase_tile_value_decoy",
        )
    ]["delta_exact_match"] == 1
    visual_hard_slice_h1p = {
        row["comparison"]: row for row in payload["visual_hard_slice_h1p_live_replay_summary"]
    }
    assert (
        visual_hard_slice_h1p["h1p argument hints vs no directive"]["candidate_exact_rate"]
        == 0.5
    )
    assert (
        visual_hard_slice_h1p["h1p argument hints vs no directive"][
            "candidate_executor_equivalence_rate"
        ]
        == 0.5
    )
    assert (
        visual_hard_slice_h1p["h1p hybrid label guard vs no directive"]["candidate_exact_rate"]
        == 0.75
    )
    assert (
        visual_hard_slice_h1p["h1p hybrid label guard vs no directive"][
            "candidate_executor_equivalence_rate"
        ]
        == 0.8333333333333334
    )
    assert (
        visual_hard_slice_h1p["h1p no-call control rescue vs no directive"][
            "candidate_exact_rate"
        ]
        == 0.5
    )
    assert (
        visual_hard_slice_h1p["h1p component-value guard vs no directive"][
            "candidate_exact_rate"
        ]
        == 0.8333333333333334
    )
    assert (
        visual_hard_slice_h1p["h1p component-value guard vs no directive"][
            "candidate_executor_equivalence_rate"
        ]
        == 0.9166666666666666
    )
    visual_hard_slice_h1p_cases = {
        (row["comparison"], row["case_id"]): row
        for row in payload["visual_hard_slice_h1p_live_replay_case_deltas"]
    }
    assert visual_hard_slice_h1p_cases[
        (
            "h1p component-value guard vs no directive",
            "h1p_compact_status_pill_summary_value_decoy",
        )
    ]["delta_exact_match"] == 1
    assert visual_hard_slice_h1p_cases[
        (
            "h1p component-value guard vs no directive",
            "h1p_surface_lane_tile_board_value_decoy",
        )
    ]["delta_executor_equivalence_match"] == 1
    assert visual_hard_slice_h1p_cases[
        (
            "h1p component-value guard vs no directive",
            "h1p_surface_lane_tile_board_value_decoy",
        )
    ]["delta_exact_match"] == 0
    assert visual_hard_slice_h1p_cases[
        (
            "h1p component-value guard vs no directive",
            "h1p_stale_phase_tile_archive_decoy",
        )
    ]["delta_exact_match"] == 0
    h1q_aggregate = {
        row["profile_label"]: row for row in payload["h1q_component_label_guard_aggregate_summary"]
    }
    assert h1q_aggregate["component_label_guard_v11"]["exact_success_count"] == 26
    assert h1q_aggregate["component_label_guard_v11"]["executor_success_count"] == 29
    assert h1q_aggregate["component_value_guard_v9"]["exact_success_count"] == 23
    assert h1q_aggregate["component_value_guard_v9"]["executor_success_count"] == 25
    h1q_packet = {
        (row["packet_label"], row["profile_label"]): row
        for row in payload["h1q_component_label_guard_packet_summary"]
    }
    assert h1q_packet[("h1o_control_factorial", "component_label_guard_v11")][
        "executor_success_count"
    ] == 12
    assert h1q_packet[("h1p_component_value", "component_label_guard_v11")][
        "exact_success_count"
    ] == 10
    h1q_failures = {
        (row["packet_label"], row["case_id"]): row for row in payload["h1q_component_label_guard_v11_failures"]
    }
    assert h1q_failures[
        ("h1n_component_value", "component_value_owner_field_stale_selection_decoy")
    ]["failure_mode"] == "wrong_tool"
    assert h1q_failures[
        ("h1p_component_value", "h1p_surface_mode_toggle_note_value_decoy")
    ]["executor_equivalence_match"] is False
    h1s_aggregate = {
        row["profile_label"]: row for row in payload["h1s_component_residual_transfer_aggregate"]
    }
    assert h1s_aggregate["component_label_guard_v11"]["exact_success_count"] == 26
    assert h1s_aggregate["component_label_guard_v11"]["executor_success_count"] == 29
    assert h1s_aggregate["component_residual_guard_v12"]["exact_success_count"] == 27
    assert h1s_aggregate["component_residual_guard_v12"]["executor_success_count"] == 27
    h1s_packet = {
        (row["packet_label"], row["profile_label"]): row
        for row in payload["h1s_component_residual_packet_summary"]
    }
    assert h1s_packet[("h1r_component_residual", "component_residual_guard_v12")][
        "exact_success_count"
    ] == 6
    assert h1s_packet[("h1n_component_value", "component_residual_guard_v12")][
        "executor_success_count"
    ] == 5
    h1s_failures = {
        (row["packet_label"], row["case_id"]): row for row in payload["h1s_component_residual_v12_failures"]
    }
    assert h1s_failures[
        ("h1p_component_value", "h1p_stale_phase_tile_archive_decoy")
    ]["failure_mode"] == "wrong_tool"
    h1x_packet = {row["profile_label"]: row for row in payload["h1x_v11_breaker_packet_summary"]}
    assert h1x_packet["no_directive"]["exact_success_count"] == 2
    assert h1x_packet["component_label_guard_v11"]["exact_success_count"] == 7
    assert h1x_packet["component_residual_guard_v12"]["exact_success_count"] == 8
    assert h1x_packet["code_label_exact_guard_v15"]["exact_success_count"] == 6
    assert h1x_packet["code_label_exact_guard_v15"]["executor_success_count"] == 7
    h1x_family = {
        (row["profile_label"], row["family"]): row for row in payload["h1x_v11_breaker_family_summary"]
    }
    assert h1x_family[("component_label_guard_v11", "h1x_oblique_stale_field")][
        "exact_success_count"
    ] == 1
    assert h1x_family[("component_residual_guard_v12", "h1x_oblique_stale_field")][
        "exact_success_count"
    ] == 2
    assert h1x_family[("code_label_exact_guard_v15", "h1x_oblique_surface_value")][
        "executor_success_count"
    ] == 2
    h1x_failures = {
        (row["profile_label"], row["case_id"]): row for row in payload["h1x_v11_breaker_non_exact_rows"]
    }
    assert h1x_failures[
        ("component_label_guard_v11", "h1x_responsible_party_field_old_owner_memo_decoy")
    ]["failure_mode"] == "wrong_tool"
    assert h1x_failures[
        ("code_label_exact_guard_v15", "h1x_resolution_chip_comment_result_decoy")
    ]["executor_equivalence_match"] is True
    h1y_packet = {row["profile_label"]: row for row in payload["h1y_routed_residual_packet_summary"]}
    assert h1y_packet["no_directive"]["exact_success_count"] == 0
    assert h1y_packet["component_label_guard_v11"]["exact_success_count"] == 5
    assert h1y_packet["component_residual_guard_v12"]["exact_success_count"] == 7
    assert h1y_packet["component_label_guard_v11_stale_selection_gate_h2a"]["exact_success_count"] == 8
    h1y_family = {
        (row["profile_label"], row["family"]): row for row in payload["h1y_routed_residual_family_summary"]
    }
    assert h1y_family[
        ("component_label_guard_v11_stale_selection_gate_h2a", "h1y_route_stale_field")
    ]["exact_success_count"] == 3
    h1y_findings = {row["finding_id"]: row["finding"] for row in payload["h1y_routed_residual_findings"]}
    assert "8/10" in h1y_findings["h2a_controller_gate_is_causal"]
    assert "Promote H2a" in h1y_findings["next_slice"]
    h1i_candidates = {row["system_id"]: row for row in payload["h1i_prompt_contract_candidate_metrics"]}
    assert h1i_candidates["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive"]["tool_turn_directive_enabled"] == "False"
    assert h1i_candidates["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_anchor"]["raw_planning_clean_rate_avg"] == "1.0"
    h1i_repeats = {row["system_id"]: row for row in payload["h1i_prompt_contract_repeat3_metrics"]}
    assert h1i_repeats["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive"]["raw_planning_clean_rate_avg"] == "1.0"
    assert h1i_repeats["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_tool_required"]["controller_repair_avg"] == "0.0"
    h1j_candidates = {row["system_id"]: row for row in payload["h1j_probe_derived_candidate_metrics"]}
    assert h1j_candidates["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive"]["real_world_readiness_avg"] == "0.9657666666666667"
    assert h1j_candidates["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_anchor"]["raw_planning_clean_rate_avg"] == "1.0"
    h1j_helpers = {row["system_id"]: row for row in payload["h1j_probe_derived_helper_metrics"]}
    assert h1j_helpers["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_repair"]["strict_interface_avg"] == "1.0"
    assert h1j_helpers["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_fallback"]["raw_planning_clean_rate_avg"] == "1.0"
    h1k_candidates = {row["system_id"]: row for row in payload["h1k_parallel_audit_candidate_metrics"]}
    assert h1k_candidates["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive"]["real_world_readiness_avg"] == "0.9178"
    assert h1k_candidates["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_parallel_array_required"]["raw_planning_clean_rate_avg"] == "1.0"
    h1k_helpers = {row["system_id"]: row for row in payload["h1k_parallel_audit_helper_metrics"]}
    assert h1k_helpers["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_repair"]["controller_repair_avg"] == "0.0"
    assert h1k_helpers["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_argument_repair"]["strict_interface_avg"] == "1.0"
    h1l_candidates = {row["system_id"]: row for row in payload["h1l_visual_executor_equivalence_candidate_metrics"]}
    assert h1l_candidates["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive"]["real_world_readiness_avg"] == "0.90406"
    assert h1l_candidates[
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints"
    ]["raw_planning_clean_rate_avg"] == "1.0"
    h1m_candidates = {row["system_id"]: row for row in payload["h1m_visual_alias_repeat_candidate_metrics"]}
    assert (
        h1m_candidates["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive"][
            "real_world_readiness_avg"
        ]
        == "0.8778333333333332"
    )
    assert h1m_candidates[
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_literal_targets"
    ]["strict_interface_avg"] == "0.75"
    assert h1m_candidates[
        "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints"
    ]["controller_repair_avg"] == "0.0"

    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "tables" / "packet_summary.csv").exists()
    assert (tmp_path / "tables" / "probe_failure_modes.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_exactness_gaps.csv").exists()
    assert (tmp_path / "tables" / "prompt_contract_candidates.csv").exists()
    assert (tmp_path / "tables" / "prompt_contract_probe_gates.csv").exists()
    assert (tmp_path / "tables" / "prompt_contract_probe_failure_modes.csv").exists()
    assert (tmp_path / "tables" / "prompt_contract_wave2_probe_gates.csv").exists()
    assert (tmp_path / "tables" / "prompt_contract_wave2_probe_failure_modes.csv").exists()
    assert (tmp_path / "tables" / "prompt_contract_wave3_probe_gates.csv").exists()
    assert (tmp_path / "tables" / "prompt_contract_wave3_probe_failure_modes.csv").exists()
    assert (tmp_path / "tables" / "tool_catalog_profile_probe_gates.csv").exists()
    assert (tmp_path / "tables" / "tool_catalog_argument_hints_vs_role_catalog_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "tool_catalog_split_selector_vs_argument_hints_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "tool_catalog_split_selector_vs_role_catalog_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "tool_catalog_split_selector_live_replay_decision.csv").exists()
    assert (tmp_path / "tables" / "tool_catalog_schema_field_hints_vs_argument_hints_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "tool_catalog_schema_field_hints_vs_split_selector_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "tool_catalog_schema_field_hints_vs_role_catalog_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "tool_catalog_schema_field_hints_live_replay_decision.csv").exists()
    assert (tmp_path / "tables" / "prompt_contract_wave6_probe_gates.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_probe_gates.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_failure_modes.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_family_summary.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_case_deltas_vs_no_directive.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_case_deltas_vs_contracted.csv").exists()
    assert (tmp_path / "tables" / "visual_catalog_live_candidate_replay_summary.csv").exists()
    assert (tmp_path / "tables" / "visual_catalog_live_candidate_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "visual_catalog_argument_hints_live_candidate_replay_summary.csv").exists()
    assert (tmp_path / "tables" / "visual_catalog_argument_hints_live_candidate_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_live_replay_summary.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_live_replay_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_stress_live_replay_summary.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_stress_live_replay_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_alias_repeat_live_replay_summary.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_alias_repeat_live_replay_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_alias_transfer_live_replay_summary.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_alias_transfer_live_replay_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_alias_transfer_oracle_live_replay_summary.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_alias_transfer_oracle_live_replay_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_post_repair_live_replay_summary.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_post_repair_live_replay_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_residual_live_replay_summary.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_residual_live_replay_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_component_value_live_replay_summary.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_component_value_live_replay_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_h1o_live_replay_summary.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_h1o_live_replay_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_h1p_live_replay_summary.csv").exists()
    assert (tmp_path / "tables" / "visual_hard_slice_h1p_live_replay_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "h1q_component_label_guard_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h1q_component_label_guard_aggregate_summary.csv").exists()
    assert (tmp_path / "tables" / "h1q_component_label_guard_v11_failures.csv").exists()
    assert (tmp_path / "tables" / "h1q_component_label_guard_findings.csv").exists()
    assert (tmp_path / "tables" / "h1s_component_residual_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h1s_component_residual_transfer_aggregate.csv").exists()
    assert (tmp_path / "tables" / "h1s_component_residual_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h1s_component_residual_v12_failures.csv").exists()
    assert (tmp_path / "tables" / "h1s_component_residual_findings.csv").exists()
    assert (tmp_path / "tables" / "h1x_v11_breaker_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h1x_v11_breaker_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h1x_v11_breaker_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h1x_v11_breaker_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h1x_v11_breaker_findings.csv").exists()
    assert (tmp_path / "tables" / "h1y_routed_residual_packet_summary.csv").exists()
    assert (tmp_path / "tables" / "h1y_routed_residual_family_summary.csv").exists()
    assert (tmp_path / "tables" / "h1y_routed_residual_comparison_summary.csv").exists()
    assert (tmp_path / "tables" / "h1y_routed_residual_non_exact_rows.csv").exists()
    assert (tmp_path / "tables" / "h1y_routed_residual_findings.csv").exists()
    assert (tmp_path / "tables" / "prompt_contract_promotion_decisions.csv").exists()
    assert (tmp_path / "tables" / "h1i_prompt_contract_candidate_metrics.csv").exists()
    assert (tmp_path / "tables" / "h1i_prompt_contract_repeat3_metrics.csv").exists()
    assert (tmp_path / "tables" / "h1j_probe_derived_candidate_metrics.csv").exists()
    assert (tmp_path / "tables" / "h1j_probe_derived_helper_metrics.csv").exists()
    assert (tmp_path / "tables" / "h1k_parallel_audit_candidate_metrics.csv").exists()
    assert (tmp_path / "tables" / "h1k_parallel_audit_helper_metrics.csv").exists()
    assert (tmp_path / "tables" / "h1l_visual_executor_equivalence_candidate_metrics.csv").exists()
    assert (tmp_path / "tables" / "h1m_visual_alias_repeat_candidate_metrics.csv").exists()
    assert (tmp_path / "tables" / "exact_probe_replay_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "exact_probe_replay_family_deltas.csv").exists()
    assert (tmp_path / "tables" / "exact_probe_replay_focus_summary.csv").exists()
    assert (tmp_path / "tables" / "live_parallel_replay_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "live_visual_replay_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "live_canonical_replay_case_deltas.csv").exists()
    assert (tmp_path / "tables" / "wave3_live_candidate_replay_summary.csv").exists()
    assert (tmp_path / "tables" / "wave3_live_candidate_case_deltas.csv").exists()
    assert (tmp_path / "figures" / "h1i_readiness_strict_recovered.svg").exists()
    assert (tmp_path / "figures" / "h1h_h1i_controller_burden.svg").exists()
    assert (tmp_path / "figures" / "prompt_contract_candidate_targets.svg").exists()
    assert (tmp_path / "figures" / "prompt_contract_probe_gate.svg").exists()
    assert (tmp_path / "figures" / "prompt_contract_wave2_probe_gate.svg").exists()
    assert (tmp_path / "figures" / "prompt_contract_wave3_probe_gate.svg").exists()
    assert (tmp_path / "figures" / "tool_catalog_profile_probe_gate.svg").exists()
    assert (tmp_path / "figures" / "prompt_contract_wave6_probe_gate.svg").exists()
    assert (tmp_path / "figures" / "visual_hard_slice_probe_gate.svg").exists()
    assert (tmp_path / "figures" / "h1i_prompt_contract_repeat3_burden.svg").exists()
    assert (tmp_path / "figures" / "h1j_probe_derived_burden.svg").exists()
    assert (tmp_path / "figures" / "h1j_probe_derived_helper_burden.svg").exists()
    assert (tmp_path / "figures" / "h1k_parallel_audit_burden.svg").exists()
    assert (tmp_path / "figures" / "h1k_parallel_audit_helper_burden.svg").exists()
    assert (tmp_path / "figures" / "h1l_visual_executor_equivalence_burden.svg").exists()
    assert (tmp_path / "figures" / "h1m_visual_alias_repeat_burden.svg").exists()
    assert (tmp_path / "figures" / "exact_probe_replay_gap.svg").exists()
    assert (tmp_path / "figures" / "exact_probe_replay_focus_gap.svg").exists()
    assert (tmp_path / "figures" / "live_parallel_replay_gap.svg").exists()
    assert (tmp_path / "figures" / "live_replay_focus_gap.svg").exists()
    assert (tmp_path / "figures" / "wave3_live_candidate_replay_gate.svg").exists()
    assert (tmp_path / "figures" / "visual_catalog_live_candidate_replay_gate.svg").exists()
    assert (tmp_path / "figures" / "visual_catalog_argument_hints_live_candidate_replay_gate.svg").exists()
    assert (tmp_path / "figures" / "visual_hard_slice_live_replay_gate.svg").exists()
    assert (tmp_path / "figures" / "visual_hard_slice_stress_live_replay_gate.svg").exists()
    assert (tmp_path / "figures" / "visual_hard_slice_alias_repeat_live_replay_gate.svg").exists()
    assert (tmp_path / "figures" / "visual_hard_slice_alias_transfer_live_replay_gate.svg").exists()
    assert (tmp_path / "figures" / "visual_hard_slice_alias_transfer_oracle_live_replay_gate.svg").exists()
    assert (tmp_path / "figures" / "visual_hard_slice_post_repair_live_replay_gate.svg").exists()
    assert (tmp_path / "figures" / "visual_hard_slice_residual_live_replay_gate.svg").exists()
    assert (tmp_path / "figures" / "visual_hard_slice_component_value_live_replay_gate.svg").exists()
    assert (tmp_path / "figures" / "visual_hard_slice_h1o_live_replay_gate.svg").exists()
    assert (tmp_path / "figures" / "visual_hard_slice_h1p_live_replay_gate.svg").exists()
    assert (tmp_path / "figures" / "h1q_component_label_guard_transfer_gate.svg").exists()
    assert (tmp_path / "figures" / "h1s_component_residual_transfer_gate.svg").exists()
    assert (tmp_path / "figures" / "h1x_v11_breaker_gate.svg").exists()
    assert (tmp_path / "figures" / "h1y_routed_residual_gate.svg").exists()
