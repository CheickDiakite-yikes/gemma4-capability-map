# Visual Hard Slice Candidate Gates

- packet_run_id: `20260509T_visual_hard_slice_v5_execute_v1`
- created_at: `2026-05-09T11:57:55.491493+00:00`
- case_count: `8`
- contracted_system_id: `mlx_gemma4_e2b_reasoner_only`
- no_directive_system_id: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive`

| System | Exact | Executable | Delta Exact vs No Directive | Delta Exec vs No Directive | Failure | Gate |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| mlx_gemma4_e2b_reasoner_only | 1.000 | 1.000 | 0.875 | 0.875 | exact | contracted_reference |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | 0.125 | 0.125 | 0.000 | 0.000 | no_tool_call | no_directive_reference |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog | 0.375 | 0.375 | 0.250 | 0.250 | argument_mismatch | hard_slice_improved_vs_no_directive |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints | 0.750 | 0.875 | 0.625 | 0.750 | exact | hard_slice_improved_vs_no_directive |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_split_selector_hints | 0.625 | 0.750 | 0.500 | 0.625 | exact | hard_slice_improved_vs_no_directive |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints | 0.750 | 1.000 | 0.625 | 0.875 | exact | hard_slice_improved_vs_no_directive |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_literal_targets | 0.625 | 0.875 | 0.500 | 0.750 | exact | hard_slice_improved_vs_no_directive |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_literal_guard | 0.375 | 0.500 | 0.250 | 0.375 | exact | hard_slice_improved_vs_no_directive |
