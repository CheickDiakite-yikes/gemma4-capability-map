# Visual Tool-Choice Diagnostics

- Packet count: `2`
- Visual case rows: `6`
- Diagnosis counts: `{'exact': 2, 'tool_ok_argument_alias_executable': 1, 'visual_tool_initiation_missing': 1, 'wrong_visual_tool_selection': 2}`

| packet | system | case | expected | actual | failure | diagnosis | next diagnostic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20260507T_visual_state_visual_tool_initiation_live_execute_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_tool_initiation | visual_form_target_literal | extract_layout | extract_layout | executable_paraphrase | tool_ok_argument_alias_executable | tighten canonical visual argument copy without losing executable aliases |
| 20260507T_visual_state_visual_tool_initiation_live_execute_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_tool_initiation | visual_latest_filter_literal | refine_selection | extract_layout | wrong_tool | wrong_visual_tool_selection | separate latest-selection filtering from locating/readback; actual first tool was extract_layout |
| 20260507T_visual_state_visual_tool_initiation_live_execute_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_tool_initiation | visual_readback_region_literal | read_region_text | read_region_text | exact | exact | no further diagnostic needed |
| 20260508T_visual_state_tool_selection_live_execute_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_state_tool_selection | visual_form_target_literal | extract_layout |  | no_tool_call | visual_tool_initiation_missing | preserve visual tool initiation before tuning selectors |
| 20260508T_visual_state_tool_selection_live_execute_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_state_tool_selection | visual_latest_filter_literal | refine_selection | extract_layout | wrong_tool | wrong_visual_tool_selection | separate latest-selection filtering from locating/readback; actual first tool was extract_layout |
| 20260508T_visual_state_tool_selection_live_execute_v1 | mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_state_tool_selection | visual_readback_region_literal | read_region_text | read_region_text | exact | exact | no further diagnostic needed |
