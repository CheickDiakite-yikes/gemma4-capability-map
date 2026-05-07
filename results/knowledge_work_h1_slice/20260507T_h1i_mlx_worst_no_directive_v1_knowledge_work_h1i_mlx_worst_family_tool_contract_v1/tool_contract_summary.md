# Tool Contract Summary

- Contracted readiness: 0.97710
- No-directive readiness: 0.97710
- No-directive controller repair/fallback/argument repair: 1.00 / 0.50 / 0.50
- No-directive raw planning clean rate: 0.00

| system_id | controls | readiness | strict | recovered | repair | fallback | arg repair | raw clean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mlx_gemma4_e2b_reasoner_only | none | 0.97710 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 | 1.00 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | disable_tool_turn_directive | 0.97710 | 1.000 | 1.000 | 1.00 | 0.50 | 0.50 | 0.00 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_argument_repair | disable_argument_repair;disable_tool_turn_directive | 0.81220 | 0.719 | 0.500 | 0.50 | 0.50 | 0.00 | 0.50 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_fallback | disable_controller_fallback;disable_tool_turn_directive | 0.83125 | 0.625 | 0.500 | 0.50 | 0.00 | 0.50 | 0.50 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_repair | disable_controller_repair;disable_tool_turn_directive | 0.64697 | 0.297 | 0.000 | 1.25 | 1.25 | 0.00 | 0.72 |
