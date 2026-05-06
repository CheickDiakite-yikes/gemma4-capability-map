# Tool Contract Summary

- Contracted readiness: 0.97936
- No-directive readiness: 0.97936
- No-directive controller repair/fallback/argument repair: 0.70 / 0.20 / 0.50
- No-directive raw planning clean rate: 0.30

| system_id | controls | readiness | strict | recovered | repair | fallback | arg repair | raw clean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mlx_gemma4_e2b_reasoner_only | none | 0.97936 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 | 1.00 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | disable_tool_turn_directive | 0.97936 | 1.000 | 1.000 | 0.70 | 0.20 | 0.50 | 0.30 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_argument_repair | disable_argument_repair;disable_tool_turn_directive | 0.82036 | 0.713 | 0.500 | 0.20 | 0.20 | 0.00 | 0.80 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_fallback | disable_controller_fallback;disable_tool_turn_directive | 0.92104 | 0.850 | 0.800 | 0.50 | 0.00 | 0.50 | 0.50 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_repair | disable_controller_repair;disable_tool_turn_directive | 0.73818 | 0.475 | 0.300 | 0.50 | 0.50 | 0.00 | 0.89 |
