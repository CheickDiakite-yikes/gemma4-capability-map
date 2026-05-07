# Tool Contract Summary

- Contracted readiness: 0.96891
- No-directive readiness: 0.96891
- No-directive controller repair/fallback/argument repair: 0.70 / 0.25 / 0.45
- No-directive raw planning clean rate: 0.30

| system_id | controls | readiness | strict | recovered | repair | fallback | arg repair | raw clean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mlx_gemma4_e2b_reasoner_only | none | 0.96891 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 | 1.00 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive | disable_tool_turn_directive | 0.96891 | 1.000 | 1.000 | 0.70 | 0.25 | 0.45 | 0.30 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_argument_repair | disable_argument_repair;disable_tool_turn_directive | 0.83016 | 0.756 | 0.550 | 0.25 | 0.25 | 0.00 | 0.75 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_fallback | disable_controller_fallback;disable_tool_turn_directive | 0.89598 | 0.812 | 0.750 | 0.45 | 0.00 | 0.45 | 0.55 |
| mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_repair | disable_controller_repair;disable_tool_turn_directive | 0.73801 | 0.481 | 0.300 | 0.70 | 0.70 | 0.00 | 0.83 |
