# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_state_tool_selection`
- Baseline exact rate: `0.0`
- Candidate exact rate: `0.3333333333333333`
- Delta exact rate: `0.3333333333333333`
- Baseline executable rate: `0.0`
- Candidate executable rate: `0.0`
- Delta executable rate: `0.0`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| visual_form_target_literal | visual_argument_copying | False | False | False | False | 0 | 0 | 0 | no_tool_call |
| visual_latest_filter_literal | visual_referent_carryover | False | False | None | None | 0 | 1 | 1 | wrong_tool |
| visual_readback_region_literal | visual_referent_carryover | False | True | None | None | 0 | 1 | 1 | exact |
