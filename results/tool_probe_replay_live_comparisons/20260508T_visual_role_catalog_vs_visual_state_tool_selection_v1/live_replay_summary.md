# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_state_tool_selection`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog`
- Baseline exact rate: `0.3333333333333333`
- Candidate exact rate: `0.3333333333333333`
- Delta exact rate: `0.0`
- Baseline executable rate: `0.0`
- Candidate executable rate: `1.0`
- Delta executable rate: `1.0`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| visual_form_target_literal | visual_argument_copying | False | False | False | True | 0 | 1 | 1 | executable_paraphrase |
| visual_latest_filter_literal | visual_referent_carryover | False | False | None | None | 1 | 1 | 0 | argument_mismatch |
| visual_readback_region_literal | visual_referent_carryover | True | True | None | None | 1 | 1 | 0 | exact |
