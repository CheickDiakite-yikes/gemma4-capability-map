# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive`
- Baseline exact rate: `0.6666666666666666`
- Candidate exact rate: `0.0`
- Delta exact rate: `-0.6666666666666666`

| case_id | family | baseline exact | candidate exact | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| visual_form_target_literal | visual_argument_copying | False | False | 1 | 0 | -1 | no_tool_call |
| visual_latest_filter_literal | visual_referent_carryover | True | False | 1 | 0 | -1 | no_tool_call |
| visual_readback_region_literal | visual_referent_carryover | True | False | 1 | 0 | -1 | no_tool_call |
