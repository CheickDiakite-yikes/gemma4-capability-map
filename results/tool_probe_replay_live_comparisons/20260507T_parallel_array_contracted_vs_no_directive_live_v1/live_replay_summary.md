# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive`
- Baseline exact rate: `1.0`
- Candidate exact rate: `0.0`
- Delta exact rate: `-1.0`

| case_id | family | baseline exact | candidate exact | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| parallel_audit_array_literal | parallel_tool_calling | True | False | 2 | 0 | -2 | no_tool_call |
