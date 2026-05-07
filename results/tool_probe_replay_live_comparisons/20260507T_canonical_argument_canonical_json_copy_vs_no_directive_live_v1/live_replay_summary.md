# Live Exact Replay Comparison

- Baseline system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive`
- Candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_canonical_json_copy`
- Baseline exact rate: `0.0`
- Candidate exact rate: `0.0`
- Delta exact rate: `0.0`
- Baseline executable rate: `None`
- Candidate executable rate: `None`
- Delta executable rate: `None`

| case_id | family | baseline exact | candidate exact | baseline executable | candidate executable | baseline calls | candidate calls | delta calls | candidate failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| api_form_issue_fetch | api_canonicalization | False | False | None | None | 1 | 1 | 0 | argument_mismatch |
| api_invoice_lock_hold_update | api_canonicalization | False | False | None | None | 1 | 0 | -1 | no_tool_call |
| cli_invoice_lock_hyphen_query | cli_canonicalization | False | False | None | None | 1 | 1 | 0 | argument_mismatch |
| cli_phone_patch_latest_only | cli_patch_copying | False | False | None | None | 1 | 0 | -1 | no_tool_call |
