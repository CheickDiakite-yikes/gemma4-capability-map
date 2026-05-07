# Prompt-Contract Probe Candidate Gate Summary

Packet: `/Users/cheickdiakite/Codex/moonie/results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1`

| contract | exact | executable | delta exact vs no-directive | improved cases | dominant failure | recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| schema_anchor_v1 | 0.125 | 0.0 | 0.125 | 1 | argument_mismatch | weak_exact_gain |
| literal_argument_guard_v1 | 0.0 | 1.0 | 0.0 | 1 | no_tool_call | visual_executable_gain_only |
| tool_required_parallel_v1 | 0.0 | 1.0 | 0.0 | 1 | no_tool_call | visual_executable_gain_only |

Interpretation:

- `weak_exact_gain` means the candidate recovered at least one exact probe case over no-directive, but remains far below contracted MLX.
- `visual_executable_gain_only` means the candidate recovered the visual executor target without improving exact JSON copy rate.
- Candidates should move to H1i only as mechanism probes, not as assumed replacements for the final tool-turn directive.
