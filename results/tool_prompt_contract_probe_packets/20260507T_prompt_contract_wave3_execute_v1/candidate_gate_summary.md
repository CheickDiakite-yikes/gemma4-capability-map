# Prompt-Contract Probe Candidate Gate Summary

Packet: `/Users/cheickdiakite/Codex/moonie/results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave3_execute_v1`

| contract | exact | executable | delta exact vs no-directive | improved cases | dominant failure | recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| canonical_json_copy_v3 | 0.125 | 0.0 | 0.125 | 1 | no_tool_call | weak_exact_gain |
| visual_tool_initiation_v3 | 0.125 | 1.0 | 0.125 | 2 | no_tool_call | weak_exact_gain |
| parallel_two_call_array_v3 | 0.0 | 0.0 | 0.0 | 0 | no_tool_call | no_probe_gain |

Interpretation:

- `weak_exact_gain` means the candidate recovered at least one exact probe case over no-directive, but remains far below contracted MLX.
- `visual_executable_gain_only` means the candidate recovered the visual executor target without improving exact JSON copy rate.
- Candidates should move to H1i only as mechanism probes, not as assumed replacements for the final tool-turn directive.
