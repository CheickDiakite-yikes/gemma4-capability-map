# Prompt-Contract Probe Candidate Gate Summary

Packet: `/Users/cheickdiakite/Codex/moonie/results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave5_execute_v1`

| contract | exact | executable | delta exact vs no-directive | improved cases | dominant failure | recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| visual_refine_selection_v5 | 0.0 | 0.0 | 0.0 | 0 | no_tool_call | no_probe_gain |

Interpretation:

- `weak_exact_gain` means the candidate recovered at least one exact probe case over no-directive, but remains far below contracted MLX.
- `visual_executable_gain_only` means the candidate recovered the visual executor target without improving exact JSON copy rate.
- Candidates should move to H1i only as mechanism probes, not as assumed replacements for the final tool-turn directive.
