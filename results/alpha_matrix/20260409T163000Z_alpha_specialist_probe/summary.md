# Alpha Matrix Summary

- Run group: `20260409T163000Z`
- Matrix: `alpha_specialist_probe`
- Completed experiments: `2/2`
- Matrix complete: `True`

## Backend Posture

- Recommended local reasoner backend: `mlx`
- HF token present: `True`
- Offline mode enabled: `False`
- Preflight source: `/Users/cheickdiakite/Codex/moonie/results/tables/backend_preflight.json`

## Failure Breakdown

- `arg_mismatch`: 2
- `failed`: 2
- `malformed_call`: 2

## Experiments

| Experiment | Status | Pipeline | Track | Backend | Success | Avg latency ms | Runs |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| hf_e2b_hybrid_retrieval_real_embeddinggemma | completed | hybrid | retrieval | hf | 1.0 | 4973.75 | 8.0 |
| hf_e2b_modular_tool_routing_real_functiongemma | completed | modular | tool_routing | hf | 0.75 | 7833.62 | 8.0 |

## Probes

No probes configured.

## Failing Variants

| Experiment | Variant | Failure tags | Interface reliability |
| --- | --- | --- | ---: |
| hf_e2b_modular_tool_routing_real_functiongemma | tool_009_parallel_context_check_schema_renamed_fields | failed, malformed_call, arg_mismatch | 0.37499999999999994 |
| hf_e2b_modular_tool_routing_real_functiongemma | tool_010_validator_ready_schema_renamed_fields | failed, malformed_call, arg_mismatch | 0.37499999999999994 |
