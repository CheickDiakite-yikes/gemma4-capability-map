# Alpha Matrix Summary

- Run group: `20260409T134126Z`
- Matrix: `alpha_drift_probe`
- Completed experiments: `4/4`
- Matrix complete: `True`

## Failure Breakdown

- `answer_mismatch`: 1
- `failed`: 1
- `image_grounding_miss`: 1
- `malformed_call`: 1

## Experiments

| Experiment | Status | Pipeline | Track | Backend | Success | Avg latency ms | Runs |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| mlx_e2b_monolith_thinking_variants | completed | monolith | thinking | mlx | 0.9166666666666666 | 2122.17 | 12.0 |
| mlx_e2b_monolith_tool_routing_variants | completed | monolith | tool_routing | mlx | 1.0 | 4056.5 | 8.0 |
| mlx_e2b_hybrid_retrieval_variants | completed | hybrid | retrieval | mlx | 1.0 | 2883.62 | 8.0 |
| mlx_e2b_modular_full_stack_variants | completed | modular | full_stack | mlx | 1.0 | 2648.91 | 11.0 |

## Probes

No probes configured.

## Improvements

| Experiment | Success delta | Latency improvement ms |
| --- | ---: | ---: |
| mlx_e2b_monolith_thinking_variants | 0.0 | 365.83 |
| mlx_e2b_monolith_tool_routing_variants | 0.0 | -154.38 |
| mlx_e2b_hybrid_retrieval_variants | 0.0 | -254.24 |
| mlx_e2b_modular_full_stack_variants | 0.0 | -322.18 |

## Failing Variants

| Experiment | Variant | Failure tags | Interface reliability |
| --- | --- | --- | ---: |
| mlx_e2b_monolith_thinking_variants | think_006_screenshot_security_language_fr | failed, answer_mismatch, image_grounding_miss | 0.0 |
