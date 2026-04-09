# Alpha Matrix Summary

- Run group: `20260409T090608Z`
- Matrix: `alpha_mlx_default`

## Failure Breakdown

- `answer_mismatch`: 2
- `failed`: 2
- `image_grounding_miss`: 2

## Experiments

| Experiment | Status | Pipeline | Track | Backend | Success | Avg latency ms | Runs |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| mlx_e2b_monolith_thinking_clean | completed | monolith | thinking | mlx | 0.8 | 2371.8 | 10.0 |
| mlx_e2b_monolith_tool_routing_clean | completed | monolith | tool_routing | mlx | 1.0 | 3735.9 | 10.0 |
| mlx_e2b_hybrid_retrieval_clean | completed | hybrid | retrieval | mlx | 1.0 | 2169.4 | 10.0 |
| mlx_e2b_modular_full_stack_clean | completed | modular | full_stack | mlx | 1.0 | 2639.7 | 10.0 |
| hf_e2b_monolith_thinking_off_clean | completed | monolith | thinking | hf | 1.0 | 6297.1 | 10.0 |
| hf_e2b_monolith_thinking_on_clean | completed | monolith | thinking | hf | 1.0 | 33371.2 | 10.0 |

## Probes

| Probe | Status | Backend | Model | Load ms | Device |
| --- | --- | --- | --- | ---: | --- |
| hf_e4b_probe_cached | completed | hf | google/gemma-4-E4B-it | 1597752 | mps |

## Improvements

| Experiment | Success delta | Latency improvement ms |
| --- | ---: | ---: |
| mlx_e2b_monolith_thinking_clean | -0.1 | 46.6 |
| mlx_e2b_monolith_tool_routing_clean | 0.0 | 733.0 |
| mlx_e2b_hybrid_retrieval_clean | 0.0 | 334.6 |
| mlx_e2b_modular_full_stack_clean | 0.0 | 674.7 |
| hf_e2b_monolith_thinking_off_clean | 0.0 | -107.2 |
| hf_e2b_monolith_thinking_on_clean | 0.3 | -7826.0 |
