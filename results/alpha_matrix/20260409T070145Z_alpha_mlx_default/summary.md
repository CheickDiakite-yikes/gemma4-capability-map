# Alpha Matrix Summary

- Run group: `20260409T070145Z`
- Matrix: `alpha_mlx_default`

## Experiments

| Experiment | Status | Pipeline | Track | Backend | Success | Avg latency ms | Runs |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| mlx_e2b_monolith_thinking_clean | completed | monolith | thinking | mlx | 0.4 | 2520.0 | 10.0 |
| mlx_e2b_monolith_tool_routing_clean | completed | monolith | tool_routing | mlx | 0.0 | 3147.4 | 10.0 |
| mlx_e2b_hybrid_retrieval_clean | completed | hybrid | retrieval | mlx | 1.0 | 2290.7 | 10.0 |
| mlx_e2b_modular_full_stack_clean | completed | modular | full_stack | mlx | 0.0 | 2290.9 | 10.0 |
| hf_e2b_monolith_thinking_off_clean | completed | monolith | thinking | hf | 0.3 | 15612.5 | 10.0 |
| hf_e2b_monolith_thinking_on_clean | completed | monolith | thinking | hf | 0.3 | 21439.3 | 10.0 |

## Probes

| Probe | Status | Backend | Model | Load ms | Device |
| --- | --- | --- | --- | ---: | --- |
| hf_e4b_probe_cached | completed | hf | google/gemma-4-E4B-it | 1597752 | mps |

## Improvements

| Experiment | Success delta | Latency improvement ms |
| --- | ---: | ---: |
| mlx_e2b_monolith_thinking_clean | 0.0 | 131.5 |
| mlx_e2b_monolith_tool_routing_clean | 0.0 | -754.0 |
| mlx_e2b_hybrid_retrieval_clean | 0.0 | -778.0 |
| mlx_e2b_modular_full_stack_clean | 0.0 | -821.8 |
| hf_e2b_monolith_thinking_off_clean | 0.0 | -446.4 |
