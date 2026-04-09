# Alpha Matrix Summary

- Run group: `20260409T074743Z`
- Matrix: `alpha_mlx_default`

## Experiments

| Experiment | Status | Pipeline | Track | Backend | Success | Avg latency ms | Runs |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| mlx_e2b_monolith_thinking_clean | completed | monolith | thinking | mlx | 0.7 | 3110.3 | 10.0 |
| mlx_e2b_monolith_tool_routing_clean | completed | monolith | tool_routing | mlx | 1.0 | 4468.9 | 10.0 |
| mlx_e2b_hybrid_retrieval_clean | completed | hybrid | retrieval | mlx | 1.0 | 2504.0 | 10.0 |
| mlx_e2b_modular_full_stack_clean | completed | modular | full_stack | mlx | 1.0 | 3314.4 | 10.0 |
| hf_e2b_monolith_thinking_off_clean | completed | monolith | thinking | hf | 0.6 | 16359.8 | 10.0 |
| hf_e2b_monolith_thinking_on_clean | completed | monolith | thinking | hf | 0.4 | 39456.3 | 10.0 |

## Probes

| Probe | Status | Backend | Model | Load ms | Device |
| --- | --- | --- | --- | ---: | --- |
| hf_e4b_probe_cached | completed | hf | google/gemma-4-E4B-it | 1597752 | mps |
