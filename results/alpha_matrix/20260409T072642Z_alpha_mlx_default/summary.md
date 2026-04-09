# Alpha Matrix Summary

- Run group: `20260409T072642Z`
- Matrix: `alpha_mlx_default`

## Experiments

| Experiment | Status | Pipeline | Track | Backend | Success | Avg latency ms | Runs |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| mlx_e2b_monolith_tool_routing_clean | completed | monolith | tool_routing | mlx | 1.0 | 6756.0 | 10.0 |
| mlx_e2b_modular_full_stack_clean | completed | modular | full_stack | mlx | 0.8 | 3856.5 | 10.0 |

## Probes

| Probe | Status | Backend | Model | Load ms | Device |
| --- | --- | --- | --- | ---: | --- |
| hf_e4b_probe_cached | completed | hf | google/gemma-4-E4B-it | 1597752 | mps |

## Improvements

| Experiment | Success delta | Latency improvement ms |
| --- | ---: | ---: |
| mlx_e2b_monolith_tool_routing_clean | 0.6 | -1759.0 |
| mlx_e2b_modular_full_stack_clean | 0.2 | -489.8 |
