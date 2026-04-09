# Alpha Matrix Summary

- Run group: `20260409T085427Z`
- Matrix: `alpha_mlx_default`

## Experiments

| Experiment | Status | Pipeline | Track | Backend | Success | Avg latency ms | Runs |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| mlx_e2b_monolith_thinking_clean | completed | monolith | thinking | mlx | 0.9 | 2418.4 | 10.0 |
| hf_e2b_monolith_thinking_off_clean | completed | monolith | thinking | hf | 1.0 | 6189.9 | 10.0 |
| hf_e2b_monolith_thinking_on_clean | completed | monolith | thinking | hf | 0.7 | 25545.2 | 10.0 |

## Probes

| Probe | Status | Backend | Model | Load ms | Device |
| --- | --- | --- | --- | ---: | --- |
| hf_e4b_probe_cached | completed | hf | google/gemma-4-E4B-it | 1597752 | mps |

## Improvements

| Experiment | Success delta | Latency improvement ms |
| --- | ---: | ---: |
| mlx_e2b_monolith_thinking_clean | 0.2 | 691.9 |
| hf_e2b_monolith_thinking_off_clean | 0.4 | 10169.9 |
| hf_e2b_monolith_thinking_on_clean | 0.3 | 13911.1 |
