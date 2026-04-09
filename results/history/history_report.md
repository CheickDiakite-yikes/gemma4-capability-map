# History Report

- Generated: `2026-04-09T19:17:09.283825+00:00`
- Logged experiments: `87`
- Completed experiments: `80`
- Blocked experiments: `1`
- Failed experiments: `6`
- Run groups: `31`
- Matrix snapshots: `25`

## Canonical Matrix Snapshots

| Matrix | Run group | Complete | Completed / Expected | Avg success | Strict | Recovered | Readiness |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| alpha_drift_probe | 20260409T134126Z | True | 4/4 | 0.9792 | 0.0 | 0.0 | 0.0 |
| alpha_integrated_specialists | 20260409T180500Z | True | 9/9 | 0.9167 | 0.0 | 0.0 | 0.0 |
| alpha_mlx_default | 20260409T090608Z | True | 6/6 | 0.9667 | 0.0 | 0.0 | 0.0 |
| alpha_real_world | 20260409T210500Z | True | 4/4 | 0.5312 | 0.5625 | 0.5938 | 0.5971 |
| alpha_specialist_drift | 20260409T190500Z | True | 3/3 | 0.8625 | 0.0 | 0.0 | 0.0 |
| alpha_specialist_probe | 20260409T163000Z | True | 2/2 | 0.875 | 0.0 | 0.0 | 0.0 |

## Latest Run Groups By Matrix

| Matrix | Run group | Complete | Completed / Expected | Blocked | Failed |
| --- | --- | --- | ---: | ---: | ---: |
| alpha_drift_probe | 20260409T133025Z | True | 4/4 | 0 | 0 |
| alpha_integrated_specialists | 20260409T180500Z | True | 9/9 | 0 | 0 |
| alpha_mlx_default | 20260409T065257Z | False | 0/1 | 0 | 1 |
| alpha_real_world | 20260409T210500Z | True | 4/4 | 0 | 0 |
| alpha_specialist_drift | 20260409T190500Z | True | 3/3 | 0 | 0 |
| alpha_specialist_probe | 20260409T120000Z | True | 1/1 | 0 | 0 |

## Latest Completed Experiments

| Experiment | Run group | Track | Backend | Success | Strict | Recovered | Readiness | Avg latency ms | Runs |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hf_e2b_hybrid_retrieval_real_embeddinggemma | 20260409T163000Z | retrieval | hf | 1.0 | 0.0 | 0.0 | 0.0 | 4973.75 | 8.0 |
| hf_e2b_hybrid_retrieval_real_embeddinggemma_clean | 20260409T180500Z | retrieval | hf | 1.0 | 0.0 | 0.0 | 0.0 | 3739.08 | 12.0 |
| hf_e2b_hybrid_retrieval_real_embeddinggemma_variants | 20260409T190500Z | retrieval | hf | 0.9 | 0.0 | 0.0 | 0.0 | 2642.1 | 20.0 |
| hf_e2b_modular_full_stack_real_specialists_clean | 20260409T180500Z | full_stack | hf | 1.0 | 0.0 | 0.0 | 0.0 | 28887.5 | 2.0 |
| hf_e2b_modular_full_stack_real_specialists_variants | 20260409T190500Z | full_stack | hf | 0.9375 | 0.0 | 0.0 | 0.0 | 13259.69 | 16.0 |
| hf_e2b_modular_tool_routing_real_functiongemma | 20260409T172000Z | tool_routing | hf | 1.0 | 0.0 | 0.0 | 0.0 | 7841.25 | 8.0 |
| hf_e2b_modular_tool_routing_real_functiongemma_clean | 20260409T180500Z | tool_routing | hf | 0.8333333333333334 | 0.0 | 0.0 | 0.0 | 13786.33 | 12.0 |
| hf_e2b_modular_tool_routing_real_functiongemma_variants | 20260409T190500Z | tool_routing | hf | 0.75 | 0.0 | 0.0 | 0.0 | 5198.25 | 16.0 |
| hf_e2b_monolith_thinking_multilingual_multimodal | 20260409T120000Z | thinking | hf | 1.0 | 0.0 | 0.0 | 0.0 | 10545.25 | 4.0 |
| hf_e2b_monolith_thinking_off_clean | 20260409T180500Z | thinking | hf | 0.9166666666666666 | 0.0 | 0.0 | 0.0 | 1518.92 | 12.0 |
| hf_e2b_monolith_thinking_on_clean | 20260409T180500Z | thinking | hf | 0.75 | 0.0 | 0.0 | 0.0 | 19631.0 | 12.0 |
| hf_e2b_real_world_full_stack_variants | 20260409T210500Z | full_stack | hf | 0.75 | 0.75 | 1.0 | 0.8696428571428572 | 12115.68 | 28.0 |
| hf_e2b_real_world_retrieval_variants | 20260409T210500Z | retrieval | hf | 0.875 | 1.0 | 0.875 | 0.88125 | 2855.44 | 16.0 |
| hf_e2b_real_world_routing_variants | 20260409T210500Z | tool_routing | hf | 0.5 | 0.5 | 0.5 | 0.6375 | 7339.0 | 16.0 |
| hf_e2b_real_world_thinking_variants | 20260409T210500Z | thinking | hf | 0.0 | 0.0 | 0.0 | 0.0 | 1016.75 | 4.0 |
| mlx_e2b_hybrid_retrieval_clean | 20260409T180500Z | retrieval | mlx | 1.0 | 0.0 | 0.0 | 0.0 | 731.67 | 12.0 |
| mlx_e2b_hybrid_retrieval_variants | 20260409T134126Z | retrieval | mlx | 1.0 | 0.0 | 0.0 | 0.0 | 2883.62 | 8.0 |
| mlx_e2b_modular_full_stack_clean | 20260409T180500Z | full_stack | mlx | 1.0 | 0.0 | 0.0 | 0.0 | 1044.17 | 12.0 |
| mlx_e2b_modular_full_stack_variants | 20260409T134126Z | full_stack | mlx | 1.0 | 0.0 | 0.0 | 0.0 | 2648.91 | 11.0 |
| mlx_e2b_monolith_thinking_clean | 20260409T180500Z | thinking | mlx | 0.8333333333333334 | 0.0 | 0.0 | 0.0 | 659.42 | 12.0 |
| mlx_e2b_monolith_thinking_variants | 20260409T134126Z | thinking | mlx | 0.9166666666666666 | 0.0 | 0.0 | 0.0 | 2122.17 | 12.0 |
| mlx_e2b_monolith_tool_routing_clean | 20260409T180500Z | tool_routing | mlx | 0.9166666666666666 | 0.0 | 0.0 | 0.0 | 1891.0 | 12.0 |
| mlx_e2b_monolith_tool_routing_variants | 20260409T134126Z | tool_routing | mlx | 1.0 | 0.0 | 0.0 | 0.0 | 4056.5 | 8.0 |

## Best Completed Experiments

| Experiment | Run group | Track | Backend | Success | Strict | Recovered | Readiness | Avg latency ms |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| hf_e2b_hybrid_retrieval_real_embeddinggemma | 20260409T163000Z | retrieval | hf | 1.0 | 0.0 | 0.0 | 0.0 | 4973.75 |
| hf_e2b_hybrid_retrieval_real_embeddinggemma_clean | 20260409T180500Z | retrieval | hf | 1.0 | 0.0 | 0.0 | 0.0 | 3739.08 |
| hf_e2b_hybrid_retrieval_real_embeddinggemma_variants | 20260409T190500Z | retrieval | hf | 0.9 | 0.0 | 0.0 | 0.0 | 2642.1 |
| hf_e2b_modular_full_stack_real_specialists_clean | 20260409T180500Z | full_stack | hf | 1.0 | 0.0 | 0.0 | 0.0 | 28887.5 |
| hf_e2b_modular_full_stack_real_specialists_variants | 20260409T190500Z | full_stack | hf | 0.9375 | 0.0 | 0.0 | 0.0 | 13259.69 |
| hf_e2b_modular_tool_routing_real_functiongemma | 20260409T172000Z | tool_routing | hf | 1.0 | 0.0 | 0.0 | 0.0 | 7841.25 |
| hf_e2b_modular_tool_routing_real_functiongemma_clean | 20260409T180500Z | tool_routing | hf | 0.8333333333333334 | 0.0 | 0.0 | 0.0 | 13786.33 |
| hf_e2b_modular_tool_routing_real_functiongemma_variants | 20260409T190500Z | tool_routing | hf | 0.75 | 0.0 | 0.0 | 0.0 | 5198.25 |
| hf_e2b_monolith_thinking_multilingual_multimodal | 20260409T120000Z | thinking | hf | 1.0 | 0.0 | 0.0 | 0.0 | 10545.25 |
| hf_e2b_monolith_thinking_off_clean | 20260409T085427Z | thinking | hf | 1.0 | 0.0 | 0.0 | 0.0 | 6189.9 |
| hf_e2b_monolith_thinking_on_clean | 20260409T090608Z | thinking | hf | 1.0 | 0.0 | 0.0 | 0.0 | 33371.2 |
| hf_e2b_real_world_full_stack_variants | 20260409T210500Z | full_stack | hf | 0.75 | 0.75 | 1.0 | 0.8696428571428572 | 12115.68 |
| hf_e2b_real_world_retrieval_variants | 20260409T200500Z | retrieval | hf | 0.9166666666666666 | 0.0 | 0.0 | 0.0 | 2430.42 |
| hf_e2b_real_world_routing_variants | 20260409T200500Z | tool_routing | hf | 0.6666666666666666 | 0.0 | 0.0 | 0.0 | 7218.25 |
| hf_e2b_real_world_thinking_variants | 20260409T203500Z | thinking | hf | 0.0 | 0.0 | 0.0 | 0.0 | 864.25 |
| mlx_e2b_hybrid_retrieval_clean | 20260409T180500Z | retrieval | mlx | 1.0 | 0.0 | 0.0 | 0.0 | 731.67 |
| mlx_e2b_hybrid_retrieval_variants | 20260409T133025Z | retrieval | mlx | 1.0 | 0.0 | 0.0 | 0.0 | 2629.38 |
| mlx_e2b_modular_full_stack_clean | 20260409T180500Z | full_stack | mlx | 1.0 | 0.0 | 0.0 | 0.0 | 1044.17 |
| mlx_e2b_modular_full_stack_variants | 20260409T133025Z | full_stack | mlx | 1.0 | 0.0 | 0.0 | 0.0 | 2326.73 |
| mlx_e2b_monolith_thinking_clean | 20260409T091827Z | thinking | mlx | 1.0 | 0.0 | 0.0 | 0.0 | 2260.4 |
| mlx_e2b_monolith_thinking_variants | 20260409T134126Z | thinking | mlx | 0.9166666666666666 | 0.0 | 0.0 | 0.0 | 2122.17 |
| mlx_e2b_monolith_tool_routing_clean | 20260409T090608Z | tool_routing | mlx | 1.0 | 0.0 | 0.0 | 0.0 | 3735.9 |
| mlx_e2b_monolith_tool_routing_variants | 20260409T133914Z | tool_routing | mlx | 1.0 | 0.0 | 0.0 | 0.0 | 3902.12 |

## Recent Run Groups

| Run group | Matrix | Completed | Blocked | Failed | Avg success | Strict | Recovered | Readiness | Avg latency ms | Complete snapshot |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20260409T210500Z | alpha_real_world | 4 | 0 | 0 | 0.5312 | 0.5625 | 0.5938 | 0.5971 | 5831.72 | True |
| 20260409T203500Z | alpha_real_world | 2 | 0 | 0 | 0.4375 | 0.5 | 0.4375 | 0.4406 | 2041.88 | unknown |
| 20260409T200500Z | alpha_real_world | 2 | 0 | 0 | 0.7917 | 0.0 | 0.0 | 0.0 | 4824.34 | unknown |
| 20260409T190500Z | alpha_specialist_drift | 3 | 0 | 0 | 0.8625 | 0.0 | 0.0 | 0.0 | 7033.35 | True |
| 20260409T180500Z | alpha_integrated_specialists | 9 | 0 | 0 | 0.9167 | 0.0 | 0.0 | 0.0 | 7987.68 | True |
| 20260409T172000Z | alpha_specialist_probe | 1 | 0 | 0 | 1.0 | 0.0 | 0.0 | 0.0 | 7841.25 | True |
| 20260409T170500Z | alpha_specialist_probe | 1 | 0 | 0 | 1.0 | 0.0 | 0.0 | 0.0 | 8180.5 | True |
| 20260409T163000Z | alpha_specialist_probe | 2 | 0 | 0 | 0.875 | 0.0 | 0.0 | 0.0 | 6403.68 | True |
| 20260409T162500Z | alpha_specialist_probe | 0 | 1 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | True |
| 20260409T161500Z | alpha_specialist_probe | 0 | 0 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | False |
| 20260409T160700Z | alpha_specialist_probe | 0 | 0 | 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | False |
| 20260409T134126Z | alpha_drift_probe | 4 | 0 | 0 | 0.9792 | 0.0 | 0.0 | 0.0 | 2927.8 | True |

## Recent Improvements

| Experiment | Current run | Previous run | Success delta | Latency improvement ms |
| --- | --- | --- | ---: | ---: |
| hf_e2b_modular_tool_routing_real_functiongemma | 20260409T170500Z | 20260409T163000Z | 0.25 | -346.88 |
| hf_e2b_modular_tool_routing_real_functiongemma | 20260409T172000Z | 20260409T170500Z | 0.0 | 339.25 |
| mlx_e2b_monolith_thinking_clean | 20260409T180500Z | 20260409T091910Z | -0.1667 | 1794.68 |
| mlx_e2b_monolith_tool_routing_clean | 20260409T180500Z | 20260409T091910Z | -0.0833 | 2298.7 |
| mlx_e2b_hybrid_retrieval_clean | 20260409T180500Z | 20260409T091910Z | 0.0 | 1930.03 |
| mlx_e2b_modular_full_stack_clean | 20260409T180500Z | 20260409T090608Z | 0.0 | 1595.53 |
| hf_e2b_monolith_thinking_off_clean | 20260409T180500Z | 20260409T090608Z | -0.0833 | 4778.18 |
| hf_e2b_monolith_thinking_on_clean | 20260409T180500Z | 20260409T090608Z | -0.25 | 13740.2 |
| hf_e2b_real_world_retrieval_variants | 20260409T203500Z | 20260409T200500Z | -0.0417 | -789.08 |
| hf_e2b_real_world_thinking_variants | 20260409T210500Z | 20260409T203500Z | 0.0 | -152.5 |
| hf_e2b_real_world_retrieval_variants | 20260409T210500Z | 20260409T203500Z | 0.0 | 364.06 |
| hf_e2b_real_world_routing_variants | 20260409T210500Z | 20260409T200500Z | -0.1667 | -120.75 |
