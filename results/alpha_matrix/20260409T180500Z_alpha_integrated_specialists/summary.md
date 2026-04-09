# Alpha Matrix Summary

- Run group: `20260409T180500Z`
- Matrix: `alpha_integrated_specialists`
- Completed experiments: `9/9`
- Matrix complete: `True`

## Backend Posture

- Recommended local reasoner backend: `mlx`
- HF token present: `True`
- Offline mode enabled: `False`
- Preflight source: `/Users/cheickdiakite/Codex/moonie/results/tables/backend_preflight.json`

## Failure Breakdown

- `answer_mismatch`: 4
- `answer_missing`: 2
- `arg_mismatch`: 3
- `failed`: 9
- `generation_truncated`: 2
- `image_grounding_miss`: 4
- `thinking_overflow`: 2
- `wrong_tool`: 2

## Experiments

| Experiment | Status | Pipeline | Track | Backend | Success | Avg latency ms | Runs |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| mlx_e2b_monolith_thinking_clean | completed | monolith | thinking | mlx | 0.8333333333333334 | 659.42 | 12.0 |
| mlx_e2b_monolith_tool_routing_clean | completed | monolith | tool_routing | mlx | 0.9166666666666666 | 1891.0 | 12.0 |
| mlx_e2b_hybrid_retrieval_clean | completed | hybrid | retrieval | mlx | 1.0 | 731.67 | 12.0 |
| mlx_e2b_modular_full_stack_clean | completed | modular | full_stack | mlx | 1.0 | 1044.17 | 12.0 |
| hf_e2b_monolith_thinking_off_clean | completed | monolith | thinking | hf | 0.9166666666666666 | 1518.92 | 12.0 |
| hf_e2b_monolith_thinking_on_clean | completed | monolith | thinking | hf | 0.75 | 19631.0 | 12.0 |
| hf_e2b_hybrid_retrieval_real_embeddinggemma_clean | completed | hybrid | retrieval | hf | 1.0 | 3739.08 | 12.0 |
| hf_e2b_modular_tool_routing_real_functiongemma_clean | completed | modular | tool_routing | hf | 0.8333333333333334 | 13786.33 | 12.0 |
| hf_e2b_modular_full_stack_real_specialists_clean | completed | modular | full_stack | hf | 1.0 | 28887.5 | 2.0 |

## Probes

| Probe | Status | Backend | Model | Load ms | Device |
| --- | --- | --- | --- | ---: | --- |
| hf_e4b_probe_cached | completed | hf | google/gemma-4-E4B-it | 1597752 | mps |

## Improvements

| Experiment | Success delta | Latency improvement ms |
| --- | ---: | ---: |
| mlx_e2b_monolith_thinking_clean | -0.1667 | 1794.68 |
| mlx_e2b_monolith_tool_routing_clean | -0.0833 | 2298.7 |
| mlx_e2b_hybrid_retrieval_clean | 0.0 | 1930.03 |
| mlx_e2b_modular_full_stack_clean | 0.0 | 1595.53 |
| hf_e2b_monolith_thinking_off_clean | -0.0833 | 4778.18 |
| hf_e2b_monolith_thinking_on_clean | -0.25 | 13740.2 |

## Failing Variants

| Experiment | Variant | Failure tags | Interface reliability |
| --- | --- | --- | ---: |
| mlx_e2b_monolith_thinking_clean | think_011_incident_screenshot_toggle_clean | failed, answer_mismatch, image_grounding_miss | 0.0 |
| mlx_e2b_monolith_thinking_clean | think_012_billing_invoice_lock_clean | failed, answer_mismatch, image_grounding_miss | 0.0 |
| mlx_e2b_monolith_tool_routing_clean | tool_012_billing_patch_record_clean | failed, arg_mismatch | 0.625 |
| hf_e2b_monolith_thinking_off_clean | think_011_incident_screenshot_toggle_clean | failed, answer_mismatch, image_grounding_miss | 0.0 |
| hf_e2b_monolith_thinking_on_clean | think_006_screenshot_security_clean | failed, answer_mismatch, image_grounding_miss | 0.0 |
| hf_e2b_monolith_thinking_on_clean | think_007_doc_image_summary_clean | failed, answer_missing, generation_truncated, thinking_overflow | 0.0 |
| hf_e2b_monolith_thinking_on_clean | think_011_incident_screenshot_toggle_clean | failed, answer_missing, generation_truncated, thinking_overflow | 0.0 |
| hf_e2b_modular_tool_routing_real_functiongemma_clean | tool_008_patch_record_clean | failed, wrong_tool, arg_mismatch | 0.25 |
| hf_e2b_modular_tool_routing_real_functiongemma_clean | tool_012_billing_patch_record_clean | failed, wrong_tool, arg_mismatch | 0.25 |
