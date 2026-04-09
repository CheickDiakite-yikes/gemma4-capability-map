# Alpha Matrix Summary

- Run group: `20260409T190500Z`
- Matrix: `alpha_specialist_drift`
- Completed experiments: `3/3`
- Matrix complete: `True`

## Backend Posture

- Recommended local reasoner backend: `mlx`
- HF token present: `True`
- Offline mode enabled: `False`
- Preflight source: `/Users/cheickdiakite/Codex/moonie/results/tables/backend_preflight.json`

## Failure Breakdown

- `answer_mismatch`: 3
- `arg_mismatch`: 6
- `failed`: 7
- `image_grounding_miss`: 1
- `wrong_tool`: 6

## Experiments

| Experiment | Status | Pipeline | Track | Backend | Success | Avg latency ms | Runs |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| hf_e2b_hybrid_retrieval_real_embeddinggemma_variants | completed | hybrid | retrieval | hf | 0.9 | 2642.1 | 20.0 |
| hf_e2b_modular_tool_routing_real_functiongemma_variants | completed | modular | tool_routing | hf | 0.75 | 5198.25 | 16.0 |
| hf_e2b_modular_full_stack_real_specialists_variants | completed | modular | full_stack | hf | 0.9375 | 13259.69 | 16.0 |

## Probes

No probes configured.

## Failing Variants

| Experiment | Variant | Failure tags | Interface reliability |
| --- | --- | --- | ---: |
| hf_e2b_hybrid_retrieval_real_embeddinggemma_variants | retr_011_approval_policy_language_fr | failed, answer_mismatch | 0.0 |
| hf_e2b_hybrid_retrieval_real_embeddinggemma_variants | retr_012_rollout_toggle_multimodal_context_long_history | failed, answer_mismatch, image_grounding_miss | 0.0 |
| hf_e2b_modular_tool_routing_real_functiongemma_variants | tool_012_billing_patch_record_clean | failed, wrong_tool, arg_mismatch | 0.25 |
| hf_e2b_modular_tool_routing_real_functiongemma_variants | tool_012_billing_patch_record_language_code_switch | failed, wrong_tool, arg_mismatch | 0.25 |
| hf_e2b_modular_tool_routing_real_functiongemma_variants | tool_012_billing_patch_record_schema_renamed_fields | failed, wrong_tool, arg_mismatch | 0.25 |
| hf_e2b_modular_tool_routing_real_functiongemma_variants | tool_012_billing_patch_record_context_irrelevant_tool_output | failed, wrong_tool, arg_mismatch | 0.25 |
| hf_e2b_modular_full_stack_real_specialists_variants | agent_011_runbook_guided_patch_language_fr | failed, answer_mismatch | 1.0 |
