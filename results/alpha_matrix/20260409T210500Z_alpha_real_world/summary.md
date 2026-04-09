# Alpha Matrix Summary

- Run group: `20260409T210500Z`
- Matrix: `alpha_real_world`
- Completed experiments: `4/4`
- Matrix complete: `True`

## Backend Posture

- Recommended local reasoner backend: `mlx`
- HF token present: `True`
- Offline mode enabled: `False`
- Preflight source: `/Users/cheickdiakite/Codex/moonie/results/tables/backend_preflight.json`

## Failure Breakdown

- `answer_mismatch`: 17
- `arg_mismatch`: 15
- `failed`: 21
- `malformed_call`: 4
- `wrong_tool`: 15

## Experiments

| Experiment | Status | Exec mode | Pipeline | Track | Backend | Success | Strict interface | Recovered execution | Readiness | Avg latency ms | Runs |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hf_e2b_real_world_thinking_variants | completed | subprocess | monolith | thinking | hf | 0.0 | 0.0 | 0.0 | 0.0 | 1016.75 | 4.0 |
| hf_e2b_real_world_retrieval_variants | completed | subprocess | hybrid | retrieval | hf | 0.875 | 1.0 | 0.875 | 0.88125 | 2855.44 | 16.0 |
| hf_e2b_real_world_routing_variants | completed | subprocess | modular | tool_routing | hf | 0.5 | 0.5 | 0.5 | 0.6375 | 7339.0 | 16.0 |
| hf_e2b_real_world_full_stack_variants | completed | subprocess | modular | full_stack | hf | 0.75 | 0.75 | 1.0 | 0.8696428571428572 | 12115.68 | 28.0 |

## Probes

No probes configured.

## Improvements

| Experiment | Success delta | Latency improvement ms |
| --- | ---: | ---: |
| hf_e2b_real_world_thinking_variants | 0.0 | -152.5 |
| hf_e2b_real_world_retrieval_variants | 0.0 | 364.06 |
| hf_e2b_real_world_routing_variants | -0.1667 | -120.75 |

## Failing Variants

| Experiment | Variant | Failure tags | Interface reliability |
| --- | --- | --- | ---: |
| hf_e2b_real_world_thinking_variants | think_013_prod_approval_escalation_clean | failed, answer_mismatch | 0.0 |
| hf_e2b_real_world_thinking_variants | think_013_prod_approval_escalation_language_fr | failed, answer_mismatch | 0.0 |
| hf_e2b_real_world_thinking_variants | think_013_prod_approval_escalation_context_stale_preference | failed, answer_mismatch | 0.0 |
| hf_e2b_real_world_thinking_variants | think_013_prod_approval_escalation_efficiency_dim_256 | failed, answer_mismatch | 0.0 |
| hf_e2b_real_world_retrieval_variants | retr_011_approval_policy_language_fr | failed, answer_mismatch | 0.0 |
| hf_e2b_real_world_retrieval_variants | retr_013_vendor_access_defer_language_fr | failed, answer_mismatch | 0.0 |
| hf_e2b_real_world_routing_variants | tool_012_billing_patch_record_clean | failed, wrong_tool, arg_mismatch | 0.25 |
| hf_e2b_real_world_routing_variants | tool_012_billing_patch_record_language_code_switch | failed, wrong_tool, arg_mismatch | 0.25 |
| hf_e2b_real_world_routing_variants | tool_012_billing_patch_record_schema_renamed_fields | failed, wrong_tool, arg_mismatch | 0.25 |
| hf_e2b_real_world_routing_variants | tool_012_billing_patch_record_context_irrelevant_tool_output | failed, wrong_tool, arg_mismatch | 0.25 |
| hf_e2b_real_world_routing_variants | tool_013_refuse_unsafe_billing_disable_clean | failed, malformed_call, wrong_tool, arg_mismatch, answer_mismatch | 0.0 |
| hf_e2b_real_world_routing_variants | tool_013_refuse_unsafe_billing_disable_language_code_switch | failed, malformed_call, wrong_tool, arg_mismatch, answer_mismatch | 0.0 |
