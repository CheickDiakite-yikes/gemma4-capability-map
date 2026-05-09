# Packaged Replay Gap Diagnostic

- Generated at: `2026-05-09T23:46:33.286582+00:00`
- Surface count: `2`
- Saturated packaged surface count: `2`

## Surface Summary

| surface_id | replay_signal | packaged_surface | replay_comparison_count | replay_max_delta_exact_rate | replay_max_delta_executor_equivalence_rate | replay_positive_executor_delta_count | packaged_system_count | packaged_readiness_span | packaged_strict_interface_span | packaged_recovered_execution_span | packaged_controller_burden_max | classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h1l_visual_executor_equivalence | preserved two-case visual hard-slice replay | five packaged visual workflows | 5 | 1.0 | 1.0 | 5 | 6 | 0.0 | 0.0 | 0.0 | 0.0 | positive_replay_saturated_packaged_surface |
| h1m_visual_alias_repeat | eight-case visual alias-repeat replay | three packaged alias-repeat visual workflows | 5 | 0.625 | 0.375 | 5 | 6 | 0.0 | 0.0 | 0.0 | 0.0 | positive_replay_saturated_packaged_surface |

## Recommendations

| recommendation_id | status | evidence |
| --- | --- | --- |
| do_not_spend_helper_budget_on_saturated_packaged_visual_surfaces | active | 2 visual surfaces have positive replay gains but zero packaged strict/readiness span. |
| preserve_replay_shape_before_packaging | active | The next visual task should keep alias/decoy pressure in the live prompt instead of decomposing it into staged packaged steps. |
| report_strict_and_executor_equivalence_separately | active | Replay gains appear first as executor-equivalence gains, while packaged readiness can saturate. |

## Interpretation

H1l and H1m now tell the same methodological story: replay-shaped visual packets can separate strict fidelity from executor-equivalent target success, while current packaged visual workflows erase row-level differences. That makes packaged workflow saturation an experimental-design finding, not a model-quality win.
