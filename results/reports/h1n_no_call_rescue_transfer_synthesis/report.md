# H1n No-Call Rescue Transfer Synthesis

Generated: `2026-05-10T16:19:11.102272+00:00`

## Aggregate

- v10 exact successes: `22 / 30`
- no-directive exact successes: `11 / 30`
- incumbent exact successes: `25 / 30`
- v10 executor-equivalent successes: `25 / 30`
- no-directive executor-equivalent successes: `12 / 30`
- incumbent executor-equivalent successes: `26 / 30`

## Findings

- `large_no_directive_lift`: v10 is a real no-directive harness improvement: 22/30 exact versus 11/30 no-directive, and 25/30 executor-equivalent versus 12/30 no-directive.
- `not_universal_replacement`: v10 is not a universal replacement for the best specialized profiles: 22/30 exact versus incumbents at 25/30, and 25/30 executor-equivalent versus incumbents at 26/30.
- `transfer_pattern`: Executor-equivalence versus incumbents is positive on component_value_v10, tied on post_repair_v7, and negative on residual_v8, oblique_v7.
- `promotion_decision`: Treat v10 as a scoped current-image/no-call activation guard. The next H1o slice should factor activation rescue, code/negation preservation, and component-label/value disambiguation instead of stacking broad prose.

## Packet Rows

| label | incumbent_label | case_count | v10_exact_rate | delta_exact_vs_no_directive | delta_exact_vs_incumbent | v10_executor_rate | delta_executor_vs_no_directive | delta_executor_vs_incumbent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| component_value_v10 | hybrid_label_guard_v8 | 8 | 0.875 | 0.25 | 0.125 | 1.0 | 0.25 | 0.125 |
| residual_v8 | hybrid_label_guard_v8 | 8 | 0.5 | 0.0 | -0.375 | 0.75 | 0.25 | -0.125 |
| post_repair_v7 | oblique_code_guard_v7 | 8 | 0.75 | 0.5 | 0.0 | 0.75 | 0.5 | 0.0 |
| oblique_v7 | oblique_code_guard_v7 | 6 | 0.8333333333333334 | 0.8333333333333334 | -0.16666666666666663 | 0.8333333333333334 | 0.8333333333333334 | -0.16666666666666663 |
