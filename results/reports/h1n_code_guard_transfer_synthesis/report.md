# H1n Code-Guard Transfer Synthesis

Generated: `2026-05-10T01:39:35.326804+00:00`

## Aggregate

- argument_hints_exact_success_count: `14`
- code_guard_exact_success_count: `14`
- argument_hints_executor_success_count: `16`
- code_guard_executor_success_count: `15`
- code_hints_exact_success_count: `11`
- code_hints_executor_success_count: `12`

## Findings

- `code_guard_beats_v6`: Code guard improves on v6 across the three-packet aggregate: 14/18 exact and 15/18 executor-equivalent versus v6 at 11/18 exact and 12/18 executor-equivalent.
- `argument_hints_still_best_executor`: Argument hints remains the stronger executor-equivalence baseline overall: 16/18 versus code guard at 15/18.
- `oblique_only_positive_vs_argument_hints`: Code guard is positive versus argument hints only on oblique_v5 and negative on oracle_transfer_v2, oracle_repeat_v1.
- `promotion_decision`: Treat code guard as a better scoped repair than v6, not a broad replacement for argument hints; the next proof point should be a fresh post-repair holdout.

## Code Guard vs Argument Hints

| label | baseline_profile | candidate_exact_rate | delta_exact_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- |
| oracle_transfer_v2 | argument_hints_v2 | 0.6666666666666666 | -0.16666666666666674 | 0.6666666666666666 | -0.33333333333333337 |
| oracle_repeat_v1 | argument_hints_v2 | 0.6666666666666666 | -0.16666666666666674 | 0.8333333333333334 | -0.16666666666666663 |
| oblique_v5 | argument_hints_v2 | 1.0 | 0.33333333333333337 | 1.0 | 0.33333333333333337 |

## Code Guard vs Code Hints

| label | baseline_profile | candidate_exact_rate | delta_exact_rate | candidate_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- |
| oracle_transfer_v2 | code_hints_v6 | 0.6666666666666666 | 0.16666666666666663 | 0.6666666666666666 | 0.16666666666666663 |
| oracle_repeat_v1 | code_hints_v6 | 0.6666666666666666 | 0.16666666666666663 | 0.8333333333333334 | 0.16666666666666674 |
| oblique_v5 | code_hints_v6 | 1.0 | 0.16666666666666663 | 1.0 | 0.16666666666666663 |
