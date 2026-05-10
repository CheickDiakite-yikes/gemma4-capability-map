# H1n Code-Hints Transfer Synthesis

Generated: `2026-05-10T01:28:30.610687+00:00`

## Aggregate

- total_case_count: `18`
- argument_hints_exact_success_count: `14`
- code_hints_exact_success_count: `11`
- argument_hints_executor_success_count: `16`
- code_hints_executor_success_count: `12`

## Findings

- `localized_oblique_repair`: Code hints improves only the oblique code-label packet; positive transfer labels: oblique_v5.
- `negative_transfer_elsewhere`: Code hints regresses against argument hints on earlier oracle transfer surfaces; negative transfer labels: oracle_transfer_v2, oracle_repeat_v1.
- `aggregate_exactness`: Across the three H1n oracle packets, argument hints has 14/18 exact successes versus code hints at 11/18.
- `aggregate_executor_equivalence`: Across the three H1n oracle packets, argument hints has 16/18 executor-equivalent successes versus code hints at 12/18.
- `promotion_decision`: Keep oblique code hints as a scoped repair candidate, not a replacement for argument hints, until a stale-selection guard or fresh holdout reverses the transfer loss.

## Comparison Summary

| label | interpretation | argument_hints_exact_rate | code_hints_exact_rate | delta_exact_rate | argument_hints_executor_equivalence_rate | code_hints_executor_equivalence_rate | delta_executor_equivalence_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_transfer_v2 | negative_transfer | 0.8333333333333334 | 0.5 | -0.33333333333333337 | 1.0 | 0.5 | -0.5 |
| oracle_repeat_v1 | negative_transfer | 0.8333333333333334 | 0.5 | -0.33333333333333337 | 1.0 | 0.6666666666666666 | -0.33333333333333337 |
| oblique_v5 | localized_repair | 0.6666666666666666 | 0.8333333333333334 | 0.16666666666666674 | 0.6666666666666666 | 0.8333333333333334 | 0.16666666666666674 |
