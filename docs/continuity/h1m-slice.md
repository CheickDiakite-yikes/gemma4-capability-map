# H1m Visual Alias-Repeat Packaged Live Slice

H1m is the packaged-workflow successor to the visual alias-repeat replay matrix.

The replay-shaped matrix found that no-directive MLX reached `2 / 8` strict and `5 / 8` executor-equivalent, schema-field hints reached `2 / 8` strict and `7 / 8` executor-equivalent, schema target literals reached `3 / 8` strict and `8 / 8` executor-equivalent, and contracted MLX remained the strict upper bound at `7 / 8` strict and `8 / 8` executor-equivalent.

H1m does not add a new frontend or free-form live entrypoint. It packages existing visual live episodes that are closer to the alias-repeat mechanisms than H1l's staged visual set:

- `executive_visual_dashboard_revision`
- `jobs_visual_latest_issue_review`
- `finance_visual_invoice_hold_review`

Candidate packet:

```bash
uv run python scripts/run_knowledge_work_h1_ablation_packet.py \
  --config configs/knowledge_work_h1m_slice.yaml \
  --packet-id mlx_visual_alias_repeat_packaged_candidates \
  --run-group-id <timestamp>_h1m_visual_alias_repeat_candidates_v1
```

Dry-run first:

```bash
uv run python scripts/run_knowledge_work_h1_ablation_packet.py \
  --config configs/knowledge_work_h1m_slice.yaml \
  --packet-id mlx_visual_alias_repeat_packaged_candidates \
  --run-group-id <timestamp>_h1m_visual_alias_repeat_candidates_dry_run_v1 \
  --dry-run
```

Interpretation guardrail:

- If all rows tie again, record H1m as a saturated packaged-workflow negative result and return to replay-shaped or raw hard-slice evidence.
- If schema-field or schema target literals separate from no-directive, run `mlx_visual_alias_repeat_helper_ablation` to attribute controller repair, fallback, and argument repair.
- Strict protocol fidelity and executor-equivalent visual success must remain separate endpoints.
