# H1l Visual Executor-Equivalence Live Slice

H1l is the packaged-workflow promotion of the visual hard-slice executor-equivalence result.

The hard-slice packet shows `visual_role_catalog_schema_field_hints_v4` at `6 / 8` strict exactness but `8 / 8` executor-equivalent target success. H1l asks whether that distinction survives live packaged visual workflows, where Moonie still records controller burden, strict interface, recovered execution, and raw planning cleanliness.

Config:

- [`configs/knowledge_work_h1l_slice.yaml`](../../configs/knowledge_work_h1l_slice.yaml)

## Workflow Shape

| Workflow | Replay pressure | Live episode |
| --- | --- | --- |
| `executive_visual_dashboard_review` | visual argument copying and region readback | `kwa_exec_live_visual_dashboard_brief` |
| `executive_visual_referent_review` | latest visual filter and referent carryover | `kwa_exec_live_visual_dashboard_referent_hold_v3` |
| `jobs_visual_constraint_override` | stale-selection and visual routing pressure | `kwa_jobs_live_visual_constraint_override_hold_v2` |
| `finance_visual_invoice_review` | invoice visual evidence under approval pressure | `kwa_finance_live_invoice_lock_direction_hold_v4` |
| `finance_visual_invoice_revision` | current invoice visual referent revision | `kwa_finance_live_visual_invoice_revision_hold_v2` |

This is deliberately narrower than H1j. It avoids API/CLI argument families and focuses on the visual endpoint split exposed by the executor-equivalence packet.

## Candidate Packet

Dry run:

```bash
uv run python scripts/run_knowledge_work_h1_ablation_packet.py \
  --config configs/knowledge_work_h1l_slice.yaml \
  --packet-id mlx_visual_executor_equivalence_candidates \
  --run-group-id <timestamp>_h1l_visual_executor_equivalence_candidates_v1 \
  --dry-run
```

Execution:

```bash
uv run python scripts/run_knowledge_work_h1_ablation_packet.py \
  --config configs/knowledge_work_h1l_slice.yaml \
  --packet-id mlx_visual_executor_equivalence_candidates \
  --run-group-id <timestamp>_h1l_visual_executor_equivalence_candidates_v1
```

Candidate rows:

- contracted MLX
- no-directive MLX
- visual role catalog v1
- visual role catalog argument hints v2
- visual role catalog schema-field hints v4
- visual role catalog schema target literals v5

## Helper Packet

Dry run:

```bash
uv run python scripts/run_knowledge_work_h1_ablation_packet.py \
  --config configs/knowledge_work_h1l_slice.yaml \
  --packet-id mlx_visual_executor_equivalence_helper_ablation \
  --run-group-id <timestamp>_h1l_visual_executor_equivalence_helpers_v1 \
  --dry-run
```

Execution:

```bash
uv run python scripts/run_knowledge_work_h1_ablation_packet.py \
  --config configs/knowledge_work_h1l_slice.yaml \
  --packet-id mlx_visual_executor_equivalence_helper_ablation \
  --run-group-id <timestamp>_h1l_visual_executor_equivalence_helpers_v1
```

## Acceptance Criteria

H1l is useful if it separates rows on at least one of these:

- raw planning clean rate
- controller repair
- controller fallback
- argument repair
- strict interface
- recovered execution
- trace-mined visual routing or stale-selection failure candidates

Interpretation guardrail: do not collapse strict exactness and executor-equivalent target success into a single pass/fail label. The point of H1l is to preserve that split in live workflow evidence.
