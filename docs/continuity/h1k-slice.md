# H1k Parallel-Audit Tool-Contract Live Slice

H1k is the packaged-workflow promotion of the deferred `parallel_audit_array_literal` replay case.

H1j saturated because it covered API/CLI argument and visual no-call pressure, but still deferred the exact parallel-tool failure. H1k closes that gap with one focused workflow: `ops_parallel_audit_review`.

Config:

- [`configs/knowledge_work_h1k_slice.yaml`](../../configs/knowledge_work_h1k_slice.yaml)

## Workflow Shape

| Workflow | Replay pressure | Replayable episode | Live episode |
| --- | --- | --- | --- |
| `ops_parallel_audit_review` | `parallel_audit_array_literal` | `kwa_ops_parallel_audit_review_v1` | `kwa_ops_live_parallel_audit_review_v1` |

The workflow is intentionally narrow. It uses existing gold tasks that require the model to inspect image evidence and read `config/settings.yaml` before recording the `safe_mode: true` patch.

## CLI Preflight

```bash
uv run moonie-agent workflows \
  --lane live_web_stress \
  --workflow-id ops_parallel_audit_review \
  --validate
```

Expected result:

- `valid = true`
- `workflow_count = 1`
- live episode ID is `kwa_ops_live_parallel_audit_review_v1`

## Candidate Packet

Executed packet:

- [`results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_candidates_v1_knowledge_work_ablation_packet)

Result:

- `5` rows, `1` live workflow each
- all rows matched readiness `0.91780`
- strict/recovered `1.0 / 1.0`
- repair/fallback/argument repair `0.0 / 0.0 / 0.0`
- raw clean `1.0`
- trace mining found `0` notes and `0` failure candidates

Interpretation: H1k is harder on artifact quality than H1j, but it still does not reproduce the raw no-directive parallel no-call failure. The existing packaged workflow decomposes the pressure into `tool_009_parallel_context_check` and `agent_010_parallel_audit_patch`, and no-directive MLX remains controller-clean on that staged form. The next discriminator should preserve the one-turn exact-call shape or add a live replay executor for exact probe cases.

Dry run:

```bash
uv run python scripts/run_knowledge_work_h1_ablation_packet.py \
  --config configs/knowledge_work_h1k_slice.yaml \
  --packet-id mlx_parallel_audit_tool_contract_candidates \
  --run-group-id <timestamp>_h1k_parallel_audit_candidates_v1 \
  --dry-run
```

Execution:

```bash
uv run python scripts/run_knowledge_work_h1_ablation_packet.py \
  --config configs/knowledge_work_h1k_slice.yaml \
  --packet-id mlx_parallel_audit_tool_contract_candidates \
  --run-group-id <timestamp>_h1k_parallel_audit_candidates_v1
```

## Helper Packet

Dry run:

```bash
uv run python scripts/run_knowledge_work_h1_ablation_packet.py \
  --config configs/knowledge_work_h1k_slice.yaml \
  --packet-id mlx_parallel_audit_helper_ablation \
  --run-group-id <timestamp>_h1k_parallel_audit_helpers_v1 \
  --dry-run
```

## Acceptance Criteria

H1k is useful if it separates rows on at least one controller-dependence metric:

- raw planning clean rate
- controller repair
- controller fallback
- argument repair
- strict interface
- recovered execution

If H1k saturates, the next surface should be exact-probe replay execution under the live operator, not another broad H1h rerun.
