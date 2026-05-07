# H1j Probe-Derived Tool-Contract Live Slice

H1j is the next saturation breaker after H1i candidate v1 and H1i repeat3 both went clean.

The goal is not to widen the benchmark for its own sake. The goal is to move the exact no-directive probe failures into a live packaged-workflow packet:

- API/CLI argument mismatch:
  - `api_form_issue_fetch`
  - `api_invoice_lock_hold_update`
  - `cli_invoice_lock_hyphen_query`
  - `cli_phone_patch_latest_only`
- visual no-tool-call and readback pressure:
  - `visual_form_target_literal`
  - `visual_latest_filter_literal`
  - `visual_readback_region_literal`
- parallel no-tool-call:
  - `parallel_audit_array_literal`
  - deferred in v1 because the current live packaged-workflow surface does not yet contain a faithful parallel-tool workflow

Config:

- [`configs/knowledge_work_h1j_slice.yaml`](../../configs/knowledge_work_h1j_slice.yaml)

## Packet Shape

H1j keeps the v1 live-entrypoint rule: packaged workflows only.

Live workflow families:

| Workflow | Probe pressure |
| --- | --- |
| `executive_visual_dashboard_review` | visual target selection and readback |
| `executive_visual_referent_review` | latest visual filter and referent carryover |
| `jobs_visual_constraint_override` | form visual target selection |
| `jobs_phone_patch_resume` | API form-issue fetch, CLI phone patch, visual referent carryover |
| `finance_visual_invoice_review` | invoice-lock API/CLI argument mismatch plus visual evidence |
| `finance_billing_patch_hold` | review-only billing direction under API/CLI disagreement |

Candidate packet:

```bash
uv run python scripts/run_knowledge_work_h1_ablation_packet.py \
  --config configs/knowledge_work_h1j_slice.yaml \
  --packet-id mlx_probe_derived_tool_contract_candidates \
  --run-group-id <timestamp>_h1j_probe_derived_candidates_v1
```

Helper-ablation packet:

```bash
uv run python scripts/run_knowledge_work_h1_ablation_packet.py \
  --config configs/knowledge_work_h1j_slice.yaml \
  --packet-id mlx_probe_derived_helper_ablation \
  --run-group-id <timestamp>_h1j_probe_derived_helpers_v1
```

## Acceptance Criteria

H1j is useful if it does at least one of these:

- restores no-directive controller burden after H1i repeat3 saturation
- separates prompt-contract candidates on raw clean rate, repair, fallback, argument repair, strict interface, or recovered execution
- produces trace-note failure candidates that point to a specific second-wave prompt-contract change

If H1j also saturates, the next move is not H1h. It is a new packaged workflow for the deferred `parallel_audit_array_literal` family or a probe runner that can replay exact probe cases through the live operator harness.
