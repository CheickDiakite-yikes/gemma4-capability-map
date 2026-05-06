# H1c Slice

H1c follows the replayable H1/H1b saturation result and the CLI live-smoke packets.

The key shift is from replayable visual semantics alone to live-policy controller dependence:

- H1 and H1b now saturate after the FunctionGemma final-turn directive.
- The real MLX CLI packets still show argument repair across visual, API, and CLI calls.
- The live-web policy packet reintroduced controller fallback on the jobs form workflow.
- Live execution now has committed evidence for completed, awaiting-approval, and sandbox-policy-blocked states.

H1c keeps packaged workflows as the only live entrypoint and treats live-web policy events as first-class benchmark signal.

## Replayable Episodes

- `kwa_exec_visual_dashboard_brief`
- `kwa_finance_invoice_lock_direction_hold_v4`
- `kwa_jobs_email_block_resume_hold_v5`
- `kwa_jobs_phone_patch_resume_hold_v4`
- `kwa_finance_diff_review_hold_v5`

## Live Episodes

- `kwa_exec_live_visual_dashboard_brief`
- `kwa_finance_live_invoice_lock_direction_hold_v4`
- `kwa_jobs_live_email_block_resume_hold_v5`
- `kwa_jobs_live_phone_patch_resume_hold_v4`
- `kwa_finance_live_diff_review_hold_v5`

## Stressors

- live-web sandbox policy blocks
- approval-safe stop behavior
- controller fallback under live visual/form pressure
- visual argument repair
- API argument repair
- CLI patch argument repair
- stale-instruction and latest-direction conflict

## Commands

Dry-run the H1c live primary row:

```bash
uv run python scripts/run_knowledge_work_h1_slice.py \
  --config configs/knowledge_work_h1c_slice.yaml \
  --dry-run \
  --run-set primary \
  --lane live_web_stress
```

Run the compact H1c live-policy helper packet:

```bash
uv run python scripts/run_knowledge_work_h1_ablation_packet.py \
  --config configs/knowledge_work_h1c_slice.yaml \
  --packet-id live_policy_controller_helpers \
  --run-group-id <timestamp>_h1c_live_policy_helpers
```

Run the full H1c HF service ablation:

```bash
uv run python scripts/run_knowledge_work_h1_ablation_packet.py \
  --config configs/knowledge_work_h1c_slice.yaml \
  --lane live_web_stress \
  --run-group-id <timestamp>_h1c_live_policy_ablation
```

## Current Status

H1c is scaffolded but not yet executed beyond dry-run validation.

The first empirical target should be the compact `live_policy_controller_helpers` packet. It is intentionally smaller than the full five-episode live slice because the HF service-backed live lane is the expensive, high-signal path.
