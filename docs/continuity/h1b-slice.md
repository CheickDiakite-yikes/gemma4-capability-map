# H1b Slice

H1b is the follow-up to the original H1 controller-dependence slice.

The first H1 slice is now saturated after the FunctionGemma final visual turn directive:

- every full-H1 HF service ablation row matches baseline
- `no_controller_repair` no longer breaks top-line readiness on H1
- `no_deterministic_visual_follow_on` is also controller-clean on H1
- trace mining finds `0` failure candidates

H1b keeps the same packaged-workflow and runtime path, but swaps in older visual/revision/resume episodes that were not part of the now-saturated five-episode H1 lane.

## Replayable Episodes

- `kwa_exec_visual_dashboard_referent_hold_v3`
- `kwa_exec_latest_action_resume_hold_v4`
- `kwa_jobs_visual_constraint_override_hold_v2`
- `kwa_jobs_phone_patch_resume_hold_v4`
- `kwa_finance_visual_invoice_revision_hold_v2`

## Live Episodes

- `kwa_exec_live_visual_dashboard_referent_hold_v3`
- `kwa_exec_live_latest_action_resume_hold_v4`
- `kwa_jobs_live_visual_constraint_override_hold_v2`
- `kwa_jobs_live_phone_patch_resume_hold_v4`
- `kwa_finance_live_visual_invoice_revision_hold_v2`

## Stressors

- longer visual referent carryover
- latest-instruction and stale-context pressure
- visual evidence followed by CLI/action dependencies
- artifact revision after visual evidence
- approval-safe stop after partial progress

## Commands

Dry-run the H1b primary row:

```bash
uv run python scripts/run_knowledge_work_h1_slice.py \
  --config configs/knowledge_work_h1b_slice.yaml \
  --dry-run \
  --run-set primary \
  --lane replayable_core
```

Run the compact H1b no-controller-repair packet:

```bash
uv run python scripts/run_knowledge_work_h1_ablation_packet.py \
  --config configs/knowledge_work_h1b_slice.yaml \
  --packet-id visual_policy_no_controller_repair \
  --run-group-id <timestamp>_h1b_visual_policy_packet
```

Run the full H1b HF service ablation:

```bash
uv run python scripts/run_knowledge_work_h1_ablation_packet.py \
  --config configs/knowledge_work_h1b_slice.yaml \
  --lane replayable_core \
  --run-group-id <timestamp>_h1b_hf_service_ablation
```
