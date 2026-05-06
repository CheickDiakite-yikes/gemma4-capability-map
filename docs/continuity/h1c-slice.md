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

## Current Result

The compact `live_policy_controller_helpers` packet is complete:

- output: [`results/knowledge_work_h1_slice/20260506T_h1c_live_policy_packet_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1c_live_policy_packet_v1_knowledge_work_ablation_packet)
- systems: baseline HF service specialists, `no_controller_repair`, `no_controller_fallback`, and `no_argument_repair`
- all four rows matched at readiness `0.9779666666666667`
- strict/recovered stayed `1.0 / 1.0`
- controller repair, argument repair, and controller fallback stayed `0.0`
- raw planning clean stayed `1.0`
- trace mining found `0` failure candidates

The H1c MLX primary live path is also complete:

- output: [`results/knowledge_work_h1_slice/20260506T_h1c_mlx_live_primary_v1_knowledge_work_h1c_live_policy_controller_dependence_v1`](../../results/knowledge_work_h1_slice/20260506T_h1c_mlx_live_primary_v1_knowledge_work_h1c_live_policy_controller_dependence_v1)
- system: `mlx_gemma4_e2b_reasoner_only`
- lane: `live_web_stress`
- episodes: all `5` H1c live episodes
- readiness averaged `0.97936`
- artifact quality averaged `0.95`
- strict/recovered stayed `1.0 / 1.0`
- controller repair, argument repair, and controller fallback stayed `0.0`
- raw planning clean stayed `1.0`
- direct trace inspection found no non-empty planning repair notes

The repeated CLI live-smoke packet is complete:

- output: [`results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet`](../../results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet)
- system: `mlx_gemma4_e2b_reasoner_only`
- lane: `live_web_stress`
- workflows: `executive_visual_dashboard_review`, `finance_visual_invoice_review`, and `jobs_visual_form_hold`
- repeat count: `3`
- status counts: `completed = 3`, `awaiting_approval = 6`, `failed = 0`
- readiness averaged `0.9818333333333334`
- strict/recovered stayed `1.0 / 1.0`
- controller repair averaged `0.6666666666666666`
- argument repair averaged `0.5`
- controller fallback averaged `0.16666666666666666`
- raw planning clean averaged `0.3333333333333333`
- all three repeats reproduced the same workflow-level repair pattern
- analyzer outputs in the packet report `stable_repair_family_count = 4` and `stable_policy_block_family_count = 7`

The corrected H1c MLX primary monolith run is complete:

- output: [`results/knowledge_work_h1_slice/20260506T_h1c_mlx_live_primary_monolith_v1_knowledge_work_h1c_live_policy_controller_dependence_v1`](../../results/knowledge_work_h1_slice/20260506T_h1c_mlx_live_primary_monolith_v1_knowledge_work_h1c_live_policy_controller_dependence_v1)
- system: `mlx_gemma4_e2b_reasoner_only`
- lane: `live_web_stress`
- reason for rerun: the earlier H1c MLX primary row used a modular benchmark bundle with a heuristic router, while the CLI live profile is monolith/reasoner-only
- readiness stayed `0.97936`
- strict/recovered stayed `1.0 / 1.0`
- controller repair averaged `0.7`
- argument repair averaged `0.5`
- controller fallback averaged `0.2`
- raw planning clean averaged `0.3`

Interpretation: the repeated CLI live signal was not a one-off anomaly. Once H1c uses the same monolith posture as `moonie-agent live`, benchmark and CLI agree that local MLX Gemma completes the workflows but still needs controller repair/fallback on live visual/API/CLI families. The next slice should add local MLX controller-helper ablation profiles or an equivalent H1c monolith ablation packet.
