# H1d Candidate Slice

H1d should be built from the corrected H1c MLX monolith evidence, not from the earlier router-mismatched clean row.

Source packets:

- [`results/knowledge_work_h1_slice/20260506T_h1c_mlx_live_primary_monolith_v1_knowledge_work_h1c_live_policy_controller_dependence_v1`](../../results/knowledge_work_h1_slice/20260506T_h1c_mlx_live_primary_monolith_v1_knowledge_work_h1c_live_policy_controller_dependence_v1)
- [`results/knowledge_work_h1_slice/20260506T_h1c_mlx_monolith_helpers_v1_knowledge_work_h1c_live_policy_controller_dependence_v1`](../../results/knowledge_work_h1_slice/20260506T_h1c_mlx_monolith_helpers_v1_knowledge_work_h1c_live_policy_controller_dependence_v1)
- [`results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet`](../../results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet)

## Evidence

Corrected H1c baseline:

- readiness `0.97936`
- strict/recovered `1.0 / 1.0`
- controller repair `0.7`
- argument repair `0.5`
- controller fallback `0.2`
- raw planning clean `0.3`

Local MLX helper ablations:

- `no_controller_repair`: readiness `0.7381800000000001`, strict/recovered `0.475 / 0.3`
- `no_controller_fallback`: readiness `0.92104`, strict/recovered `0.85 / 0.8`
- `no_argument_repair`: readiness `0.82036`, strict/recovered `0.7125 / 0.5`

Trace-mined failure modes:

- `visual_stepwise_control = 6`
- `repair_disabled = 5`
- `fallback_planner = 4`
- `argument_repair = 2`
- `fallback_disabled = 2`
- `visual_repeated_refinement = 2`
- `visual_readback_missing = 1`

## Candidate Families

### 1. Visual Stepwise Control

Observed failures:

- `visual_016_live_dashboard_stale_selection_recovery`
- `visual_022_live_form_latest_issue_referent_carryover`
- `visual_030_live_form_latest_blocked_email_refinement`

H1d stressor:

- force a sequence where a semantically broad raw visual query is tempting but only the canonical stepwise chain preserves the final referent
- require the trace to preserve a prior selection id across at least three refinements before readback
- score the difference between "close enough semantic query" and exact stateful visual control

### 2. API/CLI Canonicalization

Observed failures:

- `tool_018_jobs_api_latest_form_issue`
- `tool_019_finance_cli_log_search_latest_lock`
- `tool_021_jobs_cli_patch_only_latest_email_fix`
- `tool_016_finance_api_invoice_lock_update`

H1d stressor:

- require canonical record types, ids, paths, fields, and normalized values even when the prompt phrase is colloquial
- include near-miss raw calls such as `record_type = FORM-88`, `path = billing.log`, and `value = on hold`
- separate "semantic intent understood" from "contract-valid tool call emitted"

### 3. Fallback Boundary

Observed failures:

- jobs phone visual chain
- jobs blocked-email visual chain

H1d stressor:

- preserve the fallback-worthy malformed multi-call planner output, but require the fallback to recover only the next legal call
- add a no-fallback row to confirm that fallback is causal only on malformed chained visual output, not on ordinary argument repair

### 4. Approval-Safe Stop Under Repair Pressure

Observed failures:

- disabled helper rows still sometimes complete intermediate artifacts but fall at approval/recovery boundaries

H1d stressor:

- require a sandbox-only repair followed by an approval-required stop
- verify that the corrected artifact is present before the stop, and that no public side effect is claimed
- make approval correctness independent from artifact quality so controller helpers cannot hide a release-boundary miss

## Proposed Packet

Name: `h1d_mlx_monolith_controller_stress`

Lane:

- `live_web_stress`

Systems:

- `mlx_gemma4_e2b_reasoner_only`
- `mlx_gemma4_e2b_reasoner_only_no_controller_repair`
- `mlx_gemma4_e2b_reasoner_only_no_controller_fallback`
- `mlx_gemma4_e2b_reasoner_only_no_argument_repair`

Initial episode candidates:

- dashboard stepwise visual referent
- jobs latest phone issue API + visual carryover
- jobs blocked-email CLI patch + visual fallback + approval hold
- finance invoice-lock CLI search
- finance invoice-lock API update

Success criteria:

- baseline keeps readiness above `0.97` while exposing non-zero controller dependence
- at least one disabled helper row drops below `0.85` readiness
- failure modes are attributable to one of visual stepwise control, API/CLI canonicalization, fallback boundary, or approval-safe stop
- replayable/live mirrors remain tied to packaged workflow families

## Next Build Step

Create `configs/knowledge_work_h1d_slice.yaml` only after the candidate episode ids and any required task variants are selected from existing data or added as a focused new data patch.
