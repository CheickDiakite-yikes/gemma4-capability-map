# Current State

## Repository Scope

This repo now contains two benchmark layers:

- atomic white-box capability benchmarking
  - reasoning
  - tool routing
  - retrieval
  - full-stack execution
- `KnowledgeWorkArena`
  - role-based, job-shaped episodes built on top of the atomic substrate

## Benchmark Shape

- `64` gold atomic tasks
- `282` explicit atomic variants
- `16` real-world-tagged atomic tasks
- `8` replayable-core visual-tool orchestration atomic tasks
- `4` live-web stress visual-tool orchestration atomic tasks
- `24` replayable-core `KnowledgeWorkArena` episodes in the generated corpus
- `18` live-web stress `KnowledgeWorkArena` episodes in the generated corpus

## Canonical Atomic Benchmark Pointers

- real-world autonomy matrix:
  - [`results/alpha_matrix/20260409T210500Z_alpha_real_world`](../../results/alpha_matrix/20260409T210500Z_alpha_real_world)
- atomic benchmark history:
  - [`results/history/history_report.md`](../../results/history/history_report.md)

## Canonical Visual Tool Orchestration Pointers

Replayable core:

- [`results/visual_tool_orchestration/replayable_core/summary.json`](../../results/visual_tool_orchestration/replayable_core/summary.json)
- current metrics:
  - `runs = 11`
  - `success_rate = 1.0`
  - `strict_interface_rate = 1.0`
  - `recovered_execution_rate = 1.0`
  - `real_world_readiness_avg = 1.0`

Live-web stress:

- [`results/visual_tool_orchestration/live_web_stress/summary.json`](../../results/visual_tool_orchestration/live_web_stress/summary.json)
- current metrics:
  - `runs = 7`
  - `success_rate = 1.0`
  - `strict_interface_rate = 1.0`
  - `recovered_execution_rate = 1.0`

Interpretation:

- the repo now has a first-class atomic benchmark for visual-tool orchestration, not just generic multimodal QA
- the canonical track measures whether the reasoner picks the right visual tools, refines selections correctly, preserves referents across turns, and lands the right final answer
- stricter placeholder-aware scoring is now live: follow-up `selection_id` and `region_id` arguments must point at the latest valid visual referent, not just any non-empty placeholder replacement
- replayable scoring is seeded and deterministic; live-web stress uses the same tool surface with a local executor path

## Canonical KnowledgeWorkArena Pointers

Replayable core:

- [`results/knowledge_work/replayable_core/summary.json`](../../results/knowledge_work/replayable_core/summary.json)
- current metrics:
  - `runs = 24`
  - `artifact_quality_avg = 0.9866`
  - `browser_workflow_avg = 0.9910`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9510`
  - `escalation_correctness_avg = 1.0`

Live-web stress:

- [`results/knowledge_work/live_web_stress/summary.json`](../../results/knowledge_work/live_web_stress/summary.json)
- current metrics:
  - `runs = 18`
  - `artifact_quality_avg = 0.9822`
  - `browser_workflow_avg = 1.0`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9630`
  - `escalation_correctness_avg = 1.0`

Harder human-nuance episodes now included in the canonical oracle lanes:

- replayable:
  - `kwa_exec_stale_brief_hold`
  - `kwa_jobs_constraint_preservation_hold`
  - `kwa_finance_stale_assumption_hold`
- live:
  - `kwa_exec_live_stale_brief_hold`
  - `kwa_jobs_live_constraint_hold`
  - `kwa_finance_live_stale_assumption_hold`

Visual KWA episodes now also exist and have bounded oracle references:

- replayable visual KWA slice:
  - [`results/knowledge_work/kwa_visual_replayable_oracle_v1/summary.json`](../../results/knowledge_work/kwa_visual_replayable_oracle_v1/summary.json)
  - episodes:
    - `kwa_exec_visual_dashboard_brief`
    - `kwa_jobs_visual_form_hold`
    - `kwa_finance_visual_invoice_hold`
- live visual KWA slice:
  - [`results/knowledge_work/kwa_visual_live_oracle_v1/summary.json`](../../results/knowledge_work/kwa_visual_live_oracle_v1/summary.json)
  - episodes:
    - `kwa_exec_live_visual_dashboard_brief`
    - `kwa_jobs_live_visual_form_hold`
    - `kwa_finance_live_visual_invoice_hold`

Interpretation:

- the canonical KWA lanes now cover stale context reconciliation, original-constraint preservation under pressure, and stale-assumption repair before approval-gated release
- the generated KWA corpus is now larger than the canonical oracle lane pointers because the new visual episodes were validated in bounded slices first, then the full canonical oracle lanes were rerun on the expanded corpus
- these are harder because the right move is often “repair and stop safely,” not just “complete the workflow”
- the canonical runner no longer silently truncates the lane; `run_knowledge_work_arena.py` now defaults to full-lane execution unless `--limit` is explicitly set

## Benchmark Board / Reporting Layer

Registry and exports:

- registry:
  - [`configs/model_registry.yaml`](../../configs/model_registry.yaml)
- board latest:
  - [`results/history/knowledge_work_board_latest.csv`](../../results/history/knowledge_work_board_latest.csv)
- board runs:
  - [`results/history/knowledge_work_board_runs.csv`](../../results/history/knowledge_work_board_runs.csv)
- scatter data:
  - [`results/history/knowledge_work_scatter.csv`](../../results/history/knowledge_work_scatter.csv)
- board payload:
  - [`results/history/knowledge_work_board.json`](../../results/history/knowledge_work_board.json)
- role breakdown:
  - [`results/history/knowledge_work_role_breakdown.csv`](../../results/history/knowledge_work_role_breakdown.csv)
- category breakdown:
  - [`results/history/knowledge_work_category_breakdown.csv`](../../results/history/knowledge_work_category_breakdown.csv)
- track breakdown:
  - [`results/history/knowledge_work_track_breakdown.csv`](../../results/history/knowledge_work_track_breakdown.csv)

UI surface:

- Streamlit board mode now lives in:
  - [`src/gemma4_capability_map/app/streamlit_app.py`](../../src/gemma4_capability_map/app/streamlit_app.py)
- the board is registry-backed and keyed by:
  - `system_id`
  - `lane`
  - `run_intent`

Interpretation:

- the repo can now render internal leaderboard and scatter-style benchmark views from normalized run metadata
- the board now supports role-family, category, track, modality, and executor-mode cuts
- the board now also carries runtime-facing fields when manifests provide them:
  - `warmup_load_ms`
  - `last_request_elapsed_ms`
  - `requests_completed`
  - `total_cost_per_mtok`
- the current board is good enough for internal benchmarking and chart production
- the remaining gap to a public leaderboard is broader model coverage plus stronger latency and cost instrumentation

## Current Full-Lane Comparative Systems

The board now has five meaningful full-lane comparison rows:

- canonical oracle:
  - `oracle_gemma4_e2b`
- exploratory local reasoner-only via `hf_service`:
  - `hf_service_gemma4_reasoner_only`
- exploratory local specialist-backed via `hf_service` + real HF specialists:
  - `hf_service_gemma4_specialists_cpu`
- exploratory local reasoner-only via direct in-process HF:
  - `hf_gemma4_e2b_reasoner_only`
- exploratory local specialist-backed via direct in-process HF:
  - `hf_gemma4_e2b_specialists_cpu`

New direct in-process HF references:

- replayable:
  - [`results/knowledge_work/model_backed_hf_inprocess_reasoner_full_replayable_v1/summary.json`](../../results/knowledge_work/model_backed_hf_inprocess_reasoner_full_replayable_v1/summary.json)
  - `runs = 24`
  - `artifact_quality_avg = 0.9834`
  - `browser_workflow_avg = 0.9910`
  - `strict_interface_avg = 0.9531`
  - `recovered_execution_avg = 0.9375`
  - `real_world_readiness_avg = 0.9330`
- live:
  - [`results/knowledge_work/model_backed_hf_inprocess_reasoner_full_live_v1/summary.json`](../../results/knowledge_work/model_backed_hf_inprocess_reasoner_full_live_v1/summary.json)
  - `runs = 18`
  - `artifact_quality_avg = 0.9779`
  - `browser_workflow_avg = 1.0`
  - `strict_interface_avg = 0.9306`
  - `recovered_execution_avg = 0.9167`
  - `real_world_readiness_avg = 0.9379`

New direct in-process HF specialist-backed references:

- replayable:
  - [`results/knowledge_work/model_backed_hf_inprocess_specialists_full_replayable_v1/summary.json`](../../results/knowledge_work/model_backed_hf_inprocess_specialists_full_replayable_v1/summary.json)
  - `runs = 24`
  - `artifact_quality_avg = 0.9834`
  - `browser_workflow_avg = 0.9910`
  - `strict_interface_avg = 0.9844`
  - `recovered_execution_avg = 0.9792`
  - `real_world_readiness_avg = 0.9452`
- live:
  - [`results/knowledge_work/model_backed_hf_inprocess_specialists_full_live_v1/summary.json`](../../results/knowledge_work/model_backed_hf_inprocess_specialists_full_live_v1/summary.json)
  - `runs = 18`
  - `artifact_quality_avg = 0.9779`
  - `browser_workflow_avg = 1.0`
  - `strict_interface_avg = 0.9514`
  - `recovered_execution_avg = 0.9444`
  - `real_world_readiness_avg = 0.9460`

Interpretation:

- direct in-process HF is materially weaker than the existing `hf_service` reasoner-only baseline on the same full-lane KWA surface
- adding real HF specialists improves the direct in-process path materially:
  - replayable `strict_interface_avg`: `0.9531 -> 0.9844`
  - replayable `recovered_execution_avg`: `0.9375 -> 0.9792`
  - live `strict_interface_avg`: `0.9306 -> 0.9514`
  - live `recovered_execution_avg`: `0.9167 -> 0.9444`
- the original remaining losses were concentrated in a small visual KWA subset rather than the broader non-visual workflow corpus
- those concentrated execution failures are now closed in bounded reruns after the planner repair:
  - replayable service-backed control:
    - [`results/knowledge_work/model_backed_hf_service_specialists_smoke_finance_visual_replayable_v1/summary.json`](../../results/knowledge_work/model_backed_hf_service_specialists_smoke_finance_visual_replayable_v1/summary.json)
  - live service-backed controls:
    - [`results/knowledge_work/model_backed_hf_service_specialists_smoke_finance_visual_live_v1/summary.json`](../../results/knowledge_work/model_backed_hf_service_specialists_smoke_finance_visual_live_v1/summary.json)
    - [`results/knowledge_work/model_backed_hf_service_specialists_smoke_jobs_visual_live_v1/summary.json`](../../results/knowledge_work/model_backed_hf_service_specialists_smoke_jobs_visual_live_v1/summary.json)
  - replayable direct-HF specialist control:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_replayable_v2/summary.json`](../../results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_replayable_v2/summary.json)
  - live direct-HF specialist controls:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_live_v2/summary.json`](../../results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_live_v2/summary.json)
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_jobs_visual_live_v2/summary.json`](../../results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_jobs_visual_live_v2/summary.json)
- those reruns now show:
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - the remaining invoice weakness is `artifact_quality_avg = 0.7692`, which is shared by the service-backed control and therefore is a softer artifact/readiness issue, not a still-open visual orchestration bug
- the full-lane in-process specialist references above have not yet been rerun after this narrow fix, so use them as the last broad comparison snapshot, not as the final post-fix state for the visual invoice/form episodes
- this is a useful deployment-level benchmark result:
  - same base model family
  - different execution path and specialist composition
  - meaningfully different full-lane behavior even after specialist recovery
- `mlx` is still blocked locally because the runtime probe fails with `ModuleNotFoundError: mlx`
- `google/gemma-4-E4B-it` remains probe-only locally on this Mac and should not be treated as the next serious full-lane comparison target

## Policy-Hardening Oracle Snapshots

Replayable policy-hardening subset:

- [`results/knowledge_work/replayable_policy_hardening_oracle/summary.json`](../../results/knowledge_work/replayable_policy_hardening_oracle/summary.json)
- episodes:
  - `kwa_exec_vendor_access_hold`
  - `kwa_jobs_screening_hold`
  - `kwa_finance_billing_patch_hold`
- aggregate metrics:
  - `artifact_quality_avg = 1.0`
  - `browser_workflow_avg = 0.9818`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9363`

Live-web policy-hardening subset:

- [`results/knowledge_work/live_policy_hardening_oracle/summary.json`](../../results/knowledge_work/live_policy_hardening_oracle/summary.json)
- episodes:
  - `kwa_exec_live_vendor_access_hold`
  - `kwa_jobs_live_screening_hold`
  - `kwa_finance_live_billing_patch_hold`
- aggregate metrics:
  - `artifact_quality_avg = 1.0`
  - `browser_workflow_avg = 1.0`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9383`

## First Finished Model-Backed KnowledgeWorkArena Result

The first finished non-oracle episode baseline is:

- [`results/knowledge_work/model_backed_hf_exec_hold/summary.json`](../../results/knowledge_work/model_backed_hf_exec_hold/summary.json)

Configuration:

- backend: `hf_service`
- reasoner: `google/gemma-4-E2B-it`
- router backend: `heuristic`
- retriever backend: `heuristic`
- device: `mps`

Metrics:

- `artifact_quality_avg = 1.0`
- `browser_workflow_avg = 0.9836`
- `strict_interface_avg = 1.0`
- `recovered_execution_avg = 1.0`
- `real_world_readiness_avg = 0.9056`

## Partial Model-Backed Pilot

There is also a stopped exploratory pilot at:

- [`results/knowledge_work/model_backed_hf_reasoner_pilot/summary.json`](../../results/knowledge_work/model_backed_hf_reasoner_pilot/summary.json)

It completed:

- `kwa_jobs_tailored_packet`
- `kwa_finance_three_statement_model`

Partial aggregate:

- `real_world_readiness_avg = 0.9074`

It was intentionally stopped after two episodes so the repo could retain a finished single-episode baseline plus a clearly marked exploratory subset.

## First Fully Specialist-Backed KnowledgeWorkArena Pilot

The first bounded fully specialist-backed KWA run is:

- [`results/knowledge_work/model_backed_hf_specialists_finance/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_finance/summary.json)

Configuration:

- reasoner backend: `hf_service`
- reasoner: `google/gemma-4-E2B-it`
- router backend: `hf`
- router: `google/functiongemma-270m-it`
- router device: `cpu`
- retriever backend: `hf`
- retriever: `google/embeddinggemma-300m`
- retriever device: `cpu`
- episode: `kwa_finance_three_statement_model`

Metrics:

- `artifact_quality_avg = 1.0`
- `browser_workflow_avg = 1.0`
- `strict_interface_avg = 0.7`
- `recovered_execution_avg = 1.0`
- `real_world_readiness_avg = 0.8574`

Interpretation:

- specialist loading stability is no longer the blocker for bounded KWA runs
- the current weakness has moved to interface discipline inside the composed full-stack episode
- the failing surface is not artifact quality or browser realism; it is controller/router exactness under full-stack composition

## Cross-Role Specialist-Backed Snapshot

The newer bounded cross-role specialist-backed run is:

- [`results/knowledge_work/model_backed_hf_specialists_cross_role/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_cross_role/summary.json)

Episodes:

- `kwa_exec_board_send_hold`
- `kwa_jobs_submission_hold`
- `kwa_finance_three_statement_model`

Aggregate metrics:

- `artifact_quality_avg = 1.0`
- `browser_workflow_avg = 0.9896`
- `strict_interface_avg = 1.0`
- `recovered_execution_avg = 1.0`
- `real_world_readiness_avg = 0.9063`

Interpretation:

- the planner hardening fixed the earlier specialist-backed finance interface leak
- the stack is now stable enough across three role families to move the next hardening pass toward tougher branch realism and broader specialist-backed volume

## Broader Cross-Role Specialist-Backed Snapshot

The current larger replayable cross-role specialist-backed reference is:

- [`results/knowledge_work/model_backed_hf_specialists_cross_role_broad_v2/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_cross_role_broad_v2/summary.json)

Episodes:

- `kwa_exec_board_prep_pack`
- `kwa_exec_inbox_triage`
- `kwa_exec_vendor_access_hold`
- `kwa_jobs_tailored_packet`
- `kwa_jobs_revise_after_feedback`
- `kwa_jobs_submission_hold`
- `kwa_finance_three_statement_model`
- `kwa_finance_partner_deck_revision`
- `kwa_finance_billing_patch_hold`

Aggregate metrics:

- `artifact_quality_avg = 1.0`
- `browser_workflow_avg = 0.9939`
- `strict_interface_avg = 1.0`
- `recovered_execution_avg = 1.0`
- `real_world_readiness_avg = 0.9175`
- `escalation_correctness_avg = 1.0`

Interpretation:

- the broader replayable specialist-backed stack is now clean across policy and non-policy episodes in all three role families
- the decisive fix was controller-side, not model-side: the planner now enforces full parallel evidence batches before accepting a patch proposal and repairs patch arguments from combined image + repo feedback
- the previous broad replayable `v1` run should be treated as a diagnosis artifact for `agent_010_parallel_audit_patch`, not the current reference state

## Broader Live Cross-Role Specialist-Backed Snapshot

The current larger live-web stress cross-role specialist-backed reference is:

- [`results/knowledge_work/model_backed_hf_specialists_cross_role_live_broad_v1/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_cross_role_live_broad_v1/summary.json)

Episodes:

- `kwa_exec_live_brief`
- `kwa_exec_live_calendar_policy`
- `kwa_exec_live_vendor_access_hold`
- `kwa_jobs_live_requirements_extract`
- `kwa_jobs_live_career_plan`
- `kwa_jobs_live_submission_hold`
- `kwa_finance_live_earnings_update`
- `kwa_finance_live_comps_revision`
- `kwa_finance_live_billing_patch_hold`

Aggregate metrics:

- `artifact_quality_avg = 1.0`
- `browser_workflow_avg = 1.0`
- `strict_interface_avg = 1.0`
- `recovered_execution_avg = 1.0`
- `real_world_readiness_avg = 0.9794`
- `escalation_correctness_avg = 1.0`

Interpretation:

- the broader live specialist-backed stack is also clean across policy and non-policy episodes in all three role families
- the controller/planner hardening that fixed the replayable parallel-audit leak appears to generalize under the live-web stress lane
- the next uncertainty is no longer “can the stack stay interface-clean on a broader cross-role live slice?” but how far we can widen the model-backed matrix before new mixed-evidence or revision failures emerge

## Mixed-Pressure Specialist-Backed Cross-Role References

Replayable mixed-pressure broad reference:

- [`results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_visual_replayable_v2/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_visual_replayable_v2/summary.json)
- episodes:
  - broad cross-role core:
    - `kwa_exec_board_prep_pack`
    - `kwa_exec_inbox_triage`
    - `kwa_jobs_tailored_packet`
    - `kwa_jobs_revise_after_feedback`
    - `kwa_finance_three_statement_model`
    - `kwa_finance_partner_deck_revision`
    - `kwa_jobs_submission_hold`
    - `kwa_exec_vendor_access_hold`
    - `kwa_finance_billing_patch_hold`
  - harder human-nuance additions:
    - `kwa_exec_stale_brief_hold`
    - `kwa_jobs_constraint_preservation_hold`
    - `kwa_finance_stale_assumption_hold`
  - visual additions:
    - `kwa_exec_visual_dashboard_brief`
    - `kwa_jobs_visual_form_hold`
    - `kwa_finance_visual_invoice_hold`
- aggregate metrics:
  - `runs = 18`
  - `artifact_quality_avg = 0.9822`
  - `browser_workflow_avg = 0.9880`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9436`
  - `escalation_correctness_avg = 1.0`

Live mixed-pressure broad reference:

- [`results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_visual_live_v2/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_visual_live_v2/summary.json)
- episodes:
  - broad live cross-role core:
    - `kwa_exec_live_brief`
    - `kwa_exec_live_calendar_policy`
    - `kwa_jobs_live_requirements_extract`
    - `kwa_jobs_live_career_plan`
    - `kwa_jobs_live_submission_hold`
    - `kwa_finance_live_earnings_update`
    - `kwa_finance_live_comps_revision`
    - `kwa_exec_live_vendor_access_hold`
    - `kwa_finance_live_billing_patch_hold`
  - harder human-nuance additions:
    - `kwa_exec_live_stale_brief_hold`
    - `kwa_jobs_live_constraint_hold`
    - `kwa_finance_live_stale_assumption_hold`
  - visual additions:
    - `kwa_exec_live_visual_dashboard_brief`
    - `kwa_jobs_live_visual_form_hold`
    - `kwa_finance_live_visual_invoice_hold`
- aggregate metrics:
  - `runs = 15`
  - `artifact_quality_avg = 0.9786`
  - `browser_workflow_avg = 1.0`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9555`
  - `escalation_correctness_avg = 1.0`

Interpretation:

- the old non-visual mixed-pressure references are no longer the best summary of the current model-backed frontier
- the corrected replayable reference is `model_backed_hf_specialists_cross_role_hardmix_visual_replayable_v2`
- the corrected live reference is `model_backed_hf_specialists_cross_role_hardmix_visual_live_v2`
- the earlier `visual_*_v1` runs should be treated as diagnosis artifacts for image-id plumbing and visual answer-surface rescue, not the current reference state
- the current remaining softer realism signal is `kwa_finance_partner_deck_revision`:
  - revision responsiveness is still weak
  - memory retention is now scored correctly after the semantic scorer hardening

## Specialist-Backed Visual KWA Snapshot

Replayable visual slice:

- [`results/knowledge_work/model_backed_hf_specialists_visual_replayable_v3/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_visual_replayable_v3/summary.json)
- episodes:
  - `kwa_exec_visual_dashboard_brief`
  - `kwa_jobs_visual_form_hold`
  - `kwa_finance_visual_invoice_hold`
- aggregate metrics:
  - `artifact_quality_avg = 0.8932`
  - `browser_workflow_avg = 0.9782`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9141`

Interpretation:

- the specialist-backed stack now handles the job-shaped visual episodes cleanly in bounded form
- the first failing replayable visual slice was a benchmark plumbing issue, not a stable model weakness: the planner needed logical image ids, and visual tasks needed second-pass answer rescue
- the remaining gap on this bounded visual slice is softer artifact quality, not strict interface discipline

## Full-Lane Specialist-Backed Exploratory References

Replayable full generated corpus:

- [`results/knowledge_work/model_backed_hf_specialists_replayable_full_v1/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_replayable_full_v1/summary.json)
- aggregate metrics:
  - `runs = 24`
  - `artifact_quality_avg = 0.9866`
  - `browser_workflow_avg = 0.9910`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9510`
  - `escalation_correctness_avg = 1.0`

Live full generated corpus:

- [`results/knowledge_work/model_backed_hf_specialists_live_full_v1/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_live_full_v1/summary.json)
- aggregate metrics:
  - `runs = 18`
  - `artifact_quality_avg = 0.9822`
  - `browser_workflow_avg = 1.0`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9630`
  - `escalation_correctness_avg = 1.0`

Interpretation:

- the current local specialist-backed stack is now clean on the entire generated KWA corpus in both replayable and live exploratory form
- the remaining deltas on the full-lane runs are softer artifact/browser-readiness movements, not strict interface failures
- this shifts the next benchmark frontier away from “can the stack survive the current corpus?” and toward broader system comparisons, richer public-style reporting, and harder new episode design

## Full-Lane Reasoner-Only Exploratory References

Replayable full generated corpus:

- [`results/knowledge_work/model_backed_hf_reasoner_full_replayable_v1/summary.json`](../../results/knowledge_work/model_backed_hf_reasoner_full_replayable_v1/summary.json)
- aggregate metrics:
  - `runs = 24`
  - `artifact_quality_avg = 0.9866`
  - `browser_workflow_avg = 0.9910`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9510`
  - `escalation_correctness_avg = 1.0`

Live full generated corpus:

- [`results/knowledge_work/model_backed_hf_reasoner_full_live_v1/summary.json`](../../results/knowledge_work/model_backed_hf_reasoner_full_live_v1/summary.json)
- aggregate metrics:
  - `runs = 18`
  - `artifact_quality_avg = 0.9822`
  - `browser_workflow_avg = 1.0`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9630`
  - `escalation_correctness_avg = 1.0`

Interpretation:

- the comparative full-lane board now cleanly resolves `reasoner_only` versus `specialist_stack` through registry-backed system ids instead of ad hoc slice labels
- the board prefers `full_lane` exploratory runs over narrower subset sweeps, so the top-line comparison surface is no longer shadowed by later partial slices

## Harder Human-Nuance Specialist-Backed Snapshots

Replayable harder slice:

- [`results/knowledge_work/model_backed_hf_specialists_hard_human_replayable_v1/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_hard_human_replayable_v1/summary.json)
- episodes:
  - `kwa_exec_stale_brief_hold`
  - `kwa_jobs_constraint_preservation_hold`
  - `kwa_finance_stale_assumption_hold`
- aggregate metrics:
  - `artifact_quality_avg = 1.0`
  - `browser_workflow_avg = 0.9828`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9364`
  - `escalation_correctness_avg = 1.0`

Live harder slice:

- [`results/knowledge_work/model_backed_hf_specialists_hard_human_live_v1/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_hard_human_live_v1/summary.json)
- episodes:
  - `kwa_exec_live_stale_brief_hold`
  - `kwa_jobs_live_constraint_hold`
  - `kwa_finance_live_stale_assumption_hold`
- aggregate metrics:
  - `artifact_quality_avg = 1.0`
  - `browser_workflow_avg = 1.0`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9383`
  - `escalation_correctness_avg = 1.0`

Interpretation:

- the specialist-backed stack stays interface-clean on the new harder stale-context / constraint-preservation / stale-assumption family
- the next benchmark frontier is composition: mix these harder human-nuance patterns into broader cross-role matrices with more revision pressure instead of testing them only in isolation

## Specialist-Backed Policy-Hardening Snapshots

Replayable specialist-backed subset:

- [`results/knowledge_work/model_backed_hf_specialists_policy_replayable_v6/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_policy_replayable_v6/summary.json)
- episodes:
  - `kwa_exec_vendor_access_hold`
  - `kwa_jobs_screening_hold`
  - `kwa_finance_billing_patch_hold`
- aggregate metrics:
  - `artifact_quality_avg = 1.0`
  - `browser_workflow_avg = 0.9818`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9363`
  - `escalation_correctness_avg = 1.0`

Interpretation:

- the replayable policy subset is now aligned with the live policy subset on the bounded three-episode specialist-backed slice
- the decisive fix was not more prompt pressure alone; it was judgment-aware scoring and rescue acceptance keyed to `expected_action + basis`, plus broader semantic aliases for policy-safety language like `high-risk` and `safety control`
- exploratory replayable specialist-backed runs `v1` through `v5` should be treated as diagnosis artifacts, not the current reference state

Live specialist-backed subset:

- [`results/knowledge_work/model_backed_hf_specialists_policy_live/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_policy_live/summary.json)
- episodes:
  - `kwa_exec_live_vendor_access_hold`
  - `kwa_jobs_live_screening_hold`
  - `kwa_finance_live_billing_patch_hold`
- aggregate metrics:
  - `artifact_quality_avg = 1.0`
  - `browser_workflow_avg = 1.0`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9383`
  - `escalation_correctness_avg = 1.0`

Interpretation:

- bounded replayable and live policy-hardening specialist-backed slices are now both clean on the current three-episode subset
- the next model-backed realism step is broader volume and harder branching, not another pass on the same refusal/clarify bug

## Broader Specialist-Backed Policy Exploratory Snapshots

Replayable broad policy subset:

- [`results/knowledge_work/model_backed_hf_specialists_policy_replayable_broad_v2/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_policy_replayable_broad_v2/summary.json)
- episodes:
  - `kwa_exec_board_send_hold`
  - `kwa_exec_vendor_access_hold`
  - `kwa_jobs_submission_hold`
  - `kwa_jobs_screening_hold`
  - `kwa_finance_committee_hold`
  - `kwa_finance_billing_patch_hold`
- aggregate metrics:
  - `artifact_quality_avg = 1.0`
  - `browser_workflow_avg = 0.9827`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9364`
  - `escalation_correctness_avg = 1.0`

Live broad policy subset:

- [`results/knowledge_work/model_backed_hf_specialists_policy_live_broad_v3/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_policy_live_broad_v3/summary.json)
- episodes:
  - `kwa_exec_live_send_hold`
  - `kwa_exec_live_vendor_access_hold`
  - `kwa_jobs_live_submission_hold`
  - `kwa_jobs_live_screening_hold`
  - `kwa_finance_live_committee_hold`
  - `kwa_finance_live_billing_patch_hold`
- aggregate metrics:
  - `artifact_quality_avg = 1.0`
  - `browser_workflow_avg = 1.0`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9383`
  - `escalation_correctness_avg = 1.0`

Interpretation:

- the clean policy result now generalizes beyond the original three-episode replayable/live subset
- the remaining high-value model-backed expansion work is now broader cross-role and non-policy volume, not another pass on the same hold/refusal bug

## KnowledgeWorkArena History Semantics

The history layer now distinguishes:

- latest canonical by lane
- latest exploratory by lane
- best historical by lane

Operationally:

- canonical top-line claims should still come from the explicit summary pointers above
- exploratory model-backed pilots should use `--no-update-latest --run-intent exploratory`
- generated history now reflects that split instead of treating every newest lane snapshot as equally authoritative

Interpretation:

- the same specialist-backed stack stayed clean on the live subset
- that suggests the current weakness is not generic model instability
- it is specific to certain replayable judgment/action-label surfaces, especially refusal vs defer in the replayable billing path

## Runtime Posture

- `hf_service` is now a core research execution primitive on this Mac.
- cold `hf_service` startup remains expensive:
  - the finished executive pilot required about `345s` from service boot to ready state
- the latest bounded specialist-backed finance pilot warmed to ready in about `37.9s`, showing that the reusable service path is materially better once the runtime is already in a usable local state
- `MLX` remains useful but session-sensitive; use current preflight rather than static prose when choosing local defaults
- real specialist-backed KWA runs now consistently surface the same decoding warning on the HF specialist path:
  - `top_p` / `top_k` generation flags may be ignored
  - this is not yet tied to a measured quality regression, but it should be cleaned up because backend-specific decoding drift is benchmark noise

## Artifact and Browser Realism

`KnowledgeWorkArena` now uses:

- real `.xlsx` finance/model work products
- real `.pptx` deck work products
- real `.docx` packet/memo/form work products
- seeded browser state machines with:
  - explicit transitions
  - `validation_failed` and `recovered` recovery branches
  - validation rules
  - approval gates
  - explicit policy blocks for unsafe requests
  - blocked submissions
  - sandbox endpoints for dry-run flows

## Important Caveat

Canonical lane pointers must not be overwritten by smoke runs or pilots.

Use:

- `--no-update-latest`

for any exploratory run that is not intended to replace the published replayable or live lane snapshot.

Also note:

- history views such as `latest by lane` can surface smaller exploratory runs that share the same lane label
- when this happens, the authoritative KWA pointers are the canonical summary files above, not the generic latest-lane aggregate
