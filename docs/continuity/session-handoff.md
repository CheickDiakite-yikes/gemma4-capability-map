# Session Handoff

## Resume Here

The repo is in a stronger state after turning the direct in-process Gemma specialist stack into a publishable-default full-lane result rather than a weaker appendix lane.

## Latest Headline Result

The current strongest research claim in the repo is now:

- on the publishable-default full-lane `KnowledgeWorkArena` matrix, the direct in-process Gemma 4 specialist stack matches the oracle row on the same `26 / 20` surface
- the reasoner-only Gemma control remains materially weaker on that same surface
- that means the controller/runtime/specialist-stack work materially improved Gemma 4 as a full-stack local agent on our own benchmark

Current board-backed headline rows:

- oracle:
  - `oracle_gemma4_e2b`
  - comparison batch:
    - `20260411T142324Z_knowledge_work_full_lane`
  - replayable:
    - `runs = 26`
    - `strict_interface_avg = 0.9711538461538461`
    - `recovered_execution_avg = 0.9615384615384616`
    - `real_world_readiness_avg = 0.9668576923076924`
  - live:
    - `runs = 20`
    - `strict_interface_avg = 0.9625`
    - `recovered_execution_avg = 0.95`
    - `real_world_readiness_avg = 0.966045`
- local headline Gemma row:
  - `hf_gemma4_e2b_specialists_cpu`
  - comparison batch:
    - `20260411T152330Z_knowledge_work_publishable_core`
  - replayable:
    - `runs = 26`
    - `strict_interface_avg = 0.9711538461538461`
    - `recovered_execution_avg = 0.9615384615384616`
    - `real_world_readiness_avg = 0.9668576923076924`
  - live:
    - `runs = 20`
    - `strict_interface_avg = 0.9625`
    - `recovered_execution_avg = 0.95`
    - `real_world_readiness_avg = 0.966045`
- local control:
  - `hf_gemma4_e2b_reasoner_only`
  - replayable:
    - `strict_interface_avg = 0.9038461538461539`
    - `recovered_execution_avg = 0.8846153846153846`
    - `real_world_readiness_avg = 0.9392653846153846`
  - live:
    - `strict_interface_avg = 0.875`
    - `recovered_execution_avg = 0.85`
    - `real_world_readiness_avg = 0.9347899999999999`

Important claim boundary:

- this is now a strong Gemma-improvement claim
- it is not yet a valid Gemma-versus-Qwen claim because there is still no full-lane Qwen run in the repo
- the repo now has Qwen-ready comparator support:
  - `Qwen/Qwen3-8B` is registered as an experimental local reasoner-only system
  - the HF runner supports tokenizer-based text generation for non-Gemma models
  - the experimental matrix includes the Qwen row

What just landed:

- shared local-agent runtime substrate with:
  - persistent sessions
  - event timelines
  - approval requests
  - packaged workflows
  - trace/artifact persistence
- first-class local entrypoints:
  - `moonie-agent`
  - `moonie-agent-api`
- Streamlit `operator_console` and `mobile_companion` surfaces over the same runtime contract
- runtime/API regression coverage:
  - `tests/test_runtime_core.py`
  - `tests/test_runtime_api.py`
- new atomic `visual_tool_orchestration` benchmark family with replayable and live lanes
- seeded/local visual executor abstraction and four visual tools:
  - `segment_entities`
  - `refine_selection`
  - `extract_layout`
  - `read_region_text`
- generated visual gold corpus now totals `22` tasks; current canonical lane pointers cover `11` replayable + `7` live seeded tasks
- visual KWA bounded slices for executive, jobs, and finance episodes
- corrected visual planner/image-id plumbing and visual answer-surface rescue
- older canonical KWA oracle lane pointers still reflect the last full oracle rerun on `24` replayable and `18` live episodes
- current generated KWA corpus is now `26` replayable and `20` live episodes
- registry-backed KWA benchmark board and scatter exports
- Streamlit `knowledge_work_board` mode
- explicit `system_id` support in KWA run manifests
- role/category/track board exports
- board rows now expose `warmup_load_ms`, `last_request_elapsed_ms`, `requests_completed`, and `total_cost_per_mtok` when manifests support them
- board rows now also carry `run_scope`, and latest-board selection prefers `full_lane` exploratory runs over later subset sweeps
- mixed-pressure specialist-backed replayable broad visual reference at `model_backed_hf_specialists_cross_role_hardmix_visual_replayable_v2`
- mixed-pressure specialist-backed live broad visual reference at `model_backed_hf_specialists_cross_role_hardmix_visual_live_v2`
- replay/rescore utility for saved KWA traces
- semantic memory-retention scoring for revision-heavy artifacts
- stronger revision contract for `kwa_finance_partner_deck_revision`
- judgment-aware answer scoring for `refuse` / `defer` / `clarify` / `escalate`
- judgment-aware second-pass rescue acceptance
- broader semantic aliases for policy-safety language like `high-risk` and `safety control`
- clean replayable specialist-backed policy subset at `v6`
- clean broader 6-episode specialist-backed replayable and live policy slices
- explicit `canonical` vs `exploratory` KWA history semantics
- planner-enforced full parallel audit batches before patch proposals
- combined-feedback patch repair for image + repo audit stages
- clean broader 9-episode replayable specialist-backed cross-role slice at `model_backed_hf_specialists_cross_role_broad_v2`
- clean broader 9-episode live specialist-backed cross-role slice at `model_backed_hf_specialists_cross_role_live_broad_v1`
- `KnowledgeWorkArena` first expanded to `21` replayable and `15` live episodes, then continued growing to the current `26 / 20` generated surface
- new harder canonical episodes for stale context, constraint preservation, and stale assumptions
- canonical KWA runner no longer silently truncates to `12` episodes by default
- clean 3-episode replayable harder human-nuance specialist-backed slice at `model_backed_hf_specialists_hard_human_replayable_v1`
- clean 3-episode live harder human-nuance specialist-backed slice at `model_backed_hf_specialists_hard_human_live_v1`
- refreshed continuity and history artifacts
- publishable-default direct in-process Gemma specialist full-lane rerun at:
  - `20260411T152330Z_knowledge_work_publishable_core`
- regenerated board/history rows showing the headline Gemma specialist stack matching the oracle full-lane row on the publishable-default matrix

## Current Canonical Pointers

- visual-tool orchestration replayable core:
  - [`results/visual_tool_orchestration/replayable_core/summary.json`](../../results/visual_tool_orchestration/replayable_core/summary.json)
- visual-tool orchestration live-web stress:
  - [`results/visual_tool_orchestration/live_web_stress/summary.json`](../../results/visual_tool_orchestration/live_web_stress/summary.json)
- replayable core:
  - [`results/knowledge_work/replayable_core/summary.json`](../../results/knowledge_work/replayable_core/summary.json)
- live-web stress:
  - [`results/knowledge_work/live_web_stress/summary.json`](../../results/knowledge_work/live_web_stress/summary.json)
- replayable visual KWA oracle slice:
  - [`results/knowledge_work/kwa_visual_replayable_oracle_v1/summary.json`](../../results/knowledge_work/kwa_visual_replayable_oracle_v1/summary.json)
- live visual KWA oracle slice:
  - [`results/knowledge_work/kwa_visual_live_oracle_v1/summary.json`](../../results/knowledge_work/kwa_visual_live_oracle_v1/summary.json)
- replayable specialist-backed visual KWA slice:
  - [`results/knowledge_work/model_backed_hf_specialists_visual_replayable_v3/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_visual_replayable_v3/summary.json)
- replayable specialist-backed full-lane exploratory slice:
  - [`results/knowledge_work/model_backed_hf_specialists_replayable_full_v1/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_replayable_full_v1/summary.json)
- live specialist-backed full-lane exploratory slice:
  - [`results/knowledge_work/model_backed_hf_specialists_live_full_v1/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_live_full_v1/summary.json)
- replayable reasoner-only full-lane exploratory slice:
  - [`results/knowledge_work/model_backed_hf_reasoner_full_replayable_v1/summary.json`](../../results/knowledge_work/model_backed_hf_reasoner_full_replayable_v1/summary.json)
- live reasoner-only full-lane exploratory slice:
  - [`results/knowledge_work/model_backed_hf_reasoner_full_live_v1/summary.json`](../../results/knowledge_work/model_backed_hf_reasoner_full_live_v1/summary.json)
- replayable direct-HF in-process reasoner-only full-lane exploratory slice:
  - [`results/knowledge_work/model_backed_hf_inprocess_reasoner_full_replayable_v1/summary.json`](../../results/knowledge_work/model_backed_hf_inprocess_reasoner_full_replayable_v1/summary.json)
- live direct-HF in-process reasoner-only full-lane exploratory slice:
  - [`results/knowledge_work/model_backed_hf_inprocess_reasoner_full_live_v1/summary.json`](../../results/knowledge_work/model_backed_hf_inprocess_reasoner_full_live_v1/summary.json)
- replayable specialist-backed policy subset:
  - [`results/knowledge_work/model_backed_hf_specialists_policy_replayable_v6/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_policy_replayable_v6/summary.json)
- live specialist-backed policy subset:
  - [`results/knowledge_work/model_backed_hf_specialists_policy_live/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_policy_live/summary.json)
- replayable specialist-backed policy broad subset:
  - [`results/knowledge_work/model_backed_hf_specialists_policy_replayable_broad_v2/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_policy_replayable_broad_v2/summary.json)
- live specialist-backed policy broad subset:
  - [`results/knowledge_work/model_backed_hf_specialists_policy_live_broad_v3/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_policy_live_broad_v3/summary.json)
- broader replayable specialist-backed cross-role subset:
  - [`results/knowledge_work/model_backed_hf_specialists_cross_role_broad_v2/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_cross_role_broad_v2/summary.json)
- broader live specialist-backed cross-role subset:
  - [`results/knowledge_work/model_backed_hf_specialists_cross_role_live_broad_v1/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_cross_role_live_broad_v1/summary.json)
- replayable harder human-nuance specialist-backed subset:
  - [`results/knowledge_work/model_backed_hf_specialists_hard_human_replayable_v1/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_hard_human_replayable_v1/summary.json)
- live harder human-nuance specialist-backed subset:
  - [`results/knowledge_work/model_backed_hf_specialists_hard_human_live_v1/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_hard_human_live_v1/summary.json)
- replayable mixed-pressure specialist-backed broad subset:
  - [`results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_visual_replayable_v2/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_visual_replayable_v2/summary.json)
- live mixed-pressure specialist-backed broad subset:
  - [`results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_visual_live_v2/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_visual_live_v2/summary.json)
- benchmark board latest:
  - [`results/history/knowledge_work_board_latest.csv`](../../results/history/knowledge_work_board_latest.csv)
- benchmark scatter export:
  - [`results/history/knowledge_work_scatter.csv`](../../results/history/knowledge_work_scatter.csv)
- benchmark role breakdown:
  - [`results/history/knowledge_work_role_breakdown.csv`](../../results/history/knowledge_work_role_breakdown.csv)
- benchmark category breakdown:
  - [`results/history/knowledge_work_category_breakdown.csv`](../../results/history/knowledge_work_category_breakdown.csv)
- benchmark track breakdown:
  - [`results/history/knowledge_work_track_breakdown.csv`](../../results/history/knowledge_work_track_breakdown.csv)
- knowledge-work history:
  - [`results/history/knowledge_work_history.md`](../../results/history/knowledge_work_history.md)

## Finished Model-Backed Evidence

Primary finished baseline:

- [`results/knowledge_work/model_backed_hf_exec_hold/summary.json`](../../results/knowledge_work/model_backed_hf_exec_hold/summary.json)

Exploratory stopped pilot:

- [`results/knowledge_work/model_backed_hf_reasoner_pilot/summary.json`](../../results/knowledge_work/model_backed_hf_reasoner_pilot/summary.json)

## What To Do First In The Next Thread

1. Read [`../../AGENT_CONTEXT.md`](../../AGENT_CONTEXT.md).
2. Check `git status --short`.
3. If new result dirs were added, run:
   - `uv run python scripts/build_knowledge_work_history.py`
4. If scoring logic changed, rescore affected run dirs before treating leaderboard deltas as real:
   - `uv run python scripts/rescore_knowledge_work_runs.py <run_dir> ...`
5. If making new benchmark claims, run:
   - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q`
6. Treat [`current-state.md`](./current-state.md) as the canonical source of truth when `knowledge_work_history.md` and exploratory runs disagree.
7. If touching the product surfaces, validate both:
   - `uv run moonie-agent workflows`
   - `uv run pytest tests/test_runtime_core.py tests/test_runtime_api.py`

## Highest-Value Next Work

1. Harden the shared runtime so benchmark execution and product sessions keep one execution contract.
2. Extend the operator console, mobile companion, and board into a more polished shared product/reporting surface.
3. Broaden system coverage on the same full-lane KWA surface instead of only widening volume further.
4. Keep inspecting softer-realism signals instead of only binary failures.

## Important Operational Notes

- Use `--no-update-latest` for all ad hoc pilots.
- Do not let smoke runs overwrite `results/knowledge_work/replayable_core`.
- The `hf_service` reasoner is useful but cold startup is expensive on this Mac.
- The replayable specialist-backed policy bug is closed in `v6`; do not resume from older `v1`-`v5` policy conclusions.
- The broader 6-episode policy exploratory slices are now also clean; the next thread should not spend time re-debugging `kwa_exec_vendor_access_hold` unless a fresh regression appears.
- The broader replayable cross-role `v1` miss was a planner/controller bug in `agent_010_parallel_audit_patch`, not a general finance-deck weakness.
- The corrected reference is `model_backed_hf_specialists_cross_role_broad_v2`; do not continue from `v1` conclusions.
- The broader live specialist-backed cross-role reference is `model_backed_hf_specialists_cross_role_live_broad_v1`; it is clean and should be used as the current live counterpart to the replayable broad slice.
- The canonical oracle KWA lanes now include the harder stale-context / constraint-preservation / stale-assumption episodes; any future “canonical KWA” claim should assume `21` replayable and `15` live episodes, not the older `18/12` surface.
- The canonical oracle KWA lanes have now been rerun on the full generated corpus and should be treated as `24` replayable and `18` live episodes, not the older `21/15` surface.
- The new harder model-backed references are `model_backed_hf_specialists_hard_human_replayable_v1` and `model_backed_hf_specialists_hard_human_live_v1`; both are clean and should be treated as the current proof that the specialist-backed stack handles these harder human-nuance patterns in bounded form.
- The current mixed-pressure replayable reference is `model_backed_hf_specialists_cross_role_hardmix_visual_replayable_v2`; older non-visual or `visual_*_v1` runs are diagnosis artifacts and should not be used as the current model-backed reference.
- The current mixed-pressure live reference is `model_backed_hf_specialists_cross_role_hardmix_visual_live_v2`; it is the current live counterpart to the replayable mixed-pressure visual slice.
- The broader full-lane exploratory references are now `model_backed_hf_specialists_replayable_full_v1` and `model_backed_hf_specialists_live_full_v1`; use them when making claims about the current local specialist-backed stack on the full generated KWA corpus.
- The new direct in-process HF full-lane exploratory references are `model_backed_hf_inprocess_reasoner_full_replayable_v1` and `model_backed_hf_inprocess_reasoner_full_live_v1`; they are weaker than `hf_service_gemma4_reasoner_only` and should be treated as a real comparative baseline, not as a replacement for the existing service-backed reasoner baseline.
- The new direct in-process HF specialist-backed full-lane exploratory references are `model_backed_hf_inprocess_specialists_full_replayable_v3` and `model_backed_hf_inprocess_specialists_full_live_v3`; they materially improve on the direct-HF reasoner-only baseline, recover the visual invoice/form execution misses cleanly at the full-lane level, and now absorb the invoice memo-quality repair into the board-facing rows.
- The worst episodes for `hf_gemma4_e2b_reasoner_only` are the visual KWA episodes:
  - replayable:
    - `kwa_exec_visual_dashboard_brief`
    - `kwa_jobs_visual_form_hold`
    - `kwa_finance_visual_invoice_hold`
  - live:
    - `kwa_exec_live_visual_dashboard_brief`
    - `kwa_jobs_live_visual_form_hold`
    - `kwa_finance_live_visual_invoice_hold`
- The dashboard stale-selection regression is closed in the planner/controller path; do not resume from the earlier conclusion that `kwa_exec_visual_dashboard_brief` was the blocker for the in-process specialist lane.
- The visual invoice/form referent-repair regression is now closed in bounded reruns for both service-backed and direct-HF specialist paths:
  - replayable service-backed:
    - `results/knowledge_work/model_backed_hf_service_specialists_smoke_finance_visual_replayable_v1`
  - live service-backed:
    - `results/knowledge_work/model_backed_hf_service_specialists_smoke_finance_visual_live_v1`
    - `results/knowledge_work/model_backed_hf_service_specialists_smoke_jobs_visual_live_v1`
  - replayable direct-HF specialists:
    - `results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_replayable_v2`
  - live direct-HF specialists:
    - `results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_live_v2`
    - `results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_jobs_visual_live_v2`
- Those targeted reruns now show `strict_interface = 1.0` and `recovered_execution = 1.0` on all three previously failing episodes.
- The softer invoice artifact/readiness gap is also closed in bounded reruns after the memo generator/review path was hardened:
  - replayable:
    - `results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_replayable_v3`
    - `artifact_quality = 1.0`
  - live:
    - `results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_live_v3`
    - `artifact_quality = 1.0`
- The older bounded `v2` / `v3` direct-HF specialist reruns are now diagnostic history, not the current headline claim surface.
- The current headline claim surface is the publishable-default full-lane board:
  - `results/history/knowledge_work_board_latest.csv`
  - with the `hf_gemma4_e2b_specialists_cpu` row from `20260411T152330Z_knowledge_work_publishable_core`
- Use the bounded invoice/form reruns only as proof of where the controller and memo fixes came from, not as the main comparison artifact.
- The next meaningful comparison step is a real non-Gemma local stack, not more reruns of the now-clean Gemma specialist row.
- The new atomic visual-tool benchmark is canonical at `results/visual_tool_orchestration/...`; use it when making claims about multimodal tool orchestration rather than inferring from KWA alone.
- The board/reporting layer now depends on [`configs/model_registry.yaml`](../../configs/model_registry.yaml) plus the exports in [`results/history`](../../results/history); update both when adding new systems or public-style charts.
- The repo is no longer benchmark-only:
  - use the shared runtime entrypoints for product-surface work
  - keep packaged workflows benchmark-backed and bounded
  - do not let desktop/mobile shells invent a second orchestration model
- `kwa_finance_partner_deck_revision` is clean on the current corrected mixed-pressure visual reference; if it regresses again, treat it as a soft-realism target, not a hard interface failure.
- `mlx` should not be the next execution target on this machine until the runtime exists locally; the current probe still fails with `ModuleNotFoundError: mlx`.
- `google/gemma-4-E4B-it` remains a probe-only local lane here and should not be promoted to a full-lane comparison until hardware/runtime conditions change.
- The current branch is expected to remain `main` unless a feature branch is explicitly needed.
