# Session Handoff

## Resume Here

The repo is in a stronger state after hardening the canonical oracle KWA surface with harder human-nuance episodes.

What just landed:

- new atomic `visual_tool_orchestration` benchmark family with replayable and live lanes
- seeded/local visual executor abstraction and four visual tools:
  - `segment_entities`
  - `refine_selection`
  - `extract_layout`
  - `read_region_text`
- generated visual gold corpus at `11` replayable + `7` live tasks
- visual KWA bounded slices for executive, jobs, and finance episodes
- corrected visual planner/image-id plumbing and visual answer-surface rescue
- canonical KWA oracle lanes refreshed to the full `24` replayable and `18` live generated corpus
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
- `KnowledgeWorkArena` expanded to `21` replayable and `15` live episodes
- new harder canonical episodes for stale context, constraint preservation, and stale assumptions
- canonical KWA runner no longer silently truncates to `12` episodes by default
- clean 3-episode replayable harder human-nuance specialist-backed slice at `model_backed_hf_specialists_hard_human_replayable_v1`
- clean 3-episode live harder human-nuance specialist-backed slice at `model_backed_hf_specialists_hard_human_live_v1`
- refreshed continuity and history artifacts

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

## Highest-Value Next Work

1. Extend the board/reporting layer into richer public-style charts using the role/category/track exports plus latency and cost metadata.
2. Broaden system coverage on the same full-lane KWA surface instead of only widening volume further.
3. Keep inspecting softer-realism signals instead of only binary failures.

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
- The new direct in-process HF specialist-backed full-lane exploratory references are `model_backed_hf_inprocess_specialists_full_replayable_v1` and `model_backed_hf_inprocess_specialists_full_live_v1`; they materially improve on the direct-HF reasoner-only baseline but still remain weaker than `hf_service_gemma4_specialists_cpu`.
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
- The current concentrated misses for `hf_gemma4_e2b_specialists_cpu` are:
  - replayable:
    - `kwa_finance_visual_invoice_hold`
  - live:
    - `kwa_finance_live_visual_invoice_hold`
    - `kwa_jobs_live_visual_form_hold`
- That means the next debugging target is no longer “run the direct-HF specialists full lane.” That work is done. The next target is to isolate why those remaining visual invoice/form episodes still lose `strict_interface` or `recovered_execution` under the in-process specialist path.
- The new atomic visual-tool benchmark is canonical at `results/visual_tool_orchestration/...`; use it when making claims about multimodal tool orchestration rather than inferring from KWA alone.
- The board/reporting layer now depends on [`configs/model_registry.yaml`](../../configs/model_registry.yaml) plus the exports in [`results/history`](../../results/history); update both when adding new systems or public-style charts.
- `kwa_finance_partner_deck_revision` is clean on the current corrected mixed-pressure visual reference; if it regresses again, treat it as a soft-realism target, not a hard interface failure.
- `mlx` should not be the next execution target on this machine until the runtime exists locally; the current probe still fails with `ModuleNotFoundError: mlx`.
- `google/gemma-4-E4B-it` remains a probe-only local lane here and should not be promoted to full `24 / 18` until hardware/runtime conditions change.
- The current branch is expected to remain `main` unless a feature branch is explicitly needed.
