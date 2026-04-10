# Session Handoff

## Resume Here

The repo is in a stronger state after hardening the canonical oracle KWA surface with harder human-nuance episodes.

What just landed:

- registry-backed KWA benchmark board and scatter exports
- Streamlit `knowledge_work_board` mode
- explicit `system_id` support in KWA run manifests
- mixed-pressure specialist-backed replayable broad reference at `model_backed_hf_specialists_cross_role_hardmix_replayable_v2`
- mixed-pressure specialist-backed live broad reference at `model_backed_hf_specialists_cross_role_hardmix_live_v1`
- replay/rescore utility for saved KWA traces
- semantic memory-retention scoring for revision-heavy artifacts
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

- replayable core:
  - [`results/knowledge_work/replayable_core/summary.json`](../../results/knowledge_work/replayable_core/summary.json)
- live-web stress:
  - [`results/knowledge_work/live_web_stress/summary.json`](../../results/knowledge_work/live_web_stress/summary.json)
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
  - [`results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_replayable_v2/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_replayable_v2/summary.json)
- live mixed-pressure specialist-backed broad subset:
  - [`results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_live_v1/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_live_v1/summary.json)
- benchmark board latest:
  - [`results/history/knowledge_work_board_latest.csv`](../../results/history/knowledge_work_board_latest.csv)
- benchmark scatter export:
  - [`results/history/knowledge_work_scatter.csv`](../../results/history/knowledge_work_scatter.csv)
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

1. Extend the board/reporting layer into richer public-style charts and per-role/category cuts.
2. Widen the fully specialist-backed mixed-pressure matrix beyond the current `12` replayable and `12` live slices.
3. Investigate softer realism signals like `kwa_finance_partner_deck_revision` instead of only chasing binary failures.

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
- The new harder model-backed references are `model_backed_hf_specialists_hard_human_replayable_v1` and `model_backed_hf_specialists_hard_human_live_v1`; both are clean and should be treated as the current proof that the specialist-backed stack handles these harder human-nuance patterns in bounded form.
- The mixed-pressure replayable reference is `model_backed_hf_specialists_cross_role_hardmix_replayable_v2`; do not continue from `v1` conclusions because the refusal-versus-escalate billing miss was fixed after that run.
- The mixed-pressure live reference is `model_backed_hf_specialists_cross_role_hardmix_live_v1`; it is clean and should be treated as the current live counterpart to the replayable mixed-pressure slice.
- The board/reporting layer now depends on [`configs/model_registry.yaml`](../../configs/model_registry.yaml) plus the exports in [`results/history`](../../results/history); update both when adding new systems or public-style charts.
- After the semantic memory-retention scorer landed, `kwa_finance_partner_deck_revision` still shows weak revision responsiveness but no longer takes a false memory penalty. Treat it as a soft realism target, not a hard failure.
- The current branch is expected to remain `main` unless a feature branch is explicitly needed.
