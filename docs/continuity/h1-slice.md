# H1 Slice

## Purpose

`H1` is the next harder Moonie research slice.

It exists because the aligned exploratory `32 / 26` KWA surface is partially saturated at the top-line readiness level. The remaining useful signal is no longer just whether a row finishes; it is how much controller help, fallback, argument repair, and approval judgment were needed to get there.

## Source Config

- [`configs/knowledge_work_h1_slice.yaml`](../../configs/knowledge_work_h1_slice.yaml)

## Design Rule

`H1 v1` is packaged-workflow-first.

That keeps benchmark attribution aligned with the CLI live harness:

- every live operator run maps to a packaged workflow
- every packaged workflow maps to replayable and live KWA episode ids
- every result can be grouped by workflow family, lane, role family, and H1 stressor

## Included Workflow Families

- `executive_visual_dashboard_review`
  - visual sanity/control row for the CLI live harness
- `executive_stale_brief_packet`
  - resume, latest-instruction override, project-memory carryover, API/function-call pressure
- `jobs_visual_form_hold`
  - conflict handling, CLI patch path, blocked-email repair, approval-safe stop
- `finance_billing_patch_hold`
  - review-only finance diff, invoice-lock direction, CLI/API choice, approval-safe stop
- `finance_visual_invoice_review`
  - visual invoice evidence, stale override, locked sign-off behavior

## Replayable Episode Set

- `kwa_exec_visual_dashboard_brief`
- `kwa_exec_backlog_resume_hold_v5`
- `kwa_jobs_email_block_resume_hold_v5`
- `kwa_finance_diff_review_hold_v5`
- `kwa_finance_invoice_lock_direction_hold_v4`

## Live Episode Set

- `kwa_exec_live_visual_dashboard_brief`
- `kwa_exec_live_backlog_resume_hold_v5`
- `kwa_jobs_live_email_block_resume_hold_v5`
- `kwa_finance_live_diff_review_hold_v5`
- `kwa_finance_live_invoice_lock_direction_hold_v4`

## Primary Read

The first H1 question is:

- can `mlx_gemma4_e2b_reasoner_only` stay clean when the slice concentrates resume, approval, latest-instruction, CLI/API, and revision pressure?

The second H1 question is:

- where does `hf_gemma4_e2b_specialists_cpu` still need controller help on the same families?

## Metrics To Watch

- `real_world_readiness_avg`
- `controller_repair_avg`
- `controller_fallback_avg`
- `raw_planning_clean_rate_avg`
- `strict_interface_score_avg`
- `role_readiness_score_avg`
- approval-safe stop behavior
- sandbox policy-block count on live runs

## Execution Discipline

- H1 is exploratory until rerun and summarized.
- H1 should not update latest board exports by default.
- Live H1 work should go through packaged workflows, not ad hoc prompts.
- Use H1 before another broad `32 / 26` rerun.

## Runner

Config-backed H1 execution now starts here:

```bash
uv run python scripts/run_knowledge_work_h1_slice.py --dry-run --run-set primary --lane replayable_core
```

Useful run sets:

- `primary`: `mlx_gemma4_e2b_reasoner_only`
- `comparison`: primary plus oracle, HF Gemma specialists, and MLX Qwen
- `ablation`: HF Gemma specialists plus configured ablation rows
- `all`: primary, comparison baselines, and ablation rows

The H1 runner validates workflow/episode mappings first, writes a run manifest, and delegates real execution to `scripts/run_knowledge_work_arena.py` with explicit `--episode-id` filters and `--no-update-latest`.

Current H1 ablation rows:

- baseline HF Gemma specialists
- no controller repair
- no controller fallback
- no visual rescue
- no intent priority
- no argument repair
- no deterministic visual follow-on
