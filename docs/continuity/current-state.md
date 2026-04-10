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

- `52` gold atomic tasks
- `232` explicit atomic variants
- `16` real-world-tagged atomic tasks
- `15` replayable-core `KnowledgeWorkArena` episodes
- `9` live-web stress `KnowledgeWorkArena` episodes

## Canonical Atomic Benchmark Pointers

- real-world autonomy matrix:
  - [`results/alpha_matrix/20260409T210500Z_alpha_real_world`](../../results/alpha_matrix/20260409T210500Z_alpha_real_world)
- atomic benchmark history:
  - [`results/history/history_report.md`](../../results/history/history_report.md)

## Canonical KnowledgeWorkArena Pointers

Replayable core:

- [`results/knowledge_work/replayable_core/summary.json`](../../results/knowledge_work/replayable_core/summary.json)
- source snapshot:
  - [`results/knowledge_work/20260410T021825Z_replayable_core/summary.json`](../../results/knowledge_work/20260410T021825Z_replayable_core/summary.json)
- current metrics:
  - `artifact_quality_avg = 1.0`
  - `browser_workflow_avg = 0.9955`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9320`

Live-web stress:

- [`results/knowledge_work/live_web_stress/summary.json`](../../results/knowledge_work/live_web_stress/summary.json)
- source snapshot:
  - [`results/knowledge_work/20260410T021845Z_live_web_stress/summary.json`](../../results/knowledge_work/20260410T021845Z_live_web_stress/summary.json)
- current metrics:
  - `artifact_quality_avg = 1.0`
  - `browser_workflow_avg = 1.0`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9794`

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

## Runtime Posture

- `hf_service` is now a core research execution primitive on this Mac.
- cold `hf_service` startup remains expensive:
  - the finished executive pilot required about `345s` from service boot to ready state
- `MLX` remains useful but session-sensitive; use current preflight rather than static prose when choosing local defaults

## Artifact and Browser Realism

`KnowledgeWorkArena` now uses:

- real `.xlsx` finance/model work products
- real `.pptx` deck work products
- real `.docx` packet/memo/form work products
- seeded browser state machines with:
  - explicit transitions
  - validation rules
  - approval gates
  - blocked submissions
  - sandbox endpoints for dry-run flows

## Important Caveat

Canonical lane pointers must not be overwritten by smoke runs or pilots.

Use:

- `--no-update-latest`

for any exploratory run that is not intended to replace the published replayable or live lane snapshot.
