# KnowledgeWorkArena

`KnowledgeWorkArena` is the role-based benchmark layer built on top of the task and variant benchmark substrate in this repo.

## Design

- `replayable_core` is the canonical lane.
- `live_web_stress` is a supplementary realism lane.
- Memory is episode-scoped.
- Primary scoring truth is artifact quality plus workflow trace together.

## Role Families

- `executive_assistant`
- `job_application_ops`
- `finance`

## Episode Model

Each episode composes atomic benchmark tasks into a higher-level job-shaped workflow with:

- staged goals
- required artifacts
- revision rounds
- browser/environment metadata
- episode-level memory updates
- role-level scorecards

## Score Layers

- `strict_interface_score`
- `recovered_execution_score`
- `role_readiness_score`

The episode scorecard also records:

- `artifact_quality_score`
- `browser_workflow_score`
- `revision_responsiveness`
- `memory_retention_score`
- `escalation_correctness`
- `collateral_damage_free`
- `human_time_ratio`

## Domain-Native Artifact Grading

Artifact grading is now kind-aware instead of relying only on generic fragment checks.

- `memo`, `email`, `research_note`, and `schedule`
  - section checks
  - required bullets
  - citation requirements
- `spreadsheet` and `model`
  - required markdown-table rows
  - evidence columns
  - citation requirements
- `deck`
  - required slide titles
  - slide bullet checks
  - citation requirements
- `form_submission`
  - required field/value pairs
  - response-summary structure
  - citation requirements

The current implementation still uses deterministic markdown artifacts, but the grading contracts now reflect the structure expected from real work products rather than plain keyword presence alone.

## Browser Replay Metadata

Each episode stage can now carry an explicit browser plan. Traces retain:

- action
- target
- surface
- purpose
- expected signal
- evidence
- verification checks
- captured fields
- sandbox endpoint for dry-run submissions
- dry-run versus replayed status

Replayable-core stages use seeded workspace browser steps. Live-web stress stages use dry-run public-web steps and never record real external side effects.

`browser_workflow_score` is now part of the role-readiness blend. It rewards explicit purpose, evidence, verification, and safe submission behavior instead of counting browser activity alone as progress.

## Current Seed Set

- `12` replayable-core episodes
- `6` live-web stress episodes

Data lives under:

- [`data/knowledge_work/replayable_core`](/Users/cheickdiakite/Codex/moonie/data/knowledge_work/replayable_core)
- [`data/knowledge_work/live_web_stress`](/Users/cheickdiakite/Codex/moonie/data/knowledge_work/live_web_stress)
- [`data/knowledge_work/workspaces`](/Users/cheickdiakite/Codex/moonie/data/knowledge_work/workspaces)
- [`data/knowledge_work/artifact_goldens`](/Users/cheickdiakite/Codex/moonie/data/knowledge_work/artifact_goldens)
- [`data/knowledge_work/review_comments`](/Users/cheickdiakite/Codex/moonie/data/knowledge_work/review_comments)

## Commands

Generate the seed episode specs and fixtures:

```bash
uv run python scripts/make_knowledge_work_arena.py
```

Run the replayable-core lane:

```bash
uv run python scripts/run_knowledge_work_arena.py --lane replayable_core --backend oracle
```

Run the live-web stress lane in dry-run mode:

```bash
uv run python scripts/run_knowledge_work_arena.py --lane live_web_stress --backend oracle
```

Episode outputs are written under:

- [`results/knowledge_work`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work)
