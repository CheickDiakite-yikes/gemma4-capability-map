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
  - real `.xlsx` work products
  - required table rows
  - required formulas
  - evidence columns
  - citation requirements
- `deck`
  - real `.pptx` work products
  - required slide titles
  - slide bullet checks
  - required slide sections
  - revision diff preservation
  - citation requirements
- `form_submission`
  - required field/value pairs
  - response-summary structure
  - cross-field consistency checks
  - citation requirements
- `memo`, `email`, `research_note`, and role packets
  - real `.docx` work products when the episode contract expects them

Native artifacts are materialized through the runner and then re-read for grading. That means replayable-core now exercises real file-backed `.xlsx`, `.pptx`, and `.docx` benchmark outputs instead of markdown-only stand-ins for finance and job-application episodes.

## Browser Replay Metadata

Each episode stage can now carry an explicit browser plan. Traces retain:

- action
- target
- surface
- purpose
- expected signal
- evidence
- verification checks
- validation rules
- captured fields
- state updates
- submission-gate outcomes
- blocked reasons
- sandbox endpoint for dry-run submissions
- dry-run versus replayed status

Replayable-core stages use seeded workspace browser steps. Live-web stress stages use dry-run public-web steps and never record real external side effects. Both lanes now include partial-progress hold cases where the correct behavior is to stop at an approval gate instead of forcing completion.

`browser_workflow_score` is now part of the role-readiness blend. It rewards explicit purpose, evidence, verification, and safe submission behavior instead of counting browser activity alone as progress.

Replayable-core now also carries explicit browser state machines:

- state ids
- transition ids
- from/to state edges
- approval-required transitions
- blocked-submission transitions

That makes it possible to score not just whether a browser step happened, but whether the workflow moved through the right validation and approval states.

## Current Seed Set

- `32` replayable-core episodes in the generated corpus
- `26` live-web stress episodes in the generated corpus

Data lives under:

- [`data/knowledge_work/replayable_core`](/Users/cheickdiakite/Codex/moonie/data/knowledge_work/replayable_core)
- [`data/knowledge_work/live_web_stress`](/Users/cheickdiakite/Codex/moonie/data/knowledge_work/live_web_stress)
- [`data/knowledge_work/workspaces`](/Users/cheickdiakite/Codex/moonie/data/knowledge_work/workspaces)
- [`data/knowledge_work/artifact_goldens`](/Users/cheickdiakite/Codex/moonie/data/knowledge_work/artifact_goldens)
- [`data/knowledge_work/review_comments`](/Users/cheickdiakite/Codex/moonie/data/knowledge_work/review_comments)

The older canonical oracle lane pointers remain useful stable references, but the current research surface is wider than those older pointers. The active board-backed comparison surface is the aligned exploratory `32 / 26` matrix, and the active local Gemma harnessing surface is the H1/H1i packet family.

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

Run a model-backed replayable pilot without overwriting the canonical `latest` lane pointer:

```bash
uv run python scripts/run_knowledge_work_arena.py \
  --lane replayable_core \
  --backend hf_service \
  --router-backend heuristic \
  --retriever-backend heuristic \
  --reasoner google/gemma-4-E2B-it \
  --reasoner-device mps \
  --reasoner-max-new-tokens 96 \
  --episode-id kwa_exec_board_send_hold \
  --limit 1 \
  --output-dir results/knowledge_work/model_backed_hf_exec_hold \
  --no-update-latest
```

Long model-backed runs now checkpoint after each episode via:

- `manifest.json`
- `progress.json`
- `episode_traces.jsonl`
- `episode_leaderboard.csv`
- `summary.json`

Episode outputs are written under:

- [`results/knowledge_work`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work)

## Current Research Notes

- authoritative replayable-core full-lane snapshot:
  - [`20260410T021825Z_replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/20260410T021825Z_replayable_core/summary.json)
- authoritative live-web stress full-lane snapshot:
  - [`20260410T021845Z_live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/20260410T021845Z_live_web_stress/summary.json)
- first finished non-oracle episode baseline:
  - [`model_backed_hf_exec_hold/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_exec_hold/summary.json)
  - `artifact_quality = 1.0`
  - `browser_workflow = 0.9836`
  - `strict_interface = 1.0`
  - `recovered_execution = 1.0`
  - `role_readiness = 0.9056`
- exploratory multi-episode HF reasoner pilot:
  - [`model_backed_hf_reasoner_pilot/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_reasoner_pilot/summary.json)
  - stopped after two completed episodes to isolate a fully completed executive baseline

## Current H1/H1i Harnessing Surface

The H1 series is the current saturation-breaking layer on top of `KnowledgeWorkArena`.

H1 and descendants are deliberately restricted to packaged workflows. That keeps live runs attributable to workflow families and prevents the CLI harness from becoming an unbounded browsing or UI experiment.

Current MLX tool-contract packets:

- H1f compact no-directive packet:
  - [`results/knowledge_work_h1_slice/20260506T_h1f_mlx_no_directive_v1_knowledge_work_h1f_mlx_tool_contract_ablation_v1`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1f_mlx_no_directive_v1_knowledge_work_h1f_mlx_tool_contract_ablation_v1)
- H1h full no-directive packet:
  - [`results/knowledge_work_h1_slice/20260507T_h1h_mlx_full_no_directive_v1_knowledge_work_h1h_mlx_full_tool_contract_ablation_v1`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260507T_h1h_mlx_full_no_directive_v1_knowledge_work_h1h_mlx_full_tool_contract_ablation_v1)
- H1i worst-family packet:
  - [`configs/knowledge_work_h1i_slice.yaml`](/Users/cheickdiakite/Codex/moonie/configs/knowledge_work_h1i_slice.yaml)
  - [`results/knowledge_work_h1_slice/20260507T_h1i_mlx_worst_no_directive_v1_knowledge_work_h1i_mlx_worst_family_tool_contract_v1`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260507T_h1i_mlx_worst_no_directive_v1_knowledge_work_h1i_mlx_worst_family_tool_contract_v1)

Current H1i workflow families:

- `executive_latest_action_resume`
- `jobs_phone_patch_resume`
- `jobs_visual_form_hold`
- `executive_stale_brief_packet`

Current interpretation:

- contracted MLX is clean on H1i: readiness `0.97710`, strict/recovered `1.0 / 1.0`, raw clean `1.0`
- no-directive MLX keeps the same readiness only with controller help: repair/fallback/argument repair `1.00 / 0.50 / 0.50`, raw clean `0.00`
- no-directive plus no controller repair falls to readiness `0.64697`, strict/recovered `0.297 / 0.000`
- no-directive plus no fallback falls to `0.83125`
- no-directive plus no argument repair falls to `0.81220`

The current cross-packet report is:

- [`docs/reports/mlx-tool-contract-harnessing.md`](/Users/cheickdiakite/Codex/moonie/docs/reports/mlx-tool-contract-harnessing.md)
