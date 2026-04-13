# Session Handoff

## Resume Here

The repo is in a cleaner state now.

The important result is no longer “can we align the rows?” That is done.

Current aligned exploratory full-lane surface:

- [`results/knowledge_work_matrix/20260413Taligned_controller_burden_patch_v1_knowledge_work_alignment_32_26`](../../results/knowledge_work_matrix/20260413Taligned_controller_burden_patch_v1_knowledge_work_alignment_32_26)

Current same-surface headline rows:

- `oracle_gemma4_e2b`
- `hf_gemma4_e2b_specialists_cpu`
- `mlx_qwen3_8b_reasoner_only`
- `mlx_gemma4_e2b_reasoner_only`

All four now tie on top-line replayable and live readiness.

The remaining interesting problem is:

- HF Gemma specialist controller dependence

Not:

- MLX Gemma alignment
- MLX Gemma readiness
- whether the aligned surface exists at all

## Latest Headline Readout

Replayable `32`:

- oracle:
  - `real_world_readiness_avg = 0.976853125`
  - `controller_repair_avg = 0.578125`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 0.8395875`
- HF Gemma specialists:
  - `real_world_readiness_avg = 0.976853125`
  - `controller_repair_avg = 1.296875`
  - `controller_fallback_avg = 0.28125`
  - `raw_planning_clean_rate_avg = 0.46875`
- MLX Qwen:
  - `real_world_readiness_avg = 0.976853125`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`
- MLX Gemma:
  - `real_world_readiness_avg = 0.976853125`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`

Live `26`:

- oracle:
  - `real_world_readiness_avg = 0.9791653846153847`
- HF Gemma specialists:
  - `real_world_readiness_avg = 0.9791653846153847`
  - `controller_repair_avg = 1.5192307692307692`
  - `controller_fallback_avg = 0.23076923076923078`
- MLX Qwen:
  - `real_world_readiness_avg = 0.9791653846153847`
  - `controller_repair_avg = 0.0`
- MLX Gemma:
  - `real_world_readiness_avg = 0.9791653846153847`
  - `controller_repair_avg = 0.0`

Interpretation:

- same top-line readiness now exists across the four aligned rows
- HF Gemma specialists still need materially more controller help than the clean MLX rows
- the current research problem is now how much of that burden can be removed without dropping readiness

## What Just Changed

Two things landed in the last pass:

1. Planner/controller cleanup

- [`src/gemma4_capability_map/tools/planner.py`](../../src/gemma4_capability_map/tools/planner.py)
- the planner now synthesizes priority replacement calls directly in obvious repair/follow-on cases instead of falling through to broad fallback behavior

2. New research evidence

- focused replayable ablation packet rerun:
  - [`results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v3_knowledge_work_ablation_packet`](../../results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v3_knowledge_work_ablation_packet)
- aligned rerun after the planner patch:
  - [`results/knowledge_work_matrix/20260413Taligned_controller_burden_patch_v1_knowledge_work_alignment_32_26`](../../results/knowledge_work_matrix/20260413Taligned_controller_burden_patch_v1_knowledge_work_alignment_32_26)

The key learning from the packet rerun:

- baseline readiness stayed fixed at `0.9627777777777777`
- packet baseline `controller_fallback_avg` dropped from `2.0555555555555554` to `0.4444444444444444`
- `controller_fallback_planner` notes dropped from `37` to `8`

The key learning from the full aligned rerun:

- HF Gemma replayable `controller_repair_avg` dropped from `2.046875` to `1.296875`
- HF Gemma replayable `controller_fallback_avg` dropped from `1.03125` to `0.28125`
- HF Gemma live `controller_repair_avg` dropped from `2.3653846153846154` to `1.5192307692307692`
- HF Gemma live `controller_fallback_avg` dropped from `1.0769230769230769` to `0.23076923076923078`
- top-line readiness did not move

That is the current strongest new finding.

## What Not To Re-Learn

Do not spend time re-proving these already-established points:

- MLX Gemma can now align on the same top-line readiness tier as the other headline rows
- the older MLX Gemma executive-assistant judgment miss is closed
- Qwen MLX is a real reproduced same-surface row
- the direct in-process Gemma reasoner-only control is materially weaker than the improved Gemma stacks

Those are already settled enough for the next research pass.

## Next Best Move

1. Target the remaining HF Gemma specialist note families directly.
   Focus on:
   - `feedback_prior:refine_selection`
   - `feedback_prior:read_region_text`

2. Keep using the focused replayable packet as the main Gemma research harness.
   Do not jump to blind full-lane reruns first.

3. Install the local Gemma `31B` `GGUF` artifact and run the first real `llama.cpp` posture row once the model exists locally.

## Current Claim Boundary

Safe current claim:

- Moonie improved Gemma 4 materially as a local full-stack agent
- all four aligned rows now tie on top-line readiness
- HF Gemma specialists still need materially more controller help than the clean MLX rows

Unsafe current claim:

- Gemma broadly beats Qwen families
- Gemma beats frontier closed models on the same harness
- Gemma `31B` posture is already reproduced locally

## Verification State

Current code-side regressions from the last pass:

- targeted suite: `59 passed`

Documentation now points at:

- the aligned `20260413Taligned_controller_burden_patch_v1_knowledge_work_alignment_32_26` batch
- the focused `20260413Tresearch_ablation_focus_v3_knowledge_work_ablation_packet` batch

## Operational Notes

- The board source of truth is still:
  - [`results/history/knowledge_work_board_latest.csv`](../../results/history/knowledge_work_board_latest.csv)
- The Gemma `31B` lane is blocked by missing local artifact availability:
  - `GEMMA4_31B_GGUF_PATH` unset
  - no local bundle under `/Users/cheickdiakite/models`
