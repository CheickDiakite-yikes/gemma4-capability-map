# Current State

## Repository Scope

The repo currently contains:

- an atomic white-box benchmark for:
  - reasoning
  - tool routing
  - retrieval
  - full-stack execution
  - visual tool orchestration
- a role-based realism layer:
  - `KnowledgeWorkArena`
- a shared local runtime substrate:
  - persistent sessions
  - approval hold/resume
  - artifact and revision persistence
  - typed runtime events
  - packaged workflows
- thin product surfaces over the same substrate:
  - CLI
  - local API
  - Streamlit `operator_console`
  - Streamlit `mobile_companion`

## Benchmark Shape

Current generated and gold corpus shape:

- `91` gold atomic tasks
- `396` explicit atomic variants
- `16` real-world-tagged atomic tasks
- `30` atomic `visual_tool_orchestration` tasks in the current gold corpus
- `32` replayable `KnowledgeWorkArena` episodes in the generated corpus
- `26` live `KnowledgeWorkArena` episodes in the generated corpus

Important distinction:

- the stable canonical oracle pointers under `results/knowledge_work/replayable_core` and `results/knowledge_work/live_web_stress` still reflect the last full seeded rerun on the older `24 / 18` surface
- the current board-backed comparison story lives on the aligned exploratory `32 / 26` surface
- use `results/history/knowledge_work_board_latest.csv` as the source of truth for comparison claims

## Current Source Of Truth

Board and history:

- [`results/history/knowledge_work_board_latest.csv`](../../results/history/knowledge_work_board_latest.csv)
- [`results/history/knowledge_work_history.md`](../../results/history/knowledge_work_history.md)

Current aligned batch:

- [`results/knowledge_work_matrix/20260413Taligned_controller_burden_patch_v1_knowledge_work_alignment_32_26`](../../results/knowledge_work_matrix/20260413Taligned_controller_burden_patch_v1_knowledge_work_alignment_32_26)

Current focused replayable ablation packet:

- [`results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v3_knowledge_work_ablation_packet`](../../results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v3_knowledge_work_ablation_packet)

## Current Aligned `32 / 26` Comparison Surface

Same-surface headline rows now exist for:

- `oracle_gemma4_e2b`
- `hf_gemma4_e2b_specialists_cpu`
- `mlx_qwen3_8b_reasoner_only`
- `mlx_gemma4_e2b_reasoner_only`

Replayable `32`:

- oracle:
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.976853125`
  - `controller_repair_avg = 0.578125`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 0.8395875`
- HF Gemma specialists:
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.976853125`
  - `controller_repair_avg = 1.296875`
  - `controller_fallback_avg = 0.28125`
  - `raw_planning_clean_rate_avg = 0.46875`
- MLX Qwen:
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.976853125`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`
- MLX Gemma:
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.976853125`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`

Live `26`:

- oracle:
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9791653846153847`
  - `controller_repair_avg = 0.7115384615384616`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 0.8025692307692308`
- HF Gemma specialists:
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9791653846153847`
  - `controller_repair_avg = 1.5192307692307692`
  - `controller_fallback_avg = 0.23076923076923078`
  - `raw_planning_clean_rate_avg = 0.4230769230769231`
- MLX Qwen:
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9791653846153847`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`
- MLX Gemma:
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9791653846153847`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`

## What This Means

Current honest reading:

- all four rows now tie on top-line replayable and live readiness on the aligned exploratory surface
- HF Gemma specialists still rely on materially more controller help than the clean MLX rows
- the current headline research gap is HF controller dependence, not MLX Gemma readiness
- the direct in-process Gemma reasoner-only control remains materially weaker on the older `26 / 20` reproduced surface, so the Gemma-improvement claim is still meaningful

## Focused Ablation Result

The focused replayable packet is now the cleanest research instrument for the remaining Gemma gap.

Packet summary:

- baseline:
  - `real_world_readiness_avg = 0.9627777777777777`
  - `controller_repair_avg = 2.3333333333333335`
  - `controller_fallback_avg = 0.4444444444444444`
- `no_controller_repair`:
  - `real_world_readiness_avg = 0.6551777777777779`
- `no_controller_fallback`:
  - `real_world_readiness_avg = 0.8182333333333333`
- `no_visual_rescue`:
  - `real_world_readiness_avg = 0.9627777777777777`

Interpretation:

- controller repair is essential on this slice
- controller fallback is also essential
- visual rescue is currently low or zero leverage on this slice

The strongest planner-specific improvement from the latest packet rerun:

- `controller_fallback_planner` note family dropped from `37` to `8`
- packet readiness stayed flat
- packet baseline fallback burden dropped from `2.0555555555555554` to `0.4444444444444444`

That is the clearest sign that the latest planner patch reduced controller dependence instead of merely moving score noise around.

## Current Runtime-Posture Read

What now exists:

- HF in-process Gemma specialist headline row
- MLX Qwen reproduced row
- MLX Gemma reproduced row
- Gemma `31B` `GGUF` / `llama.cpp` runtime-posture support in code and registry

What does not exist yet:

- a reproduced local Gemma `31B` `GGUF` row

Current blocker:

- `GEMMA4_31B_GGUF_PATH` is unset
- no local Gemma `31B` `GGUF` artifact exists under `/Users/cheickdiakite/models`

## Current Product/Harness Status

The repo now has one real local-agent substrate:

- persistent runtime sessions
- approval ids and lifecycle
- event timelines
- artifact history
- workflow-backed local CLI
- workflow-backed local API
- Streamlit operator and mobile shells

This matters because the repo is explicitly studying the chatbot -> agent gap, not only model quality.

## Current Claim Boundary

The repo can now honestly say:

- Gemma 4 was materially improved as a local full-stack agent on Moonie
- those gains came from runtime/controller/specialist-stack work, not from changed model weights
- same-surface top-line parity now exists across oracle, HF Gemma specialists, MLX Qwen, and MLX Gemma
- that parity hides a real controller-burden gap under the HF Gemma specialist row

The repo cannot honestly say yet:

- that Gemma broadly beats Qwen families
- that Gemma beats frontier closed models on the same harness
- that the Gemma `31B` runtime posture has already been reproduced locally
