# Current State

## Benchmark Shape

Current generated corpus on disk:

- atomic tasks: `91`
- variants: `396`
- replayable KWA episodes: `32`
- live KWA episodes: `26`

The current headline comparison surface is the aligned exploratory `32 / 26` lane.

On the live-harness side, the active direction is now CLI-first rather than frontend-first.

Current runtime substrate:

- persistent sessions
- packaged workflow identity
- approvals and resume/deny flow
- event timelines
- artifact and revision persistence
- runtime traces and scorecards
- per-session sandbox metadata and sandboxed runtime output roots

Current CLI surface:

- `moonie-agent run`
- `moonie-agent watch`
- `moonie-agent approve` / `deny` / `resume` / `retry`
- `moonie-agent live`
  - packaged workflows only
  - defaults to `mlx_gemma4_e2b_reasoner_only`
  - launches a background run and attaches a Rich terminal operator view
- `moonie-agent attach <session_id>`
  - watches an existing run from the terminal

The React desktop harness exists and remains useful prior work, but it is no longer the main next workstream:

- `frontend`
  - defaults to `mlx_gemma4_e2b_reasoner_only`
  - groups sessions by project
  - keeps conversation in the center
  - exposes summary / review / browser context on the right
  - long-polls the real session stream endpoint instead of pretending the UI is live
  - surfaces backend connection state directly from the API health check
  - settles completed session state in the rail from the stream payload instead of relying only on a potentially stale session list snapshot
  - is no longer the active refinement target
  - runs against `moonie-agent-api`, not inside Streamlit

The prior React product loop was verified end to end:

- launched a fresh `mlx_gemma4_e2b_reasoner_only` `Dashboard Visual Review` session from the React shell
- observed `created -> instruction_updated -> warming -> running -> artifacts_ready -> completed` through the live workspace
- confirmed the browser pane renders runtime-backed preview assets and browser-state events
- confirmed the sidebar/session pills settle back to `completed` after the stream/list race fix

The current CLI pivot scaffold is verified with focused runtime/API/CLI tests:

- `uv run pytest tests/test_runtime_core.py tests/test_runtime_cli.py tests/test_runtime_api.py`
- `22 passed`
- `uv run pytest`
- `244 passed`

Relevant batches:

- aligned HF Gemma controller-burden rerun:
  - [`results/knowledge_work_matrix/20260413Taligned_controller_burden_patch_v2_knowledge_work_alignment_32_26`](../../results/knowledge_work_matrix/20260413Taligned_controller_burden_patch_v2_knowledge_work_alignment_32_26)
- aligned oracle + MLX Gemma judgment patch:
  - [`results/knowledge_work_matrix/20260413Toracle_mlx_gemma_judgment_patch_v1_knowledge_work_alignment_32_26`](../../results/knowledge_work_matrix/20260413Toracle_mlx_gemma_judgment_patch_v1_knowledge_work_alignment_32_26)
- aligned MLX Qwen row:
  - [`results/knowledge_work_matrix/20260412T235251Z_knowledge_work_alignment_32_26`](../../results/knowledge_work_matrix/20260412T235251Z_knowledge_work_alignment_32_26)

## Headline Comparison Read

Replayable `32`:

- `oracle_gemma4_e2b`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.976853125`
  - `controller_repair_avg = 0.578125`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 0.8395875`
- `hf_gemma4_e2b_specialists_cpu`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.976853125`
  - `controller_repair_avg = 0.71875`
  - `controller_fallback_avg = 0.28125`
  - `raw_planning_clean_rate_avg = 0.46875`
- `mlx_qwen3_8b_reasoner_only`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.976853125`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`
- `mlx_gemma4_e2b_reasoner_only`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.976853125`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`

Live `26`:

- `oracle_gemma4_e2b`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9791653846153847`
  - `controller_repair_avg = 0.7115384615384616`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 0.8025692307692308`
- `hf_gemma4_e2b_specialists_cpu`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9791653846153847`
  - `controller_repair_avg = 0.8076923076923077`
  - `controller_fallback_avg = 0.23076923076923078`
  - `raw_planning_clean_rate_avg = 0.4230769230769231`
- `mlx_qwen3_8b_reasoner_only`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9791653846153847`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`
- `mlx_gemma4_e2b_reasoner_only`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9791653846153847`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`

## Focused Gemma Packet

Current focused replayable research harness:

- [`results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v4_knowledge_work_ablation_packet_knowledge_work_ablation_packet`](../../results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v4_knowledge_work_ablation_packet_knowledge_work_ablation_packet)

Baseline packet metrics:

- `real_world_readiness_avg = 0.9627777777777777`
- `controller_repair_avg = 0.8888888888888888`
- `controller_fallback_avg = 0.4444444444444444`

Ablation rows:

- `no_controller_repair = 0.6551777777777779`
- `no_controller_fallback = 0.8182333333333333`
- `no_visual_rescue = 0.9627777777777777`

Important packet delta versus the older focused baseline:

- readiness unchanged at `0.9627777777777777`
- `controller_repair_avg` dropped from `2.3333333333333335` to `0.8888888888888888`
- `feedback_prior:refine_selection` dropped from `16` to `0`
- `feedback_prior:read_region_text` dropped from `10` to `0`
- `controller_fallback_planner` remains at `8`

Interpretation:

- the deterministic visual follow-on patch removed a real controller-help artifact
- repair and fallback are still causal on this slice
- visual rescue still contributes effectively nothing on this packet

## H1 Controller-Dependence Slice

Current H1 replayable output:

- MLX primary:
  - [`results/knowledge_work_h1_slice/20260506T_h1_mlx_gemma_primary_v1_knowledge_work_h1_controller_dependence_v1`](../../results/knowledge_work_h1_slice/20260506T_h1_mlx_gemma_primary_v1_knowledge_work_h1_controller_dependence_v1)
- HF service-backed ablation:
  - [`results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet)

MLX Gemma primary:

- `real_world_readiness_avg = 0.9749800000000001`
- `strict_interface_avg = 1.0`
- `recovered_execution_avg = 1.0`
- `controller_repair_avg = 0.0`
- `controller_fallback_avg = 0.0`
- `raw_planning_clean_rate_avg = 1.0`

HF service-backed Gemma baseline:

- `real_world_readiness_avg = 0.9749800000000001`
- `strict_interface_avg = 1.0`
- `recovered_execution_avg = 1.0`
- `controller_repair_avg = 0.9`
- `controller_fallback_avg = 0.6`
- `raw_planning_clean_rate_avg = 0.1`

H1 ablation result:

- `no_controller_repair = 0.7194`
- `no_controller_fallback = 0.7596999999999999`
- `no_visual_rescue = 0.9749800000000001`
- `no_intent_priority = 0.9749800000000001`
- `no_argument_repair = 0.9749800000000001`
- `no_deterministic_visual_follow_on = 0.9749800000000001`

Interpretation:

- H1 does not break MLX Gemma's controller-clean posture
- H1 does break HF Gemma when controller repair or controller fallback are disabled
- the remaining research seam is now trace-level repair/fallback mechanism design, not top-line readiness

Trace-note analysis:

- [`trace_note_summary.json`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet/trace_note_summary.json)
- [`trace_note_counts.csv`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet/trace_note_counts.csv)
- [`trace_episode_failures.csv`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet/trace_episode_failures.csv)

Current trace read:

- `102` controller-note events across `35` H1 episode rows
- `10` strict/recovered failure candidates
- failure rows now include coarse `failure_modes` labels for `raw_refusal`, `generic_tool_name`, disabled fallback/repair, argument repair, intent prior, visual follow-on, and tool canonicalization
- baseline uses `controller_fallback_planner` `6` times across all `5` H1 episodes
- `no_controller_fallback` fails all `5` H1 episodes, mostly raw refusal/no-call cases
- `no_controller_repair` fails all `5` H1 episodes, mostly generic `tool_name` hallucinations and malformed arguments

## Strongest Current Findings

1. Top-line parity is now established on the aligned `32 / 26` surface.
HF Gemma specialists, MLX Qwen, and MLX Gemma all reach the same readiness tier as oracle.

2. HF Gemma specialist controller burden is now lower, but still real.
Replayable `controller_repair_avg` improved from `1.296875` to `0.71875`.
Live `controller_repair_avg` improved from `1.5192307692307692` to `0.8076923076923077`.
Readiness did not move.

3. The old visual follow-on repair families were real benchmark signal, not noise.
Removing them via deterministic runtime sequencing changed controller metrics materially without changing outcomes.

4. The remaining Gemma gap is no longer “can Gemma tie the lane?”
The remaining gap is: how much controller help still remains after the obvious visual follow-ons are made deterministic?

5. The repo is no longer only benchmark-legible.
There is now a real runtime/session substrate, and the active live-testing surface is shifting to a sandboxed CLI operator harness over that same substrate.

6. H1 is now the best local instrument for controller-dependence work.
It keeps packaged workflow attribution while exposing the causal impact of repair and fallback on HF Gemma.

## Current Blockers

- Gemma `31B` `GGUF` / `llama.cpp` still has no local artifact:
  - `GEMMA4_31B_GGUF_PATH` unset
  - no local bundle under `/Users/cheickdiakite/models`
- board exports are updated, but README and continuity docs should always be treated as the narrative layer; `knowledge_work_board_latest.csv` alone is not a same-batch comparison argument
- the React workspace is useful prior work, but the current workstream should not spend cycles on frontend polish; browser execution still lives in the runtime trace layer

## Repo Truth

The repo now supports these statements:

- Moonie materially improved Gemma 4 as a local full-stack agent
- same-surface readiness parity is real on the aligned exploratory `32 / 26` surface
- HF Gemma specialists still need materially more controller help than the clean MLX rows
- the controller burden is reducible with harness/runtime changes alone
- live operator work should now be benchmark-backed, packaged-workflow-only, and sandboxed by default

The repo still does not support these statements:

- Gemma broadly beats Qwen families beyond the reproduced `Qwen3 8B MLX` row
- Gemma beats frontier closed models on the same harness
- Gemma `31B` runtime posture is already reproduced locally
