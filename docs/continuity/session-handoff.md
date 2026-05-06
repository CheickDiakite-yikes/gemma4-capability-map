# Session Handoff

## Resume Here

The current restart point is now:

- [`docs/continuity/cli-live-harness-pivot.md`](./cli-live-harness-pivot.md)

Use that file first in a new chat.

It supersedes the old assumption that the next main workstream is React workspace refinement.

The current research seam is no longer “make the rows tie.”

That part is done on the aligned exploratory `32 / 26` surface.

The current seam is:

- reduce HF Gemma specialist controller burden further
- without losing the current aligned readiness tier
- shift the main live-testing surface to a CLI-first sandboxed operator harness
- use Gemini CLI as a design reference and external baseline, not a replacement

The latest live-harness gain is now CLI-first:

- sessions and runtime traces carry sandbox metadata
- packaged workflow runs get a per-session sandbox with copied workflow/episode inputs
- native artifacts and runtime summaries write under the sandbox output root
- live-web dry-run holds are emitted as `sandbox_policy_block` events and stored on sessions/traces
- `moonie-agent live` launches a packaged workflow and attaches a Rich terminal operator view
- `moonie-agent attach <session_id>` watches an existing run from the terminal
- `moonie-agent attach <session_id> --action approve|deny|resume|retry|quit` applies operator actions from the same terminal path
- `moonie-agent inspect <session_id>` inspects sandbox, artifact, policy-block, and summary metadata
- a real `mlx_gemma4_e2b_reasoner_only` CLI smoke completed on `executive_visual_dashboard_review`
- `moonie-agent gemini-baseline` prepares dry-run Gemini CLI baseline packets for packaged workflows
- `H1 v1` is defined as the next packaged-workflow-first harder slice:
  - [`configs/knowledge_work_h1_slice.yaml`](../../configs/knowledge_work_h1_slice.yaml)
  - [`docs/continuity/h1-slice.md`](./h1-slice.md)
- `scripts/run_knowledge_work_h1_slice.py` validates H1 and delegates filtered runs to the existing KWA arena runner
- second-wave ablation controls now exist for intent priority, argument repair, and deterministic visual follow-on
- H1 HF ablation should use [`scripts/run_knowledge_work_h1_ablation_packet.py`](../../scripts/run_knowledge_work_h1_ablation_packet.py) so the ablation rows share one warmed HF service-backed bundle
- an attempted in-process H1 ablation launch was stopped pre-child-manifest after roughly ten minutes; no episode results were produced from that attempt
- H1 primary replayable MLX Gemma completed cleanly:
  - [`results/knowledge_work_h1_slice/20260506T_h1_mlx_gemma_primary_v1_knowledge_work_h1_controller_dependence_v1`](../../results/knowledge_work_h1_slice/20260506T_h1_mlx_gemma_primary_v1_knowledge_work_h1_controller_dependence_v1)
  - `real_world_readiness_avg = 0.9749800000000001`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`

The prior React product-side gain remains useful context:

- the React workspace now runs against the real API in a live loop
- the shell uses backend health plus long-poll session streaming
- a fresh `mlx_gemma4_e2b_reasoner_only` session was launched from the UI and observed through completion
- the stream payload now wins over stale session-list snapshots so the rail settles correctly after completion

## Current Source Runs

Aligned comparison surface:

- HF Gemma controller-burden rerun:
  - [`results/knowledge_work_matrix/20260413Taligned_controller_burden_patch_v2_knowledge_work_alignment_32_26`](../../results/knowledge_work_matrix/20260413Taligned_controller_burden_patch_v2_knowledge_work_alignment_32_26)
- oracle + MLX Gemma aligned reference:
  - [`results/knowledge_work_matrix/20260413Toracle_mlx_gemma_judgment_patch_v1_knowledge_work_alignment_32_26`](../../results/knowledge_work_matrix/20260413Toracle_mlx_gemma_judgment_patch_v1_knowledge_work_alignment_32_26)
- MLX Qwen aligned reference:
  - [`results/knowledge_work_matrix/20260412T235251Z_knowledge_work_alignment_32_26`](../../results/knowledge_work_matrix/20260412T235251Z_knowledge_work_alignment_32_26)

Focused replayable Gemma packet:

- [`results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v4_knowledge_work_ablation_packet_knowledge_work_ablation_packet`](../../results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v4_knowledge_work_ablation_packet_knowledge_work_ablation_packet)

## Latest Headline Readout

Replayable `32`:

- oracle:
  - `readiness = 0.976853125`
  - `controller_repair_avg = 0.578125`
  - `controller_fallback_avg = 0.0`
- HF Gemma specialists:
  - `readiness = 0.976853125`
  - `controller_repair_avg = 0.71875`
  - `controller_fallback_avg = 0.28125`
  - `raw_planning_clean_rate_avg = 0.46875`
- MLX Qwen:
  - `readiness = 0.976853125`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
- MLX Gemma:
  - `readiness = 0.976853125`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`

Live `26`:

- oracle:
  - `readiness = 0.9791653846153847`
  - `controller_repair_avg = 0.7115384615384616`
- HF Gemma specialists:
  - `readiness = 0.9791653846153847`
  - `controller_repair_avg = 0.8076923076923077`
  - `controller_fallback_avg = 0.23076923076923078`
- MLX Qwen:
  - `readiness = 0.9791653846153847`
  - `controller_repair_avg = 0.0`
- MLX Gemma:
  - `readiness = 0.9791653846153847`
  - `controller_repair_avg = 0.0`

## What Just Changed

The latest pass added the first CLI-first live harness slice.

Code path:

- [`src/gemma4_capability_map/runtime/sandbox.py`](../../src/gemma4_capability_map/runtime/sandbox.py)
- [`src/gemma4_capability_map/runtime/operator.py`](../../src/gemma4_capability_map/runtime/operator.py)
- [`src/gemma4_capability_map/runtime/cli.py`](../../src/gemma4_capability_map/runtime/cli.py)
- [`src/gemma4_capability_map/runtime/core.py`](../../src/gemma4_capability_map/runtime/core.py)
- [`tests/test_runtime_core.py`](../../tests/test_runtime_core.py)
- [`tests/test_runtime_cli.py`](../../tests/test_runtime_cli.py)
- [`tests/test_runtime_api.py`](../../tests/test_runtime_api.py)

What that means:

- `moonie-agent live` is now the active live-entry scaffold for packaged workflows
- `moonie-agent attach` provides a Rich terminal operator view
- new live runs are sandboxed by default with policy id `packaged_workflow_ephemeral_v1`
- runtime artifacts, summaries, and traces are attributable to the sandbox root
- live-web sandbox-only or approval-gated actions now produce explicit policy block metadata
- attach actions can approve, deny, resume, retry, or quit from the Rich operator path
- inspect commands can show sandbox roots, artifacts, policy blocks, and trace/summary paths as Rich output or JSON

Verification:

- `uv run pytest tests/test_runtime_core.py tests/test_runtime_cli.py tests/test_runtime_api.py`
- latest targeted run: `24 passed`
- `uv run pytest tests/test_runtime_cli.py tests/test_runtime_core.py`
- latest operator inspect/action run: `21 passed`
- `uv run moonie-agent inspect <latest_session> --target sandbox --json`
- completed and showed the sandbox root plus manifest path
- `uv run moonie-agent live --workflow-id executive_visual_dashboard_review --system-id mlx_gemma4_e2b_reasoner_only --lane replayable_core --once --refresh-s 0.5 --timeout-s 1.0`
- completed session: `20260506T173247139289Z_executive_visual_dashboard_review`
- smoke metrics: `strict_interface_score = 1.0`, `role_readiness_score = 0.9942`, `controller_repair_count = 0.5`, `controller_fallback_count = 0.0`, `raw_planning_clean_rate = 0.5`
- `uv run pytest tests/test_runtime_gemini_cli.py tests/test_runtime_cli.py`
- Gemini adapter scaffold: `11 passed`
- `uv run moonie-agent gemini-baseline --workflow-id executive_visual_dashboard_review --lane replayable_core --output-dir tmp/gemini-baseline-smoke`
- completed as a dry-run packet with `/usr/local/bin/gemini` detected
- `uv run pytest tests/test_knowledge_work_h1.py tests/test_knowledge_work_matrix_script.py tests/test_run_knowledge_work_arena_script.py`
- H1 runner/config scaffold: `17 passed`
- `uv run python scripts/run_knowledge_work_h1_slice.py --dry-run --run-set primary --lane replayable_core --output-root tmp/h1-dry-run-smoke --run-group-id 20260506T_h1_dry_run_smoke`
- completed and wrote a dry-run manifest for one primary replayable H1 run
- `uv run pytest tests/test_tool_planner.py tests/test_trace_metrics.py tests/test_knowledge_work_matrix_script.py tests/test_run_knowledge_work_arena_script.py tests/test_knowledge_work_h1.py`
- second-wave ablation control scaffold: `64 passed`
- `uv run python scripts/run_knowledge_work_h1_slice.py --dry-run --run-set ablation --lane replayable_core --output-root tmp/h1-ablation-dry-run-smoke --run-group-id 20260506T_h1_ablation_dry_run_smoke`
- completed and wrote `7` replayable H1 ablation run specs
- `uv run pytest`
- full repo suite after H1 + second-wave controls: `260 passed`
- `uv run python scripts/run_knowledge_work_h1_slice.py --dry-run --run-set all --lane replayable_core --output-root tmp/h1-all-dry-run-smoke --run-group-id 20260506T_h1_all_dry_run_smoke`
- completed and wrote `10` replayable H1 run specs
- `uv run python scripts/run_knowledge_work_h1_slice.py --run-set primary --lane replayable_core --system-id mlx_gemma4_e2b_reasoner_only --run-group-id 20260506T_h1_mlx_gemma_primary_v1`
- completed with `5 / 5` H1 replayable episodes and `failed_runs = 0`
- `uv run moonie-agent live --workflow-id executive_visual_dashboard_review --system-id oracle_gemma4_e2b --lane replayable_core --refresh-s 0.1 --timeout-s 0.5`
- completed through the Rich operator view with sandbox context visible
- `uv run pytest`
- earlier full live-harness suite: `244 passed`

## Prior Change

The latest pass added deterministic runtime execution for obvious visual follow-ons.

Code path:

- [`src/gemma4_capability_map/runtime/core.py`](../../src/gemma4_capability_map/runtime/core.py)
- [`src/gemma4_capability_map/tools/planner.py`](../../src/gemma4_capability_map/tools/planner.py)
- [`tests/test_tool_planner.py`](../../tests/test_tool_planner.py)
- [`tests/test_smoke_eval.py`](../../tests/test_smoke_eval.py)
- [`tests/test_trace_metrics.py`](../../tests/test_trace_metrics.py)

What that means:

- after a successful `extract_layout` or `refine_selection`, the runtime now auto-executes deterministic `refine_selection` / `read_region_text` follow-ons
- the runtime no longer asks the model again for those same obvious visual steps

## Measured Effect

Focused packet delta versus the prior packet:

- readiness unchanged at `0.9627777777777777`
- `controller_repair_avg` improved from `2.3333333333333335` to `0.8888888888888888`
- `feedback_prior:refine_selection` dropped from `16` to `0`
- `feedback_prior:read_region_text` dropped from `10` to `0`
- `controller_fallback_planner` stayed at `8`

Aligned full-lane delta for HF Gemma specialists:

- replayable:
  - `controller_repair_avg` improved from `1.296875` to `0.71875`
  - `controller_fallback_avg` stayed `0.28125`
  - readiness stayed `0.976853125`
- live:
  - `controller_repair_avg` improved from `1.5192307692307692` to `0.8076923076923077`
  - `controller_fallback_avg` stayed `0.23076923076923078`
  - readiness stayed `0.9791653846153847`

Interpretation:

- the old visual follow-on repairs were inflating controller burden
- removing them did not reduce the actual causal value of repair/fallback
- the remaining burden is now more honestly concentrated in fallback planner and non-visual repair families

## What Not To Re-Learn

Do not spend time re-proving:

- aligned top-line readiness parity exists
- MLX Gemma’s earlier executive-assistant judgment miss is closed
- MLX Qwen is a real same-surface comparator
- the direct in-process Gemma reasoner-only control is still materially weaker on the older reproduced surface

## Next Best Move

1. Follow the CLI live harness pivot file first.
Primary targets:
   - runtime sandbox model
   - `moonie-agent live`
   - `moonie-agent attach`
   - Rich terminal operator harness

2. Run the H1 replayable HF Gemma ablation packet before another broad same-surface rerun.

3. Attack the remaining HF Gemma specialist note families directly.
Primary targets:
   - `controller_fallback_planner`
   - `repaired_arguments:extract_layout`
   - `intent_prior:record_or_update`
   - `intent_prior:inspect_or_lookup`
Now scaffolded toggles:
   - `disable_intent_priority`
   - `disable_argument_repair`
   - `disable_deterministic_visual_follow_on`

4. Keep using the focused replayable packet first.
Only rerun the aligned `32 / 26` surface after the packet shifts again.

5. Use Gemini CLI as a wrapped reference/baseline after the CLI live harness exists.

6. If the next question becomes runtime posture instead of controller dependence, switch to installing the Gemma `31B` local `GGUF` artifact and run the first real `llama.cpp` row.

## Verification State

Current code-side verification from the latest CLI live harness patch:

- targeted runtime/API/CLI suite from the original live-harness slice: `22 passed`
- full suite after the CLI/H1/ablation-control pivot: `260 passed`
- H1 all-run-set dry-run: `10` replayable run specs

Benchmark outputs rebuilt:

- [`results/history/knowledge_work_board_latest.csv`](../../results/history/knowledge_work_board_latest.csv)
- [`results/history/knowledge_work_history.md`](../../results/history/knowledge_work_history.md)

## Operational Notes

- `output/` and `tmp/` remain untracked scratch dirs
- the Gemma `31B` lane is still blocked by local artifact availability:
  - `GEMMA4_31B_GGUF_PATH` unset
  - no local bundle under `/Users/cheickdiakite/models`
