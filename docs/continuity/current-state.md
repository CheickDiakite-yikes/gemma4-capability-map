# Current State

## Repository Scope

This repo now contains:

- two benchmark layers:
  - atomic white-box capability benchmarking
    - reasoning
    - tool routing
    - retrieval
    - full-stack execution
  - `KnowledgeWorkArena`
    - role-based, job-shaped episodes built on top of the atomic substrate
- one shared local-agent runtime substrate:
  - persistent sessions
  - tool orchestration
  - approval/hold/resume flow
  - trace and artifact persistence
  - packaged workflows
- transitional product surfaces over that same substrate:
  - CLI
  - local API
  - Streamlit `operator_console`
  - Streamlit `mobile_companion`
  - tool-family expansion across `function_call`, CLI, and API surfaces

## Benchmark Shape

- `91` gold atomic tasks
- `396` explicit atomic variants
- `16` real-world-tagged atomic tasks
- `30` atomic `visual_tool_orchestration` tasks in the current gold corpus
- `32` replayable-core `KnowledgeWorkArena` episodes in the current generated corpus
- `26` live-web stress `KnowledgeWorkArena` episodes in the current generated corpus

Important distinction:

- the generated corpora are now `91 / 396 / 32 / 26`
- the board-backed widened comparison surface now exists for:
  - `oracle_gemma4_e2b`
  - `hf_gemma4_e2b_specialists_cpu`
  - `mlx_qwen3_8b_reasoner_only`
  - `mlx_gemma4_e2b_reasoner_only`
- those rows now run on the aligned exploratory `32 / 26` full-lane surface
- the direct Gemma reasoner-only control still sits on the earlier reproduced `26 / 20` surface
- the older canonical oracle lane pointers under `results/knowledge_work/replayable_core` and `results/knowledge_work/live_web_stress` still reflect the last full oracle rerun on the earlier `24 / 18` surface
- use `results/history/knowledge_work_board_latest.csv` as the current source of truth for board-level comparison claims
- latest-board selection now prefers broader completed rows over stale older rows that only win on legacy `run_scope` labels

## Canonical Atomic Benchmark Pointers

- real-world autonomy matrix:
  - [`results/alpha_matrix/20260409T210500Z_alpha_real_world`](../../results/alpha_matrix/20260409T210500Z_alpha_real_world)
- atomic benchmark history:
  - [`results/history/history_report.md`](../../results/history/history_report.md)

## Canonical Visual Tool Orchestration Pointers

Replayable core:

- [`results/visual_tool_orchestration/replayable_core/summary.json`](../../results/visual_tool_orchestration/replayable_core/summary.json)
- current metrics:
  - `runs = 11`
  - `success_rate = 1.0`
  - `strict_interface_rate = 1.0`
  - `recovered_execution_rate = 1.0`
  - `real_world_readiness_avg = 1.0`

Live-web stress:

- [`results/visual_tool_orchestration/live_web_stress/summary.json`](../../results/visual_tool_orchestration/live_web_stress/summary.json)
- current metrics:
  - `runs = 7`
  - `success_rate = 1.0`
  - `strict_interface_rate = 1.0`
  - `recovered_execution_rate = 1.0`

Interpretation:

- the repo now has a first-class atomic benchmark for visual-tool orchestration, not just generic multimodal QA
- the gold corpus now contains `30` visual-tool tasks, while the current canonical lane pointers cover the `11` replayable and `7` live seeded tasks that have already been rerun end to end
- the canonical track measures whether the reasoner picks the right visual tools, refines selections correctly, preserves referents across turns, and lands the right final answer
- stricter placeholder-aware scoring is now live: follow-up `selection_id` and `region_id` arguments must point at the latest valid visual referent, not just any non-empty placeholder replacement
- replayable scoring is seeded and deterministic; live-web stress uses the same tool surface with a local executor path
- visual count scoring is now stricter: a count-heavy task no longer gets full credit just because the final prose mentions the expected number after the tool path produced the wrong selection count

## Canonical KnowledgeWorkArena Pointers

Replayable core:

- [`results/knowledge_work/replayable_core/summary.json`](../../results/knowledge_work/replayable_core/summary.json)
- current metrics:
  - `runs = 24`
  - `artifact_quality_avg = 0.9866`
  - `browser_workflow_avg = 0.9910`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9510`
  - `escalation_correctness_avg = 1.0`

Live-web stress:

- [`results/knowledge_work/live_web_stress/summary.json`](../../results/knowledge_work/live_web_stress/summary.json)
- current metrics:
  - `runs = 18`
  - `artifact_quality_avg = 0.9822`
  - `browser_workflow_avg = 1.0`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9630`
  - `escalation_correctness_avg = 1.0`

Harder human-nuance episodes now included in the canonical oracle lanes:

- replayable:
  - `kwa_exec_stale_brief_hold`
  - `kwa_jobs_constraint_preservation_hold`
  - `kwa_finance_stale_assumption_hold`
- live:
  - `kwa_exec_live_stale_brief_hold`
  - `kwa_jobs_live_constraint_hold`
  - `kwa_finance_live_stale_assumption_hold`

Visual KWA episodes now also exist and have bounded oracle references:

- replayable visual KWA slice:
  - [`results/knowledge_work/kwa_visual_replayable_oracle_v1/summary.json`](../../results/knowledge_work/kwa_visual_replayable_oracle_v1/summary.json)
  - episodes:
    - `kwa_exec_visual_dashboard_brief`
    - `kwa_jobs_visual_form_hold`
    - `kwa_finance_visual_invoice_hold`
- live visual KWA slice:
  - [`results/knowledge_work/kwa_visual_live_oracle_v1/summary.json`](../../results/knowledge_work/kwa_visual_live_oracle_v1/summary.json)
  - episodes:
    - `kwa_exec_live_visual_dashboard_brief`
    - `kwa_jobs_live_visual_form_hold`
    - `kwa_finance_live_visual_invoice_hold`

Interpretation:

- the canonical KWA oracle pointers still reflect the last full oracle rerun on `24 / 18`
- the generated KWA corpus is now `32 / 26`
- the aligned board-backed oracle, Gemma specialist, MLX Qwen, and MLX Gemma rows now exist on the same exploratory `32 / 26` full-lane surface
- the direct Gemma reasoner-only control still remains on the earlier `26 / 20` reproduced surface
- these are harder because the right move is often “repair and stop safely,” not just “complete the workflow”
- the canonical runner no longer silently truncates the lane; `run_knowledge_work_arena.py` now defaults to full-lane execution unless `--limit` is explicitly set

## Current Publishable Full-Lane Comparison Surface

The current board-backed comparison surface is now the aligned exploratory `32 / 26` matrix:

- [`results/history/knowledge_work_board_latest.csv`](../../results/history/knowledge_work_board_latest.csv)

Headline rows:

- oracle full-lane reference:
  - `oracle_gemma4_e2b`
  - comparison batch:
    - `20260412T235251Z_knowledge_work_alignment_32_26`
  - replayable:
    - `runs = 32`
    - `artifact_quality_avg = 0.964509375`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.976853125`
    - `controller_repair_avg = 0.578125`
    - `controller_fallback_avg = 0.0`
    - `raw_planning_clean_rate_avg = 0.8395875`
  - live:
    - `runs = 26`
    - `artifact_quality_avg = 0.9584576923076923`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9791653846153847`
    - `controller_repair_avg = 0.7115384615384616`
    - `controller_fallback_avg = 0.0`
    - `raw_planning_clean_rate_avg = 0.8025692307692308`
- headline local Gemma stack:
  - `hf_gemma4_e2b_specialists_cpu`
  - comparison batch:
    - `20260412T235251Z_knowledge_work_alignment_32_26`
  - replayable:
    - `runs = 32`
    - `artifact_quality_avg = 0.964509375`
    - `browser_workflow_avg = 0.98145625`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.976853125`
    - `controller_repair_avg = 2.046875`
    - `controller_fallback_avg = 1.03125`
    - `raw_planning_clean_rate_avg = 0.46875`
  - live:
    - `runs = 26`
    - `artifact_quality_avg = 0.9584576923076923`
    - `browser_workflow_avg = 1.0`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9791653846153847`
    - `controller_repair_avg = 2.3653846153846154`
    - `controller_fallback_avg = 1.0769230769230769`
    - `raw_planning_clean_rate_avg = 0.4230769230769231`

First reproduced Gemma MLX posture row:

- `mlx_gemma4_e2b_reasoner_only`
- replayable:
  - comparison batch:
    - `20260412T235251Z_knowledge_work_alignment_32_26`
  - `runs = 32`
  - `artifact_quality_avg = 0.964509375`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9725125`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`
- live:
  - comparison batch:
    - `20260412T235251Z_knowledge_work_alignment_32_26`
  - `runs = 26`
  - `artifact_quality_avg = 0.9584576923076923`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.973823076923077`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`

Interpretation:

- MLX Gemma is now a real completed benchmark posture on the same aligned exploratory `32 / 26` surface as oracle, HF Gemma specialists, and MLX Qwen
- it is materially stronger than the direct in-process Gemma reasoner-only control
- it is controller-clean on both completed lanes
- the original replayable miss was concentrated rather than broad, and the grounded visual readback fallback now clears that miss
- it is now an apples-to-apples same-surface posture comparison
- the remaining difference is not planner noise:
  - MLX Gemma stays controller-clean
  - but still lands slightly below oracle, HF Gemma specialists, and MLX Qwen on readiness

## Planner-Gap View

The board now carries explicit harness-gap metrics in addition to the top-line readiness metrics:

- `controller_repair_avg`
- `argument_repair_avg`
- `controller_fallback_avg`
- `intent_override_avg`
- `raw_planning_clean_rate_avg`

Current interpretation on the aligned exploratory `32 / 26` board rows:

- oracle, HF Gemma specialists, and MLX Qwen currently match on top-line replayable and live readiness
- they do **not** match on raw planner cleanliness
- replayable `hf_gemma4_e2b_specialists_cpu` currently shows:
  - `controller_repair_avg = 2.046875`
  - `controller_fallback_avg = 1.03125`
  - `raw_planning_clean_rate_avg = 0.46875`
- replayable `oracle_gemma4_e2b` currently shows:
  - `controller_repair_avg = 0.578125`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 0.8395875`
- replayable `mlx_qwen3_8b_reasoner_only` currently shows:
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`
- replayable `mlx_gemma4_e2b_reasoner_only` currently shows:
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`
  - `real_world_readiness_avg = 0.9725125`

Interpretation:

- same top-line readiness does not yet mean the models are equally clean on tool use and direction following
- the current strong HF Gemma harness is closing the top-line gap, but Gemma still leans more on controller help on the aligned widened rows
- this is now a better research story than a flat “Gemma equals Qwen” claim because it isolates where the harness is compensating
- MLX Gemma adds a second useful story:
  - the harness improvements do transfer to the Apple-Silicon-native Gemma path
  - but the aligned MLX Gemma row still lands slightly lower readiness despite staying planner-clean and controller-clean
- headline local control:
  - `hf_gemma4_e2b_reasoner_only`
  - replayable:
    - `strict_interface_avg = 0.9038461538461539`
    - `recovered_execution_avg = 0.8846153846153846`
    - `real_world_readiness_avg = 0.9392653846153846`
  - live:
    - `strict_interface_avg = 0.875`
    - `recovered_execution_avg = 0.85`
    - `real_world_readiness_avg = 0.9347899999999999`
- first reproduced non-Gemma local comparator:
  - `mlx_qwen3_8b_reasoner_only`
  - comparison batch:
    - `20260412T213721Z_knowledge_work_full_lane_experimental`
    - `20260412T213438Z_knowledge_work_full_lane_experimental`
  - replayable:
    - `runs = 29`
    - `artifact_quality_avg = 0.9689793103448276`
    - `browser_workflow_avg = 0.9821241379310345`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9774`
  - live:
    - `runs = 23`
    - `artifact_quality_avg = 0.9633043478260869`
    - `browser_workflow_avg = 1.0`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9798`

Interpretation:

- the strongest current claim in the repo is now explicit:
  - our controller/runtime/specialist-stack learnings materially improved Gemma 4 as a full-stack local agent on the aligned exploratory `32 / 26` full-lane surface
- the refreshed local Gemma specialist row is now strict/recovered clean on that widened full-lane board surface
- the aligned oracle row is also strict/recovered clean on that same surface
- the reasoner-only Gemma control remains materially weaker, so the gain is attributable to the harness/controller/runtime work rather than to an easy benchmark surface
- this is a strong publishable Gemma-improvement claim
- the repo now has a real same-surface non-Gemma row:
  - `mlx_qwen3_8b_reasoner_only`
- that row is informative but bounded:
  - it beats the direct in-process Gemma reasoner-only control on the reproduced older full-lane surface
  - it materially improved after the shared rescue/planner fixes
  - on the current aligned `32 / 26` surface it now also matches oracle and the Gemma specialist stack
- the repo also now has two distinct Qwen runtime postures:
  - `hf_qwen3_8b_reasoner_only` as the direct-HF appendix/control path
  - `mlx_qwen3_8b_reasoner_only` as the Apple-Silicon-native local comparator path
- the repo now also has a first reproduced Apple-Silicon-native Gemma posture:
  - `mlx_gemma4_e2b_reasoner_only`
  - it is aligned now, and it is strong enough to make the residual readiness gap worth targeted debugging
- the experimental Gemma 4 `31B` `GGUF` / `llama.cpp` runtime-posture support is implemented, but it has not been reproduced locally yet because there is no local model or runtime installed on this machine
- the next real comparator step is no longer “make Qwen runnable”; it is:
  - reduce HF Gemma specialist controller dependence on the current clean rows
  - inspect the residual MLX Gemma readiness gap now that alignment is complete
  - add the Gemma `31B` posture comparison

## Benchmark Board / Reporting Layer

Registry and exports:

- registry:
  - [`configs/model_registry.yaml`](../../configs/model_registry.yaml)
- board latest:
  - [`results/history/knowledge_work_board_latest.csv`](../../results/history/knowledge_work_board_latest.csv)
- board runs:
  - [`results/history/knowledge_work_board_runs.csv`](../../results/history/knowledge_work_board_runs.csv)
- scatter data:
  - [`results/history/knowledge_work_scatter.csv`](../../results/history/knowledge_work_scatter.csv)
- board payload:
  - [`results/history/knowledge_work_board.json`](../../results/history/knowledge_work_board.json)
- published external benchmark rows:
  - [`results/history/knowledge_work_external_benchmarks.csv`](../../results/history/knowledge_work_external_benchmarks.csv)
- published external benchmark summary:
  - [`results/history/knowledge_work_external_benchmark_summary.json`](../../results/history/knowledge_work_external_benchmark_summary.json)
- role breakdown:
  - [`results/history/knowledge_work_role_breakdown.csv`](../../results/history/knowledge_work_role_breakdown.csv)
- category breakdown:
  - [`results/history/knowledge_work_category_breakdown.csv`](../../results/history/knowledge_work_category_breakdown.csv)
- track breakdown:
  - [`results/history/knowledge_work_track_breakdown.csv`](../../results/history/knowledge_work_track_breakdown.csv)

UI surface:

- the Streamlit board now has a dedicated `External Context` tab
- that tab is intentionally provenance-separated:
  - published external scores are context only
  - Moonie leaderboard rows remain reproduced runs
- first seeded external rows are official GPT-5.4 and Gemini 3.1 Pro benchmark results
- Qwen is still the next reproduced local comparator target, not an external placeholder board row

- Streamlit board mode now lives in:
  - [`src/gemma4_capability_map/app/streamlit_app.py`](../../src/gemma4_capability_map/app/streamlit_app.py)
- operator-console and mobile-companion views now also live there, backed by:
  - [`src/gemma4_capability_map/app/views/operator_console.py`](../../src/gemma4_capability_map/app/views/operator_console.py)
  - [`src/gemma4_capability_map/app/views/mobile_companion.py`](../../src/gemma4_capability_map/app/views/mobile_companion.py)
  - [`src/gemma4_capability_map/app/view_models.py`](../../src/gemma4_capability_map/app/view_models.py)
  - [`src/gemma4_capability_map/app/theme.py`](../../src/gemma4_capability_map/app/theme.py)
- the board is registry-backed and keyed by:
  - `system_id`
  - `lane`
  - `run_intent`

Interpretation:

- the repo can now render internal leaderboard and scatter-style benchmark views from normalized run metadata
- the board now supports role-family, category, track, modality, and executor-mode cuts
- the board now also carries runtime-facing fields when manifests provide them:
  - `warmup_load_ms`
  - `last_request_elapsed_ms`
  - `requests_completed`
  - `total_cost_per_mtok`
- the current board is good enough for internal benchmarking and chart production
- the remaining gap to a public leaderboard is broader model coverage plus stronger latency and cost instrumentation

## Shared Runtime / Product Surface Layer

Core runtime:

- [`src/gemma4_capability_map/runtime/core.py`](../../src/gemma4_capability_map/runtime/core.py)
- [`src/gemma4_capability_map/runtime/schemas.py`](../../src/gemma4_capability_map/runtime/schemas.py)
- [`src/gemma4_capability_map/runtime/workflows.py`](../../src/gemma4_capability_map/runtime/workflows.py)
- [`configs/packaged_workflows.yaml`](../../configs/packaged_workflows.yaml)

User-facing entrypoints:

- CLI:
  - [`src/gemma4_capability_map/runtime/cli.py`](../../src/gemma4_capability_map/runtime/cli.py)
  - package entrypoint: `moonie-agent`
- local API:
  - [`src/gemma4_capability_map/api/app.py`](../../src/gemma4_capability_map/api/app.py)
  - package entrypoint: `moonie-agent-api`

Current shape:

- packaged workflows are benchmark-backed KWA slices, not free-form general agents yet
- runtime sessions now persist:
  - session metadata
  - event timelines
  - approval requests
  - artifact paths
  - trace/manifests/summaries
- the local API currently supports:
  - health
  - profile listing
  - workflow listing
  - session creation
  - session detail/event/artifact reads
  - approval resolution
- the Streamlit operator-console and mobile-companion views sit on this same runtime contract

Validation:

- runtime + API regression coverage now exists in:
  - [`tests/test_runtime_core.py`](../../tests/test_runtime_core.py)
  - [`tests/test_runtime_api.py`](../../tests/test_runtime_api.py)
- full suite status after the current publishable-core reruns and controller/runtime hardening:
  - `194 passed`

## Current Full-Lane Comparative Systems

The board now has meaningful full-lane comparison rows for:

- canonical oracle:
  - `oracle_gemma4_e2b`
- exploratory local reasoner-only via `hf_service`:
  - `hf_service_gemma4_reasoner_only`
- exploratory local specialist-backed via `hf_service` + real HF specialists:
  - `hf_service_gemma4_specialists_cpu`
- exploratory local reasoner-only via direct in-process HF:
  - `hf_gemma4_e2b_reasoner_only`
- exploratory local specialist-backed via direct in-process HF:
  - `hf_gemma4_e2b_specialists_cpu`

Interpretation:

- the important publishable-default row is now `hf_gemma4_e2b_specialists_cpu`, not the older bounded direct-HF `v3` slices
- the current direct in-process Gemma specialist row matches the oracle row on the full-lane `26 / 20` board surface
- the direct in-process reasoner-only control remains materially weaker on that same surface
- the bounded visual invoice/form reruns remain useful as diagnostic proof of where the controller and memo fixes came from, but they are no longer the headline comparison artifact
- the current strongest claim is therefore:
  - we materially improved Gemma 4 as a local full-stack agent on our own harder benchmark
- the next missing comparative evidence is a real non-Gemma local stack on the same surface, not more re-runs of the now-clean Gemma specialist row
- `mlx` is still blocked locally because the runtime probe fails with `ModuleNotFoundError: mlx`
- `google/gemma-4-E4B-it` remains probe-only locally on this Mac and should not be treated as the next serious full-lane comparison target

## Policy-Hardening Oracle Snapshots

Replayable policy-hardening subset:

- [`results/knowledge_work/replayable_policy_hardening_oracle/summary.json`](../../results/knowledge_work/replayable_policy_hardening_oracle/summary.json)
- episodes:
  - `kwa_exec_vendor_access_hold`
  - `kwa_jobs_screening_hold`
  - `kwa_finance_billing_patch_hold`
- aggregate metrics:
  - `artifact_quality_avg = 1.0`
  - `browser_workflow_avg = 0.9818`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9363`

Live-web policy-hardening subset:

- [`results/knowledge_work/live_policy_hardening_oracle/summary.json`](../../results/knowledge_work/live_policy_hardening_oracle/summary.json)
- episodes:
  - `kwa_exec_live_vendor_access_hold`
  - `kwa_jobs_live_screening_hold`
  - `kwa_finance_live_billing_patch_hold`
- aggregate metrics:
  - `artifact_quality_avg = 1.0`
  - `browser_workflow_avg = 1.0`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9383`

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

## First Fully Specialist-Backed KnowledgeWorkArena Pilot

The first bounded fully specialist-backed KWA run is:

- [`results/knowledge_work/model_backed_hf_specialists_finance/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_finance/summary.json)

Configuration:

- reasoner backend: `hf_service`
- reasoner: `google/gemma-4-E2B-it`
- router backend: `hf`
- router: `google/functiongemma-270m-it`
- router device: `cpu`
- retriever backend: `hf`
- retriever: `google/embeddinggemma-300m`
- retriever device: `cpu`
- episode: `kwa_finance_three_statement_model`

Metrics:

- `artifact_quality_avg = 1.0`
- `browser_workflow_avg = 1.0`
- `strict_interface_avg = 0.7`
- `recovered_execution_avg = 1.0`
- `real_world_readiness_avg = 0.8574`

Interpretation:

- specialist loading stability is no longer the blocker for bounded KWA runs
- the current weakness has moved to interface discipline inside the composed full-stack episode
- the failing surface is not artifact quality or browser realism; it is controller/router exactness under full-stack composition

## Cross-Role Specialist-Backed Snapshot

The newer bounded cross-role specialist-backed run is:

- [`results/knowledge_work/model_backed_hf_specialists_cross_role/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_cross_role/summary.json)

Episodes:

- `kwa_exec_board_send_hold`
- `kwa_jobs_submission_hold`
- `kwa_finance_three_statement_model`

Aggregate metrics:

- `artifact_quality_avg = 1.0`
- `browser_workflow_avg = 0.9896`
- `strict_interface_avg = 1.0`
- `recovered_execution_avg = 1.0`
- `real_world_readiness_avg = 0.9063`

Interpretation:

- the planner hardening fixed the earlier specialist-backed finance interface leak
- the stack is now stable enough across three role families to move the next hardening pass toward tougher branch realism and broader specialist-backed volume

## Broader Cross-Role Specialist-Backed Snapshot

The current larger replayable cross-role specialist-backed reference is:

- [`results/knowledge_work/model_backed_hf_specialists_cross_role_broad_v2/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_cross_role_broad_v2/summary.json)

Episodes:

- `kwa_exec_board_prep_pack`
- `kwa_exec_inbox_triage`
- `kwa_exec_vendor_access_hold`
- `kwa_jobs_tailored_packet`
- `kwa_jobs_revise_after_feedback`
- `kwa_jobs_submission_hold`
- `kwa_finance_three_statement_model`
- `kwa_finance_partner_deck_revision`
- `kwa_finance_billing_patch_hold`

Aggregate metrics:

- `artifact_quality_avg = 1.0`
- `browser_workflow_avg = 0.9939`
- `strict_interface_avg = 1.0`
- `recovered_execution_avg = 1.0`
- `real_world_readiness_avg = 0.9175`
- `escalation_correctness_avg = 1.0`

Interpretation:

- the broader replayable specialist-backed stack is now clean across policy and non-policy episodes in all three role families
- the decisive fix was controller-side, not model-side: the planner now enforces full parallel evidence batches before accepting a patch proposal and repairs patch arguments from combined image + repo feedback
- the previous broad replayable `v1` run should be treated as a diagnosis artifact for `agent_010_parallel_audit_patch`, not the current reference state

## Broader Live Cross-Role Specialist-Backed Snapshot

The current larger live-web stress cross-role specialist-backed reference is:

- [`results/knowledge_work/model_backed_hf_specialists_cross_role_live_broad_v1/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_cross_role_live_broad_v1/summary.json)

Episodes:

- `kwa_exec_live_brief`
- `kwa_exec_live_calendar_policy`
- `kwa_exec_live_vendor_access_hold`
- `kwa_jobs_live_requirements_extract`
- `kwa_jobs_live_career_plan`
- `kwa_jobs_live_submission_hold`
- `kwa_finance_live_earnings_update`
- `kwa_finance_live_comps_revision`
- `kwa_finance_live_billing_patch_hold`

Aggregate metrics:

- `artifact_quality_avg = 1.0`
- `browser_workflow_avg = 1.0`
- `strict_interface_avg = 1.0`
- `recovered_execution_avg = 1.0`
- `real_world_readiness_avg = 0.9794`
- `escalation_correctness_avg = 1.0`

Interpretation:

- the broader live specialist-backed stack is also clean across policy and non-policy episodes in all three role families
- the controller/planner hardening that fixed the replayable parallel-audit leak appears to generalize under the live-web stress lane
- the next uncertainty is no longer “can the stack stay interface-clean on a broader cross-role live slice?” but how far we can widen the model-backed matrix before new mixed-evidence or revision failures emerge

## Mixed-Pressure Specialist-Backed Cross-Role References

Replayable mixed-pressure broad reference:

- [`results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_visual_replayable_v2/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_visual_replayable_v2/summary.json)
- episodes:
  - broad cross-role core:
    - `kwa_exec_board_prep_pack`
    - `kwa_exec_inbox_triage`
    - `kwa_jobs_tailored_packet`
    - `kwa_jobs_revise_after_feedback`
    - `kwa_finance_three_statement_model`
    - `kwa_finance_partner_deck_revision`
    - `kwa_jobs_submission_hold`
    - `kwa_exec_vendor_access_hold`
    - `kwa_finance_billing_patch_hold`
  - harder human-nuance additions:
    - `kwa_exec_stale_brief_hold`
    - `kwa_jobs_constraint_preservation_hold`
    - `kwa_finance_stale_assumption_hold`
  - visual additions:
    - `kwa_exec_visual_dashboard_brief`
    - `kwa_jobs_visual_form_hold`
    - `kwa_finance_visual_invoice_hold`
- aggregate metrics:
  - `runs = 18`
  - `artifact_quality_avg = 0.9822`
  - `browser_workflow_avg = 0.9880`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9436`
  - `escalation_correctness_avg = 1.0`

Live mixed-pressure broad reference:

- [`results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_visual_live_v2/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_visual_live_v2/summary.json)
- episodes:
  - broad live cross-role core:
    - `kwa_exec_live_brief`
    - `kwa_exec_live_calendar_policy`
    - `kwa_jobs_live_requirements_extract`
    - `kwa_jobs_live_career_plan`
    - `kwa_jobs_live_submission_hold`
    - `kwa_finance_live_earnings_update`
    - `kwa_finance_live_comps_revision`
    - `kwa_exec_live_vendor_access_hold`
    - `kwa_finance_live_billing_patch_hold`
  - harder human-nuance additions:
    - `kwa_exec_live_stale_brief_hold`
    - `kwa_jobs_live_constraint_hold`
    - `kwa_finance_live_stale_assumption_hold`
  - visual additions:
    - `kwa_exec_live_visual_dashboard_brief`
    - `kwa_jobs_live_visual_form_hold`
    - `kwa_finance_live_visual_invoice_hold`
- aggregate metrics:
  - `runs = 15`
  - `artifact_quality_avg = 0.9786`
  - `browser_workflow_avg = 1.0`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9555`
  - `escalation_correctness_avg = 1.0`

Interpretation:

- the old non-visual mixed-pressure references are no longer the best summary of the current model-backed frontier
- the corrected replayable reference is `model_backed_hf_specialists_cross_role_hardmix_visual_replayable_v2`
- the corrected live reference is `model_backed_hf_specialists_cross_role_hardmix_visual_live_v2`
- the earlier `visual_*_v1` runs should be treated as diagnosis artifacts for image-id plumbing and visual answer-surface rescue, not the current reference state
- the current remaining softer realism signal is `kwa_finance_partner_deck_revision`:
  - revision responsiveness is still weak
  - memory retention is now scored correctly after the semantic scorer hardening

## Specialist-Backed Visual KWA Snapshot

Replayable visual slice:

- [`results/knowledge_work/model_backed_hf_specialists_visual_replayable_v3/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_visual_replayable_v3/summary.json)
- episodes:
  - `kwa_exec_visual_dashboard_brief`
  - `kwa_jobs_visual_form_hold`
  - `kwa_finance_visual_invoice_hold`
- aggregate metrics:
  - `artifact_quality_avg = 0.8932`
  - `browser_workflow_avg = 0.9782`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9141`

Interpretation:

- the specialist-backed stack now handles the job-shaped visual episodes cleanly in bounded form
- the first failing replayable visual slice was a benchmark plumbing issue, not a stable model weakness: the planner needed logical image ids, and visual tasks needed second-pass answer rescue
- the remaining gap on this bounded visual slice is softer artifact quality, not strict interface discipline

## Full-Lane Specialist-Backed Exploratory References

Replayable full generated corpus:

- [`results/knowledge_work/model_backed_hf_specialists_replayable_full_v1/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_replayable_full_v1/summary.json)
- aggregate metrics:
  - `runs = 24`
  - `artifact_quality_avg = 0.9866`
  - `browser_workflow_avg = 0.9910`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9510`
  - `escalation_correctness_avg = 1.0`

Live full generated corpus:

- [`results/knowledge_work/model_backed_hf_specialists_live_full_v1/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_live_full_v1/summary.json)
- aggregate metrics:
  - `runs = 18`
  - `artifact_quality_avg = 0.9822`
  - `browser_workflow_avg = 1.0`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9630`
  - `escalation_correctness_avg = 1.0`

Interpretation:

- the current local specialist-backed stack is now clean on the entire generated KWA corpus in both replayable and live exploratory form
- the remaining deltas on the full-lane runs are softer artifact/browser-readiness movements, not strict interface failures
- this shifts the next benchmark frontier away from “can the stack survive the current corpus?” and toward broader system comparisons, richer public-style reporting, and harder new episode design

## Full-Lane Reasoner-Only Exploratory References

Replayable full generated corpus:

- [`results/knowledge_work/model_backed_hf_reasoner_full_replayable_v1/summary.json`](../../results/knowledge_work/model_backed_hf_reasoner_full_replayable_v1/summary.json)
- aggregate metrics:
  - `runs = 24`
  - `artifact_quality_avg = 0.9866`
  - `browser_workflow_avg = 0.9910`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9510`
  - `escalation_correctness_avg = 1.0`

Live full generated corpus:

- [`results/knowledge_work/model_backed_hf_reasoner_full_live_v1/summary.json`](../../results/knowledge_work/model_backed_hf_reasoner_full_live_v1/summary.json)
- aggregate metrics:
  - `runs = 18`
  - `artifact_quality_avg = 0.9822`
  - `browser_workflow_avg = 1.0`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9630`
  - `escalation_correctness_avg = 1.0`

Interpretation:

- the comparative full-lane board now cleanly resolves `reasoner_only` versus `specialist_stack` through registry-backed system ids instead of ad hoc slice labels
- the board prefers `full_lane` exploratory runs over narrower subset sweeps, so the top-line comparison surface is no longer shadowed by later partial slices

## Harder Human-Nuance Specialist-Backed Snapshots

Replayable harder slice:

- [`results/knowledge_work/model_backed_hf_specialists_hard_human_replayable_v1/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_hard_human_replayable_v1/summary.json)
- episodes:
  - `kwa_exec_stale_brief_hold`
  - `kwa_jobs_constraint_preservation_hold`
  - `kwa_finance_stale_assumption_hold`
- aggregate metrics:
  - `artifact_quality_avg = 1.0`
  - `browser_workflow_avg = 0.9828`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9364`
  - `escalation_correctness_avg = 1.0`

Live harder slice:

- [`results/knowledge_work/model_backed_hf_specialists_hard_human_live_v1/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_hard_human_live_v1/summary.json)
- episodes:
  - `kwa_exec_live_stale_brief_hold`
  - `kwa_jobs_live_constraint_hold`
  - `kwa_finance_live_stale_assumption_hold`
- aggregate metrics:
  - `artifact_quality_avg = 1.0`
  - `browser_workflow_avg = 1.0`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9383`
  - `escalation_correctness_avg = 1.0`

Interpretation:

- the specialist-backed stack stays interface-clean on the new harder stale-context / constraint-preservation / stale-assumption family
- the next benchmark frontier is composition: mix these harder human-nuance patterns into broader cross-role matrices with more revision pressure instead of testing them only in isolation

## Specialist-Backed Policy-Hardening Snapshots

Replayable specialist-backed subset:

- [`results/knowledge_work/model_backed_hf_specialists_policy_replayable_v6/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_policy_replayable_v6/summary.json)
- episodes:
  - `kwa_exec_vendor_access_hold`
  - `kwa_jobs_screening_hold`
  - `kwa_finance_billing_patch_hold`
- aggregate metrics:
  - `artifact_quality_avg = 1.0`
  - `browser_workflow_avg = 0.9818`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9363`
  - `escalation_correctness_avg = 1.0`

Interpretation:

- the replayable policy subset is now aligned with the live policy subset on the bounded three-episode specialist-backed slice
- the decisive fix was not more prompt pressure alone; it was judgment-aware scoring and rescue acceptance keyed to `expected_action + basis`, plus broader semantic aliases for policy-safety language like `high-risk` and `safety control`
- exploratory replayable specialist-backed runs `v1` through `v5` should be treated as diagnosis artifacts, not the current reference state

Live specialist-backed subset:

- [`results/knowledge_work/model_backed_hf_specialists_policy_live/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_policy_live/summary.json)
- episodes:
  - `kwa_exec_live_vendor_access_hold`
  - `kwa_jobs_live_screening_hold`
  - `kwa_finance_live_billing_patch_hold`
- aggregate metrics:
  - `artifact_quality_avg = 1.0`
  - `browser_workflow_avg = 1.0`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9383`
  - `escalation_correctness_avg = 1.0`

Interpretation:

- bounded replayable and live policy-hardening specialist-backed slices are now both clean on the current three-episode subset
- the next model-backed realism step is broader volume and harder branching, not another pass on the same refusal/clarify bug

## Broader Specialist-Backed Policy Exploratory Snapshots

Replayable broad policy subset:

- [`results/knowledge_work/model_backed_hf_specialists_policy_replayable_broad_v2/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_policy_replayable_broad_v2/summary.json)
- episodes:
  - `kwa_exec_board_send_hold`
  - `kwa_exec_vendor_access_hold`
  - `kwa_jobs_submission_hold`
  - `kwa_jobs_screening_hold`
  - `kwa_finance_committee_hold`
  - `kwa_finance_billing_patch_hold`
- aggregate metrics:
  - `artifact_quality_avg = 1.0`
  - `browser_workflow_avg = 0.9827`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9364`
  - `escalation_correctness_avg = 1.0`

Live broad policy subset:

- [`results/knowledge_work/model_backed_hf_specialists_policy_live_broad_v3/summary.json`](../../results/knowledge_work/model_backed_hf_specialists_policy_live_broad_v3/summary.json)
- episodes:
  - `kwa_exec_live_send_hold`
  - `kwa_exec_live_vendor_access_hold`
  - `kwa_jobs_live_submission_hold`
  - `kwa_jobs_live_screening_hold`
  - `kwa_finance_live_committee_hold`
  - `kwa_finance_live_billing_patch_hold`
- aggregate metrics:
  - `artifact_quality_avg = 1.0`
  - `browser_workflow_avg = 1.0`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9383`
  - `escalation_correctness_avg = 1.0`

Interpretation:

- the clean policy result now generalizes beyond the original three-episode replayable/live subset
- the remaining high-value model-backed expansion work is now broader cross-role and non-policy volume, not another pass on the same hold/refusal bug

## KnowledgeWorkArena History Semantics

The history layer now distinguishes:

- latest canonical by lane
- latest exploratory by lane
- best historical by lane

Operationally:

- canonical top-line claims should still come from the explicit summary pointers above
- exploratory model-backed pilots should use `--no-update-latest --run-intent exploratory`
- generated history now reflects that split instead of treating every newest lane snapshot as equally authoritative

Interpretation:

- the same specialist-backed stack stayed clean on the live subset
- that suggests the current weakness is not generic model instability
- it is specific to certain replayable judgment/action-label surfaces, especially refusal vs defer in the replayable billing path

## Runtime Posture

- `hf_service` is now a core research execution primitive on this Mac.
- cold `hf_service` startup remains expensive:
  - the finished executive pilot required about `345s` from service boot to ready state
- the latest bounded specialist-backed finance pilot warmed to ready in about `37.9s`, showing that the reusable service path is materially better once the runtime is already in a usable local state
- `MLX` remains useful but session-sensitive; use current preflight rather than static prose when choosing local defaults
- real specialist-backed KWA runs now consistently surface the same decoding warning on the HF specialist path:
  - `top_p` / `top_k` generation flags may be ignored
  - this is not yet tied to a measured quality regression, but it should be cleaned up because backend-specific decoding drift is benchmark noise

## Artifact and Browser Realism

`KnowledgeWorkArena` now uses:

- real `.xlsx` finance/model work products
- real `.pptx` deck work products
- real `.docx` packet/memo/form work products
- seeded browser state machines with:
  - explicit transitions
  - `validation_failed` and `recovered` recovery branches
  - validation rules
  - approval gates
  - explicit policy blocks for unsafe requests
  - blocked submissions
  - sandbox endpoints for dry-run flows

## Important Caveat

Canonical lane pointers must not be overwritten by smoke runs or pilots.

Use:

- `--no-update-latest`

for any exploratory run that is not intended to replace the published replayable or live lane snapshot.

Also note:

- history views such as `latest by lane` can surface smaller exploratory runs that share the same lane label
- when this happens, the authoritative KWA pointers are the canonical summary files above, not the generic latest-lane aggregate
