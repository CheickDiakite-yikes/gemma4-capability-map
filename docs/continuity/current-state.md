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
- `moonie-agent inspect <session_id>`
  - inspects sandbox roots, artifacts, policy blocks, summary paths, scorecards, and per-task controller repair findings

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
- `uv run pytest tests/test_runtime_cli.py tests/test_runtime_core.py -q`
- `23 passed`
- `uv run pytest`
- `244 passed`

Latest real CLI smoke:

- command: `uv run moonie-agent live --workflow-id executive_visual_dashboard_review --system-id mlx_gemma4_e2b_reasoner_only --lane replayable_core --once --refresh-s 0.5 --timeout-s 1.0`
- session: `20260506T220039380037Z_executive_visual_dashboard_review`
- result: completed with all artifacts under the per-session sandbox
- `role_readiness_score = 0.9942`
- `strict_interface_score = 1.0`
- `recovered_execution_score = 1.0`
- `controller_repair_count = 0.5`
- `argument_repair_count = 0.5`
- `raw_planning_clean_rate = 0.5`
- `moonie-agent inspect <session_id> --target scorecard --json` attributes the repair to `visual_013_dashboard_stale_selection_recovery`, where MLX Gemma emitted a semantically reasonable but non-canonical `extract_layout` query and the controller repaired it to the benchmark-canonical visual argument

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
- HF service-backed ablation after FunctionGemma prompt patch:
  - [`results/knowledge_work_h1_slice/20260506T_h1_hf_service_prompt_patch_ablation_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_prompt_patch_ablation_v1_knowledge_work_ablation_packet)
- HF service-backed ablation after concrete FunctionGemma hints:
  - [`results/knowledge_work_h1_slice/20260506T_h1_hf_service_concrete_hint_ablation_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_concrete_hint_ablation_v1_knowledge_work_ablation_packet)
- HF service-backed ablation after visual filter repair:
  - [`results/knowledge_work_h1_slice/20260506T_h1_hf_service_visual_filter_repair_ablation_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_visual_filter_repair_ablation_v1_knowledge_work_ablation_packet)
- Compact visual semantics packet:
  - [`results/knowledge_work_h1_slice/20260506T_h1_visual_semantics_no_repair_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_visual_semantics_no_repair_v1_knowledge_work_ablation_packet)
- HF service-backed full H1 ablation after final visual turn directive:
  - [`results/knowledge_work_h1_slice/20260506T_h1_hf_service_turn_directive_ablation_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_turn_directive_ablation_v1_knowledge_work_ablation_packet)
- H1b follow-up slice scaffold:
  - [`configs/knowledge_work_h1b_slice.yaml`](../../configs/knowledge_work_h1b_slice.yaml)
  - [`docs/continuity/h1b-slice.md`](./h1b-slice.md)
- H1b compact visual policy packet:
  - [`results/knowledge_work_h1_slice/20260506T_h1b_visual_policy_packet_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1b_visual_policy_packet_v1_knowledge_work_ablation_packet)
- H1b full HF service-backed ablation:
  - [`results/knowledge_work_h1_slice/20260506T_h1b_hf_service_ablation_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1b_hf_service_ablation_v1_knowledge_work_ablation_packet)
- Visual sequencing canary:
  - [`results/knowledge_work_h1_slice/20260506T_h1_visual_sequence_hint_canary_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_visual_sequence_hint_canary_v1_knowledge_work_ablation_packet)
- Visual filter repair canary:
  - [`results/knowledge_work_h1_slice/20260506T_h1_visual_filter_repair_canary_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_visual_filter_repair_canary_v1_knowledge_work_ablation_packet)

MLX Gemma primary:

- `real_world_readiness_avg = 0.9749800000000001`
- `strict_interface_avg = 1.0`
- `recovered_execution_avg = 1.0`
- `controller_repair_avg = 0.0`
- `controller_fallback_avg = 0.0`
- `raw_planning_clean_rate_avg = 1.0`

HF service-backed Gemma baseline after concrete FunctionGemma hints:

- `real_world_readiness_avg = 0.9749800000000001`
- `strict_interface_avg = 1.0`
- `recovered_execution_avg = 1.0`
- `controller_repair_avg = 0.0`
- `controller_fallback_avg = 0.0`
- `raw_planning_clean_rate_avg = 1.0`
- `argument_repair_avg = 0.0`

H1 ablation result after concrete FunctionGemma hints:

- `no_controller_repair = 0.88748`
- `no_controller_fallback = 0.9749800000000001`
- `no_visual_rescue = 0.9749800000000001`
- `no_intent_priority = 0.9749800000000001`
- `no_argument_repair = 0.9749800000000001`
- `no_deterministic_visual_follow_on = 0.88748`

H1 ablation result after visual filter repair:

- `no_controller_repair = 0.8874599999999999`
- `no_controller_fallback = 0.9749800000000001`
- `no_visual_rescue = 0.9749800000000001`
- `no_intent_priority = 0.9749800000000001`
- `no_argument_repair = 0.9749800000000001`
- `no_deterministic_visual_follow_on = 0.9749800000000001`

Visual sequencing canary on the three visual H1 replayable episodes:

- base specialists:
  - `real_world_readiness_avg = 0.9809666666666667`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`
- `no_controller_repair`:
  - `real_world_readiness_avg = 0.8837333333333333`
  - `strict_interface_avg = 0.75`
  - `recovered_execution_avg = 0.6666666666666666`
  - `raw_planning_clean_rate_avg = 0.9583333333333334`
- `no_deterministic_visual_follow_on` before the filter-repair patch:
  - `real_world_readiness_avg = 0.8837333333333333`
  - `strict_interface_avg = 0.75`
  - `recovered_execution_avg = 0.6666666666666666`
  - `controller_repair_avg = 1.0`
  - `raw_planning_clean_rate_avg = 0.7916666666666666`
- `no_deterministic_visual_follow_on` after the filter-repair patch:
  - `real_world_readiness_avg = 0.9809666666666667`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `controller_repair_avg = 0.8333333333333334`
  - `argument_repair_avg = 0.5`
  - `controller_fallback_avg = 0.16666666666666666`
  - `raw_planning_clean_rate_avg = 0.8250000000000001`
  - `failure_candidate_count = 0`

Interpretation:

- H1 does not break MLX Gemma's controller-clean posture
- H1 still breaks HF Gemma when controller repair is disabled, even when raw syntax is mostly clean
- the concrete FunctionGemma hint removed the prior placeholder/fallback artifact from the baseline
- accepting valid-but-stale visual `refine_selection` filters was a real controller bug; repairing repeated filters to the pending visual filter restored the no-deterministic-follow-on mini-row to full recovery
- the full filter-repair rerun restored `no_deterministic_visual_follow_on` to baseline top-line readiness, but that row remains repair-heavy
- visual rescue, intent priority, argument repair, controller fallback, and deterministic visual follow-on no longer move top-line readiness on the current full H1 slice
- the only remaining top-line causal helper on H1 is controller repair
- the remaining research seam is now valid-but-semantically-wrong visual chains under disabled repair, especially repeated refinements that fail to complete readback

Trace-note analysis:

- concrete-hint full ablation:
  - [`trace_note_summary.json`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_concrete_hint_ablation_v1_knowledge_work_ablation_packet/trace_note_summary.json)
  - [`trace_note_counts.csv`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_concrete_hint_ablation_v1_knowledge_work_ablation_packet/trace_note_counts.csv)
- visual filter repair canary:
  - [`trace_note_summary.json`](../../results/knowledge_work_h1_slice/20260506T_h1_visual_filter_repair_canary_v1_knowledge_work_ablation_packet/trace_note_summary.json)
  - [`trace_note_counts.csv`](../../results/knowledge_work_h1_slice/20260506T_h1_visual_filter_repair_canary_v1_knowledge_work_ablation_packet/trace_note_counts.csv)
- visual filter repair full rerun:
  - [`trace_note_summary.json`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_visual_filter_repair_ablation_v1_knowledge_work_ablation_packet/trace_note_summary.json)
  - [`trace_note_counts.csv`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_visual_filter_repair_ablation_v1_knowledge_work_ablation_packet/trace_note_counts.csv)
  - [`trace_episode_failures.csv`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_visual_filter_repair_ablation_v1_knowledge_work_ablation_packet/trace_episode_failures.csv)
  - [`trace_failure_mode_counts.csv`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_visual_filter_repair_ablation_v1_knowledge_work_ablation_packet/trace_failure_mode_counts.csv)

Current trace read:

- concrete-hint full ablation:
  - `42` controller-note events across `35` H1 episode rows
  - `6` strict/recovered failure candidates
  - aggregate failure modes: `visual_readback_missing = 6`, `visual_stepwise_control = 6`, `fallback_planner = 4`, `argument_repair = 3`, `raw_refusal = 3`, `repair_disabled = 3`, `visual_follow_on = 3`, `visual_repeated_refinement = 3`
  - the old `generic_tool_name` mode is gone
- visual filter repair canary:
  - `5` controller-note events across `3` episode rows
  - `0` strict/recovered failure candidates
  - residual notes are `repaired_arguments:refine_selection`, `visual_stepwise_prior`, and one `controller_fallback_planner` on the visual dashboard episode
- visual filter repair full rerun:
  - `35` controller-note events across `35` H1 episode rows
  - `3` strict/recovered failure candidates
  - all failure candidates are in `hf_service_gemma4_specialists_cpu_no_controller_repair`
  - aggregate failure modes: `repair_disabled = 3`, `visual_readback_missing = 2`, `visual_repeated_refinement = 2`, `visual_stepwise_control = 2`
  - `no_deterministic_visual_follow_on` has `0` failure candidates after the pending-filter semantic repair

FunctionGemma prompt canary:

- output: [`results/knowledge_work_h1_slice/20260506T_h1_functiongemma_prompt_canary_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_functiongemma_prompt_canary_v1_knowledge_work_ablation_packet)
- readiness stayed `0.9749800000000001`
- strict/recovered stayed `1.0 / 1.0`
- `controller_fallback_avg` moved from `0.6` to `0.3`
- `controller_repair_avg` moved from `0.9` to `0.8`
- `raw_planning_clean_rate_avg` moved from `0.1` to `0.2`
- `argument_repair_avg` rose from `0.1` to `0.5`
- trace miner found `0` failure candidates and `controller_fallback_planner = 3`

FunctionGemma concrete-hint canary:

- output: [`results/knowledge_work_h1_slice/20260506T_h1_functiongemma_concrete_hint_canary_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_functiongemma_concrete_hint_canary_v1_knowledge_work_ablation_packet)
- readiness stayed `0.9749800000000001`
- strict/recovered stayed `1.0 / 1.0`
- `controller_repair_avg = 0.0`
- `argument_repair_avg = 0.0`
- `controller_fallback_avg = 0.0`
- `raw_planning_clean_rate_avg = 1.0`
- trace miner found `0` controller-note events and `0` failure candidates
- interpretation: the remaining baseline H1 controller burden was largely seeded by placeholder-shaped prompt examples; the next required check is the full H1 ablation after this stronger prompt prior

Full H1 ablation after the concrete hint:

- output: [`results/knowledge_work_h1_slice/20260506T_h1_hf_service_concrete_hint_ablation_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_concrete_hint_ablation_v1_knowledge_work_ablation_packet)
- baseline `hf_service_gemma4_specialists_cpu`: readiness `0.9749800000000001`, strict/recovered `1.0 / 1.0`, repair `0.0`, fallback `0.0`, raw clean `1.0`
- `no_controller_repair`: readiness `0.88748`, strict/recovered `0.775 / 0.7`, raw clean `0.89`
- `no_controller_fallback`: unchanged at readiness `0.9749800000000001`
- `no_visual_rescue`: unchanged at readiness `0.9749800000000001`
- `no_intent_priority`: unchanged at readiness `0.9749800000000001`
- `no_argument_repair`: unchanged at readiness `0.9749800000000001`
- `no_deterministic_visual_follow_on`: readiness `0.88748`, strict/recovered `0.775 / 0.7`, repair `0.8`, fallback `0.4`
- trace miner found `42` controller-note events and `6` failure candidates
- aggregate failure modes after the richer visual taxonomy: `visual_readback_missing = 6`, `visual_stepwise_control = 6`, `fallback_planner = 4`, `argument_repair = 3`, `raw_refusal = 3`, `repair_disabled = 3`, `visual_follow_on = 3`, `visual_repeated_refinement = 3`
- interpretation: fallback causality on H1 was prompt-artifact-heavy; repair causality now concentrates in stepwise visual control and future-state visual follow-ons

Full H1 ablation after visual filter repair:

- output: [`results/knowledge_work_h1_slice/20260506T_h1_hf_service_visual_filter_repair_ablation_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_visual_filter_repair_ablation_v1_knowledge_work_ablation_packet)
- baseline `hf_service_gemma4_specialists_cpu`: readiness `0.9749800000000001`, strict/recovered `1.0 / 1.0`, repair `0.0`, fallback `0.0`, raw clean `1.0`
- `no_controller_repair`: readiness `0.8874599999999999`, strict/recovered `0.775 / 0.7`, repair `0.1`, fallback `0.1`, raw clean `0.975`
- `no_controller_fallback`: unchanged at readiness `0.9749800000000001`
- `no_visual_rescue`: unchanged at readiness `0.9749800000000001`
- `no_intent_priority`: unchanged at readiness `0.9749800000000001`
- `no_argument_repair`: unchanged at readiness `0.9749800000000001`
- `no_deterministic_visual_follow_on`: restored to readiness `0.9749800000000001`, strict/recovered `1.0 / 1.0`, with repair `0.6`, argument repair `0.3`, fallback `0.1`, and raw clean `0.845`
- trace miner found `35` controller-note events and `3` failure candidates
- all remaining failure candidates are in `no_controller_repair`; the disabled-repair row is accepting valid calls that are semantically stale for the visual readback sequence
- interpretation: deterministic visual follow-on is not top-line causal after the pending-filter repair, but it still reduces controller burden; controller repair is the remaining top-line causal helper on H1

Compact H1 visual semantics packet:

- output: [`results/knowledge_work_h1_slice/20260506T_h1_visual_semantics_no_repair_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_visual_semantics_no_repair_v1_knowledge_work_ablation_packet)
- systems: `hf_service_gemma4_specialists_cpu`, `hf_service_gemma4_specialists_cpu_no_controller_repair`, `hf_service_gemma4_specialists_cpu_no_deterministic_visual_follow_on`
- episodes: `kwa_exec_backlog_resume_hold_v5`, `kwa_jobs_email_block_resume_hold_v5`, `kwa_finance_invoice_lock_direction_hold_v4`
- baseline specialists: readiness `0.9715666666666666`, strict/recovered `1.0 / 1.0`, repair/fallback `0.0 / 0.0`, raw clean `1.0`
- `no_controller_repair`: readiness `0.8257`, strict/recovered `0.625 / 0.5`, repair/fallback `0.0 / 0.0`, raw clean `1.0`
- `no_deterministic_visual_follow_on`: readiness `0.9715666666666666`, strict/recovered `1.0 / 1.0`, repair `0.8333333333333334`, argument repair `0.5`, fallback `0.0`, raw clean `0.7833333333333333`
- trace miner found `18` controller-note events and `3` failure candidates
- all failure candidates are in `no_controller_repair`
- failure modes: `repair_disabled = 3`, `visual_readback_missing = 2`, `visual_repeated_refinement = 2`, `visual_stepwise_control = 2`
- interpretation: this packet is now the fast replayable target for candidate visual sequence fixes; it reproduces the controller-repair dependence without the full 35-row H1 cost

FunctionGemma visual prompt-contract candidate:

- output: [`results/knowledge_work_h1_slice/20260506T_h1_visual_semantics_prompt_contract_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_visual_semantics_prompt_contract_v1_knowledge_work_ablation_packet)
- change under test: stronger system prompt wording that says to return exactly the next visual call and not replay prior visual calls
- baseline specialists: unchanged at readiness `0.9715666666666666`, strict/recovered `1.0 / 1.0`, raw clean `1.0`
- `no_controller_repair`: unchanged at readiness `0.8257`, strict/recovered `0.625 / 0.5`, raw clean `1.0`
- `no_deterministic_visual_follow_on`: unchanged top-line at readiness `0.9715666666666666`, strict/recovered `1.0 / 1.0`; argument repair dropped from `0.5` to `0.3333333333333333`
- trace miner found `17` controller-note events and `3` failure candidates
- interpretation: system-level wording alone is not enough for the disabled-repair row; the next candidate should move the exact next-call directive closer to the generation point or change the routing contract shape

FunctionGemma final visual turn directive candidate:

- output: [`results/knowledge_work_h1_slice/20260506T_h1_visual_semantics_turn_directive_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_visual_semantics_turn_directive_v1_knowledge_work_ablation_packet)
- change under test: append a final router directive after tool-result messages with the exact next visual call for this turn
- baseline specialists: readiness `0.9715666666666666`, strict/recovered `1.0 / 1.0`, repair/fallback `0.0 / 0.0`, raw clean `1.0`
- `no_controller_repair`: restored to readiness `0.9715666666666666`, strict/recovered `1.0 / 1.0`, repair/fallback `0.0 / 0.0`, raw clean `1.0`
- `no_deterministic_visual_follow_on`: restored to readiness `0.9715666666666666`, strict/recovered `1.0 / 1.0`, repair/fallback `0.0 / 0.0`, raw clean `1.0`
- trace miner found `15` controller-note events and `0` failure candidates
- the only remaining note family is `controller_repair_disabled`, which is the expected ablation marker rather than a repair event
- raw disabled-repair visual chains are now clean:
  - finance invoice: `extract_layout -> read_region_text`
  - executive backlog: `extract_layout -> needs review -> backlog -> enablement ops -> read_region_text`
  - jobs blocked email: `extract_layout -> latest -> blocked -> email -> read_region_text`
- interpretation: the residual H1 visual controller dependence was materially reducible through model-side prompt recency. System-level prose did not work; a final turn-level exact-call directive did.

Full H1 ablation after final visual turn directive:

- output: [`results/knowledge_work_h1_slice/20260506T_h1_hf_service_turn_directive_ablation_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_turn_directive_ablation_v1_knowledge_work_ablation_packet)
- all seven rows match baseline:
  - `hf_service_gemma4_specialists_cpu`
  - `hf_service_gemma4_specialists_cpu_no_controller_repair`
  - `hf_service_gemma4_specialists_cpu_no_controller_fallback`
  - `hf_service_gemma4_specialists_cpu_no_visual_rescue`
  - `hf_service_gemma4_specialists_cpu_no_intent_priority`
  - `hf_service_gemma4_specialists_cpu_no_argument_repair`
  - `hf_service_gemma4_specialists_cpu_no_deterministic_visual_follow_on`
- every row has readiness `0.9749800000000001`, strict/recovered `1.0 / 1.0`, repair/fallback `0.0 / 0.0`, and raw clean `1.0`
- trace miner found `30` notes and `0` failure candidates
- remaining notes are only ablation markers:
  - `controller_repair_disabled = 21`
  - `intent_priority_disabled = 9`
- interpretation: on current H1, the prior causal controller-repair signal is eliminated by moving the exact visual next-call contract into the final FunctionGemma routing turn. H1 is now saturated again and needs a harder follow-up slice to keep measuring controller dependence.

H1b follow-up:

- config: [`configs/knowledge_work_h1b_slice.yaml`](../../configs/knowledge_work_h1b_slice.yaml)
- doc: [`docs/continuity/h1b-slice.md`](./h1b-slice.md)
- H1b reuses existing replayable/live packaged-workflow episode pairs that were not in H1:
  - executive visual referent review
  - executive latest-action resume
  - jobs visual constraint override
  - jobs phone patch resume
  - finance visual invoice revision
- purpose: re-break the now-saturated H1 surface with longer visual referent carryover, latest-instruction pressure, CLI/action dependencies after visual evidence, artifact revision, and approval-safe stop pressure

H1b compact visual policy packet:

- output: [`results/knowledge_work_h1_slice/20260506T_h1b_visual_policy_packet_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1b_visual_policy_packet_v1_knowledge_work_ablation_packet)
- systems: `hf_service_gemma4_specialists_cpu`, `hf_service_gemma4_specialists_cpu_no_controller_repair`, `hf_service_gemma4_specialists_cpu_no_deterministic_visual_follow_on`
- episodes: `kwa_exec_visual_dashboard_referent_hold_v3`, `kwa_jobs_visual_constraint_override_hold_v2`, `kwa_finance_visual_invoice_revision_hold_v2`
- all three rows match:
  - readiness `0.9472999999999999`
  - strict/recovered `1.0 / 1.0`
  - repair/fallback `0.0 / 0.0`
  - raw clean `1.0`
- trace miner found `12` notes and `0` failure candidates
- remaining notes are only the expected `controller_repair_disabled` ablation markers
- interpretation: the first H1b compact packet did not re-break controller dependence. H1b's selected episodes are harsher on artifact/readiness level, but the visual turn directive still removes the visual controller burden on this subset.

H1b full HF service-backed ablation:

- output: [`results/knowledge_work_h1_slice/20260506T_h1b_hf_service_ablation_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1b_hf_service_ablation_v1_knowledge_work_ablation_packet)
- all seven rows match:
  - `hf_service_gemma4_specialists_cpu`
  - `hf_service_gemma4_specialists_cpu_no_controller_repair`
  - `hf_service_gemma4_specialists_cpu_no_controller_fallback`
  - `hf_service_gemma4_specialists_cpu_no_visual_rescue`
  - `hf_service_gemma4_specialists_cpu_no_intent_priority`
  - `hf_service_gemma4_specialists_cpu_no_argument_repair`
  - `hf_service_gemma4_specialists_cpu_no_deterministic_visual_follow_on`
- every row has readiness `0.9581199999999999`, strict/recovered `1.0 / 1.0`, repair/fallback `0.0 / 0.0`, argument repair `0.0`, and raw clean `1.0`
- trace mining found `30` notes and `0` failure candidates
- remaining notes are only ablation markers:
  - `controller_repair_disabled = 22`
  - `intent_priority_disabled = 8`
- interpretation: H1b is harsher than H1 on artifact/readiness level, but it is also saturated with respect to the current controller-helper ablations after the FunctionGemma final-turn directive. The next signal should come from live CLI execution and a genuinely new H1c stress slice, not another same-shape replay-only H1b packet.

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

6. H1 exposed the visual sequencing bottleneck, but H1 and H1b are both saturated after the final FunctionGemma turn directive.
They keep packaged workflow attribution and show that repair dependence was reducible through model-side routing-contract recency. The next useful signal is live CLI behavior plus a genuinely harder H1c slice.

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
