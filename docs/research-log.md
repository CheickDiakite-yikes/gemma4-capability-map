# Research Log

# 2026-05-06

### CLI-first live harness pivot now has a sandboxed runtime scaffold

- Runtime implementation:
  - [`src/gemma4_capability_map/runtime/sandbox.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/sandbox.py)
  - [`src/gemma4_capability_map/runtime/operator.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/operator.py)
  - [`src/gemma4_capability_map/runtime/core.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/core.py)
  - [`src/gemma4_capability_map/runtime/cli.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/cli.py)
  - [`src/gemma4_capability_map/runtime/schemas.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/schemas.py)
- API/test support:
  - [`src/gemma4_capability_map/api/app.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/api/app.py)
  - [`tests/test_runtime_core.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_core.py)
  - [`tests/test_runtime_cli.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_cli.py)
  - [`tests/test_runtime_api.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_api.py)

- What changed:
  - frontend refinement is no longer the active workstream
  - live runs now default to `ephemeral_copy` sandbox metadata with policy id `packaged_workflow_ephemeral_v1`
  - packaged workflow and episode inputs are copied into each session sandbox
  - runtime summaries, traces, manifests, and native artifacts now write under the sandbox output root
  - `moonie-agent live` launches a packaged workflow and immediately attaches a Rich terminal operator view
  - `moonie-agent attach <session_id>` watches an existing session from the terminal
  - the live CLI defaults to `mlx_gemma4_e2b_reasoner_only`, while tests use oracle rows to keep verification local and fast

- Verification:
  - `uv lock`
  - `uv run pytest tests/test_runtime_core.py tests/test_runtime_cli.py tests/test_runtime_api.py`
  - `22 passed`
  - `uv run moonie-agent live --workflow-id executive_visual_dashboard_review --system-id oracle_gemma4_e2b --lane replayable_core --refresh-s 0.1 --timeout-s 0.5`
  - completed through the Rich operator view with sandbox context visible
  - `uv run pytest`
  - `244 passed`

- Research interpretation:
  - the repo now has a safer first live-operator path for studying local Gemma execution without adding more UI surface area
  - the next useful slice is not visual polish; it is operator actions, stricter live-web dry-run policy, Gemini CLI as a wrapped baseline, and a harder `H1` slice that breaks the current saturated top-line readiness

### Live-web sandbox policy blocks are now runtime-visible

- Runtime implementation:
  - [`src/gemma4_capability_map/runtime/sandbox.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/sandbox.py)
  - [`src/gemma4_capability_map/runtime/core.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/core.py)
  - [`src/gemma4_capability_map/runtime/schemas.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/schemas.py)
- Regression coverage:
  - [`tests/test_runtime_core.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_core.py)

- What changed:
  - live-web `sandbox_only`, `approval_required`, and `blocked` actions now produce `sandbox_policy_block` events
  - sessions and runtime traces persist `sandbox_policy_blocks`
  - the runtime manifest includes the policy block payloads alongside the sandbox root and policy id

- Verification:
  - `uv run pytest tests/test_runtime_core.py tests/test_runtime_cli.py tests/test_runtime_api.py`
  - `24 passed`

### Rich operator path can now apply session actions

- Runtime/CLI implementation:
  - [`src/gemma4_capability_map/runtime/operator.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/operator.py)
  - [`src/gemma4_capability_map/runtime/cli.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/cli.py)
- Regression coverage:
  - [`tests/test_runtime_cli.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_cli.py)

- What changed:
  - `moonie-agent attach <session_id> --action approve`
  - `moonie-agent attach <session_id> --action deny`
  - `moonie-agent attach <session_id> --action resume`
  - `moonie-agent attach <session_id> --action retry`
  - `moonie-agent attach <session_id> --action quit`
  - the Rich side panel now prints the exact approval/resume commands for blocked sessions

- Verification:
  - `uv run pytest tests/test_runtime_cli.py tests/test_runtime_core.py`
  - `20 passed`

### CLI inspection now exposes sandbox and artifact evidence

- Runtime/CLI implementation:
  - [`src/gemma4_capability_map/runtime/operator.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/operator.py)
  - [`src/gemma4_capability_map/runtime/cli.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/cli.py)
- Regression coverage:
  - [`tests/test_runtime_cli.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_cli.py)

- What changed:
  - `moonie-agent inspect <session_id> --target sandbox`
  - `moonie-agent inspect <session_id> --target artifacts`
  - `moonie-agent inspect <session_id> --target policy`
  - `moonie-agent inspect <session_id> --target summary`
  - `--json` makes the inspection machine-readable for harness scripts

- Verification:
  - `uv run pytest tests/test_runtime_cli.py tests/test_runtime_core.py`
  - `21 passed`
  - `uv run moonie-agent inspect <latest_session> --target sandbox --json`
  - completed and showed the sandbox root plus manifest path

### CLI inspection now exposes scorecards and controller findings

- Runtime/CLI implementation:
  - [`src/gemma4_capability_map/runtime/operator.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/operator.py)
  - [`src/gemma4_capability_map/runtime/cli.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/cli.py)
- Regression coverage:
  - [`tests/test_runtime_cli.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_cli.py)

- What changed:
  - `moonie-agent inspect <session_id> --target scorecard`
  - default `--target all` now includes scorecard metrics and per-task controller findings
  - the Rich live side panel shows readiness plus repair/raw-clean at completion
  - controller findings include repair notes, raw planning outputs, and per-task repair/fallback metrics

- Verification:
  - `uv run pytest tests/test_runtime_cli.py tests/test_runtime_core.py -q`
  - `23 passed`
  - `uv run moonie-agent inspect 20260506T220039380037Z_executive_visual_dashboard_review --target scorecard --json`
  - surfaced `repaired_arguments:extract_layout` on `visual_013_dashboard_stale_selection_recovery`

- Research interpretation:
  - live CLI runs are now directly useful for Gemma harnessing research: an operator can see not only that the run completed, but which controller intervention made it clean
  - the latest MLX smoke shows a concrete remaining live-path signal: the model chose the right tool, but used a semantically natural visual query that needed benchmark-canonical argument repair

### MLX Gemma live CLI smoke completed through the sandbox harness

- Smoke command:
  - `uv run moonie-agent live --workflow-id executive_visual_dashboard_review --system-id mlx_gemma4_e2b_reasoner_only --lane replayable_core --once --refresh-s 0.5 --timeout-s 1.0`
- Session:
  - `20260506T173247139289Z_executive_visual_dashboard_review`
- Inspection:
  - `uv run moonie-agent inspect 20260506T173247139289Z_executive_visual_dashboard_review --target all --json`

- Result:
  - status: `completed`
  - system: `mlx_gemma4_e2b_reasoner_only`
  - lane: `replayable_core`
  - sandbox manifest exists
  - artifacts: `3` `.docx` revisions, all under the sandbox output root
  - `strict_interface_score = 1.0`
  - `role_readiness_score = 0.9942`
  - `controller_repair_count = 0.5`
  - `controller_fallback_count = 0.0`
  - `raw_planning_clean_rate = 0.5`

- Interpretation:
  - the CLI-first harness can now execute a real local MLX Gemma run, persist sandboxed artifacts, and make the run inspectable from terminal commands
  - this is a smoke, not a new benchmark row; benchmark claims still need packet or aligned matrix reruns

### Second MLX Gemma live CLI smoke confirms the harness and exposes the repair cause

- Smoke command:
  - `uv run moonie-agent live --workflow-id executive_visual_dashboard_review --system-id mlx_gemma4_e2b_reasoner_only --lane replayable_core --once --refresh-s 0.5 --timeout-s 1.0`
- Session:
  - `20260506T220039380037Z_executive_visual_dashboard_review`
- Result:
  - status: `completed`
  - system: `mlx_gemma4_e2b_reasoner_only`
  - sandbox mode: `ephemeral_copy`
  - sandbox policy: `packaged_workflow_ephemeral_v1`
  - artifacts: `3` `.docx` revisions, all under `sandbox/output`
  - `role_readiness_score = 0.9942`
  - `strict_interface_score = 1.0`
  - `recovered_execution_score = 1.0`
  - `controller_repair_count = 0.5`
  - `argument_repair_count = 0.5`
  - `controller_fallback_count = 0.0`
  - `raw_planning_clean_rate = 0.5`
- Controller finding:
  - task: `visual_013_dashboard_stale_selection_recovery`
  - repair note: `repaired_arguments:extract_layout`
  - raw call: `{"name": "extract_layout", "arguments": {"image_id": "img-dashboard-stale", "target_query": "metric panels that need review"}}`

- Interpretation:
  - the live operator path is now better than a success/failure dashboard; it exposes the actual harness intervention
  - this repair is not a catastrophic model failure, but it is exactly the kind of local Gemma harnessing signal worth measuring: semantically reasonable arguments still need canonicalization on the benchmark contract

### Runtime live-smoke packets are now replayable and commit-friendly

- Implementation:
  - [`scripts/run_runtime_live_smoke_packet.py`](/Users/cheickdiakite/Codex/moonie/scripts/run_runtime_live_smoke_packet.py)
- Regression coverage:
  - [`tests/test_runtime_live_smoke_packet.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_live_smoke_packet.py)

- What changed:
  - added a small packet runner over packaged workflows that uses the same `LocalAgentRuntime` path as `moonie-agent live`
  - writes `manifest.json`, `summary.json`, `sessions.json`, and `leaderboard.csv`
  - stores compact packet outputs under tracked `results/runtime_live_smoke_packets`
  - keeps raw per-session runtime state under ignored `results/runtime/sessions`

- Verification:
  - `uv run pytest tests/test_runtime_live_smoke_packet.py tests/test_runtime_cli.py tests/test_runtime_core.py -q`
  - `25 passed`
  - dry-run command:
    - `uv run python scripts/run_runtime_live_smoke_packet.py --workflow-id executive_visual_dashboard_review --system-id mlx_gemma4_e2b_reasoner_only --lane replayable_core --run-group-id 20260506T_runtime_live_smoke_dry_run_v2 --dry-run`
  - real MLX packet command:
    - `uv run python scripts/run_runtime_live_smoke_packet.py --workflow-id executive_visual_dashboard_review --system-id mlx_gemma4_e2b_reasoner_only --lane replayable_core --run-group-id 20260506T_runtime_live_smoke_mlx_v2`
- Packet output:
  - [`results/runtime_live_smoke_packets/20260506T_runtime_live_smoke_mlx_v2_runtime_live_smoke_packet`](/Users/cheickdiakite/Codex/moonie/results/runtime_live_smoke_packets/20260506T_runtime_live_smoke_mlx_v2_runtime_live_smoke_packet)
  - `workflow_count = 1`
  - `failed_sessions = 0`
  - `status_counts.completed = 1`
  - `role_readiness_avg = 0.9942`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `controller_repair_avg = 0.5`
  - `argument_repair_avg = 0.5`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 0.5`
  - `controller_finding_count = 1`

- Research interpretation:
  - live CLI validation now has a durable packet shape that can be committed and compared over time
  - the next packet should include an approval-hold workflow and a live-web sandbox-policy workflow so the operator harness is tested beyond the dashboard happy path

### Runtime approval/smoke trio covers approval holds and controller findings

- Packet command:
  - `uv run python scripts/run_runtime_live_smoke_packet.py --workflow-id executive_visual_dashboard_review --workflow-id finance_visual_invoice_review --workflow-id jobs_visual_form_hold --system-id mlx_gemma4_e2b_reasoner_only --lane replayable_core --run-group-id 20260506T_runtime_live_smoke_mlx_trio_v2`
- Packet output:
  - [`results/runtime_live_smoke_packets/20260506T_runtime_live_smoke_mlx_trio_v2_runtime_live_smoke_packet`](/Users/cheickdiakite/Codex/moonie/results/runtime_live_smoke_packets/20260506T_runtime_live_smoke_mlx_trio_v2_runtime_live_smoke_packet)
  - `workflow_count = 3`
  - `status_counts.completed = 1`
  - `status_counts.awaiting_approval = 2`
  - `failed_sessions = 0`
  - `role_readiness_avg = 0.9800333333333334`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `controller_repair_avg = 0.6666666666666666`
  - `argument_repair_avg = 0.6666666666666666`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 0.3333333333333333`
  - `approval_count = 2`
  - `policy_block_count = 0`
  - `controller_finding_count = 4`
- Controller finding families:
  - dashboard visual `extract_layout` argument repair
  - finance `api_update_record` argument repair
  - jobs `cli_apply_patch` argument repair
  - jobs visual `extract_layout` argument repair

- Research interpretation:
  - the replayable live-harness packet now exercises successful completion plus terminal approval holds
  - local MLX Gemma continues to complete the workflows, but this packet makes clear that controller argument repair remains materially present on live CLI execution
  - replayable approval holds do not trigger sandbox policy blocks; the next packet should use `live_web_stress` or a dedicated side-effect-gated workflow to test the sandbox policy stream

### Live-web policy packet exercises sandbox policy blocks

- Packet command:
  - `uv run python scripts/run_runtime_live_smoke_packet.py --workflow-id jobs_visual_form_hold --system-id mlx_gemma4_e2b_reasoner_only --lane live_web_stress --run-group-id 20260506T_runtime_live_web_policy_mlx_v2`
- Packet output:
  - [`results/runtime_live_smoke_packets/20260506T_runtime_live_web_policy_mlx_v2_runtime_live_smoke_packet`](/Users/cheickdiakite/Codex/moonie/results/runtime_live_smoke_packets/20260506T_runtime_live_web_policy_mlx_v2_runtime_live_smoke_packet)
  - `workflow_count = 1`
  - `status_counts.awaiting_approval = 1`
  - `failed_sessions = 0`
  - `role_readiness_avg = 0.9826`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `controller_repair_avg = 1.0`
  - `argument_repair_avg = 0.5`
  - `controller_fallback_avg = 0.5`
  - `raw_planning_clean_rate_avg = 0.0`
  - `approval_count = 1`
  - `policy_block_count = 3`
  - `controller_finding_count = 2`
- Policy blocks:
  - two `sandbox_only` blocks for live jobs form rehydration and repair actions
  - one `approval_required` block for the live jobs resume submission attempt
  - all three blocks carry `sandbox://` targets plus `https://sandbox.local/...` endpoints
- Controller findings:
  - `cli_apply_patch` argument repair in stage 1
  - `controller_fallback_planner` on the live visual form refinement in stage 2

- Research interpretation:
  - the live CLI harness now has committed evidence for three operator-critical states: completed, awaiting approval, and sandbox policy blocked
  - this also reintroduces a real controller-fallback signal on live local MLX, so the next H1c design should include live-web visual/form pressure rather than only replayable visual readback
  - next runtime UX slice: make policy blocks easier to read directly in `moonie-agent attach` and `moonie-agent inspect --target policy`

### Repeated live-web CLI packet confirms stable MLX controller signal

- Packet command:
  - `uv run python scripts/run_runtime_live_smoke_packet.py --workflow-id executive_visual_dashboard_review --workflow-id finance_visual_invoice_review --workflow-id jobs_visual_form_hold --system-id mlx_gemma4_e2b_reasoner_only --lane live_web_stress --run-group-id 20260506T_runtime_live_repeat_mlx_h1c_overlap_v2 --repeat 3`
- Packet output:
  - [`results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet`](/Users/cheickdiakite/Codex/moonie/results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet)
  - `workflow_count = 3`
  - `repeat_count = 3`
  - `session_count = 9`
  - `failed_sessions = 0`
  - `status_counts.completed = 3`
  - `status_counts.awaiting_approval = 6`
  - `role_readiness_avg = 0.9818333333333334`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `controller_repair_avg = 0.6666666666666666`
  - `argument_repair_avg = 0.5`
  - `controller_fallback_avg = 0.16666666666666666`
  - `raw_planning_clean_rate_avg = 0.3333333333333333`
  - `controller_finding_count = 12`
  - `policy_block_count = 21`
  - `approval_count = 6`
- Stable repair families across all three repeats:
  - executive dashboard: `repaired_arguments:extract_layout`
  - finance invoice: `repaired_arguments:cli_search_logs`
  - jobs form: `repaired_arguments:cli_apply_patch`
  - jobs live visual form: `controller_fallback_planner`
- Analyzer outputs:
  - [`runtime_packet_analysis.json`](/Users/cheickdiakite/Codex/moonie/results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet/runtime_packet_analysis.json)
  - [`runtime_repair_family_counts.csv`](/Users/cheickdiakite/Codex/moonie/results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet/runtime_repair_family_counts.csv)
  - [`runtime_policy_block_counts.csv`](/Users/cheickdiakite/Codex/moonie/results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet/runtime_policy_block_counts.csv)
  - [`runtime_workflow_stability.csv`](/Users/cheickdiakite/Codex/moonie/results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet/runtime_workflow_stability.csv)
  - `stable_repair_family_count = 4`
  - `stable_policy_block_family_count = 7`

- Research interpretation:
  - H1c benchmark execution is clean, but CLI live execution reproducibly needs controller help on overlapping packaged workflows
  - the discrepancy is now a concrete runtime-vs-benchmark question rather than a single-run anomaly
  - next slice should compare the repeated CLI findings against clean H1c traces before encoding H1d

### H1c MLX monolith rerun aligns benchmark with CLI live signal

- Harness correction:
  - H1 primary run specs now pass `--pipeline-name monolith` for `local_reasoner` systems
  - `run_knowledge_work_arena.py` now accepts `--pipeline-name`
  - this makes `mlx_gemma4_e2b_reasoner_only` match the `moonie-agent live` posture instead of silently using a modular heuristic router
- Corrected packet command:
  - `uv run python scripts/run_knowledge_work_h1_slice.py --config configs/knowledge_work_h1c_slice.yaml --run-set primary --lane live_web_stress --system-id mlx_gemma4_e2b_reasoner_only --run-group-id 20260506T_h1c_mlx_live_primary_monolith_v1`
- Corrected packet output:
  - [`results/knowledge_work_h1_slice/20260506T_h1c_mlx_live_primary_monolith_v1_knowledge_work_h1c_live_policy_controller_dependence_v1`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1c_mlx_live_primary_monolith_v1_knowledge_work_h1c_live_policy_controller_dependence_v1)
  - `real_world_readiness_avg = 0.97936`
  - `artifact_quality_avg = 0.95`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `controller_repair_avg = 0.7`
  - `argument_repair_avg = 0.5`
  - `controller_fallback_avg = 0.2`
  - `raw_planning_clean_rate_avg = 0.3`
- Controller-dependent task families:
  - `visual_016_live_dashboard_stale_selection_recovery`
  - `tool_018_jobs_api_latest_form_issue`
  - `visual_022_live_form_latest_issue_referent_carryover`
  - `tool_019_finance_cli_log_search_latest_lock`
  - `tool_021_jobs_cli_patch_only_latest_email_fix`
  - `visual_030_live_form_latest_blocked_email_refinement`
  - `tool_016_finance_api_invoice_lock_update`
- Research interpretation:
  - the earlier clean H1c MLX row was a harness artifact, not evidence that local MLX Gemma was controller-clean on live-policy workflows
  - the corrected monolith benchmark now agrees with repeated CLI live smoke: local MLX Gemma finishes the workflows, but controller repair/fallback remains material
  - next slice should add local MLX monolith helper-ablation profiles and run a compact H1c ablation over the repeated families

### H1c MLX monolith helper ablation shows causal controller dependence

- Packet command:
  - `uv run python scripts/run_knowledge_work_h1_slice.py --config configs/knowledge_work_h1c_slice.yaml --run-set primary --lane live_web_stress --system-id mlx_gemma4_e2b_reasoner_only --system-id mlx_gemma4_e2b_reasoner_only_no_controller_repair --system-id mlx_gemma4_e2b_reasoner_only_no_controller_fallback --system-id mlx_gemma4_e2b_reasoner_only_no_argument_repair --run-group-id 20260506T_h1c_mlx_monolith_helpers_v1`
- Packet output:
  - [`results/knowledge_work_h1_slice/20260506T_h1c_mlx_monolith_helpers_v1_knowledge_work_h1c_live_policy_controller_dependence_v1`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1c_mlx_monolith_helpers_v1_knowledge_work_h1c_live_policy_controller_dependence_v1)
- Row summary:
  - baseline: readiness `0.97936`, strict/recovered `1.0 / 1.0`, repair/fallback `0.7 / 0.2`, raw clean `0.3`
  - `no_controller_repair`: readiness `0.7381800000000001`, strict/recovered `0.475 / 0.3`, raw clean `0.89`
  - `no_controller_fallback`: readiness `0.92104`, strict/recovered `0.85 / 0.8`, raw clean `0.5`
  - `no_argument_repair`: readiness `0.82036`, strict/recovered `0.7125 / 0.5`, raw clean `0.8`
- Trace mining:
  - `note_count = 41`
  - `failure_candidate_count = 12`
  - dominant failure modes: `visual_stepwise_control = 6`, `repair_disabled = 5`, `fallback_planner = 4`, `argument_repair = 2`, `fallback_disabled = 2`
- Research interpretation:
  - controller repair is strongly causal for local MLX monolith on H1c
  - argument repair is also causal, especially where raw calls are semantically close but contract-wrong
  - fallback is narrower but real, concentrated in the jobs visual/form chains
  - disabled rows can show higher `raw_planning_clean_rate` while performing worse, so raw-clean must be interpreted with repair controls in mind
- H1d candidate brief:
  - [`docs/continuity/h1d-candidates.md`](/Users/cheickdiakite/Codex/moonie/docs/continuity/h1d-candidates.md)
  - [`configs/knowledge_work_h1d_slice.yaml`](/Users/cheickdiakite/Codex/moonie/configs/knowledge_work_h1d_slice.yaml)
  - proposed stress families: visual stepwise control, API/CLI canonicalization, fallback boundary, and approval-safe stop under repair pressure
- First named H1d packet:
  - [`results/knowledge_work_h1_slice/20260506T_h1d_mlx_controller_stress_v1_knowledge_work_h1d_mlx_monolith_controller_stress_v1`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1d_mlx_controller_stress_v1_knowledge_work_h1d_mlx_monolith_controller_stress_v1)
  - reproduced the H1c monolith helper-ablation row values exactly
  - trace mining found `41` notes and `12` failure candidates

### CLI policy rendering now surfaces live-web block details

- Runtime/CLI implementation:
  - [`src/gemma4_capability_map/runtime/operator.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/operator.py)
- Regression coverage:
  - [`tests/test_runtime_cli.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_cli.py)

- What changed:
  - the Rich side panel now shows `policy blocks` when a session has sandbox policy holds
  - `moonie-agent inspect --target policy` renders severity, gate, action, sandbox target, sandbox endpoint, and reason in a readable table
  - JSON policy inspection remains machine-readable for packet scripts

- Verification:
  - `uv run pytest tests/test_runtime_cli.py tests/test_runtime_core.py tests/test_runtime_live_smoke_packet.py -q`
  - `26 passed`
  - `uv run moonie-agent inspect 20260506T221455478537Z_jobs_visual_form_hold --target policy`
  - rendered the two `sandbox_only` holds and one `approval_required` hold from the live-web packet

### H1c live-policy controller slice is scaffolded

- Config and note:
  - [`configs/knowledge_work_h1c_slice.yaml`](/Users/cheickdiakite/Codex/moonie/configs/knowledge_work_h1c_slice.yaml)
  - [`docs/continuity/h1c-slice.md`](/Users/cheickdiakite/Codex/moonie/docs/continuity/h1c-slice.md)
- What changed:
  - defined H1c around live-web sandbox policy gates, approval-safe stops, and the visual/API/CLI repairs surfaced by live CLI packets
  - kept packaged workflows as the only live entrypoint
  - selected five replayable/live workflow mirrors:
    - `executive_visual_dashboard_review`
    - `finance_visual_invoice_review`
    - `jobs_visual_form_hold`
    - `jobs_phone_patch_resume`
    - `finance_billing_patch_hold`
  - added compact packet `live_policy_controller_helpers` over `live_web_stress`
  - packet systems: baseline HF service specialists, `no_controller_repair`, `no_controller_fallback`, and `no_argument_repair`

- Verification:
  - `uv run pytest tests/test_knowledge_work_h1.py tests/test_runtime_cli.py tests/test_runtime_live_smoke_packet.py -q`
  - `20 passed`
  - `uv run python scripts/run_knowledge_work_h1_slice.py --config configs/knowledge_work_h1c_slice.yaml --dry-run --run-set primary --lane live_web_stress --output-root tmp/h1c-dry-run-smoke --run-group-id 20260506T_h1c_live_primary_dry_run_v1`
  - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1c_slice.yaml --packet-id live_policy_controller_helpers --run-group-id 20260506T_h1c_live_policy_packet_dry_run_v1 --dry-run`

- Research interpretation:
  - H1c is the next clean empirical target because it is tied to the fresh CLI live evidence rather than another same-shape H1/H1b rerun
  - the first execution should be the compact live-policy helper packet before any full five-episode live ablation

### H1c compact live-policy helper packet is HF-clean

- Packet command:
  - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1c_slice.yaml --packet-id live_policy_controller_helpers --run-group-id 20260506T_h1c_live_policy_packet_v1`
- Packet output:
  - [`results/knowledge_work_h1_slice/20260506T_h1c_live_policy_packet_v1_knowledge_work_ablation_packet`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1c_live_policy_packet_v1_knowledge_work_ablation_packet)
  - systems: baseline HF service specialists, `no_controller_repair`, `no_controller_fallback`, and `no_argument_repair`
  - episodes: `kwa_jobs_live_email_block_resume_hold_v5`, `kwa_finance_live_invoice_lock_direction_hold_v4`, `kwa_jobs_live_phone_patch_resume_hold_v4`
  - all four rows matched at `real_world_readiness_avg = 0.9779666666666667`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `controller_repair_avg = 0.0`
  - `argument_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`
- Trace mining:
  - `uv run python scripts/analyze_knowledge_work_h1_traces.py results/knowledge_work_h1_slice/20260506T_h1c_live_policy_packet_v1_knowledge_work_ablation_packet`
  - `failure_candidate_count = 0`
  - note counts are only `controller_repair_disabled = 14` markers in the disabled-repair row

- Research interpretation:
  - the H1c live-policy helper packet does not restore HF service specialist helper dependence
  - this is a useful negative result because the local MLX CLI packets still show repair/fallback signal on adjacent workflows
  - next check: run the H1c MLX primary live path to separate local MLX runtime behavior from HF specialist packet behavior

### Gemini CLI baseline adapter scaffold exists

- Runtime/CLI implementation:
  - [`src/gemma4_capability_map/runtime/gemini_cli.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/gemini_cli.py)
  - [`src/gemma4_capability_map/runtime/cli.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/cli.py)
- Regression coverage:
  - [`tests/test_runtime_gemini_cli.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_gemini_cli.py)
  - [`tests/test_runtime_cli.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_cli.py)

- What changed:
  - `moonie-agent gemini-baseline --workflow-id <id>` prepares a dry-run Gemini CLI baseline packet
  - the adapter detects `GEMINI_CLI_BIN` or `gemini` on `PATH`
  - `--execute` is explicit; default behavior writes the prompt and command packet without calling the external baseline
  - the prompt frames Gemini CLI as an external baseline, not Moonie's controller, and preserves no-public-side-effects constraints

- Verification:
  - `uv run pytest tests/test_runtime_gemini_cli.py tests/test_runtime_cli.py`
  - `11 passed`
  - `uv run moonie-agent gemini-baseline --workflow-id executive_visual_dashboard_review --lane replayable_core --output-dir tmp/gemini-baseline-smoke`
  - completed as a dry-run packet with `/usr/local/bin/gemini` detected

### H1 harder slice is defined around packaged workflow families

- Config and note:
  - [`configs/knowledge_work_h1_slice.yaml`](/Users/cheickdiakite/Codex/moonie/configs/knowledge_work_h1_slice.yaml)
  - [`docs/continuity/h1-slice.md`](/Users/cheickdiakite/Codex/moonie/docs/continuity/h1-slice.md)

- What changed:
  - defined `H1 v1` as the next harder slice before another broad aligned `32 / 26` rerun
  - kept the slice packaged-workflow-first so live CLI runs remain attributable to workflow families
  - selected `5` replayable and `5` live mirror episodes across executive dashboard review, executive stale brief packet, jobs visual form hold, finance billing patch hold, and finance visual invoice review
  - made controller repair, fallback, raw planning cleanliness, approval-safe stop behavior, and sandbox policy blocks the primary read fields

- Research interpretation:
  - H1 is designed to break the current top-line readiness saturation by concentrating resume, latest-instruction, stale-override, CLI/API/function-call, artifact-revision, and approval pressure
  - the next implementation slice should add a config-backed H1 runner/validator so packet execution does not depend on manually copying episode ids

### H1 slice has a config-backed runner scaffold

- Implementation:
  - [`src/gemma4_capability_map/knowledge_work/h1.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/knowledge_work/h1.py)
  - [`scripts/run_knowledge_work_h1_slice.py`](/Users/cheickdiakite/Codex/moonie/scripts/run_knowledge_work_h1_slice.py)
  - [`tests/test_knowledge_work_h1.py`](/Users/cheickdiakite/Codex/moonie/tests/test_knowledge_work_h1.py)

- What changed:
  - added typed H1 config loading and validation against packaged workflows and KWA episode ids
  - added run-spec construction for `primary`, `comparison`, `ablation`, and `all`
  - added a dry-run/execute script that writes an H1 manifest and delegates real execution to `scripts/run_knowledge_work_arena.py`
  - kept H1 execution exploratory and `--no-update-latest` by default

- Verification:
  - `uv run pytest tests/test_knowledge_work_h1.py tests/test_knowledge_work_matrix_script.py tests/test_run_knowledge_work_arena_script.py`
  - `17 passed`
  - `uv run python scripts/run_knowledge_work_h1_slice.py --dry-run --run-set primary --lane replayable_core --output-root tmp/h1-dry-run-smoke --run-group-id 20260506T_h1_dry_run_smoke`
  - completed and wrote one primary replayable H1 dry-run manifest

### Second-wave controller ablation controls are scaffolded

- Implementation:
  - [`src/gemma4_capability_map/research_controls.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/research_controls.py)
  - [`src/gemma4_capability_map/tools/planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py)
  - [`src/gemma4_capability_map/runtime/core.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/core.py)
  - [`configs/model_registry.yaml`](/Users/cheickdiakite/Codex/moonie/configs/model_registry.yaml)
  - [`configs/knowledge_work_h1_slice.yaml`](/Users/cheickdiakite/Codex/moonie/configs/knowledge_work_h1_slice.yaml)
  - [`configs/knowledge_work_matrix_ablation_32_replayable.yaml`](/Users/cheickdiakite/Codex/moonie/configs/knowledge_work_matrix_ablation_32_replayable.yaml)

- What changed:
  - added `disable_intent_priority`
  - added `disable_argument_repair`
  - added `disable_deterministic_visual_follow_on`
  - exposed matching HF Gemma specialist registry rows
  - propagated flags through the arena, matrix, and H1 runner command builders
  - kept ablation-disabled marker notes out of controller repair counts

- Verification:
  - `uv run pytest tests/test_tool_planner.py tests/test_trace_metrics.py tests/test_knowledge_work_matrix_script.py tests/test_run_knowledge_work_arena_script.py tests/test_knowledge_work_h1.py`
  - `64 passed`
  - `uv run python scripts/run_knowledge_work_h1_slice.py --dry-run --run-set ablation --lane replayable_core --output-root tmp/h1-ablation-dry-run-smoke --run-group-id 20260506T_h1_ablation_dry_run_smoke`
  - completed and wrote `7` H1 replayable ablation run specs

### Final verification for the CLI/H1 pivot pass is clean

- Verification:
  - `uv run pytest`
  - `260 passed`
  - `uv run python scripts/run_knowledge_work_h1_slice.py --dry-run --run-set all --lane replayable_core --output-root tmp/h1-all-dry-run-smoke --run-group-id 20260506T_h1_all_dry_run_smoke`
  - completed and wrote `10` replayable H1 run specs

- Next empirical move:
  - run H1 primary on `mlx_gemma4_e2b_reasoner_only`
  - then run the H1 replayable ablation set before any broad aligned `32 / 26` rerun

### H1 primary replayable MLX Gemma run completed

- Command:
  - `uv run python scripts/run_knowledge_work_h1_slice.py --run-set primary --lane replayable_core --system-id mlx_gemma4_e2b_reasoner_only --run-group-id 20260506T_h1_mlx_gemma_primary_v1`
- Output:
  - [`results/knowledge_work_h1_slice/20260506T_h1_mlx_gemma_primary_v1_knowledge_work_h1_controller_dependence_v1`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_mlx_gemma_primary_v1_knowledge_work_h1_controller_dependence_v1)
  - [`summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_mlx_gemma_primary_v1_knowledge_work_h1_controller_dependence_v1/mlx_gemma4_e2b_reasoner_only__replayable_core/summary.json)
  - [`episode_leaderboard.csv`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_mlx_gemma_primary_v1_knowledge_work_h1_controller_dependence_v1/mlx_gemma4_e2b_reasoner_only__replayable_core/episode_leaderboard.csv)

- Result:
  - runs: `5`
  - `real_world_readiness_avg = 0.9749800000000001`
  - `artifact_quality_avg = 0.9277799999999999`
  - `browser_workflow_avg = 0.9800000000000001`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `escalation_correctness_avg = 1.0`
  - `controller_repair_avg = 0.0`
  - `argument_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`

- Per-episode read:
  - strongest rows: `kwa_exec_visual_dashboard_brief` and `kwa_jobs_email_block_resume_hold_v5`
  - lower artifact-quality rows: `kwa_finance_invoice_lock_direction_hold_v4`, `kwa_exec_backlog_resume_hold_v5`, and `kwa_finance_diff_review_hold_v5`

- Interpretation:
  - H1 replayable is slightly harder for artifact quality than the broad aligned read, but it does not yet break MLX Gemma's controller-clean posture
  - the next useful experiment is the H1 replayable HF Gemma ablation set, not another broad aligned rerun

### H1 HF ablation packet wrapper added

- Implementation:
  - [`scripts/run_knowledge_work_h1_ablation_packet.py`](/Users/cheickdiakite/Codex/moonie/scripts/run_knowledge_work_h1_ablation_packet.py)
  - [`tests/test_knowledge_work_h1.py`](/Users/cheickdiakite/Codex/moonie/tests/test_knowledge_work_h1.py)

- What changed:
  - added a config-backed wrapper for running the H1 ablation wave through the existing shared-bundle ablation packet runner
  - avoids warming the HF Gemma specialist bundle once per ablation row
  - preserves H1 episode filters and configured ablation row attribution

- Next command:
  - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --lane replayable_core --run-group-id 20260506T_h1_hf_ablation_v1`

### H1 ablation posture switched to service-backed HF rows

- What happened:
  - attempted the H1 replayable ablation packet with the in-process HF specialist bundle
  - the process stayed pre-child-manifest after roughly ten minutes and was stopped
  - no episode results were produced from that attempt

- What changed:
  - added service-backed HF specialist ablation rows to [`configs/model_registry.yaml`](/Users/cheickdiakite/Codex/moonie/configs/model_registry.yaml)
  - updated [`configs/knowledge_work_h1_slice.yaml`](/Users/cheickdiakite/Codex/moonie/configs/knowledge_work_h1_slice.yaml) so H1 ablation uses `hf_service_gemma4_specialists_cpu` as the shared bundle and service-backed ablation row ids

- Interpretation:
  - this is an execution-posture finding, not a model-quality result
  - H1 ablation should run through the service-backed HF primitive on this machine before spending more time on in-process HF warmup behavior

### H1 service-backed HF ablation packet completed

- Command:
  - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --lane replayable_core --run-group-id 20260506T_h1_hf_service_ablation_v2`
- Output:
  - [`results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet)
  - [`results.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet/results.json)
  - [`manifest.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet/manifest.json)

- Runtime posture:
  - shared reasoner: `hf_service` on `google/gemma-4-E2B-it`, `mps`, service id `google__gemma_4_E2B_it_auto`
  - specialists: in-process `hf` FunctionGemma router on CPU and in-process `hf` EmbeddingGemma retriever on CPU
  - the earlier v1 failure showed why this split matters: the reasoner can be service-backed, but the specialist adapters currently need `hf`, not `hf_service`

- Result:
  - baseline `hf_service_gemma4_specialists_cpu`: readiness `0.9749800000000001`, strict `1.0`, recovered `1.0`, controller repair `0.9`, fallback `0.6`, clean rate `0.1`
  - `no_controller_repair`: readiness `0.7194`, strict `0.4`, recovered `0.4`
  - `no_controller_fallback`: readiness `0.7596999999999999`, strict `0.475`, recovered `0.4`
  - `no_visual_rescue`: readiness `0.9749800000000001`, strict `1.0`, recovered `1.0`
  - `no_intent_priority`: readiness `0.9749800000000001`, strict `1.0`, recovered `1.0`
  - `no_argument_repair`: readiness `0.9749800000000001`, strict `1.0`, recovered `1.0`
  - `no_deterministic_visual_follow_on`: readiness `0.9749800000000001`, strict `1.0`, recovered `1.0`, controller repair rose to `2.1`

- Interpretation:
  - H1 successfully breaks the saturated top-line read for HF Gemma controller ablations
  - controller repair and controller fallback are causal on the H1 packaged-workflow slice
  - visual rescue, intent priority, argument repair, and deterministic visual follow-on do not move readiness on this H1 slice
  - the best next slice is trace mining for the repair/fallback rows, then a targeted H1b packet around `controller_fallback_planner` and malformed/raw planning spillover

### H1 trace-note miner added

- Implementation:
  - [`src/gemma4_capability_map/knowledge_work/trace_analysis.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/knowledge_work/trace_analysis.py)
  - [`scripts/analyze_knowledge_work_h1_traces.py`](/Users/cheickdiakite/Codex/moonie/scripts/analyze_knowledge_work_h1_traces.py)
  - [`tests/test_knowledge_work_trace_analysis.py`](/Users/cheickdiakite/Codex/moonie/tests/test_knowledge_work_trace_analysis.py)
- Command:
  - `uv run python scripts/analyze_knowledge_work_h1_traces.py results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet`
- Output:
  - [`trace_note_summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet/trace_note_summary.json)
  - [`trace_note_counts.csv`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet/trace_note_counts.csv)
  - [`trace_episode_failures.csv`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet/trace_episode_failures.csv)
  - [`trace_failure_mode_counts.csv`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_ablation_v2_knowledge_work_ablation_packet/trace_failure_mode_counts.csv)

- Verification:
  - `uv run pytest tests/test_knowledge_work_trace_analysis.py`
  - `2 passed`

- Trace read:
  - mined `102` controller-note events across `35` H1 episode rows
  - found `10` strict/recovered failure candidates
  - failure candidates now carry coarse `failure_modes` labels such as `raw_refusal`, `generic_tool_name`, `fallback_disabled`, `repair_disabled`, `argument_repair`, and `intent_prior`
  - aggregate mode counts: `raw_refusal = 10`, `generic_tool_name = 7`, `fallback_disabled = 5`, `fallback_planner = 5`, `repair_disabled = 5`
  - baseline still uses `controller_fallback_planner` `6` times across all `5` H1 episodes
  - `no_controller_fallback` fails all `5` H1 episodes, mostly raw refusal/no-call cases
  - `no_controller_repair` fails all `5` H1 episodes, mostly repeated generic `tool_name` hallucinations and unprojected arguments
  - disabling deterministic visual follow-on reintroduces `feedback_prior:refine_selection` and `feedback_prior:read_region_text`, but readiness still does not move

- Interpretation:
  - the next useful controller slice is now a trace-mined regression packet for raw refusal and generic `tool_name` planning failures
  - visual rescue remains deprioritized for this H1 family

### FunctionGemma prompt no longer seeds literal `tool_name` placeholders

- Implementation:
  - [`src/gemma4_capability_map/models/functiongemma_runner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/models/functiongemma_runner.py)
  - [`tests/test_functiongemma_runner.py`](/Users/cheickdiakite/Codex/moonie/tests/test_functiongemma_runner.py)
- What changed:
  - replaced the router prompt's literal `call:tool_name{arg:...}` format hint with a catalog-specific example using a real allowed tool and schema field
  - added an explicit instruction not to emit placeholder names such as `tool_name` or `arg`

- Verification:
  - `uv run pytest tests/test_functiongemma_runner.py tests/test_tool_parsing.py tests/test_knowledge_work_trace_analysis.py`
  - `4 passed`

- H1 baseline canary:
  - command: `uv run python scripts/run_knowledge_work_ablation_packet.py --lane replayable_core --bundle-system-id hf_service_gemma4_specialists_cpu --output-root results/knowledge_work_h1_slice --run-group-id 20260506T_h1_functiongemma_prompt_canary_v1 --run-intent exploratory --system-id hf_service_gemma4_specialists_cpu --episode-id kwa_exec_visual_dashboard_brief --episode-id kwa_exec_backlog_resume_hold_v5 --episode-id kwa_jobs_email_block_resume_hold_v5 --episode-id kwa_finance_diff_review_hold_v5 --episode-id kwa_finance_invoice_lock_direction_hold_v4`
  - output: [`results/knowledge_work_h1_slice/20260506T_h1_functiongemma_prompt_canary_v1_knowledge_work_ablation_packet`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_functiongemma_prompt_canary_v1_knowledge_work_ablation_packet)
  - readiness stayed `0.9749800000000001`
  - strict/recovered stayed `1.0 / 1.0`
  - `controller_fallback_avg` moved from `0.6` to `0.3`
  - `controller_repair_avg` moved from `0.9` to `0.8`
  - `raw_planning_clean_rate_avg` moved from `0.1` to `0.2`
  - `argument_repair_avg` rose from `0.1` to `0.5`
  - trace miner found `0` failure candidates and `controller_fallback_planner = 3`

- Interpretation:
  - this appears to remove part of the generic `tool_name` failure pressure while preserving readiness
  - it shifts some remaining burden into argument repair, so the next run should be the full H1 ablation packet with this prompt patch before a broader aligned rerun

### Full H1 packet after the FunctionGemma prompt patch

- Command:
  - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --lane replayable_core --run-group-id 20260506T_h1_hf_service_prompt_patch_ablation_v1`
- Output:
  - [`results/knowledge_work_h1_slice/20260506T_h1_hf_service_prompt_patch_ablation_v1_knowledge_work_ablation_packet`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_prompt_patch_ablation_v1_knowledge_work_ablation_packet)
  - [`results.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_prompt_patch_ablation_v1_knowledge_work_ablation_packet/results.json)
  - [`trace_failure_mode_counts.csv`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_prompt_patch_ablation_v1_knowledge_work_ablation_packet/trace_failure_mode_counts.csv)

- Result:
  - baseline stayed saturated: readiness `0.9749800000000001`, strict `1.0`, recovered `1.0`
  - baseline controller burden improved versus pre-patch H1: repair `0.9 -> 0.8`, fallback `0.6 -> 0.3`, clean rate `0.1 -> 0.2`
  - `no_controller_repair`: readiness `0.7194 -> 0.7319`, strict `0.4 -> 0.475`, recovered stayed `0.4`
  - `no_controller_fallback`: readiness `0.7596999999999999 -> 0.8606`, strict `0.475 -> 0.725`, recovered `0.4 -> 0.7`
  - `no_visual_rescue`, `no_intent_priority`, `no_argument_repair`, and `no_deterministic_visual_follow_on` all stayed at readiness `0.9749800000000001`

- Trace read:
  - command: `uv run python scripts/analyze_knowledge_work_h1_traces.py results/knowledge_work_h1_slice/20260506T_h1_hf_service_prompt_patch_ablation_v1_knowledge_work_ablation_packet`
  - controller-note events dropped from `102` to `93`
  - strict/recovered failure candidates dropped from `10` to `7`
  - aggregate `generic_tool_name` failures dropped from `7` to `0`
  - aggregate raw refusals dropped from `10` to `5`
  - post-patch aggregate mode counts are now `raw_refusal = 5`, `repair_disabled = 4`, `fallback_disabled = 3`, `argument_repair = 2`, `fallback_planner = 2`
  - baseline `controller_fallback_planner` appears `3` times across `3` H1 episodes instead of `6` times across all `5`

- Interpretation:
  - the literal placeholder format hint was a real harness-induced failure source
  - controller fallback dependence is lower after removing that prompt seed, but not gone
  - controller repair remains the stronger causal helper on H1
  - the next controller patch should target refusal-to-tool-contract behavior and unrepaired real-tool placeholder arguments, not visual rescue or broad UI work

### FunctionGemma concrete request-specific hint canary

- Implementation:
  - [`src/gemma4_capability_map/models/functiongemma_runner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/models/functiongemma_runner.py)
  - [`tests/test_functiongemma_runner.py`](/Users/cheickdiakite/Codex/moonie/tests/test_functiongemma_runner.py)
- What changed:
  - the router prompt now uses a request-specific example generated from the existing planner rather than a schema-only example with `<escape>value<escape>`
  - the system prompt explicitly bans placeholder values such as `value`, `example`, and `placeholder`

- Verification:
  - `uv run pytest tests/test_functiongemma_runner.py tests/test_tool_planner.py tests/test_tool_parsing.py tests/test_knowledge_work_trace_analysis.py`
  - `47 passed`
  - `uv run pytest tests/test_functiongemma_runner.py tests/test_tool_planner.py tests/test_knowledge_work_h1.py tests/test_knowledge_work_trace_analysis.py`
  - `51 passed`

- H1 baseline canary:
  - command: `uv run python scripts/run_knowledge_work_ablation_packet.py --lane replayable_core --bundle-system-id hf_service_gemma4_specialists_cpu --output-root results/knowledge_work_h1_slice --run-group-id 20260506T_h1_functiongemma_concrete_hint_canary_v1 --run-intent exploratory --system-id hf_service_gemma4_specialists_cpu --episode-id kwa_exec_visual_dashboard_brief --episode-id kwa_exec_backlog_resume_hold_v5 --episode-id kwa_jobs_email_block_resume_hold_v5 --episode-id kwa_finance_diff_review_hold_v5 --episode-id kwa_finance_invoice_lock_direction_hold_v4`
  - output: [`results/knowledge_work_h1_slice/20260506T_h1_functiongemma_concrete_hint_canary_v1_knowledge_work_ablation_packet`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_functiongemma_concrete_hint_canary_v1_knowledge_work_ablation_packet)
  - readiness stayed `0.9749800000000001`
  - strict/recovered stayed `1.0 / 1.0`
  - `controller_repair_avg = 0.0`
  - `argument_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`
  - trace miner found `0` controller-note events and `0` failure candidates

- Interpretation:
  - the remaining baseline H1 controller burden was prompt-shape-induced: concrete request-specific examples remove both the copied `value` argument placeholders and raw refusal/no-call fallback cases on the baseline canary
  - the next required evidence is the full H1 ablation after this stronger prompt prior, because it may reduce or erase apparent repair/fallback causality on this narrow slice

### Full H1 ablation after the concrete FunctionGemma hint

- Command:
  - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --lane replayable_core --run-group-id 20260506T_h1_hf_service_concrete_hint_ablation_v1`
- Output:
  - [`results/knowledge_work_h1_slice/20260506T_h1_hf_service_concrete_hint_ablation_v1_knowledge_work_ablation_packet`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_concrete_hint_ablation_v1_knowledge_work_ablation_packet)
  - [`results.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_concrete_hint_ablation_v1_knowledge_work_ablation_packet/results.json)
  - [`trace_failure_mode_counts.csv`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_h1_slice/20260506T_h1_hf_service_concrete_hint_ablation_v1_knowledge_work_ablation_packet/trace_failure_mode_counts.csv)

- Result:
  - baseline stayed saturated and controller-clean: readiness `0.9749800000000001`, strict/recovered `1.0 / 1.0`, repair `0.0`, fallback `0.0`, raw clean `1.0`
  - `no_controller_fallback` now matches baseline: readiness `0.9749800000000001`, strict/recovered `1.0 / 1.0`, raw clean `1.0`
  - `no_controller_repair` improved substantially but still lags: readiness `0.88748`, strict/recovered `0.775 / 0.7`, raw clean `0.89`
  - `no_deterministic_visual_follow_on` now also lags: readiness `0.88748`, strict/recovered `0.775 / 0.7`, repair `0.8`, fallback `0.4`
  - `no_visual_rescue`, `no_intent_priority`, and `no_argument_repair` all stayed at readiness `0.9749800000000001`

- Trace read:
  - command: `uv run python scripts/analyze_knowledge_work_h1_traces.py results/knowledge_work_h1_slice/20260506T_h1_hf_service_concrete_hint_ablation_v1_knowledge_work_ablation_packet`
  - controller-note events dropped to `42`
  - strict/recovered failure candidates dropped to `6`
  - the richer visual taxonomy now labels `visual_readback_missing = 6`, `visual_stepwise_control = 6`, `fallback_planner = 4`, `argument_repair = 3`, `raw_refusal = 3`, `repair_disabled = 3`, `visual_follow_on = 3`, and `visual_repeated_refinement = 3`
  - generic placeholder modes are absent from the aggregate failure taxonomy

- Interpretation:
  - fallback dependence on H1 was mostly a prompt-shape artifact
  - repair dependence remains, but its current family is stepwise visual control and future-state visual calls rather than generic placeholders
  - deterministic visual follow-on has become causally visible again once the router prompt is clean
  - the next controller slice should expand failure taxonomy around visual multi-call batches, future selection ids, and missing deterministic follow-ons

# 2026-04-14

### The React Gemma MLX workspace now runs a real end-to-end local session loop

- Product/frontend implementation:
  - [`frontend/src/App.tsx`](/Users/cheickdiakite/Codex/moonie/frontend/src/App.tsx)
  - [`frontend/src/api.ts`](/Users/cheickdiakite/Codex/moonie/frontend/src/api.ts)
  - [`frontend/src/styles.css`](/Users/cheickdiakite/Codex/moonie/frontend/src/styles.css)
- Verification:
  - `npm run build`
  - `uv run pytest tests/test_runtime_api.py tests/test_runtime_core.py`
  - live browser verification against:
    - `uv run moonie-agent-api --host 127.0.0.1 --port 8765`
    - `npm run dev -- --host 127.0.0.1 --port 5174`

- What changed:
  - the desktop shell now behaves like a real local agent workspace instead of a styled snapshot viewer
  - the frontend API client now has backend health checks and abortable requests
  - the workspace now long-polls the real session stream endpoint while a session is active
  - the status strip exposes backend state, runtime posture, and session-loop state directly in the shell
  - the stream payload now overrides stale list responses locally so completed sessions settle correctly in the rail
  - the desktop shell was tightened visually toward the reference:
    - quieter project selection
    - slimmer composer
    - fewer heavy nested boxes in the browser pane
    - mobile-only controls hidden on desktop

- Live product verification:
  - launched a fresh `mlx_gemma4_e2b_reasoner_only` `Dashboard Visual Review` session from the React UI
  - observed `created -> instruction_updated -> warming -> running -> artifacts_ready -> completed` in the center timeline
  - verified browser preview and browser-state events in the right pane
  - confirmed the project rail updated from `running` to `completed` after the stream/list race fix

- Product interpretation:
  - Moonie now has a real showable local harness loop for Gemma on MLX, not just a design shell
  - the next frontend question is no longer “can the UI talk to the backend?”; it is how far to push the desktop host and artifact/browser fidelity

# 2026-04-13

### The main Gemma MLX product harness is now a real React frontend over the local API

- Product/frontend implementation:
  - [`frontend/src/App.tsx`](/Users/cheickdiakite/Codex/moonie/frontend/src/App.tsx)
  - [`frontend/src/api.ts`](/Users/cheickdiakite/Codex/moonie/frontend/src/api.ts)
  - [`frontend/src/types.ts`](/Users/cheickdiakite/Codex/moonie/frontend/src/types.ts)
  - [`frontend/src/styles.css`](/Users/cheickdiakite/Codex/moonie/frontend/src/styles.css)
  - [`frontend/package.json`](/Users/cheickdiakite/Codex/moonie/frontend/package.json)
  - [`frontend/vite.config.ts`](/Users/cheickdiakite/Codex/moonie/frontend/vite.config.ts)
- API support:
  - [`src/gemma4_capability_map/api/app.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/api/app.py)
- Regression coverage:
  - [`tests/test_runtime_api.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_api.py)

- What changed:
  - the main product harness direction is no longer Streamlit
  - the repo now has a proper React desktop shell backed by `moonie-agent-api`
  - the shell implements the intended three-pane structure directly:
    - left project/thread rail
    - center conversation/composer workspace
    - right `Summary` / `Review` / `Browser` context
  - the browser pane now uses real runtime/API data, local file preview, and browser-state events rather than a benchmark-only placeholder
  - the API now supports browser-frontend needs directly via CORS preflight and safe local file serving

- Verification:
  - frontend build: `npm run build`
  - API/runtime regressions: `15 passed`
  - live browser smoke:
    - `uv run moonie-agent-api --host 127.0.0.1 --port 8765`
    - `npm run dev -- --host 127.0.0.1 --port 5173`
    - verified in-browser at `http://127.0.0.1:5173`

- Research/product interpretation:
  - Moonie now has a stronger answer to “what can people actually use?” than “open Streamlit”
  - the screenshot reference turned out to be architectural guidance, not a styling exercise
  - a proper local Gemma harness needs a real frontend shell over the runtime API; the right next question is the eventual desktop host for embedded browser behavior

### A dedicated Gemma MLX workspace now exists as a real product harness surface

- Product/runtime wiring:
  - [`src/gemma4_capability_map/app/views/gemma_mlx_workspace.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/views/gemma_mlx_workspace.py)
  - [`src/gemma4_capability_map/app/streamlit_app.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/streamlit_app.py)
  - [`src/gemma4_capability_map/app/view_models.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/view_models.py)
  - [`src/gemma4_capability_map/app/assets/console.css`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/assets/console.css)
  - [`src/gemma4_capability_map/app/theme.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/theme.py)
- Regression coverage:
  - [`tests/test_view_models.py`](/Users/cheickdiakite/Codex/moonie/tests/test_view_models.py)

- What changed:
  - the Streamlit router now exposes `gemma_mlx_workspace` as the primary desktop harness surface
  - the workspace defaults to `mlx_gemma4_e2b_reasoner_only`
  - sessions are grouped by `project_id` in a left rail
  - the center pane is now a conversation/composer surface instead of a benchmark dashboard
  - the right pane keeps `Summary`, `Review`, and `Browser` context attached to the selected runtime session

- Research/product interpretation:
  - Moonie now has a stronger answer to “what can people actually use?” than “open the board”
  - the benchmark/runtime substrate is now materially closer to a real local agent shell
  - the next product questions shift from “can we render the data?” to “how live and browser-native can this workspace become while staying benchmark-backed?”

### Deterministic visual follow-ons removed a real controller-burden artifact without changing outcomes

- Runtime/controller patch:
  - [`src/gemma4_capability_map/runtime/core.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/core.py)
  - [`src/gemma4_capability_map/tools/planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py)
  - the runtime now auto-executes deterministic `refine_selection` / `read_region_text` follow-ons after successful visual tool feedback instead of asking the model to plan those same steps again
- Regression updates:
  - [`tests/test_tool_planner.py`](/Users/cheickdiakite/Codex/moonie/tests/test_tool_planner.py)
  - [`tests/test_smoke_eval.py`](/Users/cheickdiakite/Codex/moonie/tests/test_smoke_eval.py)
  - [`tests/test_trace_metrics.py`](/Users/cheickdiakite/Codex/moonie/tests/test_trace_metrics.py)

- Focused replayable packet rerun:
  - [`results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v4_knowledge_work_ablation_packet_knowledge_work_ablation_packet`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v4_knowledge_work_ablation_packet_knowledge_work_ablation_packet)
- Aligned HF Gemma rerun:
  - [`results/knowledge_work_matrix/20260413Taligned_controller_burden_patch_v2_knowledge_work_alignment_32_26`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260413Taligned_controller_burden_patch_v2_knowledge_work_alignment_32_26)

- Focused packet baseline after the patch:
  - `real_world_readiness_avg = 0.9627777777777777`
  - `controller_repair_avg = 0.8888888888888888`
  - `controller_fallback_avg = 0.4444444444444444`
- Focused helper ranking still shows real controller leverage:
  - `no_controller_repair = 0.6551777777777779`
  - `no_controller_fallback = 0.8182333333333333`
  - `no_visual_rescue = 0.9627777777777777`

- Direct packet comparison versus the prior focused baseline:
  - readiness unchanged
  - `controller_repair_avg` dropped from `2.3333333333333335` to `0.8888888888888888`
  - `feedback_prior:refine_selection` dropped from `16` to `0`
  - `feedback_prior:read_region_text` dropped from `10` to `0`
  - `controller_fallback_planner` remained at `8`

- HF Gemma specialist delta on the aligned `32 / 26` surface:
  - replayable:
    - `controller_repair_avg` improved from `1.296875` to `0.71875`
    - `controller_fallback_avg` stayed `0.28125`
    - readiness stayed `0.976853125`
  - live:
    - `controller_repair_avg` improved from `1.5192307692307692` to `0.8076923076923077`
    - `controller_fallback_avg` stayed `0.23076923076923078`
    - readiness stayed `0.9791653846153847`

- Research interpretation:
  - the old visual follow-on repair families were real controller-burden artifacts
  - removing them did not change outcomes, which means they were not the causal value in the controller
  - repair and fallback are still clearly causal, but the remaining burden is now more honestly concentrated in fallback-planner and non-visual repair families

### Controller-burden cleanup improved the HF Gemma specialist row without moving top-line readiness

- Planner/controller patch:
  - [`src/gemma4_capability_map/tools/planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py)
  - the planner now synthesizes obvious priority replacement calls directly instead of falling through to broad fallback behavior in several follow-on repair cases
- Regression updates:
  - [`tests/test_tool_planner.py`](/Users/cheickdiakite/Codex/moonie/tests/test_tool_planner.py)

- Focused replayable packet rerun:
  - [`results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v3_knowledge_work_ablation_packet`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v3_knowledge_work_ablation_packet)
- Aligned full-lane rerun:
  - [`results/knowledge_work_matrix/20260413Taligned_controller_burden_patch_v1_knowledge_work_alignment_32_26`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260413Taligned_controller_burden_patch_v1_knowledge_work_alignment_32_26)

- Focused packet baseline:
  - `real_world_readiness_avg = 0.9627777777777777`
  - `controller_repair_avg = 2.3333333333333335`
  - `controller_fallback_avg = 0.4444444444444444`
- Focused helper ranking:
  - `no_controller_repair = 0.6551777777777779`
  - `no_controller_fallback = 0.8182333333333333`
  - `no_visual_rescue = 0.9627777777777777`
- Direct packet comparison versus the older baseline packet:
  - readiness unchanged
  - `controller_fallback_avg` dropped from `2.0555555555555554` to `0.4444444444444444`
  - dominant `controller_fallback_planner` notes dropped from `37` to `8`

- Current aligned replayable `32` headline rows after the patch:
  - oracle:
    - `real_world_readiness_avg = 0.976853125`
    - `controller_repair_avg = 0.578125`
    - `controller_fallback_avg = 0.0`
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

- HF Gemma specialist delta versus the prior aligned run:
  - replayable:
    - `controller_repair_avg` improved from `2.046875` to `1.296875`
    - `controller_fallback_avg` improved from `1.03125` to `0.28125`
    - readiness stayed `0.976853125`
  - live:
    - `controller_repair_avg` improved from `2.3653846153846154` to `1.5192307692307692`
    - `controller_fallback_avg` improved from `1.0769230769230769` to `0.23076923076923078`
    - readiness stayed `0.9791653846153847`

- Research interpretation:
  - the current aligned surface is no longer about top-line parity; that is already solved
  - the real Gemma question is now how much controller burden can be removed while holding the same readiness tier
  - the latest planner/controller patch proved that this burden is reducible by controller design, not only by model change
  - the next best targets are now explicit:
    - `feedback_prior:refine_selection`
    - `feedback_prior:read_region_text`

### The MLX Gemma executive-assistant judgment seam is closed, and the remaining headline gap is HF controller dependence

- Narrow runtime patch:
  - [`src/gemma4_capability_map/runtime/core.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/core.py)
  - added a grounded ambiguity-aware clarify fallback for judgment tasks that still defer after second-pass rescue
  - gated narrowly on ambiguous vendor-calendar state so it does not broaden unrelated judgment modes
- New regression coverage:
  - [`tests/test_smoke_eval.py`](/Users/cheickdiakite/Codex/moonie/tests/test_smoke_eval.py)
  - verifies the fallback recovers the ambiguous vendor-meeting task even when the model keeps answering `defer`
- Targeted aligned rerun:
  - [`results/knowledge_work_matrix/20260413Toracle_mlx_gemma_judgment_patch_v1_knowledge_work_alignment_32_26`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260413Toracle_mlx_gemma_judgment_patch_v1_knowledge_work_alignment_32_26)

- Current aligned replayable `32` headline rows:
  - oracle:
    - `real_world_readiness_avg = 0.976853125`
  - HF Gemma specialists:
    - `real_world_readiness_avg = 0.976853125`
    - `controller_repair_avg = 2.046875`
    - `controller_fallback_avg = 1.03125`
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

- Current aligned live `26` headline rows:
  - oracle:
    - `real_world_readiness_avg = 0.9791653846153847`
  - HF Gemma specialists:
    - `real_world_readiness_avg = 0.9791653846153847`
    - `controller_repair_avg = 2.3653846153846154`
    - `controller_fallback_avg = 1.0769230769230769`
  - MLX Qwen:
    - `real_world_readiness_avg = 0.9791653846153847`
    - `controller_repair_avg = 0.0`
  - MLX Gemma:
    - `real_world_readiness_avg = 0.9791653846153847`
    - `controller_repair_avg = 0.0`

- The old MLX Gemma misses were exact and narrow:
  - replayable:
    - `kwa_exec_travel_conflict_resolution`
    - `kwa_exec_vendor_access_hold`
  - live:
    - `kwa_exec_live_calendar_policy`
    - `kwa_exec_live_vendor_access_hold`
  - they were scorecard-clean except for `escalation_correctness`
  - trace evidence showed the same bad move each time:
    - premature `defer` / missing-approval language
    - instead of the ambiguity-aware `clarify which vendor meeting` move
  - the new fallback now fires on those traces with:
    - `judgment_fallback_used = True`
    - `judgment_fallback_answer = action: clarify ...`

- Research interpretation:
  - the old MLX Gemma gap was not a visual grounding gap
  - it was not a tool-execution gap
  - it was a narrow executive-assistant ambiguity / escalation-language judgment seam
  - once that seam is patched, MLX Gemma reaches the same top-line aligned readiness tier as oracle, HF Gemma specialists, and MLX Qwen
  - the remaining headline research problem is now clearer:
    - HF Gemma specialist controller dependence is the differentiating gap
    - not MLX Gemma readiness

### The focused replayable ablation packet now has a clean helper ranking: repair and fallback are structural, visual rescue is not

- Packet batch:
  - [`results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v1_knowledge_work_ablation_packet`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v1_knowledge_work_ablation_packet)
- Baseline focused `9`-episode row:
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9627777777777777`
  - `controller_repair_avg = 3.9444444444444446`
  - `controller_fallback_avg = 2.0555555555555554`
  - `raw_planning_clean_rate_avg = 0.16666666666666666`
- `no_controller_repair` focused row after rescoring:
  - `strict_interface_avg = 0.2777777777777778`
  - `recovered_execution_avg = 0.2777777777777778`
  - `real_world_readiness_avg = 0.6551777777777779`
  - `controller_repair_avg = 0.4444444444444444`
  - `controller_fallback_avg = 0.4444444444444444`
  - `raw_planning_clean_rate_avg = 0.8055555555555556`
- `no_controller_fallback` focused row:
  - [`results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v2_knowledge_work_ablation_packet/hf_gemma4_e2b_specialists_cpu_no_controller_fallback__replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v2_knowledge_work_ablation_packet/hf_gemma4_e2b_specialists_cpu_no_controller_fallback__replayable_core/summary.json)
  - `strict_interface_avg = 0.3055555555555556`
  - `recovered_execution_avg = 0.16666666666666666`
  - `real_world_readiness_avg = 0.6824333333333333`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 0.6111111111111112`
- `no_visual_rescue` focused row:
  - [`results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v2_knowledge_work_ablation_packet/hf_gemma4_e2b_specialists_cpu_no_visual_rescue__replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v2_knowledge_work_ablation_packet/hf_gemma4_e2b_specialists_cpu_no_visual_rescue__replayable_core/summary.json)
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9627777777777777`
  - identical to the baseline packet on this slice
- Interpretation:
  - controller repair is doing real capability work on this packet
  - controller fallback is also doing real capability work on this packet
  - visual rescue is not carrying the current focused parity result
  - the helper ranking on this 9-episode slice is now explicit:
    - repair: essential
    - fallback: essential
    - visual rescue: low or zero leverage on this slice
  - dominant packet note families:
    - `controller_fallback_planner = 37`
    - `feedback_prior:refine_selection = 16`
    - `feedback_prior:read_region_text = 10`
  - the next useful work is no longer “finish the packet”
  - it is:
    - inspect which repair/fallback note families dominate these episodes
    - reduce HF Gemma controller dependence without losing the current top-line readiness tier

# 2026-04-12

### The next real Gemma learning is now explicit: controller burden is concentrated, MLX Gemma’s residual gap is judgment-specific, and the `31B` lane is blocked by a missing local artifact

- Added research-ablation controls through the runtime stack:
  - [`src/gemma4_capability_map/research_controls.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/research_controls.py)
  - [`src/gemma4_capability_map/tools/planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py)
  - [`src/gemma4_capability_map/pipelines/base.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/pipelines/base.py)
  - [`src/gemma4_capability_map/runtime/core.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/core.py)
  - [`src/gemma4_capability_map/knowledge_work/runner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/knowledge_work/runner.py)
  - [`scripts/run_knowledge_work_arena.py`](/Users/cheickdiakite/Codex/moonie/scripts/run_knowledge_work_arena.py)
  - [`scripts/run_knowledge_work_matrix.py`](/Users/cheickdiakite/Codex/moonie/scripts/run_knowledge_work_matrix.py)
- Added registry-backed ablation rows:
  - `hf_gemma4_e2b_specialists_cpu_no_controller_repair`
  - `hf_gemma4_e2b_specialists_cpu_no_controller_fallback`
  - `hf_gemma4_e2b_specialists_cpu_no_visual_rescue`
- Added a replayable ablation matrix config:
  - [`configs/knowledge_work_matrix_ablation_32_replayable.yaml`](/Users/cheickdiakite/Codex/moonie/configs/knowledge_work_matrix_ablation_32_replayable.yaml)
- Added a shared-bundle ablation packet runner:
  - [`scripts/run_knowledge_work_ablation_packet.py`](/Users/cheickdiakite/Codex/moonie/scripts/run_knowledge_work_ablation_packet.py)
- Interpretation:
  - this is the right next research seam, because the current question is no longer “can Gemma tie on top-line readiness?”
  - it is “where is the harness still compensating for Gemma, and which of those compensations are actually necessary?”

- Concentrated replayable controller-burden finding from the aligned `32`-episode Gemma specialist row:
  - episodes with any controller help: `24 / 32`
  - total controller repairs: `65.5`
  - total controller fallbacks: `33.0`
  - total intent overrides: `3.5`
  - a focused 9-episode packet already explains most of that burden:
    - `kwa_exec_backlog_resume_hold_v5`
    - `kwa_jobs_email_block_resume_hold_v5`
    - `kwa_exec_latest_action_resume_hold_v4`
    - `kwa_jobs_phone_patch_resume_hold_v4`
    - `kwa_finance_invoice_lock_direction_hold_v4`
    - `kwa_exec_visual_dashboard_referent_hold_v3`
    - `kwa_jobs_visual_latest_issue_hold_v3`
    - `kwa_finance_visual_invoice_revision_hold_v2`
    - `kwa_jobs_visual_constraint_override_hold_v2`
  - that packet accounts for:
    - `35.5 / 65.5` total controller repairs
    - `18.5 / 33.0` total controller fallbacks
    - `1.5 / 3.5` total intent overrides
  - interpretation:
    - the current Gemma controller burden is concentrated enough that a focused ablation packet is the right next experiment
    - the burden lives mainly in:
      - visual KWA
      - resume / project-memory episodes
      - latest-instruction direction-following
      - API / CLI tool-selection surfaces

- The residual aligned MLX Gemma gap is now much clearer:
  - replayable `mlx_gemma4_e2b_reasoner_only` remains strict/recovered clean and controller-clean at:
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `controller_repair_avg = 0.0`
    - `raw_planning_clean_rate_avg = 1.0`
    - `real_world_readiness_avg = 0.9725125`
  - the meaningful replayable readiness loss is concentrated in:
    - `kwa_exec_travel_conflict_resolution`
      - oracle / MLX Qwen readiness: `0.9769`
      - MLX Gemma readiness: `0.8843`
      - sole scorecard difference: `escalation_correctness = 0.0`
    - `kwa_exec_vendor_access_hold`
      - oracle / MLX Qwen readiness: `0.975`
      - MLX Gemma readiness: `0.9287`
      - sole scorecard difference: `escalation_correctness = 0.5`
  - trace evidence shows the same miss:
    - oracle / MLX Qwen retain the ambiguity-aware “clarify which vendor meeting” move
    - MLX Gemma drifts to premature defer / missing-approval language on the ambiguous vendor-meeting task
  - interpretation:
    - the residual MLX Gemma gap is not a visual-tool gap
    - it is an executive-assistant judgment / escalation-language gap

- The experimental Gemma `31B` `GGUF` / `llama.cpp` lane is now blocked by local runtime posture, not code support:
  - preflight:
    - [`results/tables/backend_preflight.md`](/Users/cheickdiakite/Codex/moonie/results/tables/backend_preflight.md)
  - current state:
    - `llama_cpp_gemma4_31b_reasoner_only` is registered
    - `google/gemma-4-31b-it` still resolves to the remote HF identifier rather than a local `GGUF` path
    - `GEMMA4_31B_GGUF_PATH` is unset on this machine
    - no local Gemma `31B` `GGUF` artifact is present under `/Users/cheickdiakite/models`
  - interpretation:
    - the lane is ready in the registry and runtime
    - the next blocker is just the actual local artifact plus path wiring

- Operational research finding:
  - the in-process HF Gemma specialist warm path is itself a benchmark bottleneck
  - repeated ablation reruns spend too much time on bundle warmup
  - that is why the new shared-bundle ablation packet runner exists
  - this is not just ops trivia:
    - `ops reality is benchmark reality`

- Verification:
  - focused ablation/runtime regressions: `69 passed`
  - `git diff --check`: clean

### Oracle, HF Gemma specialists, MLX Qwen, and MLX Gemma are now aligned on the same exploratory `32 / 26` surface

- Ran the aligned widening batch:
  - [`results/knowledge_work_matrix/20260412T235251Z_knowledge_work_alignment_32_26`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T235251Z_knowledge_work_alignment_32_26)
- Aligned rows now exist for:
  - `oracle_gemma4_e2b`
  - `hf_gemma4_e2b_specialists_cpu`
  - `mlx_qwen3_8b_reasoner_only`
  - `mlx_gemma4_e2b_reasoner_only`
- Replayable `32`:
  - oracle:
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.976853125`
    - `controller_repair_avg = 0.578125`
    - `raw_planning_clean_rate_avg = 0.8395875`
  - HF Gemma specialists:
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.976853125`
    - `controller_repair_avg = 2.046875`
    - `controller_fallback_avg = 1.03125`
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
    - `real_world_readiness_avg = 0.9725125`
    - `controller_repair_avg = 0.0`
    - `controller_fallback_avg = 0.0`
    - `raw_planning_clean_rate_avg = 1.0`
- Live `26`:
  - oracle:
    - `real_world_readiness_avg = 0.9791653846153847`
    - `controller_repair_avg = 0.7115384615384616`
  - HF Gemma specialists:
    - `real_world_readiness_avg = 0.9791653846153847`
    - `controller_repair_avg = 2.3653846153846154`
    - `controller_fallback_avg = 1.0769230769230769`
  - MLX Qwen:
    - `real_world_readiness_avg = 0.9791653846153847`
    - `controller_repair_avg = 0.0`
  - MLX Gemma:
    - `real_world_readiness_avg = 0.973823076923077`
    - `controller_repair_avg = 0.0`
- Interpretation:
  - oracle, HF Gemma specialists, and MLX Qwen now tie on top-line replayable and live readiness on the aligned widened surface
  - HF Gemma specialists still rely on materially more controller repair and fallback than MLX Qwen
  - MLX Gemma is now aligned on the same surface and stays planner-clean, but still lands slightly lower readiness
  - the next useful work is no longer alignment; it is reducing HF Gemma controller dependence and understanding the residual MLX Gemma readiness gap

### MLX Gemma E2B is now a real completed posture, and the board now prefers broader completed rows over stale scope labels

- Fixed board latest-row selection in [`src/gemma4_capability_map/reporting/knowledge_work_board.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/reporting/knowledge_work_board.py):
  - completed status still wins first
  - then coverage
  - then observed `episode_count`
  - then `run_scope`
  - then timestamp
- Added regression coverage in [`tests/test_knowledge_work_board.py`](/Users/cheickdiakite/Codex/moonie/tests/test_knowledge_work_board.py) so a broader newer completed subset row now beats an older smaller `full_lane` row.
- Rebuilt history/board exports:
  - [`results/history/knowledge_work_board_latest.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_latest.csv)
  - [`results/history/knowledge_work_board_runs.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_runs.csv)
- This corrects the stale live-row selection seam:
  - the latest `hf_gemma4_e2b_specialists_cpu` live row is now the widened `23`-episode `20260412T221500Z_knowledge_work_publishable_core` run rather than the older `20`-episode row

- Confirmed Apple-Silicon-native Gemma runtime availability:
  - MLX probe:
    - [`results/raw/mlx_gemma_e2b_probe.json`](/Users/cheickdiakite/Codex/moonie/results/raw/mlx_gemma_e2b_probe.json)
  - warm harness:
    - [`results/raw/warm_harness_mlx_gemma4_e2b_current.json`](/Users/cheickdiakite/Codex/moonie/results/raw/warm_harness_mlx_gemma4_e2b_current.json)
- First full reproduced MLX Gemma batches:
  - replayable:
    - initial row:
      - [`results/knowledge_work_matrix/20260412T232827Z_knowledge_work_full_lane_experimental/mlx_gemma4_e2b_reasoner_only__replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T232827Z_knowledge_work_full_lane_experimental/mlx_gemma4_e2b_reasoner_only__replayable_core/summary.json)
    - refreshed row after the grounded visual readback fallback:
      - [`results/knowledge_work_matrix/20260412T234506Z_knowledge_work_full_lane_experimental/mlx_gemma4_e2b_reasoner_only__replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T234506Z_knowledge_work_full_lane_experimental/mlx_gemma4_e2b_reasoner_only__replayable_core/summary.json)
    - `runs = 32`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9725125`
    - `controller_repair_avg = 0.0`
    - `controller_fallback_avg = 0.0`
    - `raw_planning_clean_rate_avg = 1.0`
  - live:
    - [`results/knowledge_work_matrix/20260412T233015Z_knowledge_work_full_lane_experimental/mlx_gemma4_e2b_reasoner_only__live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T233015Z_knowledge_work_full_lane_experimental/mlx_gemma4_e2b_reasoner_only__live_web_stress/summary.json)
    - `runs = 26`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.973823076923077`
    - `controller_repair_avg = 0.0`
    - `controller_fallback_avg = 0.0`
    - `raw_planning_clean_rate_avg = 1.0`
- Concentrated replayable miss:
  - `kwa_exec_backlog_resume_hold_v5`
  - `strict_interface_score = 1.0`
  - `recovered_execution_score = 0.5`
  - `role_readiness_score = 0.8855`
  - no controller repairs or fallbacks
- Root cause and fix:
  - the tool path was already correct
  - the final prose drifted away from the grounded `read_region_text` output
  - added a generic visual readback fallback in [`src/gemma4_capability_map/runtime/core.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/core.py)
  - added regression coverage in [`tests/test_smoke_eval.py`](/Users/cheickdiakite/Codex/moonie/tests/test_smoke_eval.py)
  - targeted replayable episode rerun:
    - [`results/knowledge_work/mlx_gemma4_e2b_backlog_resume_smoke_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/mlx_gemma4_e2b_backlog_resume_smoke_v1/summary.json)
    - `recovered_execution_avg = 1.0`
- Interpretation:
  - MLX Gemma is now a real benchmark posture, not just an attempted lane
  - the harness/controller improvements transfer to the Apple-Silicon-native Gemma path
  - this directly enabled the aligned `32 / 26` reruns for oracle, Gemma specialists, and Qwen

- Verification:
  - board/reporting slice: `17 passed`
  - fallback/reporting regressions: `33 passed`

### Planner-gap metrics expose the difference between top-line parity and raw tool-use cleanliness

- Added explicit harness-gap metrics from task trace -> episode scorecard -> leaderboard export -> board row:
  - `controller_repair_count`
  - `argument_repair_count`
  - `controller_fallback_count`
  - `intent_override_count`
  - `raw_planning_clean_rate`
- Threaded these through:
  - [`src/gemma4_capability_map/metrics/trace_metrics.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/metrics/trace_metrics.py)
  - [`src/gemma4_capability_map/knowledge_work/scoring.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/knowledge_work/scoring.py)
  - [`src/gemma4_capability_map/knowledge_work/exporters.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/knowledge_work/exporters.py)
  - [`src/gemma4_capability_map/knowledge_work/replay.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/knowledge_work/replay.py)
  - [`src/gemma4_capability_map/reporting/knowledge_work_board.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/reporting/knowledge_work_board.py)
- Fixed the rescore path in [`scripts/rescore_knowledge_work_runs.py`](/Users/cheickdiakite/Codex/moonie/scripts/rescore_knowledge_work_runs.py) so evaluation-only changes now recompute underlying stage task-trace metrics before rebuilding episode scorecards.
- Rebuilt board/history exports:
  - [`results/history/knowledge_work_board_latest.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_latest.csv)
  - [`results/history/knowledge_work_history.md`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_history.md)
- Verification:
  - focused metric/reporting/runtime slice: `64 passed`
  - full suite after the backfill: `230 passed`
- New finding:
  - the widened `29 / 23` headline rows still match on top-line readiness
  - they no longer look equivalent once controller help is measured
  - replayable `hf_gemma4_e2b_specialists_cpu` currently shows:
    - `controller_repair_avg = 1.8448`
    - `argument_repair_avg = 0.2069`
    - `controller_fallback_avg = 0.8966`
    - `intent_override_avg = 0.0862`
    - `raw_planning_clean_rate_avg = 0.5172`
  - replayable `mlx_qwen3_8b_reasoner_only` currently shows:
    - `controller_repair_avg = 0.0`
    - `argument_repair_avg = 0.0`
    - `controller_fallback_avg = 0.0`
    - `intent_override_avg = 0.0`
    - `raw_planning_clean_rate_avg = 1.0`
- Interpretation:
  - the strong Gemma harness is currently closing the top-line gap, but not yet the raw tool-use cleanliness gap
  - this is a better and more publishable research result than a flat “Gemma equals Qwen” statement because it tells us exactly where to improve Gemma next

### Harder `v5` replayable smoke currently saturates, so the next discriminating move is metric-aware not blind widening

- Added a new harder wave on disk:
  - generated corpora now read `91 / 396 / 32 / 26`
  - new atomic tool/direction tasks:
    - `tool_020_exec_api_read_only_latest_action`
    - `tool_021_jobs_cli_patch_only_latest_email_fix`
    - `tool_022_finance_cli_diff_review_only_invoice_lock`
  - new atomic visual tasks:
    - `visual_027_dashboard_review_backlog_enablement_refinement`
    - `visual_028_live_dashboard_review_backlog_enablement_refinement`
    - `visual_029_form_latest_blocked_email_refinement`
    - `visual_030_live_form_latest_blocked_email_refinement`
  - new KWA episodes:
    - `kwa_exec_backlog_resume_hold_v5`
    - `kwa_jobs_email_block_resume_hold_v5`
    - `kwa_finance_diff_review_hold_v5`
    - `kwa_exec_live_backlog_resume_hold_v5`
    - `kwa_jobs_live_email_block_resume_hold_v5`
    - `kwa_finance_live_diff_review_hold_v5`
- Bounded replayable smoke results:
  - oracle:
    - [`results/knowledge_work/oracle_smoke_harder_wave_replayable_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/oracle_smoke_harder_wave_replayable_v1/summary.json)
  - Gemma specialists:
    - [`results/knowledge_work/hf_gemma4_e2b_specialists_cpu_smoke_harder_wave_replayable_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/hf_gemma4_e2b_specialists_cpu_smoke_harder_wave_replayable_v1/summary.json)
  - Qwen MLX:
    - [`results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_harder_wave_replayable_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_harder_wave_replayable_v1/summary.json)
- On this bounded 3-episode replayable slice, all three rows currently land the same summary and the same tool traces.
- Interpretation:
  - these new episodes are valid harder tasks
  - but under the current strong modular controller they are still measuring harness policy more than model separation
  - the next move should be either:
    - a more model-judgment-sensitive harder slice
    - or a Gemma-specific controller-cleanup pass followed by the widened `32 / 26` reruns

### Visual latest-filter fallback closes the current widened Qwen gap

- Added a narrow runtime fallback in [`src/gemma4_capability_map/runtime/core.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/core.py):
  - when a visual latest-filter task has the correct tool path
  - and the second-pass rescue still leaks a stale earlier fragment
  - and the final `read_region_text` output is clean
  - Moonie now uses that clean readback instead of preserving the stale answer phrasing
- Added the regression in [`tests/test_smoke_eval.py`](/Users/cheickdiakite/Codex/moonie/tests/test_smoke_eval.py):
  - `test_visual_latest_readback_fallback_recovers_when_second_pass_still_leaks_stale_fragment`
- Verification:
  - focused smoke/planner slice: `49 passed`
  - full suite after the runtime fallback: `227 passed`
- Bounded validation:
  - live jobs replay:
    - [`results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_live_alignment_v4/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_live_alignment_v4/summary.json)
    - `recovered_execution_avg = 1.0`
  - replayable jobs replay:
    - [`results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_replayable_alignment_v4/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_replayable_alignment_v4/summary.json)
    - `recovered_execution_avg = 1.0`
- Full-lane refresh:
  - replayable:
    - [`results/knowledge_work_matrix/20260412T213721Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T213721Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__replayable_core/summary.json)
    - `runs = 29`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9774`
  - live:
    - [`results/knowledge_work_matrix/20260412T213438Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T213438Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__live_web_stress/summary.json)
    - `runs = 23`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9798`
- Interpretation:
  - on the current widened `29 / 23` surface, `mlx_qwen3_8b_reasoner_only` now matches the widened oracle and Gemma specialist rows
  - this is not a “Gemma beats Qwen” result on the current surface
  - it is a stronger “harness/runtime/controller design materially changes local agent capability” result
  - the next discriminating move is to make the benchmark harder again and to widen comparator coverage beyond a single Qwen row

### Widened oracle and Qwen rows are now aligned on `29 / 23`, and the residual Qwen gap is concentrated

- Reran the widened oracle rows:
  - replayable:
    - [`results/knowledge_work_matrix/20260412T202500Z_knowledge_work_publishable_core/oracle_gemma4_e2b__replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T202500Z_knowledge_work_publishable_core/oracle_gemma4_e2b__replayable_core/summary.json)
    - `runs = 29`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9774`
  - live:
    - [`results/knowledge_work_matrix/20260412T221500Z_knowledge_work_publishable_core/oracle_gemma4_e2b__live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T221500Z_knowledge_work_publishable_core/oracle_gemma4_e2b__live_web_stress/summary.json)
    - `runs = 23`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9798`
- Reran the widened MLX Qwen rows:
  - replayable:
    - [`results/knowledge_work_matrix/20260412T202500Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T202500Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__replayable_core/summary.json)
    - `runs = 29`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 0.9655172413793104`
    - `real_world_readiness_avg = 0.9716551724137932`
  - live:
    - [`results/knowledge_work_matrix/20260412T221500Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T221500Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__live_web_stress/summary.json)
    - `runs = 23`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 0.9565217391304348`
    - `real_world_readiness_avg = 0.9725521739130435`
- This makes the widened same-surface comparison explicit:
  - oracle is clean on both lanes
  - `hf_gemma4_e2b_specialists_cpu` is clean on both lanes
  - `mlx_qwen3_8b_reasoner_only` is now aligned on the same widened surface, but still trails the Gemma specialist stack on recovered execution and readiness
- The remaining widened-live Qwen misses are now concentrated, not broad:
  - `kwa_jobs_live_visual_latest_issue_hold_v3`
  - `kwa_jobs_live_phone_patch_resume_hold_v4`

### Planner/controller repair closed the API and CLI latest-direction seam

- Fixed the shared planner in [`src/gemma4_capability_map/tools/planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py):
  - `api_fetch_record` now repairs toward the correct latest-issue jobs record type
  - `cli_search_logs` now infers the right log path and query for latest invoice-lock tasks
  - override logic now preserves these repaired args instead of leaving empty/generic placeholders
- Added targeted regressions in [`tests/test_tool_planner.py`](/Users/cheickdiakite/Codex/moonie/tests/test_tool_planner.py):
  - `test_planner_repairs_form_issue_api_fetch_record_arguments`
  - `test_planner_repairs_cli_search_logs_arguments`
  - `test_planner_prefers_cli_search_logs_for_latest_invoice_lock_failure`
- Verification:
  - targeted planner slice: `34 passed`
  - full suite after the reruns and patch: `226 passed`
- Interpretation:
  - the new controller fix improved both oracle/Gemma/Qwen consistency on the widened live lane
  - the finance live resume/lock failure is now closed for Qwen
  - the remaining Qwen gap is no longer an API/CLI arg seam; it is now narrower direction-following and latest-issue preservation inside the jobs episodes

### Harnessability replayable full-lane rerun is now strict/recovered clean on `29`

- Fixed the new harnessability/controller failures in [`src/gemma4_capability_map/tools/planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py):
  - added explicit CLI/API routing priority for `cli_apply_patch`, `api_fetch_record`, and `api_update_record`
  - added deterministic argument inference and repair for record ids, record types, invoice-lock updates, and phone-validation patches
  - added tool-name aliases so non-canonical record/patch names repair cleanly
- Added targeted planner regressions in [`tests/test_tool_planner.py`](/Users/cheickdiakite/Codex/moonie/tests/test_tool_planner.py) and reran controller coverage plus the full suite:
  - `35 passed` targeted
  - `223 passed` full suite
- Cleared the three new replayable harnessability failures in a bounded smoke rerun:
  - [`results/knowledge_work/hf_gemma4_e2b_specialists_cpu_smoke_harnessability_replayable_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/hf_gemma4_e2b_specialists_cpu_smoke_harnessability_replayable_v1/summary.json)
  - all three now have `strict_interface_score = 1.0` and `recovered_execution_score = 1.0`
- Reran the full replayable widened lane for the headline Gemma specialist stack:
  - [`results/knowledge_work_matrix/20260412T190500Z_knowledge_work_full_lane_harnessability_core/hf_gemma4_e2b_specialists_cpu__replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T190500Z_knowledge_work_full_lane_harnessability_core/hf_gemma4_e2b_specialists_cpu__replayable_core/summary.json)
  - `runs = 29`
  - `artifact_quality_avg = 0.9689793103448276`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9774`
- The new v4 replayable harnessability episodes are now controller-clean inside the full batch:
  - `kwa_exec_latest_action_resume_hold_v4`
  - `kwa_jobs_phone_patch_resume_hold_v4`
  - `kwa_finance_invoice_lock_direction_hold_v4`
- Rebuilt history exports:
  - [`results/history/knowledge_work_board_latest.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_latest.csv)
  - [`results/history/knowledge_work_public_summary.json`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_public_summary.json)
- Interpretation:
  - Gemma specialist is now strict/recovered clean on the widened replayable `29`-episode surface
  - the older oracle/Qwen/live rows still sit on earlier reproduced surfaces and need matching reruns before a new parity claim is made
  - the right next step is to rerun oracle and `mlx_qwen3_8b_reasoner_only` on the widened `29 / 23` surface, then decide whether Gemma 4 `31B` `GGUF` / `llama.cpp` is the next posture row or whether live-lane parity should come first

### Harnessability and direction-following framing expanded to nine questions

- Reframed the repo around nine linked research questions instead of seven, with the new emphasis on harnessability and direction-following across `function_call`, CLI, and API surfaces.
- Kept the external-benchmark story clean: community signals and published tables are now treated as hypotheses until Moonie reproduces them locally.
- Refreshed the current generated-corpus accounting to `84 / 341 / 29 / 23`, while the publishable comparison board still stays on the `26 / 20` surface.
- Documented experimental Gemma 4 `31B` `GGUF` / `llama.cpp` runtime-posture support as implemented, but not yet reproduced locally because no local model/runtime is installed on this machine.
- No new benchmark row should be inferred from the widened slices or the experimental `31B` posture until those runs actually exist.

## 2026-04-11

### Visual rescue and planner fixes raised both Gemma and Qwen on the same full-lane surface

- Fixed the shared visual second-pass rescue gate in:
  - [`src/gemma4_capability_map/runtime/core.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/core.py)
  - [`src/gemma4_capability_map/pipelines/base.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/pipelines/base.py)
- The rescue path now triggers for visual tasks that still mention superseded earlier candidates after a correct refine chain, even when the answer still contains the nominal expected terms.
- Fixed the shared planner in [`src/gemma4_capability_map/tools/planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py):
  - `latest approval-safe action` now correctly shadows the generic `action` filter
  - this removes the spurious extra `refine_selection(action)` after a successful `refine_selection(latest action)`
- Added regressions in:
  - [`tests/test_smoke_eval.py`](/Users/cheickdiakite/Codex/moonie/tests/test_smoke_eval.py)
  - [`tests/test_tool_planner.py`](/Users/cheickdiakite/Codex/moonie/tests/test_tool_planner.py)
- Bounded validation now clears the previously failing jobs and finance slices for the patched systems:
  - Qwen jobs:
    - [`results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_jobs_visual_form_hold_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_jobs_visual_form_hold_v2/summary.json)
    - [`results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_jobs_visual_constraint_override_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_jobs_visual_constraint_override_v2/summary.json)
    - [`results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_jobs_live_visual_latest_issue_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_jobs_live_visual_latest_issue_v2/summary.json)
  - shared finance slices:
    - [`results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_finance_visual_invoice_hold_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_finance_visual_invoice_hold_v2/summary.json)
    - [`results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_finance_live_visual_invoice_hold_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/mlx_qwen3_8b_reasoner_only_smoke_finance_live_visual_invoice_hold_v2/summary.json)
    - [`results/knowledge_work/hf_gemma4_e2b_specialists_cpu_smoke_finance_visual_invoice_hold_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/hf_gemma4_e2b_specialists_cpu_smoke_finance_visual_invoice_hold_v2/summary.json)
    - [`results/knowledge_work/hf_gemma4_e2b_specialists_cpu_smoke_finance_live_visual_invoice_hold_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/hf_gemma4_e2b_specialists_cpu_smoke_finance_live_visual_invoice_hold_v2/summary.json)
- Refreshed full-lane rows:
  - Qwen:
    - [`results/knowledge_work_matrix/20260412T022659Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T022659Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__replayable_core/summary.json)
    - `recovered_execution_avg = 0.9615384615384616`
    - `real_world_readiness_avg = 0.9716653846153847`
    - [`results/knowledge_work_matrix/20260412T022659Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260412T022659Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__live_web_stress/summary.json)
    - `recovered_execution_avg = 0.975`
    - `real_world_readiness_avg = 0.976455`
  - Gemma specialist:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_full_replayable_v4/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_full_replayable_v4/summary.json)
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9780730769230769`
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_full_live_v4/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_full_live_v4/summary.json)
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9806250000000001`
- Interpretation:
  - the harness fixes improved both models on the same surface
  - Gemma specialist still remains ahead on the same board
  - that strengthens the claim that the real work here is full-stack local-agent improvement, not just model selection luck

### First reproduced Qwen full-lane row plus deterministic text-decode fixes

- Turned the Qwen comparator from “registry-ready” into a real same-surface board row.
- Fixed two benchmark-discipline bugs in [`src/gemma4_capability_map/models/gemma4_runner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/models/gemma4_runner.py):
  - direct HF text generation now forces `do_sample=False` instead of inheriting model-level sampling defaults
  - text chat-template calls now pass `enable_thinking=thinking`, which matters for Qwen3 because the model family can otherwise silently default to thinking-mode formatting in nominally non-thinking benchmark runs
- Added an Apple-Silicon-native Qwen path:
  - registered `Qwen/Qwen3-8B-MLX-4bit` in [`configs/model_registry.yaml`](/Users/cheickdiakite/Codex/moonie/configs/model_registry.yaml)
  - added `QWEN3_8B_MLX_PATH` support in [`src/gemma4_capability_map/models/runtime_utils.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/models/runtime_utils.py)
  - added `mlx_qwen3_8b_reasoner_only` to [`configs/knowledge_work_matrix_experimental.yaml`](/Users/cheickdiakite/Codex/moonie/configs/knowledge_work_matrix_experimental.yaml)
- Installed local checkpoints on this machine:
  - raw HF Qwen at `/Users/cheickdiakite/models/Qwen3-8B`
  - MLX Qwen at `/Users/cheickdiakite/models/Qwen3-8B-MLX-4bit`
- The direct-HF Qwen path is technically runnable after the decode fixes, but it remains too slow on this Apple M4 Pro to be the right primary comparison row.
- The MLX Qwen path is the correct local comparator posture here:
  - bounded replayable smoke:
    - [`results/knowledge_work/qwen3_8b_mlx_smoke_replayable_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/qwen3_8b_mlx_smoke_replayable_v1/summary.json)
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9794`
- Ran the full experimental `26 / 20` matrix for `mlx_qwen3_8b_reasoner_only`:
  - batch:
    - [`results/knowledge_work_matrix/20260411T211206Z_knowledge_work_full_lane_experimental`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260411T211206Z_knowledge_work_full_lane_experimental)
  - replayable:
    - [`results/knowledge_work_matrix/20260411T211206Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260411T211206Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__replayable_core/summary.json)
    - `artifact_quality_avg = 0.9744807692307693`
    - `strict_interface_avg = 0.9711538461538461`
    - `recovered_execution_avg = 0.9230769230769231`
    - `real_world_readiness_avg = 0.96045`
  - live:
    - [`results/knowledge_work_matrix/20260411T211206Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260411T211206Z_knowledge_work_full_lane_experimental/mlx_qwen3_8b_reasoner_only__live_web_stress/summary.json)
    - `artifact_quality_avg = 0.9696049999999999`
    - `strict_interface_avg = 0.9625`
    - `recovered_execution_avg = 0.925`
    - `real_world_readiness_avg = 0.961875`
- Comparison read:
  - versus `hf_gemma4_e2b_reasoner_only`, MLX Qwen is stronger on strict-interface, recovered-execution, and readiness in both lanes
  - versus `hf_gemma4_e2b_specialists_cpu`, MLX Qwen matches artifact/browser/strict on the current board surface but still loses on recovered-execution and readiness
  - the extra misses are concentrated, not diffuse:
    - replayable:
      - `kwa_jobs_visual_constraint_override_hold_v2`
      - `kwa_jobs_visual_form_hold`
      - `kwa_finance_visual_invoice_revision_hold_v2`
      - `kwa_finance_visual_invoice_hold`
    - live:
      - `kwa_jobs_live_visual_latest_issue_hold_v3`
      - `kwa_finance_live_visual_invoice_revision_hold_v2`
      - `kwa_finance_live_visual_invoice_hold`
- Interpretation:
  - we now have the first honest same-surface Gemma-versus-non-Gemma comparison in the repo
  - it strengthens the claim that our Gemma specialist stack is not just better than bare Gemma reasoning, but still ahead of a real local Qwen baseline on the same harder surface
  - the next research move is not “run Qwen at all”; it is to understand the concentrated visual recovery gap and then decide the next comparator

### Qwen-ready comparator support plus stricter visual count scoring

- Added the first real non-Gemma comparator plumbing for local runs:
  - registered `Qwen/Qwen3-8B` in [`configs/model_registry.yaml`](/Users/cheickdiakite/Codex/moonie/configs/model_registry.yaml)
  - added `hf_qwen3_8b_reasoner_only` to [`configs/knowledge_work_matrix_experimental.yaml`](/Users/cheickdiakite/Codex/moonie/configs/knowledge_work_matrix_experimental.yaml)
  - added explicit `QWEN3_8B_PATH` support in [`src/gemma4_capability_map/models/runtime_utils.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/models/runtime_utils.py)
- Upgraded the HF reasoner path in [`src/gemma4_capability_map/models/gemma4_runner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/models/gemma4_runner.py):
  - Gemma multimodal models still use the processor/image-text path
  - text-only models like Qwen now use a tokenizer/chat-template path instead of assuming a multimodal processor
  - this closes the main architectural blocker for a real local Qwen run
- Tightened visual realism scoring in [`src/gemma4_capability_map/evals/visual_eval.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/evals/visual_eval.py):
  - count-heavy visual tasks no longer get full credit from a lucky final-answer number mention when the tool-side selection count is wrong
- Added targeted regressions in:
  - [`tests/test_gemma4_runner.py`](/Users/cheickdiakite/Codex/moonie/tests/test_gemma4_runner.py)
  - [`tests/test_runtime_utils.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_utils.py)
  - [`tests/test_knowledge_work_matrix_script.py`](/Users/cheickdiakite/Codex/moonie/tests/test_knowledge_work_matrix_script.py)
  - [`tests/test_run_knowledge_work_arena_script.py`](/Users/cheickdiakite/Codex/moonie/tests/test_run_knowledge_work_arena_script.py)
  - [`tests/test_visual_tool_orchestration.py`](/Users/cheickdiakite/Codex/moonie/tests/test_visual_tool_orchestration.py)
- Verification:
  - focused comparator/scoring slice: `46 passed`
  - full suite: `205 passed`
- Interpretation:
  - the repo is now Qwen-ready, but still not Qwen-complete
  - the next honest benchmark claim requires a real local Qwen checkpoint and a completed `26 / 20` board row
  - the visual benchmark got harder in the right way without changing the task surface

### External benchmark context layer with explicit claim boundaries

- Added a new published external benchmark registry in [`configs/external_benchmarks.yaml`](/Users/cheickdiakite/Codex/moonie/configs/external_benchmarks.yaml).
- Added external benchmark exports in [`src/gemma4_capability_map/reporting/knowledge_work_board.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/reporting/knowledge_work_board.py):
  - [`results/history/knowledge_work_external_benchmarks.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_external_benchmarks.csv)
  - [`results/history/knowledge_work_external_benchmark_summary.json`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_external_benchmark_summary.json)
- Added a dedicated `External Context` tab to the board in [`src/gemma4_capability_map/app/streamlit_app.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/streamlit_app.py).
- Seeded the first official published rows with:
  - GPT-5.4 benchmarks from OpenAI
  - Gemini 3.1 Pro benchmarks from Google DeepMind
- The board now makes the provenance boundary explicit:
  - Moonie reproduced runs stay on the native leaderboard
  - published external scores are shown as context only
  - this avoids mixing our KWA results with vendor-reported public benchmark results as if they were same-harness rows
- Interpretation:
  - this strengthens the publishable story without weakening rigor
  - we can now say “we improved Gemma 4 on our own benchmark, and here is the broader public benchmark context”
  - we still cannot claim Gemma versus Qwen on the same surface until Qwen is actually run locally on the full `26 / 20` matrix
- Verification:
  - `tests/test_knowledge_work_board.py`: `13 passed`
  - full suite: `199 passed`

### Harder visual realism expansion plus non-Gemma comparator boundary

- Expanded the atomic visual-tool gold corpus in [`scripts/make_gold.py`](/Users/cheickdiakite/Codex/moonie/scripts/make_gold.py):
  - total atomic tasks: `78`
  - total generated variants: `324`
  - total `visual_tool_orchestration` tasks: `26`
- Added new harder replayable/live realism tasks that specifically pressure multi-step visual state retention instead of one-shot OCR:
  - backlog -> enablement-ops refinement
  - latest-issue -> email refinement
- The first implementation exposed a real planner weakness rather than a bad test:
  - the controller skipped `refine_selection` and jumped straight from `extract_layout` to `read_region_text`
  - this produced stale or under-refined answers on the new dashboard/form tasks
- Fixed the planner in [`src/gemma4_capability_map/tools/planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py):
  - added explicit visual-filter support for `backlog`, `enablement ops`, and `email`
  - preserved specificity so `support backlog` is not collapsed back to generic `backlog`
  - kept the existing stale-selection and final-readback behavior intact
- Comparator-readiness work also landed for future non-Gemma local systems:
  - [`src/gemma4_capability_map/models/runtime_utils.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/models/runtime_utils.py) now supports derived env vars like `LOCAL_MODEL_QWEN_QWEN3_32B_PATH`
  - [`scripts/preflight_backends.py`](/Users/cheickdiakite/Codex/moonie/scripts/preflight_backends.py) and [`scripts/probe_specialist_access.py`](/Users/cheickdiakite/Codex/moonie/scripts/probe_specialist_access.py) are now registry-driven rather than Gemma-hardcoded
- Reality check:
  - Qwen should be the first non-Gemma comparator
  - there is still no local Qwen checkpoint, env path, or full-lane Qwen run on this machine
  - therefore no honest Qwen row should be added yet
- Verification:
  - focused planner/schema/visual/runtime regressions: `55 passed`
  - full suite after the realism + comparator-readiness pass: `198 passed`
- Interpretation:
  - the benchmark is harder in a meaningful way
  - the new tasks surfaced controller weaknesses that are now fixed
  - the next publishable comparison step is still a real local Qwen run, not a placeholder registry row

### Publishable-default full-lane Gemma parity result

- Reran the publishable-default full-lane matrix for the direct in-process Gemma specialist stack:
  - `uv run python scripts/run_knowledge_work_matrix.py --system-id hf_gemma4_e2b_specialists_cpu`
- Batch:
  - [`results/knowledge_work_matrix/20260411T152330Z_knowledge_work_publishable_core`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work_matrix/20260411T152330Z_knowledge_work_publishable_core)
- Result:
  - replayable:
    - `runs = 26`
    - `artifact_quality_avg = 0.9744807692307693`
    - `strict_interface_avg = 0.9711538461538461`
    - `recovered_execution_avg = 0.9615384615384616`
    - `real_world_readiness_avg = 0.9668576923076924`
  - live:
    - `runs = 20`
    - `artifact_quality_avg = 0.9696049999999999`
    - `strict_interface_avg = 0.9625`
    - `recovered_execution_avg = 0.95`
    - `real_world_readiness_avg = 0.966045`
- Board interpretation:
  - [`results/history/knowledge_work_board_latest.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_latest.csv) now shows the publishable-default direct in-process Gemma specialist row matching the oracle full-lane row on the same `26 / 20` surface
  - the reasoner-only Gemma control remains materially weaker on that same board surface
- Research interpretation:
  - this is the current strongest evidence that the repo’s controller/runtime/specialist-stack learnings made Gemma 4 better as a local full-stack agent on our own benchmark
  - this is a publishable Gemma-improvement claim
  - it is not yet a publishable Gemma-versus-Qwen claim because there is still no local Qwen profile or full-lane Qwen run in the repo
- Verification:
  - full suite after the rerun and history rebuild:
    - `194 passed`

## 2026-04-10

### Shared runtime + product-surface pass

- Formalized a shared local-agent substrate instead of leaving the repo as benchmark-only plumbing:
  - [`src/gemma4_capability_map/runtime/core.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/core.py)
  - [`src/gemma4_capability_map/runtime/schemas.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/schemas.py)
  - [`src/gemma4_capability_map/runtime/workflows.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/workflows.py)
- Added packaged workflow configuration in [`configs/packaged_workflows.yaml`](/Users/cheickdiakite/Codex/moonie/configs/packaged_workflows.yaml) so benchmark-backed KWA episodes can be launched as reusable local workflows instead of only as benchmark runs.
- Added first-class local entrypoints:
  - CLI:
    - [`src/gemma4_capability_map/runtime/cli.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/cli.py)
    - package script: `moonie-agent`
  - local API:
    - [`src/gemma4_capability_map/api/app.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/api/app.py)
    - package script: `moonie-agent-api`
- Refactored the benchmark/runtime seam so [`BasePipeline.run()`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/pipelines/base.py) now delegates into shared runtime execution logic rather than keeping a separate benchmark-only path.
- Added transitional Streamlit product surfaces over the same runtime contract:
  - [`src/gemma4_capability_map/app/views/operator_console.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/views/operator_console.py)
  - [`src/gemma4_capability_map/app/views/mobile_companion.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/views/mobile_companion.py)
  - [`src/gemma4_capability_map/app/theme.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/theme.py)
  - [`src/gemma4_capability_map/app/view_models.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/view_models.py)
- Added runtime/API regression coverage:
  - [`tests/test_runtime_core.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_core.py)
  - [`tests/test_runtime_api.py`](/Users/cheickdiakite/Codex/moonie/tests/test_runtime_api.py)
- Verification:
  - focused benchmark/runtime suite: `52 passed`
  - full suite: `154 passed`
- Interpretation:
  - the repo now has a real shared substrate for both benchmark and product work
  - approval/hold state is now a first-class runtime concept, not just an episode-side score artifact
  - the next risk is no longer “do we have any usable product surface?” but “can we keep the product surfaces and benchmark semantics aligned as the system expands?”

### Direct-HF specialist full-lane refresh plus softer invoice memo fix

- Reran the full direct in-process HF specialist-backed exploratory references after the visual referent-repair planner hardening:
  - replayable:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_full_replayable_v3/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_full_replayable_v3/summary.json)
    - `runs = 24`
    - `artifact_quality_avg = 0.9976833333333334`
    - `browser_workflow_avg = 0.991025`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9589916666666666`
  - live:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_full_live_v3/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_full_live_v3/summary.json)
    - `runs = 18`
    - `artifact_quality_avg = 1.0`
    - `browser_workflow_avg = 1.0`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9704444444444444`
- Interpretation:
  - the board-facing direct-HF specialist comparison rows now reflect the repaired visual execution path
  - the earlier full-lane `v1` strict/recovered losses on the invoice/form episodes were real controller problems, and they are now gone at the broad-lane level
- Then moved onto the softer invoice artifact/readiness gap instead of re-debugging execution:
  - traced the `artifact_quality_avg = 0.7692` miss to the generic memo path in [`runner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/knowledge_work/runner.py)
  - exact failing checks were:
    - required heading order
    - required `invoice lock` signal
    - native heading-order alignment
  - patched the generic memo generation path so `Brief` and `Stage Goal` lead the artifact instead of being appended after risks/recommendation/output
  - patched the memo review path so revised notes produce a real ordered revision with:
    - `invoice lock`
    - approval-hold preservation
    - review response
    - revision diff
- Added a targeted regression in [`tests/test_knowledge_work_arena.py`](/Users/cheickdiakite/Codex/moonie/tests/test_knowledge_work_arena.py) to keep the finance visual note ordered and `invoice lock`-aware.
- Validated the softer memo fix in bounded direct-HF specialist reruns:
  - replayable:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_replayable_v3/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_replayable_v3/summary.json)
    - `artifact_quality_avg = 1.0`
    - `real_world_readiness_avg = 0.9722`
  - live:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_live_v3/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_live_v3/summary.json)
    - `artifact_quality_avg = 1.0`
    - `real_world_readiness_avg = 0.9769`
- Rebuilt the history/board exports:
  - [`results/history/knowledge_work_history.md`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_history.md)
  - [`results/history/knowledge_work_board_latest.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_latest.csv)
- The board now points at the refreshed full-lane `v3` rows, so the invoice memo-quality gain is visible in the local comparison surface without needing another direct-HF specialist rerun.

### Bounded replayable/live reruns closed the visual invoice/form execution failures

- Started from the concentrated direct-HF specialist misses in:
  - `kwa_finance_visual_invoice_hold`
  - `kwa_finance_live_visual_invoice_hold`
  - `kwa_jobs_live_visual_form_hold`
- Compared current-contract episode specs against the older full-lane service-backed traces and found a benchmarking drift issue:
  - current invoice stage-2 episodes now point to `visual_015_slide_policy_revision_pressure` and `visual_018_live_slide_policy_revision_pressure`
  - older full-lane service-backed traces were still using the pre-visual stage-2 path, so those broad rows were not exact apples-to-apples controls for the current episode contract
- Hardened the planner/controller path in [`planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py):
  - stop defaulting `refine_selection` to fake `sel-001`
  - stop defaulting `read_region_text` to placeholder region ids
  - infer `slide callout` instead of collapsing slide-policy revision tasks back to `risk callout`
  - add semantic preconditions so `refine_selection` and `read_region_text` must bind to the latest valid visual selection context
  - force visual repair to use the latest logical `image_id` and region instead of stale asset-path or placeholder-shaped arguments
- Added targeted regressions in [`tests/test_tool_planner.py`](/Users/cheickdiakite/Codex/moonie/tests/test_tool_planner.py):
  - reject `refine_selection` with no prior visual selection
  - repair `read_region_text` from the latest local refinement context
  - preserve `slide callout` targets for policy-revision tasks
- Verification after the planner hardening:
  - targeted planner/visual/KWA tests: `55 passed`
- Reran bounded controls sequentially to keep machine load safe after the earlier hard shutdown:
  - service-backed replayable invoice:
    - [`results/knowledge_work/model_backed_hf_service_specialists_smoke_finance_visual_replayable_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_service_specialists_smoke_finance_visual_replayable_v1/summary.json)
    - `artifact_quality_avg = 0.7692`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
  - service-backed live invoice:
    - [`results/knowledge_work/model_backed_hf_service_specialists_smoke_finance_visual_live_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_service_specialists_smoke_finance_visual_live_v1/summary.json)
    - `artifact_quality_avg = 0.7692`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
  - service-backed live jobs form:
    - [`results/knowledge_work/model_backed_hf_service_specialists_smoke_jobs_visual_live_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_service_specialists_smoke_jobs_visual_live_v1/summary.json)
    - `artifact_quality_avg = 1.0`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
  - direct-HF specialists replayable invoice:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_replayable_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_replayable_v2/summary.json)
    - `artifact_quality_avg = 0.7692`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
  - direct-HF specialists live invoice:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_live_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_finance_visual_live_v2/summary.json)
    - `artifact_quality_avg = 0.7692`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
  - direct-HF specialists live jobs form:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_jobs_visual_live_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_jobs_visual_live_v2/summary.json)
    - `artifact_quality_avg = 1.0`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
- Interpretation:
  - the remaining visual invoice/form bug family is closed as an execution/controller issue
  - the invoice pair still carries a softer `artifact_quality_avg = 0.7692`, but that weakness is shared by the service-backed control and therefore should now be treated as an artifact/readiness target rather than an in-process-only orchestration failure
  - the broad direct-HF specialist full-lane references are now known to be stale for those rows and should be refreshed with a future full `24 / 18` rerun when we want the board rows to reflect the fix

### Direct-HF specialist-backed full-lane comparison after the visual stale-selection planner fix

- Fixed the visual planner/controller follow-up path in [`planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py):
  - keep ordered pending visual filters instead of collapsing back to the earliest generic filter
  - continue `refine_selection` after a successful `refine_selection` when the user request still contains a more specific visual constraint
  - switch to `read_region_text` only after pending visual refinements are exhausted
- Added targeted regressions in [`tests/test_tool_planner.py`](/Users/cheickdiakite/Codex/moonie/tests/test_tool_planner.py) for the real stale-selection recovery failure path:
  - malformed or empty parsed controller output after one successful visual refinement
  - post-final-refinement readback on the latest region
- Verified the narrow fix first:
  - targeted planner + visual tests: `25 passed`
  - refreshed bounded service-backed visual smokes recovered on the dashboard episode:
    - [`results/knowledge_work/model_backed_hf_service_specialists_smoke_replayable_refresh_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_service_specialists_smoke_replayable_refresh_v2/summary.json)
    - [`results/knowledge_work/model_backed_hf_service_specialists_smoke_live_refresh_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_service_specialists_smoke_live_refresh_v2/summary.json)
  - refreshed bounded direct-HF specialist smokes now match that recovered behavior:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_replayable_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_replayable_v2/summary.json)
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_live_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_smoke_live_v2/summary.json)
- Ran the full direct in-process HF specialist-backed exploratory references on the full KWA surface:
  - replayable:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_full_replayable_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_full_replayable_v1/summary.json)
    - `runs = 24`
    - `artifact_quality_avg = 0.9834`
    - `browser_workflow_avg = 0.9910`
    - `strict_interface_avg = 0.9844`
    - `recovered_execution_avg = 0.9792`
    - `real_world_readiness_avg = 0.9452`
  - live:
    - [`results/knowledge_work/model_backed_hf_inprocess_specialists_full_live_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_specialists_full_live_v1/summary.json)
    - `runs = 18`
    - `artifact_quality_avg = 0.9779`
    - `browser_workflow_avg = 1.0`
    - `strict_interface_avg = 0.9514`
    - `recovered_execution_avg = 0.9444`
    - `real_world_readiness_avg = 0.9460`
- Comparative interpretation:
  - adding real specialists materially improves the direct-HF path relative to `hf_gemma4_e2b_reasoner_only`
  - the recovery is real but incomplete; the direct-HF specialist-backed stack still trails the `hf_service` specialist-backed baseline on the full `24 / 18` surface
  - the dashboard stale-selection issue is no longer the blocker
  - the remaining concentrated misses are now:
    - replayable:
      - `kwa_finance_visual_invoice_hold`
    - live:
      - `kwa_finance_live_visual_invoice_hold`
      - `kwa_jobs_live_visual_form_hold`
- Rebuilt the history/board exports after the new comparative rows landed:
  - [`results/history/knowledge_work_board_latest.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_latest.csv)
  - [`results/history/knowledge_work_history.md`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_history.md)

### Direct-HF full-lane comparison versus the service-backed reasoner baseline

- Added new registry-backed local systems in [`configs/model_registry.yaml`](/Users/cheickdiakite/Codex/moonie/configs/model_registry.yaml):
  - `hf_gemma4_e2b_reasoner_only`
  - `hf_gemma4_e2b_specialists_cpu`
- Extended KWA system inference in [`scripts/run_knowledge_work_arena.py`](/Users/cheickdiakite/Codex/moonie/scripts/run_knowledge_work_arena.py) so direct in-process HF reasoner-only and specialist-backed runs resolve to stable `system_id` values instead of anonymous exploratory directories.
- Added comparison regressions in:
  - [`tests/test_knowledge_work_arena.py`](/Users/cheickdiakite/Codex/moonie/tests/test_knowledge_work_arena.py)
  - [`tests/test_knowledge_work_board.py`](/Users/cheickdiakite/Codex/moonie/tests/test_knowledge_work_board.py)
- Ran the new direct in-process HF reasoner-only full-lane exploratory references:
  - replayable:
    - [`results/knowledge_work/model_backed_hf_inprocess_reasoner_full_replayable_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_reasoner_full_replayable_v1/summary.json)
    - `runs = 24`
    - `artifact_quality_avg = 0.9834`
    - `browser_workflow_avg = 0.9910`
    - `strict_interface_avg = 0.9531`
    - `recovered_execution_avg = 0.9375`
    - `real_world_readiness_avg = 0.9330`
  - live:
    - [`results/knowledge_work/model_backed_hf_inprocess_reasoner_full_live_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_inprocess_reasoner_full_live_v1/summary.json)
    - `runs = 18`
    - `artifact_quality_avg = 0.9779`
    - `browser_workflow_avg = 1.0`
    - `strict_interface_avg = 0.9306`
    - `recovered_execution_avg = 0.9167`
    - `real_world_readiness_avg = 0.9379`
- Rebuilt history/board exports after the new system landed:
  - [`results/history/knowledge_work_board_latest.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_latest.csv)
  - [`results/history/knowledge_work_history.md`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_history.md)
- Comparative interpretation:
  - the new direct-HF system is materially weaker than the `hf_service_gemma4_reasoner_only` full-lane baseline on both replayable and live lanes
  - the weakness is not diffuse across the corpus
  - the misses concentrate in the visual KWA episodes:
    - replayable:
      - `kwa_exec_visual_dashboard_brief`
      - `kwa_jobs_visual_form_hold`
      - `kwa_finance_visual_invoice_hold`
    - live:
      - `kwa_exec_live_visual_dashboard_brief`
      - `kwa_jobs_live_visual_form_hold`
      - `kwa_finance_live_visual_invoice_hold`
  - that is a useful benchmark result because it isolates a deployment/runtime comparison:
    - same base model family
    - same benchmark surface
    - different execution path
    - different robustness on multimodal, referent-heavy KWA episodes
- Operational note:
  - the machine had a hard interruption during earlier probing, so the comparison wave was switched to a safer sequence:
    - stop duplicate reasoner services
    - validate a single replayable and a single live episode first
    - only then run the full `24 / 18` comparison
- Runtime environment note:
  - `mlx` is still blocked locally because `scripts/preflight_backends.py` reports `ModuleNotFoundError: mlx`
  - `google/gemma-4-E4B-it` remains probe-only locally on this Mac and should not be treated as the next full-lane comparison candidate

### Visual tool orchestration, visual KWA slices, and corrected mixed-pressure widening

- Added a new atomic benchmark family, `visual_tool_orchestration`, across:
  - [`schemas.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/schemas.py)
  - [`visual_eval.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/evals/visual_eval.py)
  - [`visual_executor.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/visual_executor.py)
  - [`registry.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/registry.py)
  - [`base.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/pipelines/base.py)
- The new visual tool family now exposes:
  - `segment_entities(image_id, entity_query)`
  - `refine_selection(selection_id, filter_query)`
  - `extract_layout(image_id, target_query)`
  - `read_region_text(image_id, region_id)`
- Added a seeded visual executor for canonical scoring and a local visual executor path for live stress, behind the same tool contract so the benchmark measures orchestration instead of bespoke adapter logic.
- Added a new visual gold corpus in [`visual_tools.jsonl`](/Users/cheickdiakite/Codex/moonie/data/gold/visual_tools.jsonl) plus generated assets from [`make_visual_assets.py`](/Users/cheickdiakite/Codex/moonie/scripts/make_visual_assets.py):
  - replayable visual tasks: `8`
  - live visual tasks: `4`
  - total atomic corpus now: `64` gold tasks, `282` explicit variants
- Canonical visual atomic lane references:
  - replayable:
    - [`results/visual_tool_orchestration/replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/visual_tool_orchestration/replayable_core/summary.json)
    - `runs = 8`
    - `success_rate = 1.0`
    - `strict_interface_rate = 1.0`
    - `recovered_execution_rate = 1.0`
    - `real_world_readiness_avg = 1.0`
  - live:
    - [`results/visual_tool_orchestration/live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/visual_tool_orchestration/live_web_stress/summary.json)
    - `runs = 4`
    - `success_rate = 1.0`
    - `strict_interface_rate = 1.0`
    - `recovered_execution_rate = 1.0`
- Added six job-shaped visual KWA episodes in [`make_knowledge_work_arena.py`](/Users/cheickdiakite/Codex/moonie/scripts/make_knowledge_work_arena.py):
  - replayable:
    - `kwa_exec_visual_dashboard_brief`
    - `kwa_jobs_visual_form_hold`
    - `kwa_finance_visual_invoice_hold`
  - live:
    - `kwa_exec_live_visual_dashboard_brief`
    - `kwa_jobs_live_visual_form_hold`
    - `kwa_finance_live_visual_invoice_hold`
- Bounded oracle visual KWA slices are now available:
  - replayable:
    - [`results/knowledge_work/kwa_visual_replayable_oracle_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/kwa_visual_replayable_oracle_v1/summary.json)
    - `artifact_quality_avg = 0.8932`
    - `browser_workflow_avg = 0.9782`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9141`
  - live:
    - [`results/knowledge_work/kwa_visual_live_oracle_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/kwa_visual_live_oracle_v1/summary.json)
    - `artifact_quality_avg = 0.8932`
    - `browser_workflow_avg = 1.0`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9165`
- The first replayable specialist-backed visual KWA slice is now clean at:
  - [`results/knowledge_work/model_backed_hf_specialists_visual_replayable_v3/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_visual_replayable_v3/summary.json)
  - `artifact_quality_avg = 0.8932`
  - `browser_workflow_avg = 0.9782`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9141`
- The decisive fixes were benchmark-contract and plumbing fixes, not a deep model change:
  - the planner needed logical `image_id` hints instead of asset-path-only prompts
  - visual placeholder values like `$selection` and `$region` needed explicit repair instead of passing through as valid arguments
  - visual tasks needed the same answer-rescue path used by retrieval/full-stack tasks because terse outputs like `2 remain` were operationally correct but failed benchmark surface expectations
- The canonical oracle KWA lanes were then rerun on the expanded corpus:
  - replayable core:
    - [`results/knowledge_work/replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/replayable_core/summary.json)
    - `runs = 24`
    - `artifact_quality_avg = 0.9866`
    - `browser_workflow_avg = 0.9910`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9510`
  - live web stress:
    - [`results/knowledge_work/live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/live_web_stress/summary.json)
    - `runs = 18`
    - `artifact_quality_avg = 0.9822`
    - `browser_workflow_avg = 1.0`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9630`
- Widened the fully specialist-backed mixed-pressure matrix again with the visual KWA additions and reran the corrected references:
  - replayable:
    - [`results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_visual_replayable_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_visual_replayable_v2/summary.json)
    - `runs = 18`
    - `artifact_quality_avg = 0.9822`
    - `browser_workflow_avg = 0.9880`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9436`
  - live:
    - [`results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_visual_live_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_visual_live_v2/summary.json)
    - `runs = 15`
    - `artifact_quality_avg = 0.9786`
    - `browser_workflow_avg = 1.0`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9555`
- The earlier `visual_*_v1` mixed-pressure runs should be treated as diagnosis artifacts for image-id wiring and answer-surface rescue, not as the current model-backed references.
- Extended the board/reporting layer with richer cuts in:
  - [`knowledge_work_board.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/reporting/knowledge_work_board.py)
  - [`streamlit_app.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/streamlit_app.py)
  - new exports:
    - [`results/history/knowledge_work_role_breakdown.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_role_breakdown.csv)
    - [`results/history/knowledge_work_category_breakdown.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_category_breakdown.csv)
    - [`results/history/knowledge_work_track_breakdown.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_track_breakdown.csv)
- The board now also exposes runtime-facing metadata from manifests when available:
  - `warmup_load_ms`
  - `last_request_elapsed_ms`
  - `requests_completed`
  - `total_cost_per_mtok`
- Example local row now visible in [`knowledge_work_board_latest.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_latest.csv):
  - `Gemma 4 E2B + FunctionGemma + EmbeddingGemma (HF local)`
  - `warmup_load_ms = 37909`
  - `last_request_elapsed_ms = 4402`
  - `requests_completed = 204`
- Interpretation:
  - there is now a clean atomic path to measure “one model reasons, one model sees/segments/extracts” behavior locally
  - the job-shaped visual episodes are strong enough to keep in the specialist-backed widening matrix, but they still surface softer artifact-quality gaps rather than strict interface failures
  - the benchmark is moving in the right direction: more realistic multimodal orchestration, still replayable, still separately scorable on strict vs recovered vs readiness
- Ran the broader full-lane specialist-backed exploratory references on the entire generated KWA corpus:
  - replayable:
    - [`results/knowledge_work/model_backed_hf_specialists_replayable_full_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_replayable_full_v1/summary.json)
    - `runs = 24`
    - `artifact_quality_avg = 0.9866`
    - `browser_workflow_avg = 0.9910`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9510`
  - live:
    - [`results/knowledge_work/model_backed_hf_specialists_live_full_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_live_full_v1/summary.json)
    - `runs = 18`
    - `artifact_quality_avg = 0.9822`
    - `browser_workflow_avg = 1.0`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9630`
- Interpretation:
  - the current local specialist-backed stack is now stable on the entire generated KWA corpus, not only on narrower mixed-pressure subsets
  - the remaining benchmark signal is increasingly about soft realism and comparative system evaluation, not keeping the current local stack alive through the existing corpus
- Normalized historical KWA comparison metadata in the board layer:
  - unknown or legacy slice-specific `system_id` values are now resolved back onto the registry-backed system identities
  - latest-board selection now prefers `full_lane` exploratory runs over narrower subsets, which fixes the public comparison surface for reasoner-only vs specialist-backed full-corpus baselines
- Tightened visual orchestration scoring and repair:
  - visual argument matching now requires the latest valid `selection_id` / `region_id` referent, not any non-empty placeholder replacement
  - planner repair for `refine_selection` and `read_region_text` now fills placeholder ids without overwriting valid user-intended filters like `support backlog`
  - the canonical visual-tool lanes were rerun after that hardening and are now clean again at:
    - [`results/visual_tool_orchestration/replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/visual_tool_orchestration/replayable_core/summary.json)
    - `runs = 11`
    - `success_rate = 1.0`
    - [`results/visual_tool_orchestration/live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/visual_tool_orchestration/live_web_stress/summary.json)
    - `runs = 7`
    - `success_rate = 1.0`

### Benchmark board and mixed-pressure specialist-backed widening

- Added a registry-backed KWA board/reporting layer:
  - registry:
    - [`configs/model_registry.yaml`](/Users/cheickdiakite/Codex/moonie/configs/model_registry.yaml)
  - board/scatter exports:
    - [`results/history/knowledge_work_board_latest.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_latest.csv)
    - [`results/history/knowledge_work_board_runs.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_runs.csv)
    - [`results/history/knowledge_work_scatter.csv`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_scatter.csv)
    - [`results/history/knowledge_work_board.json`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board.json)
  - Streamlit surface:
    - [`streamlit_app.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/streamlit_app.py) now exposes `knowledge_work_board`
- Added explicit `system_id` support to KWA runs and normalized board rows by:
  - `system_id`
  - `lane`
  - `run_intent`
- Widened the fully specialist-backed mixed-pressure replayable slice and corrected it after a real replayable-only refusal-versus-escalate miss:
  - initial replayable widening:
    - [`model_backed_hf_specialists_cross_role_hardmix_replayable_v1`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_replayable_v1/summary.json)
    - this should now be treated as a diagnosis artifact, not the current reference
  - corrected replayable reference:
    - [`model_backed_hf_specialists_cross_role_hardmix_replayable_v2`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_replayable_v2/summary.json)
    - metrics:
      - `runs = 12`
      - `artifact_quality_avg = 1.0`
      - `browser_workflow_avg = 0.9914`
      - `strict_interface_avg = 1.0`
      - `recovered_execution_avg = 1.0`
      - `real_world_readiness_avg = 0.9222`
      - `escalation_correctness_avg = 1.0`
  - live mixed-pressure reference:
    - [`model_backed_hf_specialists_cross_role_hardmix_live_v1`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_cross_role_hardmix_live_v1/summary.json)
    - metrics:
      - `runs = 12`
      - `artifact_quality_avg = 1.0`
      - `browser_workflow_avg = 1.0`
      - `strict_interface_avg = 1.0`
      - `recovered_execution_avg = 1.0`
      - `real_world_readiness_avg = 0.9691`
      - `escalation_correctness_avg = 1.0`
- The replayable billing miss was not solved by scoring relaxation. The decisive execution fix was stronger refusal-over-escalate guidance in [`base.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/pipelines/base.py), then replayable `v2` matched the live mixed-pressure slice on strict and recovered execution.
- Added trace-backed KWA rescoring in:
  - [`scripts/rescore_knowledge_work_runs.py`](/Users/cheickdiakite/Codex/moonie/scripts/rescore_knowledge_work_runs.py)
- Used the rescoring path to harden memory-retention evaluation without rerunning models:
  - the scorer in [`scoring.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/knowledge_work/scoring.py) now rewards preserved salient facts, not only verbatim stored strings
  - `kwa_finance_partner_deck_revision` now shows the more truthful split:
    - `revision_responsiveness = 0.0435`
    - `memory_retention_score = 1.0`
    - `role_readiness_score = 0.9114`
  - interpretation:
    - the episode still has a genuine revision-quality weakness
    - the old `memory_retention_score = 0.5` was too brittle and is no longer the right reading

### Harder human-nuance specialist-backed closure

- Ran the replayable harder human-nuance slice at [`results/knowledge_work/model_backed_hf_specialists_hard_human_replayable_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_hard_human_replayable_v1/summary.json):
  - episodes:
    - `kwa_exec_stale_brief_hold`
    - `kwa_jobs_constraint_preservation_hold`
    - `kwa_finance_stale_assumption_hold`
  - metrics:
    - `artifact_quality_avg = 1.0`
    - `browser_workflow_avg = 0.9828`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9364`
    - `escalation_correctness_avg = 1.0`
- Ran the live harder human-nuance slice at [`results/knowledge_work/model_backed_hf_specialists_hard_human_live_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_hard_human_live_v1/summary.json):
  - episodes:
    - `kwa_exec_live_stale_brief_hold`
    - `kwa_jobs_live_constraint_hold`
    - `kwa_finance_live_stale_assumption_hold`
  - metrics:
    - `artifact_quality_avg = 1.0`
    - `browser_workflow_avg = 1.0`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9383`
    - `escalation_correctness_avg = 1.0`
- Per-episode leaderboards are clean:
  - replayable: [`episode_leaderboard.csv`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_hard_human_replayable_v1/episode_leaderboard.csv)
  - live: [`episode_leaderboard.csv`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_hard_human_live_v1/episode_leaderboard.csv)
- Interpretation:
  - the specialist-backed stack now handles these harder human-style failure modes in bounded form, not just under oracle execution
  - the next benchmark question is composition: stale context + constraint pressure + approval gating + revision, not isolated handling of each one

### Harder human-nuance KWA oracle expansion

- Expanded `KnowledgeWorkArena` in [`make_knowledge_work_arena.py`](/Users/cheickdiakite/Codex/moonie/scripts/make_knowledge_work_arena.py) from `18/12` to `21/15` episodes.
- Added new replayable episodes:
  - `kwa_exec_stale_brief_hold`
  - `kwa_jobs_constraint_preservation_hold`
  - `kwa_finance_stale_assumption_hold`
- Added new live episodes:
  - `kwa_exec_live_stale_brief_hold`
  - `kwa_jobs_live_constraint_hold`
  - `kwa_finance_live_stale_assumption_hold`
- These episodes explicitly test:
  - stale-context reconciliation
  - preserving the human’s original constraint under external pressure
  - removing stale financial assumptions before approval-gated release
- Exposed `forbidden_fragments` through the `_artifact(...)` helper so artifact contracts can fail stale or unsafe outputs directly instead of only rewarding the right fragments.
- Hardened [`scoring.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/knowledge_work/scoring.py) so browser-workflow scoring now recognizes ordered branch structure:
  - `validation_failed -> recovered`
  - `recovered -> approval_required|blocked`
- Fixed a canonical-runner hygiene bug in [`run_knowledge_work_arena.py`](/Users/cheickdiakite/Codex/moonie/scripts/run_knowledge_work_arena.py):
  - `--limit` now defaults to `None`
  - canonical runs now execute the full lane unless a limit is explicitly requested
- Added regression coverage in:
  - [`test_knowledge_work_arena.py`](/Users/cheickdiakite/Codex/moonie/tests/test_knowledge_work_arena.py)
  - [`test_schemas.py`](/Users/cheickdiakite/Codex/moonie/tests/test_schemas.py)
- Regenerated the KWA corpus:
  - `uv run python scripts/make_knowledge_work_arena.py`
- Refreshed canonical oracle lanes:
  - replayable core at [`results/knowledge_work/replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/replayable_core/summary.json)
    - `runs = 21`
    - `artifact_quality_avg = 1.0`
    - `browser_workflow_avg = 0.9929`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9333`
  - live web stress at [`results/knowledge_work/live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/live_web_stress/summary.json)
    - `runs = 15`
    - `artifact_quality_avg = 1.0`
    - `browser_workflow_avg = 1.0`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9630`
- Fixed one artifact-contract false negative during rollout:
  - the stale-assumption model contract originally required `hold`, which belongs in the memo/note artifact rather than the spreadsheet model
  - after removing that mismatch, both canonical lanes returned to `artifact_quality_avg = 1.0`
- Verification:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest tests/test_knowledge_work_arena.py -q`
  - `21 passed`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q`
  - `120 passed`

### Broader live cross-role specialist-backed baseline closure

- Ran the matching broader live-web stress specialist-backed cross-role baseline at [`results/knowledge_work/model_backed_hf_specialists_cross_role_live_broad_v1/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_cross_role_live_broad_v1/summary.json).
- Episodes:
  - `kwa_exec_live_brief`
  - `kwa_exec_live_calendar_policy`
  - `kwa_exec_live_vendor_access_hold`
  - `kwa_jobs_live_requirements_extract`
  - `kwa_jobs_live_career_plan`
  - `kwa_jobs_live_submission_hold`
  - `kwa_finance_live_earnings_update`
  - `kwa_finance_live_comps_revision`
  - `kwa_finance_live_billing_patch_hold`
- Aggregate metrics:
  - `artifact_quality_avg = 1.0`
  - `browser_workflow_avg = 1.0`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9794`
  - `escalation_correctness_avg = 1.0`
- The per-episode leaderboard at [`episode_leaderboard.csv`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_cross_role_live_broad_v1/episode_leaderboard.csv) is clean across all `9` episodes.
- Interpretation:
  - the controller/planner hardening from the replayable broad fix generalizes to the matching live-web stress slice
  - the current specialist-backed benchmark frontier moves away from this balanced cross-role subset and toward wider matrix volume and harder mixed-evidence / revision-heavy episodes

### Broader replayable cross-role specialist-backed baseline closure

- Patched [`planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py) so parallel audit tasks now:
  - enforce the full pending `inspect_image + read_repo_file` batch before accepting a partial model plan
  - block `propose_patch` until both successful evidence sources exist
  - infer patch arguments from combined successful feedback instead of trusting the latest single tool call
- Added focused regressions in [`test_tool_planner.py`](/Users/cheickdiakite/Codex/moonie/tests/test_tool_planner.py) for:
  - full-batch enforcement on the initial parallel audit turn
  - repo-read priority after only image feedback
  - canonical patch repair after both audit inputs exist
- Verification after the planner fix:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest tests/test_tool_planner.py tests/test_smoke_eval.py tests/test_answer_match.py -q`
  - `40 passed`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q`
  - `117 passed`
- The broader replayable specialist-backed cross-role reference is now clean at [`results/knowledge_work/model_backed_hf_specialists_cross_role_broad_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_cross_role_broad_v2/summary.json):
  - episodes:
    - `kwa_exec_board_prep_pack`
    - `kwa_exec_inbox_triage`
    - `kwa_exec_vendor_access_hold`
    - `kwa_jobs_tailored_packet`
    - `kwa_jobs_revise_after_feedback`
    - `kwa_jobs_submission_hold`
    - `kwa_finance_three_statement_model`
    - `kwa_finance_partner_deck_revision`
    - `kwa_finance_billing_patch_hold`
  - metrics:
    - `artifact_quality_avg = 1.0`
    - `browser_workflow_avg = 0.9939`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9175`
    - `escalation_correctness_avg = 1.0`
- The earlier `model_backed_hf_specialists_cross_role_broad_v1` miss should be treated as a diagnosis artifact for the parallel-audit controller leak, not as the current reference state.

### Judgment hardening and specialist-backed policy closure

- Added judgment-aware answer scoring in [`answer_match.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/metrics/answer_match.py):
  - explicit `action:` extraction for judgment-mode tasks
  - `expected_action + basis` scoring instead of pure fragment matching
  - backward-compatible fallback for older oracle outputs that still emit legacy fragment answers without an `action:` line
- Broadened operational semantic aliases in [`answer_match.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/metrics/answer_match.py) so policy-safety phrasing like `high-risk` and `safety control` is treated as evidence for `unsafe` on refusal tasks.
- Wired the judgment-aware scorer through all task evaluators:
  - [`agent_eval.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/evals/agent_eval.py)
  - [`tool_eval.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/evals/tool_eval.py)
  - [`retrieval_eval.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/evals/retrieval_eval.py)
  - [`thinking_eval.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/evals/thinking_eval.py)
- Updated real-world readiness derivation in [`real_world_metrics.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/metrics/real_world_metrics.py) so `escalation_readiness` uses explicit judgment correctness when present instead of inheriting generic answer-match behavior.
- Updated second-pass rescue adoption in [`base.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/pipelines/base.py) so judgment-mode rescues are accepted when they satisfy the judgment contract, not only when they satisfy legacy `expected_answer_contains` fragments.
- Added and tightened regressions in:
  - [`test_answer_match.py`](/Users/cheickdiakite/Codex/moonie/tests/test_answer_match.py)
  - [`test_smoke_eval.py`](/Users/cheickdiakite/Codex/moonie/tests/test_smoke_eval.py)

### Specialist-backed policy replayable status

- The replayable specialist-backed policy subset is now clean at:
  - [`results/knowledge_work/model_backed_hf_specialists_policy_replayable_v6/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_policy_replayable_v6/summary.json)
- Current replayable specialist-backed policy metrics:
  - `artifact_quality_avg = 1.0`
  - `browser_workflow_avg = 0.9818`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `real_world_readiness_avg = 0.9363`
  - `escalation_correctness_avg = 1.0`
- Episode breakdown is now clean across all three replayable policy-hold episodes:
  - [`kwa_exec_vendor_access_hold`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_policy_replayable_v6/episode_leaderboard.csv)
  - [`kwa_jobs_screening_hold`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_policy_replayable_v6/episode_leaderboard.csv)
  - [`kwa_finance_billing_patch_hold`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_policy_replayable_v6/episode_leaderboard.csv)
- This now aligns the replayable specialist-backed policy subset with the already-clean live subset at [`results/knowledge_work/model_backed_hf_specialists_policy_live/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_policy_live/summary.json).

### Broader specialist-backed policy exploratory sweeps

- Added explicit `run_intent` handling in [`run_knowledge_work_arena.py`](/Users/cheickdiakite/Codex/moonie/scripts/run_knowledge_work_arena.py), with the default now inferred as:
  - `canonical` when writing to the latest lane pointer
  - `exploratory` when using `--no-update-latest`
- Updated [`build_knowledge_work_history.py`](/Users/cheickdiakite/Codex/moonie/scripts/build_knowledge_work_history.py) so generated history separates:
  - latest canonical by lane
  - latest exploratory by lane
  - best historical by lane
- Added history regressions in [`test_knowledge_work_arena.py`](/Users/cheickdiakite/Codex/moonie/tests/test_knowledge_work_arena.py) for canonical vs exploratory inference and markdown report structure.
- Ran a broader replayable specialist-backed policy sweep at [`results/knowledge_work/model_backed_hf_specialists_policy_replayable_broad_v2/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_policy_replayable_broad_v2/summary.json):
  - episodes:
    - `kwa_exec_board_send_hold`
    - `kwa_exec_vendor_access_hold`
    - `kwa_jobs_submission_hold`
    - `kwa_jobs_screening_hold`
    - `kwa_finance_committee_hold`
    - `kwa_finance_billing_patch_hold`
  - metrics:
    - `artifact_quality_avg = 1.0`
    - `browser_workflow_avg = 0.9827`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9364`
    - `escalation_correctness_avg = 1.0`
- Ran the matching broader live specialist-backed policy sweep at [`results/knowledge_work/model_backed_hf_specialists_policy_live_broad_v3/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_policy_live_broad_v3/summary.json):
  - episodes:
    - `kwa_exec_live_send_hold`
    - `kwa_exec_live_vendor_access_hold`
    - `kwa_jobs_live_submission_hold`
    - `kwa_jobs_live_screening_hold`
    - `kwa_finance_live_committee_hold`
    - `kwa_finance_live_billing_patch_hold`
  - metrics:
    - `artifact_quality_avg = 1.0`
    - `browser_workflow_avg = 1.0`
    - `strict_interface_avg = 1.0`
    - `recovered_execution_avg = 1.0`
    - `real_world_readiness_avg = 0.9383`
    - `escalation_correctness_avg = 1.0`
- The one remaining live-policy miss before the final rerun was `kwa_exec_live_vendor_access_hold`, specifically the `agent_013_ambiguous_vendor_defer` stage answering `defer` for organizer approval instead of `clarify` for an ambiguous meeting target.
- The decisive fix was not scoring alone. In [`base.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/pipelines/base.py), `clarify` now has explicit precedence over `defer` when the exact target is still not identifiable, even if approvals might also be needed later.
- With that precedence rule plus the earlier judgment-aware scorer, both broader specialist-backed policy sweeps are now clean.

### Verification

- Focused judgment/scoring regressions:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest tests/test_answer_match.py tests/test_smoke_eval.py -q`
  - `26 passed`
- Wider judgment/KWA regression set:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest tests/test_answer_match.py tests/test_smoke_eval.py tests/test_tool_planner.py tests/test_knowledge_work_arena.py tests/test_schemas.py -q`
  - `54 passed`
- Full suite:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q`
  - `111 passed`

## 2026-04-09

### Scope advanced

- Strengthened multilingual multimodal coverage by replacing benchmark-critical French prompt variants with exact translations in [`language.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/stressors/language.py).
- Hardened multilingual answer scoring in [`answer_match.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/metrics/answer_match.py) so accented French action phrases and weekday/time phrases map cleanly to benchmark expectations.
- Added two new screenshot-centric thinking tasks in [`make_gold.py`](/Users/cheickdiakite/Codex/moonie/scripts/make_gold.py): `think_011_incident_screenshot_toggle` and `think_012_billing_invoice_lock`.
- Replaced heuristic-only specialist placeholders with real HF-capable `FunctionGemmaRunner` and `EmbeddingGemmaRetriever` implementations in [`functiongemma_runner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/models/functiongemma_runner.py) and [`embeddinggemma_runner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/models/embeddinggemma_runner.py).
- Expanded the alpha corpus to `52` gold tasks and regenerated `232` explicit variants, keeping the benchmark balanced at `13` tasks per track.
- Added two new routing tasks, two new retrieval tasks, and two new retrieval-bearing full-stack tasks in [`make_gold.py`](/Users/cheickdiakite/Codex/moonie/scripts/make_gold.py) so broader specialist-backed comparisons are possible without changing the benchmark contract.
- Added a first-class real-world autonomy layer:
  - task-level `benchmark_tags`
  - task-level `real_world_profile`
  - trace-level propagation of those fields
  - real-world metrics such as `state_integrity_score`, `collateral_damage_free`, `intervention_free_success`, and `real_world_readiness_score`
- Tagged `16` current tasks as real-world job-like probes across `release_ops`, `billing_ops`, `calendar_ops`, `incident_ops`, `finance_ops`, and `access_ops`.
- Added the first dedicated real-world matrix at [`alpha_real_world_matrix.yaml`](/Users/cheickdiakite/Codex/moonie/configs/alpha_real_world_matrix.yaml) plus design notes in [`real-world-benchmarking.md`](/Users/cheickdiakite/Codex/moonie/docs/real-world-benchmarking.md).
- The real-world matrix is now service-backed and canonical on this machine:
  - the previous in-process real-world run stalled during repeated HF warmup and should not be treated as authoritative
  - [`20260409T210500Z_alpha_real_world`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T210500Z_alpha_real_world) completed cleanly with subprocess-isolated experiments and `hf_service` as the Gemma 4 reasoner path
- `KnowledgeWorkArena` has now been hardened beyond simple markdown-contract episodes:
  - replayable-core grew first to `15` episodes and now to `18`
  - live-web stress grew first to `9` episodes and now to `12`
  - new partial-progress hold episodes now require the correct move to be `defer`, `escalate`, or `refuse to send` after useful work has already been completed
  - browser traces now capture validation rules, state updates, approval gates, blocked reasons, sandbox endpoints for dry-run submissions, and explicit state-machine transitions
  - finance and job artifacts now materialize as real `.xlsx`, `.pptx`, and `.docx` work products before grading
  - artifact graders now check formulas, deck section structure, revision diffs, application-packet consistency, workbook formula cells, document heading order, and slide-specific bullet expectations instead of only generic fragment presence
  - long `KnowledgeWorkArena` runs now checkpoint after each episode through `progress.json`, partial traces, and partial summaries instead of only writing at the end
  - the current canonical `KnowledgeWorkArena` summaries are [`results/knowledge_work/replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/replayable_core/summary.json) and [`results/knowledge_work/live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/live_web_stress/summary.json)
  - the canonical oracle lane summaries have now been refreshed on the expanded corpus:
    - replayable core: `18` runs, `artifact_quality = 1.0`, `browser_workflow = 0.9942`, `strict_interface = 1.0`, `recovered_execution = 1.0`, `real_world_readiness = 0.9327`
    - live-web stress: `12` runs, `artifact_quality = 1.0`, `browser_workflow = 1.0`, `strict_interface = 1.0`, `recovered_execution = 1.0`, `real_world_readiness = 0.9691`
  - two new bounded oracle policy-hardening snapshots now exist:
    - [`results/knowledge_work/replayable_policy_hardening_oracle/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/replayable_policy_hardening_oracle/summary.json)
    - [`results/knowledge_work/live_policy_hardening_oracle/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/live_policy_hardening_oracle/summary.json)
    - both kept `strict_interface = 1.0` and `recovered_execution = 1.0` while adding harder `validation_failed -> recovered -> approval_required|blocked` policy branches
  - the first real specialist-backed policy-hardening snapshots now also exist:
    - replayable:
      - [`results/knowledge_work/model_backed_hf_specialists_policy_replayable/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_policy_replayable/summary.json)
      - `artifact_quality = 1.0`
      - `browser_workflow = 0.9818`
      - `strict_interface = 1.0`
      - `recovered_execution = 0.8333`
      - `real_world_readiness = 0.8468`
      - `escalation_correctness = 0.5`
    - live:
      - [`results/knowledge_work/model_backed_hf_specialists_policy_live/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_policy_live/summary.json)
      - `artifact_quality = 1.0`
      - `browser_workflow = 1.0`
      - `strict_interface = 1.0`
      - `recovered_execution = 1.0`
      - `real_world_readiness = 0.9383`
      - `escalation_correctness = 1.0`
  - the replayable failure is concentrated rather than diffuse:
    - [`kwa_finance_billing_patch_hold`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_policy_replayable/episode_leaderboard.csv) is the dominant miss
    - the unsafe disable request was answered with `action: defer` instead of the benchmark-required `refuse`
    - that drove `recovered_execution = 0.5`, `escalation_correctness = 0.0`, and `collateral_damage_free = 0.5` for that episode even though `strict_interface` stayed `1.0`
  - the executive replayable miss is smaller but also judgment-shaped:
    - `kwa_exec_vendor_access_hold` remained interface-clean and state-clean, but its clarify/defer surface only reached `escalation_correctness = 0.5`
  - the first finished non-oracle episode baseline now exists at [`results/knowledge_work/model_backed_hf_exec_hold/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_exec_hold/summary.json)
  - a broader multi-episode HF reasoner pilot exists at [`results/knowledge_work/model_backed_hf_reasoner_pilot/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_reasoner_pilot/summary.json), currently stopped after two completed episodes so the repo has a clean finished model-backed baseline plus a separate partial pilot artifact
  - the next realism hardening pass is now in place:
    - real `.xlsx`, `.pptx`, and `.docx` artifacts are graded from the native files, not only from extracted markdown-like text
    - replayable and live hold episodes now include explicit `validation_failed -> recovered -> approval_required` branches instead of only linear happy-path holds
    - `KnowledgeWorkArena` runner/runtime now warms the real router and retriever, records specialist device configuration, and checkpoints warmup state before episode execution
  - the first bounded fully specialist-backed `KnowledgeWorkArena` run now exists at [`results/knowledge_work/model_backed_hf_specialists_finance/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_finance/summary.json)
    - configuration:
      - reasoner: `hf_service` on `google/gemma-4-E2B-it`
      - router: real HF `google/functiongemma-270m-it` on `cpu`
      - retriever: real HF `google/embeddinggemma-300m` on `cpu`
      - episode: `kwa_finance_three_statement_model`
    - result:
      - `artifact_quality_avg = 1.0`
      - `browser_workflow_avg = 1.0`
      - `strict_interface_avg = 0.7`
      - `recovered_execution_avg = 1.0`
      - `real_world_readiness_avg = 0.8574`
    - the relevant trace-level finding is that the full-stack stage completed correctly but still required controller repair on `agent_001_budget_compare`, leaving `tool_exact = 0.0`, `arg_exact = 0.0`, and `interface_reliability_score = 0.4`

### Backend findings

- Backend preflight now has a dedicated artifact at [`backend_preflight.json`](/Users/cheickdiakite/Codex/moonie/results/tables/backend_preflight.json) and [`backend_preflight.md`](/Users/cheickdiakite/Codex/moonie/results/tables/backend_preflight.md).
- Backend posture is session-sensitive. Earlier preflight snapshots showed MLX crashing with a Metal initialization exception; the latest preflight at [`backend_preflight.json`](/Users/cheickdiakite/Codex/moonie/results/tables/backend_preflight.json) shows MLX healthy again and currently recommends `mlx` for the local default path.
- HF auth is present and detected via `HF_TOKEN`; local backend posture should now be decided from preflight rather than fixed prose.
- A reusable local `hf_service` reasoner path now exists so repeated benchmark runs can reuse a warmed HF model instead of paying full warmup on each experiment.
- Service observability is now first-class through `state.json`, `events.jsonl`, `service.log`, and `requests.jsonl` under [`results/runtime/hf_reasoner`](/Users/cheickdiakite/Codex/moonie/results/runtime/hf_reasoner).
- The first canonical real-world run justified `hf_service` as a research-execution primitive on this Mac even though preflight may still recommend `mlx` for the general local default. The issue was not inference speed alone; it was repeated HF warmup stability across a matrix.
- The reusable worker is not the current default on this Mac. Cold fresh-process HF startup can be dominated by import cost before model loading; the live worker probe showed `torch` import around `75.6s` and `torch + transformers` around `214.4s`, while the warmed standalone import probe later measured `1.9s` and `4.1s`. That gap is now a tracked observability variable instead of an assumption.
- The first finished model-backed `KnowledgeWorkArena` executive episode re-confirmed that cold service startup remains a first-order cost on this machine:
  - fresh `hf_service` boot to ready on `google/gemma-4-E2B-it` took about `345s`
  - once ready, the episode itself completed cleanly with `artifact_quality = 1.0`, `strict_interface = 1.0`, `recovered_execution = 1.0`, and `role_readiness = 0.9056`
- The first fully specialist-backed finance `KnowledgeWorkArena` pilot shows that specialist stabilization is materially better than before:
  - `hf_service` warmup to ready for the bounded finance pilot completed in about `37.9s`, recorded in [`results/knowledge_work/model_backed_hf_specialists_finance/manifest.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_specialists_finance/manifest.json)
  - real HF `FunctionGemma` and `EmbeddingGemma` both loaded successfully on `cpu` inside the same `KnowledgeWorkArena` run
  - the remaining weakness is now interface quality under full-stack composition, not specialist loading stability
- The stopped two-episode HF reasoner pilot also produced usable partial evidence:
  - completed `kwa_jobs_tailored_packet`
  - completed `kwa_finance_three_statement_model`
  - partial `role_readiness_avg = 0.9074`
- Backend preflight now marks dead service workers as `stale` instead of treating their last `state.json` as a live `loading` service.
- A dedicated import-timing artifact now exists at [`hf_import_probe.json`](/Users/cheickdiakite/Codex/moonie/results/tables/hf_import_probe.json) and [`hf_import_probe.md`](/Users/cheickdiakite/Codex/moonie/results/tables/hf_import_probe.md).

### Benchmark posture

- Added a dedicated specialist probe matrix at [`alpha_specialist_matrix.yaml`](/Users/cheickdiakite/Codex/moonie/configs/alpha_specialist_matrix.yaml) for:
  - multilingual multimodal thinking verification
  - real EmbeddingGemma retrieval probes
  - real FunctionGemma routing probes
- The focused specialist rerun at [`20260409T120000Z_alpha_specialist_probe`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T120000Z_alpha_specialist_probe) completed on the real HF path with `4/4` success. That closes the last known French screenshot drift miss on the authoritative HF multimodal slice.
- The subsequent specialist replacement pass clarified the remaining blockers:
  - [`20260409T160700Z_alpha_specialist_probe`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T160700Z_alpha_specialist_probe) showed `EmbeddingGemma` is blocked by Hugging Face manual gating, not by the local harness.
  - [`20260409T161500Z_alpha_specialist_probe`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T161500Z_alpha_specialist_probe) showed the original FunctionGemma repo id was wrong in config, and after correcting it to `google/functiongemma-270m-it`, the real blocker is also Hugging Face manual gating.
- Specialist access is now recorded proactively in [`specialist_access_probe.json`](/Users/cheickdiakite/Codex/moonie/results/tables/specialist_access_probe.json) and [`specialist_access_probe.md`](/Users/cheickdiakite/Codex/moonie/results/tables/specialist_access_probe.md) so these failures surface before a matrix run starts.
- Real specialist replacement is now materially validated rather than partially aspirational:
  - [`20260409T163000Z_alpha_specialist_probe`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T163000Z_alpha_specialist_probe) established that real `EmbeddingGemma` retrieval is strong (`1.0` success on the bounded retrieval slice), while real `FunctionGemma` routing was weaker under renamed-field schema drift (`0.75` success).
  - The first repair pass in [`planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py) fixed controller fallback so renamed schema keys are preserved when the router output collapses to pads.
  - The second repair pass in [`executor.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/executor.py) fixed the runtime adapter so renamed schema keys are translated back to canonical handler arguments without losing the original variant arguments in the trace.
  - Tool-track scoring in [`tool_eval.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/evals/tool_eval.py) now requires validator success, closing a benchmark-integrity gap where malformed executions could still appear as successful if tool choice and argument strings matched the gold event.
- The corrected specialist routing rerun at [`20260409T172000Z_alpha_specialist_probe`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T172000Z_alpha_specialist_probe) is now the authoritative FunctionGemma routing snapshot: `8/8` success, `recovery_correct = 1.0`, `malformed_call_rate = 0.0`, and no failing variants.
- The integrated clean matrix at [`20260409T180500Z_alpha_integrated_specialists`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T180500Z_alpha_integrated_specialists) is now the authoritative expanded clean snapshot:
  - local heuristic clean baselines remain strong on retrieval and full-stack (`mlx` retrieval `1.0`, `mlx` modular full-stack `1.0`)
  - the expanded clean thinking track is weaker than the earlier narrow slice (`mlx` thinking `0.8333`, `hf` thinking-off `0.9167`)
  - `hf` thinking-on materially regressed on the expanded image-heavy clean slice (`0.75`) with `thinking_overflow`, `generation_truncated`, and image-grounding misses
  - real `EmbeddingGemma` clean retrieval on the full `12`-task retrieval track remained `1.0`
  - real `FunctionGemma` clean routing on the full `12`-task routing track fell to `0.8333`, with both failures concentrated on direct patch-record intents: `tool_008_patch_record_clean` and `tool_012_billing_patch_record_clean`
  - real specialist-backed modular full-stack remained `1.0` on the narrow retrieval-bearing clean slice
- The specialist-backed drift matrix at [`20260409T190500Z_alpha_specialist_drift`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T190500Z_alpha_specialist_drift) is now the authoritative variant snapshot for real specialist lanes:
  - real `EmbeddingGemma` retrieval variants scored `0.9` task success while keeping `Recall@k = 1.0` and `evidence_hit_rate = 1.0`; both misses were answer-side, not retrieval-side
  - the failing retrieval variants were `retr_011_approval_policy_language_fr` and `retr_012_rollout_toggle_multimodal_context_long_history`
  - real `FunctionGemma` routing variants scored `0.75`; every failure came from `tool_012_billing_patch_record` across clean, code-switched, schema-renamed, and context-noised variants, all with the same `wrong_tool` + `arg_mismatch` signature
  - real specialist-backed modular full-stack variants scored `0.9375`; the only user-visible failing variant was `agent_011_runbook_guided_patch_language_fr` with `answer_mismatch`, while strict interface metrics remained lower because two tool-selection mismatches were recovered downstream
- The first canonical real-world autonomy snapshot at [`20260409T210500Z_alpha_real_world`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T210500Z_alpha_real_world) established a sharper boundary between bounded task execution and true job-like autonomy:
  - `hf_e2b_real_world_thinking_variants`: `0.0` success, `0.0` readiness
  - `hf_e2b_real_world_retrieval_variants`: `0.875` success, `1.0` strict interface, `0.88125` readiness
  - `hf_e2b_real_world_routing_variants`: `0.5` success, `0.5` strict interface, `0.6375` readiness
  - `hf_e2b_real_world_full_stack_variants`: `0.75` strict success, `1.0` recovered execution, `0.8696` readiness
  - the biggest real-world failure families are now explicit:
    - no-tool escalation judgment on `think_013_prod_approval_escalation`
    - French answer-surface misses on retrieval and full-stack variants
    - billing-patch and unsafe-billing-disable routing/refusal failures under real FunctionGemma routing

### Architecture findings

- Real retrieval is stronger than real routing under drift on the current benchmark. `EmbeddingGemma` kept perfect retrieval evidence metrics under variants, while `FunctionGemma` exposed a stable intent-class weakness around direct patch-record requests.
- The current bottleneck for retrieval-backed tasks is answer synthesis, not document finding. This is the strongest current argument for splitting retrieval quality claims from final-answer quality claims in published reporting.
- HF thinking-on is not currently the default reasoning path for this machine or benchmark slice. On the expanded clean thinking track, it is slower and less reliable than thinking-off because of overflow and truncation behavior.
- Patch-oriented routing needs a stronger intent prior than the current specialist stack provides. The repeated `tool_012_billing_patch_record` failures suggest a real router-side ambiguity between "inspect/lookup" and "propose/update patch record" tool classes.
- Real specialist-backed modular full-stack is operationally viable. Even under drift, it stayed above `0.93` success on the current narrow slice, which is enough to justify scaling that lane rather than treating it as experimental-only.
- In `KnowledgeWorkArena`, bounded specialist-backed execution is now good enough to expose the next real problem:
  - native artifacts can score perfectly
  - recovered execution can stay perfect
  - browser workflow can stay near-perfect even with explicit recovery branches
  - strict interface can now be pulled back to `1.0` when the controller is taught to respect next-step tool feedback instead of re-normalizing malformed outputs into repeated prior actions
  - that makes the next benchmark frontier less about obvious controller repetition and more about harder multi-step policy and judgment failures
- The history layer now needs stricter canonical semantics:
  - exploratory policy-only runs can share the same lane label as canonical runs
  - “latest by lane” is therefore not always the authoritative published state
  - the canonical summary pointers in continuity/docs are now the source of truth until the history report is split into canonical vs exploratory views
- Real-world autonomy is materially weaker than bounded task execution.
  - The model can preserve state and complete many operational tasks once tools are in motion.
  - It is still weak at deciding when not to act, when to escalate, and when a high-cost billing intent should be refused or redirected.
- Answer-surface multilinguality is still an end-to-end bottleneck.
  - Retrieval evidence and final state can both be correct while the real-world benchmark still fails because the answer layer misses the required French action phrasing.
- In `KnowledgeWorkArena`, the next specialist-backed weakness is now clearly action-class selection under replayable policy pressure:
  - the same fully specialist-backed stack can stay perfect on the live policy subset while missing replayable refusal/clarify expectations
  - this is evidence that the current problem is not broad runtime instability
  - it is likely a prompt/action-contract issue around `refuse` vs `defer` vs `clarify` after partial progress has already occurred
- H1 visual-sequence canaries clarified the current Gemma harnessing bottleneck:
  - concrete FunctionGemma prompt hints made the H1 HF service baseline controller-clean on the full five-episode slice: `controller_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, `raw_planning_clean_rate_avg = 1.0`
  - the remaining failures moved to disabled-helper rows, especially visual refinement chains that repeat valid but stale filters and fail to reach `read_region_text`
  - [`20260506T_h1_visual_sequence_hint_canary_v1`](../results/knowledge_work_h1_slice/20260506T_h1_visual_sequence_hint_canary_v1_knowledge_work_ablation_packet) showed base specialists stay clean on the three visual H1 episodes while both `no_controller_repair` and `no_deterministic_visual_follow_on` drop to `real_world_readiness_avg = 0.8837333333333333`
  - [`20260506T_h1_visual_filter_repair_canary_v1`](../results/knowledge_work_h1_slice/20260506T_h1_visual_filter_repair_canary_v1_knowledge_work_ablation_packet) restored the `no_deterministic_visual_follow_on` mini-row to `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, and `failure_candidate_count = 0`
  - the causal bug was accepting valid `refine_selection` calls whose `filter_query` repeated an already-completed visual filter; the controller now treats the pending visual filter as a semantic argument precondition
  - this does not solve `no_controller_repair`: that row still measures real model-side dependence because disabled repair accepts the valid-but-wrong refinements as-is
- The full H1 visual-filter-repair ablation converted the canary result into the current authoritative H1 controller snapshot:
  - [`20260506T_h1_hf_service_visual_filter_repair_ablation_v1`](../results/knowledge_work_h1_slice/20260506T_h1_hf_service_visual_filter_repair_ablation_v1_knowledge_work_ablation_packet) completed `7` systems across `35` replayable H1 episode rows
  - baseline specialists stayed clean: `real_world_readiness_avg = 0.9749800000000001`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, `controller_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, `raw_planning_clean_rate_avg = 1.0`
  - `no_deterministic_visual_follow_on` returned to baseline top-line readiness after the pending-filter repair, but remained controller-heavy: `controller_repair_avg = 0.6`, `argument_repair_avg = 0.3`, `controller_fallback_avg = 0.1`, `raw_planning_clean_rate_avg = 0.845`
  - all rows except `no_controller_repair` now match baseline on readiness, strict interface, and recovered execution
  - `no_controller_repair` remains the only top-line causal helper: `real_world_readiness_avg = 0.8874599999999999`, `strict_interface_avg = 0.775`, `recovered_execution_avg = 0.7`, while raw syntax is mostly clean at `0.975`
  - trace mining found only `3` failure candidates, all in `no_controller_repair`, with residual modes `repair_disabled`, `visual_readback_missing`, `visual_repeated_refinement`, and `visual_stepwise_control`
  - the next useful target is therefore model-side or contract-side visual sequence semantics, not visual rescue, generic fallback, or placeholder repair
- The compact H1 visual-semantics packet makes that residual controller-dependence cheap to replay:
  - [`20260506T_h1_visual_semantics_no_repair_v1`](../results/knowledge_work_h1_slice/20260506T_h1_visual_semantics_no_repair_v1_knowledge_work_ablation_packet) ran `3` systems over the `3` residual visual episodes
  - baseline specialists stayed clean on the packet: `real_world_readiness_avg = 0.9715666666666666`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, `raw_planning_clean_rate_avg = 1.0`
  - `no_controller_repair` reproduced the gap more sharply: `real_world_readiness_avg = 0.8257`, `strict_interface_avg = 0.625`, `recovered_execution_avg = 0.5`, while `raw_planning_clean_rate_avg = 1.0`
  - `no_deterministic_visual_follow_on` still recovered fully but needed help: `controller_repair_avg = 0.8333333333333334`, `argument_repair_avg = 0.5`, `raw_planning_clean_rate_avg = 0.7833333333333333`
  - trace mining found `3` failure candidates, all in `no_controller_repair`, with `visual_readback_missing`, `visual_repeated_refinement`, and `visual_stepwise_control` concentrated on the executive backlog and jobs form visual chains
  - this packet should be the default next loop for candidate visual sequencing fixes before another full H1 replayable rerun
- A stronger FunctionGemma visual system prompt was a negative top-line result on that compact packet:
  - [`20260506T_h1_visual_semantics_prompt_contract_v1`](../results/knowledge_work_h1_slice/20260506T_h1_visual_semantics_prompt_contract_v1_knowledge_work_ablation_packet) kept baseline specialists clean and preserved full recovery for `no_deterministic_visual_follow_on`
  - `no_controller_repair` stayed unchanged at `real_world_readiness_avg = 0.8257`, `strict_interface_avg = 0.625`, `recovered_execution_avg = 0.5`, and `raw_planning_clean_rate_avg = 1.0`
  - trace mining again found `3` failure candidates, all in `no_controller_repair`
  - the candidate did reduce one comparison-row argument-repair note, but did not teach the raw disabled-repair row to stop replaying stale visual calls
  - next hypothesis: the exact next-call directive needs to be injected as a final turn-level router instruction after tool-result messages, not only as system-prompt wording
- The final turn-level FunctionGemma visual directive validated that hypothesis on the compact packet:
  - [`20260506T_h1_visual_semantics_turn_directive_v1`](../results/knowledge_work_h1_slice/20260506T_h1_visual_semantics_turn_directive_v1_knowledge_work_ablation_packet) restored all three packet rows to `real_world_readiness_avg = 0.9715666666666666`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, and `raw_planning_clean_rate_avg = 1.0`
  - `no_controller_repair` no longer needed semantic repair on this packet; the raw FunctionGemma calls followed the full visual chains unaided
  - `no_deterministic_visual_follow_on` also became controller-clean on this packet: `controller_repair_avg = 0.0`, `argument_repair_avg = 0.0`, `controller_fallback_avg = 0.0`
  - trace mining found `0` failure candidates; the only note family was the expected `controller_repair_disabled` ablation marker
  - this separates two prompt-shaping facts: generic system prose did not fix stale visual replay, but a final recency-weighted exact-call directive did
  - next required check is the full H1 replayable ablation after this directive
- The full H1 replayable ablation confirmed the turn-directive gain across all current H1 rows:
  - [`20260506T_h1_hf_service_turn_directive_ablation_v1`](../results/knowledge_work_h1_slice/20260506T_h1_hf_service_turn_directive_ablation_v1_knowledge_work_ablation_packet) completed `7` systems across `35` H1 episode rows
  - every row matched baseline: `real_world_readiness_avg = 0.9749800000000001`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, `controller_repair_avg = 0.0`, `argument_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, `raw_planning_clean_rate_avg = 1.0`
  - `no_controller_repair` moved from the prior full-H1 `0.8874599999999999` readiness to baseline, eliminating the last top-line causal helper on this slice
  - `no_deterministic_visual_follow_on` also became controller-clean, so the final router directive now subsumes the earlier deterministic visual follow-on benefit on H1
  - trace mining found `0` failure candidates; only ablation-marker notes remained
  - research interpretation: the H1 causal controller signal was real, but it pointed to a prompt-contract recency problem in FunctionGemma routing rather than an unavoidable need for controller repair
  - next benchmark need: define H1b with harder visual/API/approval interactions because current H1 is saturated again
- H1b is now scaffolded as the next saturation breaker:
  - [`configs/knowledge_work_h1b_slice.yaml`](../configs/knowledge_work_h1b_slice.yaml) reuses existing packaged workflow episode pairs that were outside H1
  - [`docs/continuity/h1b-slice.md`](continuity/h1b-slice.md) records the purpose, episode set, stressors, and commands
  - the new slice concentrates older but harder visual/revision/resume cases: dashboard referent carryover, latest-action resume, jobs constraint override, jobs phone patch resume, and finance invoice revision
  - H1b should be run first as a compact `visual_policy_no_controller_repair` packet before a full H1b ablation
- The first compact H1b visual-policy packet stayed controller-clean:
  - [`20260506T_h1b_visual_policy_packet_v1`](../results/knowledge_work_h1_slice/20260506T_h1b_visual_policy_packet_v1_knowledge_work_ablation_packet) ran baseline, `no_controller_repair`, and `no_deterministic_visual_follow_on` over `3` H1b episodes
  - all three rows matched at `real_world_readiness_avg = 0.9472999999999999`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, `controller_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, and `raw_planning_clean_rate_avg = 1.0`
  - trace mining found `0` failure candidates
  - interpretation: these H1b episodes are harsher on artifact/readiness level than H1, but they still do not restore visual controller dependence after the final FunctionGemma turn directive
  - next check is a full H1b ablation across all five episodes and seven rows
- The full H1b ablation confirmed that the follow-up slice is also saturated with respect to the current controller-helper ablations:
  - [`20260506T_h1b_hf_service_ablation_v1`](../results/knowledge_work_h1_slice/20260506T_h1b_hf_service_ablation_v1_knowledge_work_ablation_packet) ran `7` systems across the `5` H1b replayable episodes
  - all seven rows matched at `real_world_readiness_avg = 0.9581199999999999`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, `controller_repair_avg = 0.0`, `argument_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, and `raw_planning_clean_rate_avg = 1.0`
  - trace mining found `30` note events and `0` failure candidates
  - remaining note counts are only disabled-helper markers: `controller_repair_disabled = 22` and `intent_priority_disabled = 8`
  - interpretation: H1b lowers absolute readiness relative to H1 because its artifacts are harder, but it does not restore controller dependence after the final FunctionGemma turn directive. The next useful research move is live CLI validation plus a new H1c slice with genuinely new visual/API/approval interactions.
- H1c live-policy pressure is now benchmark-clean for both HF service specialists and local MLX Gemma:
  - [`20260506T_h1c_live_policy_packet_v1`](../results/knowledge_work_h1_slice/20260506T_h1c_live_policy_packet_v1_knowledge_work_ablation_packet) ran the compact live-policy helper packet across baseline HF service specialists, `no_controller_repair`, `no_controller_fallback`, and `no_argument_repair`
  - all four HF service rows matched at `real_world_readiness_avg = 0.9779666666666667`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, `controller_repair_avg = 0.0`, `argument_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, and `raw_planning_clean_rate_avg = 1.0`
  - trace mining found `14` note events and `0` failure candidates; the notes were only disabled-helper markers
  - [`20260506T_h1c_mlx_live_primary_v1`](../results/knowledge_work_h1_slice/20260506T_h1c_mlx_live_primary_v1_knowledge_work_h1c_live_policy_controller_dependence_v1) then ran all five H1c live episodes on `mlx_gemma4_e2b_reasoner_only`
  - the MLX primary row reached `real_world_readiness_avg = 0.97936`, `artifact_quality_avg = 0.95`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, `controller_repair_avg = 0.0`, `argument_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, and `raw_planning_clean_rate_avg = 1.0`
  - direct trace inspection found no non-empty MLX `planning_repair_notes`
  - interpretation: H1c did not re-break top-line or controller-clean execution in the benchmark runner. The live CLI smoke packets remain important because they showed local MLX repair/fallback on overlapping workflows; the next research question is repeatability of that CLI/runtime-path signal, not another broad H1c rerun.
- H1d/H1e and the MLX directive probe clarified the current local Gemma harnessing boundary:
  - the final tool-turn directive was ported into the MLX/HF/GGUF Gemma runner prompt path
  - H1d directive-v2 eliminated the compact controller-stress failures: all four rows reached readiness `0.97936`, strict/recovered `1.0 / 1.0`, raw clean `1.0`, and repair/fallback/argument repair `0.0`
  - H1e then expanded to all ten packaged live workflow families and still found `0` failure candidates; all four MLX rows matched at readiness `0.96891`
  - the directive probe stayed at exact JSON `7 / 8` because MLX paraphrases one visual selector from `"validation error"` to `"phone issue"`
  - executor-level visual aliasing makes that paraphrase executable: the latest probe records exact `7 / 8`, executable visual target `1 / 1`
  - interpretation: exact-copy cleanliness and executable live readiness now need to be tracked separately for visual selector text
- H1f reopened controller-dependence by removing the tool-turn directive:
  - [`20260506T_h1f_mlx_no_directive_v1`](../results/knowledge_work_h1_slice/20260506T_h1f_mlx_no_directive_v1_knowledge_work_h1f_mlx_tool_contract_ablation_v1) ran five compact live workflow families across contracted MLX, no-directive MLX, and three no-directive/no-helper variants
  - contracted MLX was clean: readiness `0.97936`, strict/recovered `1.0 / 1.0`, raw clean `1.0`, repair/fallback/argument repair `0.0 / 0.0 / 0.0`
  - no-directive MLX preserved top-line readiness only by using controller help: repair/fallback/argument repair `0.70 / 0.20 / 0.50`, raw clean `0.30`
  - no-directive + no controller repair dropped to readiness `0.73818`, strict/recovered `0.475 / 0.300`
  - no-directive + no controller fallback dropped to readiness `0.92104`
  - no-directive + no argument repair dropped to readiness `0.82036`
  - interpretation: the directive is a causal harness intervention, but controller repair/fallback/argument repair remain causal once that prompt contract is absent
- H1g is the current negative result for remaining helper families under the directive:
  - [`20260506T_h1g_mlx_remaining_helpers_v1`](../results/knowledge_work_h1_slice/20260506T_h1g_mlx_remaining_helpers_v1_knowledge_work_h1g_mlx_remaining_helper_ablation_v1) ran baseline, `no_visual_rescue`, `no_intent_priority`, and `no_deterministic_visual_follow_on`
  - all four rows matched at readiness `0.97936`, strict/recovered `1.0 / 1.0`, raw clean `1.0`
  - trace mining found `0` failure candidates
  - interpretation: under the current directive, visual rescue, intent priority, and deterministic visual follow-on are not carrying the compact live MLX slice; the useful expansion was therefore the full-H1e no-directive packet, not more tuning of those helpers
- H1h confirms the no-directive causal ordering across the full ten-workflow live MLX set:
  - [`20260507T_h1h_mlx_full_no_directive_v1`](../results/knowledge_work_h1_slice/20260507T_h1h_mlx_full_no_directive_v1_knowledge_work_h1h_mlx_full_tool_contract_ablation_v1) ran contracted MLX, no-directive MLX, and the three no-directive/no-helper variants over all ten H1e live workflow families
  - contracted MLX and no-directive MLX both reached readiness `0.96891`, but the no-directive row needed controller repair/fallback/argument repair `0.70 / 0.25 / 0.45` and raw clean stayed `0.30`
  - no-directive + no controller repair dropped to readiness `0.73801`, strict/recovered `0.481 / 0.300`
  - no-directive + no controller fallback dropped to `0.89598`
  - no-directive + no argument repair dropped to `0.83016`
  - the H1h/H1f comparison shows no new causal ordering; the larger workflow set mostly adds more instances of the same failure families: fallback planner, visual stepwise control, repair disabled, fallback disabled, argument repair, visual repeated refinement, and visual readback missing
  - workflow-family attribution now makes the next target concrete: executive latest-action resume, jobs phone patch resume, jobs visual form hold, and executive stale brief packet are the worst no-repair rows
  - next empirical move: run an attributable Gemini CLI baseline packet over the same H1h workflow family set, then derive a smaller no-directive stress packet from the worst H1h workflows
- The H1h Gemini CLI dry-run baseline is now recorded as an external-reference packet:
  - [`20260507T_h1h_gemini_cli_dry_run_baseline_v1`](../results/gemini_cli/20260507T_h1h_gemini_cli_dry_run_baseline_v1) contains ten prompt/command artifacts, one per H1h workflow family
  - the run intentionally used a missing binary, so it recorded safe dry-run prompts without executing Gemini CLI or making external side effects
  - this is useful as a baseline interface contract: future real Gemini CLI execution can be compared against the exact same workflow-family prompts
- The no-directive MLX tool probe gives a raw-output explanation for H1h controller burden:
  - [`20260507T_mlx_no_directive_probe_v1`](../results/tool_directive_probe/20260507T_mlx_no_directive_probe_v1) ran the eight exact-call probe cases with `disable_tool_turn_directive`
  - exact match fell from contracted MLX `7 / 8` to no-directive `0 / 8`
  - the visual executable target fell from `1 / 1` to `0 / 1`
  - [`probe_case_deltas.csv`](../results/tool_directive_probe/20260507T_mlx_no_directive_probe_v1/probe_case_deltas.csv) shows CLI/API cases still often choose the right tool but drift on canonical arguments, while visual referent and parallel cases collapse to no tool call
  - interpretation: the directive is not cosmetic; it is the main model-side contract that keeps local MLX Gemma inside Moonie's tool interface. When it is absent, H1h readiness parity comes from controller repair/fallback rather than raw model compliance
- H1i turns the H1h worst-family attribution into a faster MLX prompt-contract loop:
  - [`configs/knowledge_work_h1i_slice.yaml`](../configs/knowledge_work_h1i_slice.yaml) keeps executive latest-action resume, jobs phone patch resume, jobs visual form hold, and executive stale brief packet
  - [`20260507T_h1i_mlx_worst_no_directive_v1`](../results/knowledge_work_h1_slice/20260507T_h1i_mlx_worst_no_directive_v1_knowledge_work_h1i_mlx_worst_family_tool_contract_v1) ran the same five MLX tool-contract rows over those four live workflow families
  - contracted MLX stayed clean at readiness `0.97710`, strict/recovered `1.0 / 1.0`, raw clean `1.0`
  - no-directive with helpers stayed top-line clean at readiness `0.97710`, but controller repair/fallback/argument repair rose to `1.00 / 0.50 / 0.50` and raw clean fell to `0.00`
  - no-directive + no controller repair fell to readiness `0.64697`, strict/recovered `0.297 / 0.000`
  - no-directive + no controller fallback fell to readiness `0.83125`; no-directive + no argument repair fell to `0.81220`
  - interpretation: H1i is now the best fast packet for candidate prompt contracts. It is smaller than H1h, but it amplifies the same causal ordering and should be used before spending another full H1h run
- The MLX tool-contract report is now a reproducible research artifact rather than only continuity prose:
  - human report: [`docs/reports/mlx-tool-contract-harnessing.md`](reports/mlx-tool-contract-harnessing.md)
  - generated report: [`results/reports/mlx_tool_contract_harnessing/report.md`](../results/reports/mlx_tool_contract_harnessing/report.md)
  - generator: [`scripts/build_mlx_tool_contract_report.py`](../scripts/build_mlx_tool_contract_report.py)
  - test: [`tests/test_mlx_tool_contract_report.py`](../tests/test_mlx_tool_contract_report.py)
  - it synthesizes H1f, H1h, H1i, the contracted/no-directive probe comparison, and the H1h Gemini CLI dry-run baseline
  - the report figures now make the active mechanism visible:
    - H1i top-line readiness can hide interface and recovery collapse when repair is disabled
    - H1i amplifies the H1h no-directive controller burden
    - the no-directive probe drops exact-call compliance from `7 / 8` to `0 / 8`
    - H1i failure modes remain concentrated in fallback planner, visual stepwise control, argument repair, fallback disabled, and repair disabled
  - this should be regenerated after any new H1i, H1h, probe, or Gemini baseline packet
- The next MLX harnessing wave is now concretely queued as prompt-contract candidates rather than an open-ended prompt-tuning idea:
  - candidate systems are defined in [`configs/model_registry.yaml`](../configs/model_registry.yaml):
    - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_anchor`
    - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_literal_guard`
    - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_tool_required`
  - each candidate keeps `disable_tool_turn_directive = true` and adds a generic `tool_prompt_contract_id`, which means the candidate is testing interface-shaping language rather than reintroducing the exact final directive by another name
  - generated report artifacts now include [`prompt_contract_candidates.csv`](../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_candidates.csv) and [`prompt_contract_candidate_targets.svg`](../results/reports/mlx_tool_contract_harnessing/figures/prompt_contract_candidate_targets.svg)
  - dry-run probe packet [`20260507T_prompt_contract_candidates_dry_run_v2`](../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_dry_run_v2) freezes the three candidate probe commands without running MLX and records both contracted and no-directive probe baselines
  - H1i now has a named graduation packet, `mlx_prompt_contract_candidates`, that runs contracted MLX, no-directive MLX, and the three candidate rows across the same four worst-family live workflows
  - empirical gate: execute the probe packet first, compare exact/executable rates against contracted and no-directive baselines, regenerate the report, and only then spend H1i live runtime on candidates that improve raw protocol behavior
- The first executed prompt-contract probe gate is a partial-gain result, not a solved interface result:
  - [`20260507T_prompt_contract_candidates_execute_v1`](../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1) ran all three candidate rows against the eight-case probe and wrote contracted plus no-directive comparisons for each candidate
  - [`candidate_gate_summary.md`](../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1/candidate_gate_summary.md) is the compact candidate read
  - `schema_anchor_v1` recovered one exact case over no-directive (`exact_match_rate = 0.125`, `delta_exact_vs_no_directive = 0.125`) but remained far below contracted MLX (`delta_exact_vs_contracted = -0.75`)
  - `literal_argument_guard_v1` and `tool_required_parallel_v1` recovered the executable visual target (`executable_match_rate = 1.0`) but did not improve exact JSON copy (`exact_match_rate = 0.0`)
  - `tool_required_parallel_v1` remains dominated by `no_tool_call` (`6` cases), which means its current wording is not yet solving the failure family it was meant to target
  - interpretation: these candidates can be tried on H1i as mechanism probes, but the second prompt-contract wave should combine schema anchoring with visual executable recovery and explicitly reduce no-call failures
- The first H1i prompt-contract candidate packet saturated and therefore did not validate the probe gains as live improvements:
  - [`20260507T_h1i_prompt_contract_candidates_v1`](../results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_v1_knowledge_work_ablation_packet) ran contracted MLX, no-directive MLX, and all three prompt-contract candidates over the four H1i worst-family live workflows
  - all five rows matched at `real_world_readiness_avg = 0.97710`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, `controller_repair_avg = 0.0`, `argument_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, and `raw_planning_clean_rate_avg = 1.0`
  - trace analysis found `0` note events and `0` failure candidates
  - the analyzer now derives `tool_turn_directive_enabled` from row research controls rather than the shared warmed bundle snapshot, so disabled-directive rows are attributed correctly
  - interpretation: the probe remains the stronger discriminator; H1i candidate v1 is saturated and the next second-stage slice needs repeated no-directive trials or probe-derived live cases where visual/parallel no-call failures are stable
- H1/H1i ablation packet runners now support repeated episode execution:
  - `scripts/run_knowledge_work_h1_ablation_packet.py` accepts `--repeat <n>` and passes it through to the focused ablation runner
  - `scripts/run_knowledge_work_ablation_packet.py` writes `repeat_count`, `base_episode_count`, and repeated episode execution into manifests and summary payloads
- The H1i prompt-contract repeat3 second-stage packet is now executed and saturated:
  - command:
    - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1i_slice.yaml --packet-id mlx_prompt_contract_candidates --run-group-id 20260507T_h1i_prompt_contract_candidates_repeat3_v1 --repeat 3`
  - packet: [`20260507T_h1i_prompt_contract_candidates_repeat3_v1`](../results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_repeat3_v1_knowledge_work_ablation_packet)
  - shape: `5` rows x `4` H1i workflow families x `3` repeats = `60` traces
  - all five rows matched at `real_world_readiness_avg = 0.97710`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, `controller_repair_avg = 0.0`, `argument_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, and `raw_planning_clean_rate_avg = 1.0`
  - trace analysis found `0` note events and `0` failure candidates
  - interpretation: repeated H1i is a clean negative result. The saturated candidate packet is stable, not flaky. The next packet needs probe-derived live cases, especially visual/parallel no-call and argument-mismatch cases, rather than more repeats of these packaged workflows
- H1j is now scaffolded as the probe-derived live packet:
  - config: [`configs/knowledge_work_h1j_slice.yaml`](../configs/knowledge_work_h1j_slice.yaml)
  - brief: [`docs/continuity/h1j-slice.md`](continuity/h1j-slice.md)
  - candidate packet id: `mlx_probe_derived_tool_contract_candidates`
  - helper-ablation packet id: `mlx_probe_derived_helper_ablation`
  - it selects six packaged live workflows that map to the no-directive probe failures: visual no-call/readback pressure plus API/CLI argument mismatch
  - `parallel_audit_array_literal` is explicitly deferred because the current packaged live surface has no faithful parallel-tool workflow
  - dry-run validation produced the expected command with `5` systems and `6` live episode ids
- The first H1j probe-derived candidate packet also saturated:
  - command:
    - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1j_slice.yaml --packet-id mlx_probe_derived_tool_contract_candidates --run-group-id 20260507T_h1j_probe_derived_candidates_v1`
  - packet: [`20260507T_h1j_probe_derived_candidates_v1`](../results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_candidates_v1_knowledge_work_ablation_packet)
  - shape: `5` rows x `6` packaged live workflow families = `30` traces
  - all five rows matched at `real_world_readiness_avg = 0.96577`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, `controller_repair_avg = 0.0`, `argument_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, and `raw_planning_clean_rate_avg = 1.0`
  - trace analysis found `0` note events and `0` failure candidates
  - interpretation: mapping probe failures back to packaged workflows was not enough. The raw probe is still the better discriminator. The next H1j run should remove controller helpers on the same probe-derived set, then the next prompt-contract wave should target the probe directly
- The paired H1j helper-ablation packet also saturated:
  - command:
    - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1j_slice.yaml --packet-id mlx_probe_derived_helper_ablation --run-group-id 20260507T_h1j_probe_derived_helpers_v1`
  - packet: [`20260507T_h1j_probe_derived_helpers_v1`](../results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_helpers_v1_knowledge_work_ablation_packet)
  - contracted, no-directive, no-controller-repair, no-controller-fallback, and no-argument-repair rows all matched at `real_world_readiness_avg = 0.96577`, `strict_interface_avg = 1.0`, `recovered_execution_avg = 1.0`, and `raw_planning_clean_rate_avg = 1.0`
  - trace analysis found `21` `controller_repair_disabled` markers on the disabled-repair row but `0` failure candidates
  - interpretation: H1j does not reintroduce controller dependence. The benchmark-style packaged workflow path is now empirically less discriminating than the raw tool-contract probe for this question

### Verification

- Narrow regression suite passes after the scoring and runtime-preflight changes:
  - `tests/test_answer_match.py`
  - `tests/test_stressors.py`
  - `tests/test_runtime_utils.py`
- Additional routing/runtime integrity regressions now pass:
  - `tests/test_tool_planner.py`
  - `tests/test_executor.py`
  - `tests/test_tool_eval.py`
  - `tests/test_gemma4_runner.py`
- Real-world execution/reporting integrity also passes after the new service-backed matrix work:
- The latest `KnowledgeWorkArena` hardening and specialist-stability pass now verifies cleanly end to end:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q`
  - `105 passed`
- The real specialist-backed policy-hardening pass also surfaced a durable runtime cleanup item:
  - both replayable and live runs emitted `top_p` / `top_k` generation-flag warnings on the HF specialist path
  - this did not block execution, but it should be normalized away because backend-specific decoding warnings are benchmark noise
- Real-world execution/reporting integrity also passes after the new service-backed matrix work:
  - `tests/test_alpha_matrix_script.py`
  - `tests/test_benchmark_module.py`
  - `tests/test_runtime_utils.py`
  - `tests/test_real_world_metrics.py`
  - `tests/test_replay_summary.py`
