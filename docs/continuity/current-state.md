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
  - `--target policy` renders live-web sandbox targets, sandbox endpoints, gates, and reasons
- `moonie-agent report`
  - inspects generated research report directories from the terminal
  - current default is the MLX tool-contract report with packet counts, table/figure inventory, Gemini baseline status, and prompt-contract candidate ids
  - supports `--json` for scripted harness checks
- `moonie-agent packet`
  - inspects generated research packet directories from the terminal
  - current default kind is `prompt-contract-probe`, including candidate rows, command counts, dry-run/executed counts, and packet files
  - supports `--packet-id latest`, explicit `--packet-dir`, and `--json`

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

Latest MLX tool-contract research:

- H1d/H1e saturated after the final tool-turn directive:
  - H1d directive-v2 packet: [`results/knowledge_work_h1_slice/20260506T_h1d_mlx_tool_directive_v2_knowledge_work_h1d_mlx_monolith_controller_stress_v1`](../../results/knowledge_work_h1_slice/20260506T_h1d_mlx_tool_directive_v2_knowledge_work_h1d_mlx_monolith_controller_stress_v1)
  - H1e full live packet: [`results/knowledge_work_h1_slice/20260506T_h1e_mlx_full_live_v1_knowledge_work_h1e_mlx_full_live_packaged_workflows_v1`](../../results/knowledge_work_h1_slice/20260506T_h1e_mlx_full_live_v1_knowledge_work_h1e_mlx_full_live_packaged_workflows_v1)
  - H1e result: all `4` MLX rows matched at `real_world_readiness_avg = 0.96891`, strict/recovered `1.0 / 1.0`, raw clean `1.0`, and repair/fallback/argument repair `0.0`
- Tool directive probe:
  - latest packet: [`results/tool_directive_probe/20260506T_mlx_tool_directive_probe_v4`](../../results/tool_directive_probe/20260506T_mlx_tool_directive_probe_v4)
  - exact JSON copy: `7 / 8`
  - executable visual paraphrase: `1 / 1`
  - remaining exact-copy miss is `target_query = "phone issue"` instead of benchmark-canonical `"validation error"`, but the local visual executor now resolves it to `form-err-202`
- H1f tool-contract ablation:
  - packet: [`results/knowledge_work_h1_slice/20260506T_h1f_mlx_no_directive_v1_knowledge_work_h1f_mlx_tool_contract_ablation_v1`](../../results/knowledge_work_h1_slice/20260506T_h1f_mlx_no_directive_v1_knowledge_work_h1f_mlx_tool_contract_ablation_v1)
  - summary: [`tool_contract_summary.md`](../../results/knowledge_work_h1_slice/20260506T_h1f_mlx_no_directive_v1_knowledge_work_h1f_mlx_tool_contract_ablation_v1/tool_contract_summary.md)
  - contracted MLX: readiness `0.97936`, strict/recovered `1.0 / 1.0`, raw clean `1.0`, repair/fallback/argument repair `0.0`
  - no directive with helpers: readiness still `0.97936`, but controller repair/fallback/argument repair `0.70 / 0.20 / 0.50`, raw clean `0.30`
  - no directive + no controller repair: readiness `0.73818`, strict/recovered `0.475 / 0.300`
  - no directive + no controller fallback: readiness `0.92104`
  - no directive + no argument repair: readiness `0.82036`
- H1g remaining-helper ablation:
  - packet: [`results/knowledge_work_h1_slice/20260506T_h1g_mlx_remaining_helpers_v1_knowledge_work_h1g_mlx_remaining_helper_ablation_v1`](../../results/knowledge_work_h1_slice/20260506T_h1g_mlx_remaining_helpers_v1_knowledge_work_h1g_mlx_remaining_helper_ablation_v1)
  - baseline, `no_visual_rescue`, `no_intent_priority`, and `no_deterministic_visual_follow_on` all matched at readiness `0.97936`, strict/recovered `1.0 / 1.0`, raw clean `1.0`
  - trace mining found `0` failure candidates
- H1h full no-directive tool-contract ablation:
  - config: [`configs/knowledge_work_h1h_slice.yaml`](../../configs/knowledge_work_h1h_slice.yaml)
  - packet: [`results/knowledge_work_h1_slice/20260507T_h1h_mlx_full_no_directive_v1_knowledge_work_h1h_mlx_full_tool_contract_ablation_v1`](../../results/knowledge_work_h1_slice/20260507T_h1h_mlx_full_no_directive_v1_knowledge_work_h1h_mlx_full_tool_contract_ablation_v1)
  - contracted MLX: readiness `0.96891`, strict/recovered `1.0 / 1.0`, raw clean `1.0`, repair/fallback/argument repair `0.0`
  - no directive with helpers: readiness still `0.96891`, but controller repair/fallback/argument repair `0.70 / 0.25 / 0.45`, raw clean `0.30`
  - no directive + no controller repair: readiness `0.73801`, strict/recovered `0.481 / 0.300`
  - no directive + no controller fallback: readiness `0.89598`
  - no directive + no argument repair: readiness `0.83016`
  - comparison to H1f: the causal ordering survives the full `10` workflow family set; extra workflows mainly add more instances of the same failure modes
  - workflow-family attribution: [`workflow_family_failures.csv`](../../results/knowledge_work_h1_slice/20260507T_h1h_mlx_full_no_directive_v1_knowledge_work_h1h_mlx_full_tool_contract_ablation_v1/workflow_family_failures.csv) shows the worst no-repair failures on executive latest-action resume, jobs phone patch resume, jobs visual form hold, and executive stale brief packet
- H1h Gemini CLI dry-run baseline:
  - packet: [`results/gemini_cli/20260507T_h1h_gemini_cli_dry_run_baseline_v1`](../../results/gemini_cli/20260507T_h1h_gemini_cli_dry_run_baseline_v1)
  - `workflow_count = 10`
  - `dry_run_count = 10`
  - `available_count = 0` because the run intentionally used `definitely-missing-gemini-cli`
  - interpretation: the repo now has attributable Gemini CLI prompt/command artifacts for the same H1h workflow families without executing any external side effects
- MLX no-directive tool probe:
  - packet: [`results/tool_directive_probe/20260507T_mlx_no_directive_probe_v1`](../../results/tool_directive_probe/20260507T_mlx_no_directive_probe_v1)
  - exact match `0 / 8`
  - executable visual match `0 / 1`
  - comparison against contracted MLX probe: [`probe_comparison.json`](../../results/tool_directive_probe/20260507T_mlx_no_directive_probe_v1/probe_comparison.json)
  - contracted probe exact rate `0.875`; no-directive exact rate `0.0`; delta `-0.875`
  - interpretation: H1h top-line parity under no-directive is entirely controller-mediated; raw no-directive MLX loses exact copying across CLI, API, visual, and parallel-tool families
- H1i compact worst-family MLX packet:
  - config: [`configs/knowledge_work_h1i_slice.yaml`](../../configs/knowledge_work_h1i_slice.yaml)
  - packet: [`results/knowledge_work_h1_slice/20260507T_h1i_mlx_worst_no_directive_v1_knowledge_work_h1i_mlx_worst_family_tool_contract_v1`](../../results/knowledge_work_h1_slice/20260507T_h1i_mlx_worst_no_directive_v1_knowledge_work_h1i_mlx_worst_family_tool_contract_v1)
  - contracted MLX: readiness `0.97710`, strict/recovered `1.0 / 1.0`, raw clean `1.0`
  - no directive with helpers: readiness `0.97710`, strict/recovered `1.0 / 1.0`, repair/fallback/argument repair `1.00 / 0.50 / 0.50`, raw clean `0.00`
  - no directive + no controller repair: readiness `0.64697`, strict/recovered `0.297 / 0.000`
  - no directive + no controller fallback: readiness `0.83125`
  - no directive + no argument repair: readiness `0.81220`
  - interpretation: H1i is the current fast loop for MLX prompt-contract/controller experiments; it preserves H1h's causal ordering while making the no-repair and no-fallback gaps larger
- Prompt-contract candidate queue:
  - candidate systems:
    - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_anchor`
    - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_literal_guard`
    - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_tool_required`
  - candidates deliberately keep `disable_tool_turn_directive = true` and add only generic contract reminders through `tool_prompt_contract_id`; they do not leak the exact planned tool call
  - generated candidate table: [`results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_candidates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_candidates.csv)
  - generated candidate target figure: [`results/reports/mlx_tool_contract_harnessing/figures/prompt_contract_candidate_targets.svg`](../../results/reports/mlx_tool_contract_harnessing/figures/prompt_contract_candidate_targets.svg)
  - dry-run probe packet: [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_dry_run_v2`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_dry_run_v2)
  - v2 packet schema records both the contracted probe baseline and the no-directive probe baseline, so executed candidates can be gated on improvement over no-directive before any H1i spend
  - executed probe packet: [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1)
  - executed gate summary: [`candidate_gate_summary.md`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1/candidate_gate_summary.md)
  - executed result:
    - `schema_anchor_v1`: exact `0.125`, executable `0.0`, recommendation `weak_exact_gain`
    - `literal_argument_guard_v1`: exact `0.0`, executable `1.0`, recommendation `visual_executable_gain_only`
    - `tool_required_parallel_v1`: exact `0.0`, executable `1.0`, recommendation `visual_executable_gain_only`
  - interpretation: all three candidates improve one probe case over no-directive, but none approaches the contracted row; H1i should treat them as mechanism probes rather than assumed fixes
  - H1i graduation packet: `mlx_prompt_contract_candidates` in [`configs/knowledge_work_h1i_slice.yaml`](../../configs/knowledge_work_h1i_slice.yaml)
  - executed H1i candidate packet: [`results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_v1_knowledge_work_ablation_packet)
  - H1i candidate result: all five rows matched readiness `0.97710`, strict/recovered `1.0 / 1.0`, repair/fallback/argument repair `0.0 / 0.0 / 0.0`, raw clean `1.0`, with `0` trace notes and `0` failure candidates
  - repeated H1i candidate packet: [`results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_repeat3_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_repeat3_v1_knowledge_work_ablation_packet)
  - H1i repeat3 result: `60` traces, all five rows matched readiness `0.97710`, strict/recovered `1.0 / 1.0`, repair/fallback/argument repair `0.0 / 0.0 / 0.0`, raw clean `1.0`, with `0` trace notes and `0` failure candidates
  - interpretation: the probe remains the stronger discriminator; one-pass and repeated H1i are both saturated, so the next packet should be probe-derived live cases before another broad H1h run
- H1j probe-derived live scaffold:
  - config: [`configs/knowledge_work_h1j_slice.yaml`](../../configs/knowledge_work_h1j_slice.yaml)
  - brief: [`docs/continuity/h1j-slice.md`](./h1j-slice.md)
  - candidate packet id: `mlx_probe_derived_tool_contract_candidates`
  - helper-ablation packet id: `mlx_probe_derived_helper_ablation`
  - shape: six packaged live workflows mapping no-directive probe failures into live API/CLI argument mismatch and visual no-call/readback pressure
  - parallel no-call remains deferred until a faithful live packaged workflow exists

Current generated research report:

- human report: [`docs/reports/mlx-tool-contract-harnessing.md`](../reports/mlx-tool-contract-harnessing.md)
- generated report: [`results/reports/mlx_tool_contract_harnessing/report.md`](../../results/reports/mlx_tool_contract_harnessing/report.md)
- packet summary table: [`results/reports/mlx_tool_contract_harnessing/tables/packet_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/packet_summary.csv)
- prompt-contract candidate table: [`results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_candidates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_candidates.csv)
- prompt-contract probe gate table: [`results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_probe_gates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_probe_gates.csv)
- H1i prompt-contract candidate metrics: [`results/reports/mlx_tool_contract_harnessing/tables/h1i_prompt_contract_candidate_metrics.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/h1i_prompt_contract_candidate_metrics.csv)
- H1i prompt-contract repeat3 metrics: [`results/reports/mlx_tool_contract_harnessing/tables/h1i_prompt_contract_repeat3_metrics.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/h1i_prompt_contract_repeat3_metrics.csv)
- figures: [`results/reports/mlx_tool_contract_harnessing/figures`](../../results/reports/mlx_tool_contract_harnessing/figures)
- regeneration command:

```bash
uv run python scripts/build_mlx_tool_contract_report.py
uv run pytest tests/test_mlx_tool_contract_report.py -q
```

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

Latest runtime live-smoke packet:

- script: [`scripts/run_runtime_live_smoke_packet.py`](../../scripts/run_runtime_live_smoke_packet.py)
- output: [`results/runtime_live_smoke_packets/20260506T_runtime_live_smoke_mlx_v2_runtime_live_smoke_packet`](../../results/runtime_live_smoke_packets/20260506T_runtime_live_smoke_mlx_v2_runtime_live_smoke_packet)
- workflow: `executive_visual_dashboard_review`
- system: `mlx_gemma4_e2b_reasoner_only`
- status: `completed`
- `role_readiness_avg = 0.9942`
- `strict_interface_avg = 1.0`
- `recovered_execution_avg = 1.0`
- `controller_repair_avg = 0.5`
- `argument_repair_avg = 0.5`
- `raw_planning_clean_rate_avg = 0.5`
- `controller_finding_count = 1`

Latest runtime approval/smoke trio:

- output: [`results/runtime_live_smoke_packets/20260506T_runtime_live_smoke_mlx_trio_v2_runtime_live_smoke_packet`](../../results/runtime_live_smoke_packets/20260506T_runtime_live_smoke_mlx_trio_v2_runtime_live_smoke_packet)
- workflows:
  - `executive_visual_dashboard_review`
  - `finance_visual_invoice_review`
  - `jobs_visual_form_hold`
- status counts: `completed = 1`, `awaiting_approval = 2`
- `role_readiness_avg = 0.9800333333333334`
- `strict_interface_avg = 1.0`
- `recovered_execution_avg = 1.0`
- `controller_repair_avg = 0.6666666666666666`
- `argument_repair_avg = 0.6666666666666666`
- `raw_planning_clean_rate_avg = 0.3333333333333333`
- `approval_count = 2`
- `policy_block_count = 0`
- `controller_finding_count = 4`
- [`controller_findings.json`](../../results/runtime_live_smoke_packets/20260506T_runtime_live_smoke_mlx_trio_v2_runtime_live_smoke_packet/controller_findings.json) records exact repair notes and raw calls for the visual, API, and CLI argument repairs

Latest live-web policy packet:

- output: [`results/runtime_live_smoke_packets/20260506T_runtime_live_web_policy_mlx_v2_runtime_live_smoke_packet`](../../results/runtime_live_smoke_packets/20260506T_runtime_live_web_policy_mlx_v2_runtime_live_smoke_packet)
- workflow: `jobs_visual_form_hold`
- lane: `live_web_stress`
- status counts: `awaiting_approval = 1`
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
- [`policy_blocks.json`](../../results/runtime_live_smoke_packets/20260506T_runtime_live_web_policy_mlx_v2_runtime_live_smoke_packet/policy_blocks.json) records two `sandbox_only` blocks and one `approval_required` block with sandbox endpoints

Latest repeated live-web CLI packet:

- output: [`results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet`](../../results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet)
- command shape: `3` live-web workflows x `3` repeats on `mlx_gemma4_e2b_reasoner_only`
- workflows:
  - `executive_visual_dashboard_review`
  - `finance_visual_invoice_review`
  - `jobs_visual_form_hold`
- status counts: `completed = 3`, `awaiting_approval = 6`, `failed = 0`
- packet averages:
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
- stable per-workflow pattern across all three repeats:
  - executive dashboard: `repaired_arguments:extract_layout`
  - finance invoice: `repaired_arguments:cli_search_logs`
  - jobs form: `repaired_arguments:cli_apply_patch` plus `controller_fallback_planner`
- analyzer outputs:
  - [`runtime_packet_analysis.json`](../../results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet/runtime_packet_analysis.json)
  - [`runtime_repair_family_counts.csv`](../../results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet/runtime_repair_family_counts.csv)
  - [`runtime_policy_block_counts.csv`](../../results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet/runtime_policy_block_counts.csv)
  - [`runtime_workflow_stability.csv`](../../results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet/runtime_workflow_stability.csv)
- analyzer summary: `stable_repair_family_count = 4`, `stable_policy_block_family_count = 7`
- interpretation: the CLI live path has stable controller-dependence signal even though the H1c benchmark runner is clean. The next research move should isolate runtime/session execution differences and then ablate the repeated CLI families directly.

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
- H1c live-policy controller slice scaffold:
  - [`configs/knowledge_work_h1c_slice.yaml`](../../configs/knowledge_work_h1c_slice.yaml)
  - [`docs/continuity/h1c-slice.md`](./h1c-slice.md)
- H1b compact visual policy packet:
  - [`results/knowledge_work_h1_slice/20260506T_h1b_visual_policy_packet_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1b_visual_policy_packet_v1_knowledge_work_ablation_packet)
- H1b full HF service-backed ablation:
  - [`results/knowledge_work_h1_slice/20260506T_h1b_hf_service_ablation_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1b_hf_service_ablation_v1_knowledge_work_ablation_packet)
- H1c compact live-policy helper packet:
  - [`results/knowledge_work_h1_slice/20260506T_h1c_live_policy_packet_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1c_live_policy_packet_v1_knowledge_work_ablation_packet)
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

H1c scaffold:

- config: [`configs/knowledge_work_h1c_slice.yaml`](../../configs/knowledge_work_h1c_slice.yaml)
- doc: [`docs/continuity/h1c-slice.md`](./h1c-slice.md)
- purpose: shift the saturation-breaker from replayable visual semantics to live-web policy gates, approval-safe stops, and visual/API/CLI argument repair
- live packet: `live_policy_controller_helpers`
  - lane: `live_web_stress`
  - systems: baseline HF service specialists plus `no_controller_repair`, `no_controller_fallback`, and `no_argument_repair`
  - episodes: `kwa_jobs_live_email_block_resume_hold_v5`, `kwa_finance_live_invoice_lock_direction_hold_v4`, `kwa_jobs_live_phone_patch_resume_hold_v4`
- dry-run validation:
  - `uv run pytest tests/test_knowledge_work_h1.py tests/test_runtime_cli.py tests/test_runtime_live_smoke_packet.py -q`
  - `20 passed`
  - `uv run python scripts/run_knowledge_work_h1_slice.py --config configs/knowledge_work_h1c_slice.yaml --dry-run --run-set primary --lane live_web_stress --output-root tmp/h1c-dry-run-smoke --run-group-id 20260506T_h1c_live_primary_dry_run_v1`
  - `uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1c_slice.yaml --packet-id live_policy_controller_helpers --run-group-id 20260506T_h1c_live_policy_packet_dry_run_v1 --dry-run`

H1c compact live-policy helper packet:

- output: [`results/knowledge_work_h1_slice/20260506T_h1c_live_policy_packet_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1c_live_policy_packet_v1_knowledge_work_ablation_packet)
- systems: baseline HF service specialists plus `no_controller_repair`, `no_controller_fallback`, and `no_argument_repair`
- episodes: `kwa_jobs_live_email_block_resume_hold_v5`, `kwa_finance_live_invoice_lock_direction_hold_v4`, `kwa_jobs_live_phone_patch_resume_hold_v4`
- all four rows match:
  - `real_world_readiness_avg = 0.9779666666666667`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `controller_repair_avg = 0.0`
  - `argument_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`
- trace mining found `14` notes and `0` failure candidates
- remaining notes are only `controller_repair_disabled` markers in the disabled-repair row
- interpretation: H1c live-policy pressure does not re-break HF service specialists on this compact packet. The next comparison should run the H1c MLX primary path because the CLI live packets still show MLX repair/fallback signal.

H1c MLX primary live path:

- output: [`results/knowledge_work_h1_slice/20260506T_h1c_mlx_live_primary_v1_knowledge_work_h1c_live_policy_controller_dependence_v1`](../../results/knowledge_work_h1_slice/20260506T_h1c_mlx_live_primary_v1_knowledge_work_h1c_live_policy_controller_dependence_v1)
- system: `mlx_gemma4_e2b_reasoner_only`
- lane: `live_web_stress`
- episodes: all `5` H1c live episodes
- result:
  - `real_world_readiness_avg = 0.97936`
  - `artifact_quality_avg = 0.95`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `controller_repair_avg = 0.0`
  - `argument_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`
- per-episode trace inspection found no non-empty `planning_repair_notes`
- interpretation: the benchmark H1c runner is clean for local MLX Gemma on this live-policy slice. The remaining discrepancy is with earlier CLI live-smoke packets that did surface repair/fallback on overlapping workflows, so the next best measurement is repeated CLI live-smoke execution over the same packaged workflows.

H1c MLX primary live path, corrected monolith posture:

- output: [`results/knowledge_work_h1_slice/20260506T_h1c_mlx_live_primary_monolith_v1_knowledge_work_h1c_live_policy_controller_dependence_v1`](../../results/knowledge_work_h1_slice/20260506T_h1c_mlx_live_primary_monolith_v1_knowledge_work_h1c_live_policy_controller_dependence_v1)
- system: `mlx_gemma4_e2b_reasoner_only`
- lane: `live_web_stress`
- change: H1 primary rows now pass `--pipeline-name monolith` for `local_reasoner` systems, matching `moonie-agent live`
- result:
  - `real_world_readiness_avg = 0.97936`
  - `artifact_quality_avg = 0.95`
  - `strict_interface_avg = 1.0`
  - `recovered_execution_avg = 1.0`
  - `controller_repair_avg = 0.7`
  - `argument_repair_avg = 0.5`
  - `controller_fallback_avg = 0.2`
  - `raw_planning_clean_rate_avg = 0.3`
- controller-dependent families:
  - `visual_016_live_dashboard_stale_selection_recovery`
  - `tool_018_jobs_api_latest_form_issue`
  - `visual_022_live_form_latest_issue_referent_carryover`
  - `tool_019_finance_cli_log_search_latest_lock`
  - `tool_021_jobs_cli_patch_only_latest_email_fix`
  - `visual_030_live_form_latest_blocked_email_refinement`
  - `tool_016_finance_api_invoice_lock_update`
- interpretation: the prior clean H1c MLX primary row was a harness mismatch caused by a modular heuristic router in the benchmark path. The corrected monolith row aligns with the repeated CLI live-smoke packet: local MLX Gemma still completes the workflows, but it reproducibly needs controller repair/fallback on live policy/tool families.

H1c MLX monolith helper ablation:

- output: [`results/knowledge_work_h1_slice/20260506T_h1c_mlx_monolith_helpers_v1_knowledge_work_h1c_live_policy_controller_dependence_v1`](../../results/knowledge_work_h1_slice/20260506T_h1c_mlx_monolith_helpers_v1_knowledge_work_h1c_live_policy_controller_dependence_v1)
- systems:
  - `mlx_gemma4_e2b_reasoner_only`
  - `mlx_gemma4_e2b_reasoner_only_no_controller_repair`
  - `mlx_gemma4_e2b_reasoner_only_no_controller_fallback`
  - `mlx_gemma4_e2b_reasoner_only_no_argument_repair`
- baseline row: readiness `0.97936`, strict/recovered `1.0 / 1.0`, repair/fallback `0.7 / 0.2`, raw clean `0.3`
- `no_controller_repair`: readiness `0.7381800000000001`, strict/recovered `0.475 / 0.3`, raw clean `0.89`
- `no_controller_fallback`: readiness `0.92104`, strict/recovered `0.85 / 0.8`, raw clean `0.5`
- `no_argument_repair`: readiness `0.82036`, strict/recovered `0.7125 / 0.5`, raw clean `0.8`
- trace mining: `41` notes, `12` failure candidates
- dominant failure modes:
  - `visual_stepwise_control = 6`
  - `repair_disabled = 5`
  - `fallback_planner = 4`
  - `argument_repair = 2`
  - `fallback_disabled = 2`
- interpretation: controller repair, argument repair, and fallback are all causal for local MLX monolith on H1c. Higher raw-clean rates in disabled rows are not better behavior; they mean the controller stopped repairing semantically wrong raw calls.

H1d candidate direction:

- doc: [`docs/continuity/h1d-candidates.md`](./h1d-candidates.md)
- config: [`configs/knowledge_work_h1d_slice.yaml`](../../configs/knowledge_work_h1d_slice.yaml)
- first named packet: [`results/knowledge_work_h1_slice/20260506T_h1d_mlx_controller_stress_v1_knowledge_work_h1d_mlx_monolith_controller_stress_v1`](../../results/knowledge_work_h1_slice/20260506T_h1d_mlx_controller_stress_v1_knowledge_work_h1d_mlx_monolith_controller_stress_v1)
- H1d reproduced the H1c monolith helper-ablation result exactly:
  - baseline readiness `0.97936`
  - `no_controller_repair` readiness `0.7381800000000001`
  - `no_controller_fallback` readiness `0.92104`
  - `no_argument_repair` readiness `0.82036`
  - trace mining found `41` notes and `12` failure candidates
- proposed stress families:
  - visual stepwise control
  - API/CLI canonicalization
  - fallback boundary
  - approval-safe stop under repair pressure

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
