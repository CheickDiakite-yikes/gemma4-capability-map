# Current State

## Latest Restart Point

The active research frontier is now H2a, not H1x.

H1x broke v11 saturation but could not justify promoting v12 globally because H1s had already shown transfer cost. H1y then tested whether prompt/catalog prose could route the residual cases more selectively. It could not: v16 and v17 both reached only `5 / 10` on the H1y packet, tying v11 and below v12's `7 / 10`.

The next meaningful result was H2a: keep v11's component-label prompt profile, but add a controller-side stale visual selection gate that rewrites missing `selection_id` calls into current-image `extract_layout` calls when the live visual state proves the user-mentioned selection id is stale. On the same H1y packet:

- no-directive: `0 / 10`
- v11 component-label guard: `5 / 10`
- v12 component-residual guard: `7 / 10`
- v16 routed-residual prompt guard: `5 / 10`
- v17 selection-origin prompt guard: `5 / 10`
- H2a v11 + controller stale-selection gate: `8 / 10`

The conclusion is important for the paper: stale selection-origin errors are currently controller-addressable and not reliably solved by more catalog prose. H2a fixed all three stale-field route rows and preserved both surface-value holdouts; the two remaining failures are argument-alias/code-label residuals.

Primary artifacts:

- H1y/H2a synthesis: [`results/reports/h1y_routed_residual_synthesis/report.md`](../../results/reports/h1y_routed_residual_synthesis/report.md)
- H2a live packet: [`results/tool_probe_replay_live/20260510T_h2a_visual_stale_selection_gate_on_h1y_execute_v1`](../../results/tool_probe_replay_live/20260510T_h2a_visual_stale_selection_gate_on_h1y_execute_v1)
- H2a-vs-v11 comparison: [`results/tool_probe_replay_live_comparisons/20260510T_h2a_visual_stale_selection_gate_vs_component_label_guard_on_h1y_v1`](../../results/tool_probe_replay_live_comparisons/20260510T_h2a_visual_stale_selection_gate_vs_component_label_guard_on_h1y_v1)
- H2a-vs-v12 comparison: [`results/tool_probe_replay_live_comparisons/20260510T_h2a_visual_stale_selection_gate_vs_component_residual_guard_on_h1y_v1`](../../results/tool_probe_replay_live_comparisons/20260510T_h2a_visual_stale_selection_gate_vs_component_residual_guard_on_h1y_v1)
- main report table: [`results/reports/mlx_tool_contract_harnessing/tables/h1y_routed_residual_packet_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/h1y_routed_residual_packet_summary.csv)
- main report figure: [`results/reports/mlx_tool_contract_harnessing/figures/h1y_routed_residual_gate.svg`](../../results/reports/mlx_tool_contract_harnessing/figures/h1y_routed_residual_gate.svg)

Current reporting snapshot:

- MLX tool-contract report: `97` tables / `42` figures
- publication evidence ledger: `38` claims / `197` evidence sources / `0` missing
- publication readiness audit: `paper_draft_ready`, `0` blocking failures
- latest publication claim: `C38_h2a_controller_stale_selection_gate_is_causal`

Next restart move:

- transfer-test H2a across H1n/H1o/H1p/H1x before promoting it as a broader runtime default
- then isolate the remaining H1y residuals: `state tag`/`alert s92` argument-alias and code-label exactness

## Benchmark Shape

Current generated corpus on disk:

- atomic tasks: `91`
- variants: `396`
- replayable KWA episodes: `33`
- live KWA episodes: `27`

The current headline comparison surface is still the saturated aligned exploratory `32 / 26` lane. The new `33 / 27` corpus additions are the parallel-audit scaffold for the next successor slice, not a completed replacement board yet.

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
  - `--kind tool-probe-replay` inspects exact probe replay packets, including case rows, failure modes, command counts, and packet files
  - `--kind tool-probe-replay-live` inspects live exact-replay packets, including case status rows, exact rate, executor-equivalence rate, command count, and packet files
  - `--kind tool-probe-replay-live-comparison` inspects live replay A/B comparison packets, including exact-rate deltas, executor-equivalence deltas, and case-level call deltas
  - `--kind tool-probe-replay-live-diagnostic` inspects visual tool-choice diagnostic packets, including diagnosis transitions and expected-vs-actual visual tool rows
  - supports `--packet-id latest`, explicit `--packet-dir`, and `--json`
- `moonie-agent replay-live`
  - previews or executes exact tool-probe replay cases through a Rich CLI operator view
  - defaults to the no-directive MLX replay system and supports `--case-id`, `--packet-dir`, `--execute`, and `--json`
  - writes live replay packets under `results/tool_probe_replay_live/`

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
  - replay-to-live promotion brief: [`docs/continuity/replay-to-live-packaged-workflows.md`](./replay-to-live-packaged-workflows.md)
  - candidate packet id: `mlx_probe_derived_tool_contract_candidates`
  - helper-ablation packet id: `mlx_probe_derived_helper_ablation`
  - shape: six packaged live workflows mapping no-directive probe failures into live API/CLI argument mismatch and visual no-call/readback pressure
  - parallel no-call remains deferred until a faithful live packaged workflow exists
  - candidate packet: [`results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_candidates_v1_knowledge_work_ablation_packet)
  - H1j candidate result: `30` traces, all five rows matched readiness `0.96577`, strict/recovered `1.0 / 1.0`, repair/fallback/argument repair `0.0 / 0.0 / 0.0`, raw clean `1.0`, with `0` trace notes and `0` failure candidates
  - helper packet: [`results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_helpers_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_helpers_v1_knowledge_work_ablation_packet)
  - H1j helper result: `30` traces, all five helper rows matched readiness `0.96577`, strict/recovered `1.0 / 1.0`, repair/fallback/argument repair `0.0 / 0.0 / 0.0`, raw clean `1.0`; trace mining found `21` disabled-repair markers but `0` failure candidates
- H1k parallel-audit live scaffold:
  - config: [`configs/knowledge_work_h1k_slice.yaml`](../../configs/knowledge_work_h1k_slice.yaml)
  - brief: [`docs/continuity/h1k-slice.md`](./h1k-slice.md)
  - workflow: `ops_parallel_audit_review`
  - replay pressure: `parallel_audit_array_literal`
  - replayable/live episodes: `kwa_ops_parallel_audit_review_v1` / `kwa_ops_live_parallel_audit_review_v1`
  - candidate packet id: `mlx_parallel_audit_tool_contract_candidates`
  - helper-ablation packet id: `mlx_parallel_audit_helper_ablation`
  - CLI validation:

    ```bash
    uv run moonie-agent workflows --lane live_web_stress --workflow-id ops_parallel_audit_review --validate
    ```

  - candidate dry run:

    ```bash
    uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1k_slice.yaml --packet-id mlx_parallel_audit_tool_contract_candidates --run-group-id 20260507T_h1k_parallel_audit_candidates_dry_run_v1 --dry-run
    ```

  - candidate packet: [`results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_candidates_v1_knowledge_work_ablation_packet)
  - H1k candidate result: `5` traces, all rows matched readiness `0.91780`, strict/recovered `1.0 / 1.0`, repair/fallback/argument repair `0.0 / 0.0 / 0.0`, raw clean `1.0`, with `0` trace notes and `0` failure candidates
  - helper packet: [`results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_helpers_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_helpers_v1_knowledge_work_ablation_packet)
  - H1k helper result: `5` traces, all helper-ablation rows matched readiness `0.91780`, strict/recovered `1.0 / 1.0`, repair/fallback/argument repair `0.0 / 0.0 / 0.0`, raw clean `1.0`; trace mining found `3` expected `controller_repair_disabled` markers and `0` failure candidates
  - interpretation: H1k is a useful negative result. It adds the deferred parallel-audit packaged workflow, but the staged live workflow is still easier than the raw one-turn `parallel_audit_array_literal` exact probe. Removing controller repair, controller fallback, or argument repair does not move this packaged slice, so the next discriminator needs to preserve exact-call replay shape.
- H1l visual executor-equivalence live scaffold:
  - config: [`configs/knowledge_work_h1l_slice.yaml`](../../configs/knowledge_work_h1l_slice.yaml)
  - brief: [`docs/continuity/h1l-slice.md`](./h1l-slice.md)
  - purpose: promote the visual hard-slice executor-equivalence split into packaged visual live workflows
  - candidate packet id: `mlx_visual_executor_equivalence_candidates`
  - helper-ablation packet id: `mlx_visual_executor_equivalence_helper_ablation`
  - live workflows: `executive_visual_dashboard_review`, `executive_visual_referent_review`, `jobs_visual_constraint_override`, `finance_visual_invoice_review`, `finance_visual_invoice_revision`
  - candidate packet: [`results/knowledge_work_h1_slice/20260509T_h1l_visual_executor_equivalence_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260509T_h1l_visual_executor_equivalence_candidates_v1_knowledge_work_ablation_packet)
  - result: all six rows tie at readiness `0.90406`, strict `0.85`, recovered `0.8`, raw clean `1.0`, and repair/fallback/argument repair `0.0 / 0.0 / 0.0`
  - interpretation: current packaged visual workflows are saturated and do not preserve the hard-slice executor-equivalence discriminator; defer helper spend until a non-saturated visual live surface exists
- Replay-shaped visual hard-slice CLI-live result:
  - replay source packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_no_directive_replay_dry_run_v1`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_no_directive_replay_dry_run_v1)
  - no-directive live packet: [`results/tool_probe_replay_live/20260509T_visual_hard_slice_no_directive_hard_replay_live_execute_v2`](../../results/tool_probe_replay_live/20260509T_visual_hard_slice_no_directive_hard_replay_live_execute_v2)
  - schema-field live packet: [`results/tool_probe_replay_live/20260509T_visual_hard_slice_schema_field_hints_hard_replay_live_execute_v2`](../../results/tool_probe_replay_live/20260509T_visual_hard_slice_schema_field_hints_hard_replay_live_execute_v2)
  - comparison: [`results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_schema_field_hints_vs_no_directive_live_v2`](../../results/tool_probe_replay_live_comparisons/20260509T_visual_hard_slice_schema_field_hints_vs_no_directive_live_v2)
  - result: no-directive stays strict/executor-equivalent `0 / 2`; schema-field hints reaches strict `1 / 2` and executor-equivalent `2 / 2`
  - interpretation: the v4 hard-slice signal survives in the CLI live operator path when raw replay shape is preserved; H1l saturation is evidence about staged packaged workflows, not evidence that the v4 mechanism is gone
- Visual hard-slice alias-repeat matrix:
  - source packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_repeat_dry_run_v1`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_repeat_dry_run_v1)
  - summary table: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_alias_repeat_live_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_alias_repeat_live_replay_summary.csv)
  - diagnostic: [`results/reports/visual_alias_repeat_diagnostic/diagnostic.md`](../../results/reports/visual_alias_repeat_diagnostic/diagnostic.md)
  - result: no-directive MLX is strict `2 / 8` and executor-equivalent `5 / 8`; contracted MLX is `7 / 8` strict and `8 / 8` executor-equivalent; schema-field hints v4 is strict `2 / 8` and executor-equivalent `7 / 8`; schema target literals v5 is strict `3 / 8` and executor-equivalent `8 / 8`
  - interpretation: repeated alias/decoy pressure preserves the executor-grounding gain; schema target literals are no longer purely negative on this repeated slice, but contracted MLX remains the strict upper bound
- H1m visual alias-repeat packaged live result:
  - config: [`configs/knowledge_work_h1m_slice.yaml`](../../configs/knowledge_work_h1m_slice.yaml)
  - brief: [`docs/continuity/h1m-slice.md`](./h1m-slice.md)
  - workflows packaged for CLI live use: `executive_visual_dashboard_revision`, `jobs_visual_latest_issue_review`, `finance_visual_invoice_hold_review`
  - candidate packet id: `mlx_visual_alias_repeat_packaged_candidates`
  - helper packet id: `mlx_visual_alias_repeat_helper_ablation`
  - executed packet: [`results/knowledge_work_h1_slice/20260509T_h1m_visual_alias_repeat_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260509T_h1m_visual_alias_repeat_candidates_v1_knowledge_work_ablation_packet)
  - result: all six rows tie at readiness `0.87783`, strict interface `0.75`, recovered execution `0.667`, raw clean `1.0`, and repair/fallback/argument repair `0.0 / 0.0 / 0.0`
  - interpretation: the replay-shaped alias-repeat signal is real, but these packaged workflows still wash it out; do not spend H1m helper-ablation budget until a packaged or non-packaged live visual surface separates rows
- Packaged replay gap diagnostic:
  - diagnostic: [`results/reports/packaged_replay_gap_diagnostic/diagnostic.md`](../../results/reports/packaged_replay_gap_diagnostic/diagnostic.md)
  - result: `2 / 2` visual promotion surfaces have positive replay gains and zero packaged readiness/strict-interface span
  - interpretation: packaged workflow design is now an experimental variable; current visual packaged workflows are useful as live harness scaffolds but not as discriminators for the visual alias/decoy question
- H1n visual alias-transfer replay slice:
  - brief: [`docs/continuity/h1n-slice.md`](./h1n-slice.md)
  - source packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_dry_run_v1`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_dry_run_v1)
  - oracle v2 packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_dry_run_v2`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_dry_run_v2)
  - suite: `alias_transfer_v3`
  - diagnostic: [`results/reports/visual_alias_transfer_diagnostic/diagnostic.md`](../../results/reports/visual_alias_transfer_diagnostic/diagnostic.md)
  - contract-split diagnostic: [`results/reports/h1n_alias_transfer_contract_split/diagnostic.md`](../../results/reports/h1n_alias_transfer_contract_split/diagnostic.md)
  - oracle diagnostic: [`results/reports/visual_alias_transfer_oracle_diagnostic/diagnostic.md`](../../results/reports/visual_alias_transfer_oracle_diagnostic/diagnostic.md)
  - legacy v1 result: no-directive MLX is strict `0 / 6` and executor-equivalent `2 / 6`; argument hints v2 is strict `1 / 6` and executor-equivalent `6 / 6`; schema target literals v5 is strict `1 / 6` and executor-equivalent `4 / 6`; contracted MLX is strict `5 / 6` but executor-equivalent `1 / 6`
  - contract finding: `5 / 6` generated expected-call contracts do not satisfy the packet oracle, so legacy H1n strict exactness measured heuristic planner-call fidelity more than visual target success
  - runtime update: `moonie-agent replay-live` now honors serialized packet expected calls, so oracle v2 strict scoring uses the packet contract rather than recomputed planner calls
  - oracle v2 result: no-directive is `2 / 6`; contracted is `1 / 6`; role catalog v1 is `3 / 6`; argument hints v2 is `5 / 6` strict and `6 / 6` executor-equivalent; schema-field hints v4 is `2 / 6`; schema target literals v5 is `4 / 6`
  - helper-ablation diagnostic: [`results/reports/h1n_oracle_helper_ablation/diagnostic.md`](../../results/reports/h1n_oracle_helper_ablation/diagnostic.md)
  - helper-ablation result: argument hints remains `5 / 6` strict and `6 / 6` executor-equivalent with controller repair disabled, controller fallback disabled, and argument repair disabled one at a time
  - repeat packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_repeat_oracle_dry_run_v1`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_repeat_oracle_dry_run_v1)
  - repeat diagnostic: [`results/reports/visual_alias_transfer_repeat_diagnostic/diagnostic.md`](../../results/reports/visual_alias_transfer_repeat_diagnostic/diagnostic.md)
  - repeat result: no-directive is `2 / 6`; contracted is `0 / 6`; role catalog v1 and schema-field hints v4 are `4 / 6`; argument hints v2 and schema target literals v5 are `5 / 6` strict and `6 / 6` executor-equivalent
  - synthesis: [`results/reports/h1n_oracle_transfer_synthesis/report.md`](../../results/reports/h1n_oracle_transfer_synthesis/report.md)
  - synthesis result: argument hints is executor-equivalent in both oracle packets, schema target literals catches up on the repeat, contracted prompting is not an upper bound, and the tested controller helpers do not explain the argument-hints gain
  - oblique-label packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_oblique_oracle_dry_run_v1`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_oblique_oracle_dry_run_v1)
  - oblique-label result before repair: no-directive `0 / 6`; contracted `1 / 6`; role catalog v1 `2 / 6`; argument hints v2 `4 / 6`; schema-field hints v4 `3 / 6`; schema target literals v5 `0 / 6`
  - oblique-label diagnostic: [`results/reports/visual_alias_transfer_oblique_diagnostic/diagnostic.md`](../../results/reports/visual_alias_transfer_oblique_diagnostic/diagnostic.md)
  - miss analysis: [`results/reports/h1n_oblique_miss_analysis/diagnostic.md`](../../results/reports/h1n_oblique_miss_analysis/diagnostic.md)
  - code-hints profile: `visual_role_catalog_oblique_code_hints_v6`
  - code-hints live packet: [`results/tool_probe_replay_live/20260509T_h1n_oracle_oblique_code_hints_execute_v1`](../../results/tool_probe_replay_live/20260509T_h1n_oracle_oblique_code_hints_execute_v1)
  - code-hints comparison: [`results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_oblique_code_hints_vs_argument_hints_v1`](../../results/tool_probe_replay_live_comparisons/20260509T_h1n_oracle_oblique_code_hints_vs_argument_hints_v1)
  - code-hints delta diagnostic: [`results/reports/h1n_oblique_code_hints_delta/diagnostic.md`](../../results/reports/h1n_oblique_code_hints_delta/diagnostic.md)
  - code-hints result: `5 / 6` exact and executor-equivalent, improving over argument hints by `+0.167` on both metrics
  - code-hints transfer synthesis: [`results/reports/h1n_code_hints_transfer_synthesis/report.md`](../../results/reports/h1n_code_hints_transfer_synthesis/report.md)
  - code-hints transfer result: across the first oracle, repeat, and oblique packets, argument hints has `14 / 18` exact and `16 / 18` executor-equivalent successes; code hints has `11 / 18` exact and `12 / 18` executor-equivalent successes
  - code-guard profile: `visual_role_catalog_oblique_code_guard_v7`
  - code-guard live packet: [`results/tool_probe_replay_live/20260510T_h1n_oracle_oblique_code_guard_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1n_oracle_oblique_code_guard_execute_v1)
  - code-guard result: `6 / 6` exact and executor-equivalent on the oblique packet, improving over argument hints by `+0.333` and over v6 code hints by `+0.167`
  - code-guard transfer synthesis: [`results/reports/h1n_code_guard_transfer_synthesis/report.md`](../../results/reports/h1n_code_guard_transfer_synthesis/report.md)
  - code-guard transfer result: across the three oracle packets, code guard reaches `14 / 18` exact and `15 / 18` executor-equivalent successes, improving over v6 code hints at `11 / 18` and `12 / 18`, while argument hints remains `14 / 18` and `16 / 18`
  - interpretation: fresh transfer cases favor narrow catalog-profile mechanisms once the expected-call contract is oracle-backed; the code-like oblique packet breaks the repeat tie in favor of argument hints over schema target literals, and the activation-gated code guard fixes the v6 `field e19` regression. It is better than v6 but still not a broad replacement for argument hints.
- Prompt-contract wave 2:
  - contracts: `schema_literal_tool_required_v2`, `visual_next_call_state_v2`, `parallel_array_required_v2`
  - runner flag: `scripts/run_tool_prompt_contract_probe_packet.py --candidate-wave v2`
  - dry-run packet: [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_dry_run_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_dry_run_v1)
  - executed packet: [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_execute_v1)
  - gate result:
    - `schema_literal_tool_required_v2`: exact `0.125`, executable `0.0`, recommendation `weak_exact_gain`
    - `visual_next_call_state_v2`: exact `0.0`, executable `1.0`, recommendation `visual_executable_gain_only`
    - `parallel_array_required_v2`: exact `0.0`, executable `0.0`, recommendation `no_probe_gain`
  - interpretation: wave 2 confirms the prior shape rather than solving it. The next move is stricter exact-probe live replay or a faithful live packaged workflow for the deferred parallel no-call family, not promotion back to H1
- Prompt-contract wave 3:
  - contracts: `canonical_json_copy_v3`, `visual_tool_initiation_v3`, `parallel_two_call_array_v3`
  - runner flag: `scripts/run_tool_prompt_contract_probe_packet.py --candidate-wave v3`
  - dry-run packet: [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave3_dry_run_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave3_dry_run_v1)
  - executed packet: [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave3_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave3_execute_v1)
  - gate result:
    - `canonical_json_copy_v3`: exact `0.125`, executable `0.0`, recommendation `weak_exact_gain`
    - `visual_tool_initiation_v3`: exact `0.125`, executable `1.0`, recommendation `weak_exact_gain`
    - `parallel_two_call_array_v3`: exact `0.0`, executable `0.0`, recommendation `no_probe_gain`
  - interpretation: visual initiation is the strongest candidate so far, but wave three still does not replace the final tool-turn directive. Parallel no-call remains unsolved.
- Prompt-contract wave 4:
  - contract: `visual_state_tool_selection_v4`
  - runner flag: `scripts/run_tool_prompt_contract_probe_packet.py --candidate-wave v4`
  - dry-run packet: [`results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_dry_run_v1`](../../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_dry_run_v1)
  - executed packet: [`results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_execute_v1)
  - gate result: exact `0.125`, executable `0.0`, delta exact vs no-directive `+0.125`, recommendation `weak_exact_gain`
  - interpretation: v4 is a useful negative result. It preserves one exact visual referent recovery but does not improve over the wave-three live ceiling and does not fix the targeted wrong-tool filter case.
- Prompt-contract wave 5:
  - contract: `visual_refine_selection_v5`
  - runner flag: `scripts/run_tool_prompt_contract_probe_packet.py --candidate-wave v5`
  - dry-run packet: [`results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave5_dry_run_v1`](../../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave5_dry_run_v1)
  - executed packet: [`results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave5_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave5_execute_v1)
  - gate result: exact `0.0`, executable `0.0`, delta exact vs no-directive `0.0`, recommendation `no_probe_gain`
  - interpretation: v5 is rejected before live replay. The surgical `refine_selection` wording did not preserve visual tool initiation and increased no-call concentration.
- Tool-catalog visual role profile:
  - profile: `visual_role_catalog_v1`
  - runner: [`scripts/run_tool_catalog_profile_probe_packet.py`](../../scripts/run_tool_catalog_profile_probe_packet.py)
  - dry-run packet: [`results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_v1_dry_run`](../../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_v1_dry_run)
  - executed probe packet: [`results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_v1_probe`](../../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_v1_probe)
  - raw gate result: exact `0.125`, executable `1.0`, delta exact vs no-directive `+0.125`
  - live replay packet: [`results/tool_probe_replay_live/20260508T_visual_role_catalog_live_execute_v1`](../../results/tool_probe_replay_live/20260508T_visual_role_catalog_live_execute_v1)
  - live replay result: exact `1 / 3`, executable visual-form recovery `1.0`, all three visual cases enter the tool protocol
  - comparisons:
    - [`results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_no_directive_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_no_directive_v1)
    - [`results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_visual_tool_initiation_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_visual_tool_initiation_v1)
    - [`results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_visual_state_tool_selection_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_visual_state_tool_selection_v1)
  - interpretation: the catalog profile changes the remaining visual failure from `wrong_tool` / `no_tool_call` into `argument_mismatch`. The useful mechanism is tool-role separability in the catalog, not another standalone visual prompt rule.
- Tool-catalog visual argument-hints profile:
  - profile: `visual_role_catalog_argument_hints_v2`
  - candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_argument_hints`
  - dry-run packet: [`results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_argument_hints_v2_dry_run`](../../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_argument_hints_v2_dry_run)
  - executed probe packet: [`results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_argument_hints_v2_probe`](../../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_argument_hints_v2_probe)
  - raw gate result: exact `0.25`, executable `0.0`, delta exact vs no-directive `+0.25`
  - comparison vs v1 catalog: [`results/tool_catalog_profile_probe_comparisons/20260508T_visual_argument_hints_vs_role_catalog_v1`](../../results/tool_catalog_profile_probe_comparisons/20260508T_visual_argument_hints_vs_role_catalog_v1)
  - live replay packet: [`results/tool_probe_replay_live/20260508T_visual_catalog_argument_hints_live_execute_v1`](../../results/tool_probe_replay_live/20260508T_visual_catalog_argument_hints_live_execute_v1), exact `2 / 3`
  - live comparisons:
    - [`results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_no_directive_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_no_directive_v1)
    - [`results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_contracted_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_contracted_v1)
    - [`results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_role_catalog_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_role_catalog_v1)
  - interpretation: this is the best focused-replay exact visual no-directive candidate. It fixes `visual_latest_filter_literal` and preserves exact readback, matching contracted MLX at `2 / 3` exact there. It is not solved because it loses the executable `visual_form_target_literal` rescue, and the fresh hard slice now gives a broader visual read.
- Tool-catalog visual split-selector profile:
  - profile: `visual_role_catalog_split_selector_hints_v3`
  - candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_split_selector_hints`
  - dry-run packet: [`results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_split_selector_hints_v3_dry_run`](../../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_split_selector_hints_v3_dry_run)
  - executed probe packet: [`results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_split_selector_hints_v3_probe`](../../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_split_selector_hints_v3_probe)
  - raw gate result: exact `0.125`, executable `0.0`, delta exact vs no-directive `+0.125`
  - comparison vs v2: [`results/tool_catalog_profile_probe_comparisons/20260508T_visual_split_selector_hints_vs_argument_hints_v2`](../../results/tool_catalog_profile_probe_comparisons/20260508T_visual_split_selector_hints_vs_argument_hints_v2), delta exact `-0.125`
  - comparison vs v1: [`results/tool_catalog_profile_probe_comparisons/20260508T_visual_split_selector_hints_vs_role_catalog_v1`](../../results/tool_catalog_profile_probe_comparisons/20260508T_visual_split_selector_hints_vs_role_catalog_v1), delta exact `0.0`, executable regression from v1
  - skipped-live decision: [`results/tool_probe_replay_live/20260508T_visual_split_selector_hints_live_replay_skipped_v1`](../../results/tool_probe_replay_live/20260508T_visual_split_selector_hints_live_replay_skipped_v1)
  - interpretation: v3 is negative evidence. It preserved the v2 latest-filter exact case but regressed readback by emitting `tool_name` instead of `name`, did not recover form-target executability, and did not earn live replay.
- Tool-catalog visual schema-field profile:
  - profile: `visual_role_catalog_schema_field_hints_v4`
  - candidate system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints`
  - dry-run packet: [`results/tool_catalog_profile_probe_packets/20260509T_visual_role_catalog_schema_field_hints_v4_dry_run`](../../results/tool_catalog_profile_probe_packets/20260509T_visual_role_catalog_schema_field_hints_v4_dry_run)
  - executed probe packet: [`results/tool_catalog_profile_probe_packets/20260509T_visual_role_catalog_schema_field_hints_v4_probe`](../../results/tool_catalog_profile_probe_packets/20260509T_visual_role_catalog_schema_field_hints_v4_probe)
  - raw gate result: exact `0.25`, executable `0.0`, delta exact vs no-directive `+0.25`
  - comparison vs v2: [`results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_argument_hints_v2`](../../results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_argument_hints_v2), delta exact `0.0`
  - comparison vs v3: [`results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_split_selector_v3`](../../results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_split_selector_v3), delta exact `+0.125`
  - comparison vs v1: [`results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_role_catalog_v1`](../../results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_role_catalog_v1), delta exact `+0.125`, executable regression vs v1
  - skipped-live decision: [`results/tool_probe_replay_live/20260509T_visual_schema_field_hints_live_replay_skipped_v1`](../../results/tool_probe_replay_live/20260509T_visual_schema_field_hints_live_replay_skipped_v1)
  - focused-slice interpretation: v4 is cleaner than v3 because it keeps the schema field hints local and restores exact readback. On the original focused replay/probe slice it still does not beat v2 on exactness, remains below v1 on executable form targeting, and over-prefers `refine_selection(selection_id="latest", filter_query="phone issue")` on the form-target case.
- Fresh visual hard-slice packet:
  - script: [`scripts/build_visual_hard_slice_design.py`](../../scripts/build_visual_hard_slice_design.py)
  - runner: [`scripts/run_visual_hard_slice_probe_packet.py`](../../scripts/run_visual_hard_slice_probe_packet.py)
  - design packet: [`results/reports/visual_hard_slice_design/design.md`](../../results/reports/visual_hard_slice_design/design.md)
  - dry-run packet: [`results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_dry_run_v1`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_dry_run_v1)
  - first executed packet: [`results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_execute_v1`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_execute_v1)
  - v5 dry-run packet: [`results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_v5_dry_run`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_v5_dry_run)
  - latest executed packet: [`results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1)
  - latest gate summary: [`candidate_gate_summary.md`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1/candidate_gate_summary.md)
  - v5-vs-v4 comparison: [`schema_literal_targets_vs_schema_field_hints`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1/schema_literal_targets_vs_schema_field_hints)
  - exactness-vs-executor diagnostic: [`results/reports/visual_hard_slice_exactness_diagnostic`](../../results/reports/visual_hard_slice_exactness_diagnostic)
  - shape: `8` executable cases across visual argument copying, visual tool routing, referent carryover, and region readback
  - latest result:
    - contracted MLX: strict/executable/executor-equivalent `8 / 8`
    - no-directive MLX: strict/executable/executor-equivalent `1 / 8`, dominant failure `no_tool_call`
    - `visual_role_catalog_v1`: strict/executable/executor-equivalent `3 / 8`
    - `visual_role_catalog_argument_hints_v2`: strict `6 / 8`, executable/executor-equivalent `7 / 8`
    - `visual_role_catalog_split_selector_hints_v3`: strict `5 / 8`, executable/executor-equivalent `6 / 8`
    - `visual_role_catalog_schema_field_hints_v4`: strict `6 / 8`, executable/executor-equivalent `8 / 8`
    - `visual_role_catalog_schema_literal_targets_v5`: strict `5 / 8`, executable/executor-equivalent `7 / 8`
    - `visual_role_catalog_v1 + literal_guard`: strict `3 / 8`, executable/executor-equivalent `4 / 8`
  - interpretation: the hard slice breaks prior top-line saturation. Schema-field hints move from a focused-replay negative result to the strongest fresh-slice no-directive candidate because they preserve full executor-visible target success, but they still trail contracted MLX on strict exact protocol fidelity. The exactness diagnostic classifies both v4 non-exact rows as executor-success selector aliases, and the latest packet now scores those rows directly with first-class executor-equivalence. The v5 schema-target-literal repair is negative evidence: it keeps those two aliases and adds a wrong-tool stale-selection regression.
- Prompt-contract wave 6:
  - candidate: `literal_argument_guard_v1` + `visual_role_catalog_v1`
  - runner flag: `scripts/run_tool_prompt_contract_probe_packet.py --candidate-wave v6`
  - dry-run packet: [`results/tool_prompt_contract_probe_packets/20260508T_visual_catalog_literal_guard_v6_dry_run`](../../results/tool_prompt_contract_probe_packets/20260508T_visual_catalog_literal_guard_v6_dry_run)
  - executed packet: [`results/tool_prompt_contract_probe_packets/20260508T_visual_catalog_literal_guard_v6_probe`](../../results/tool_prompt_contract_probe_packets/20260508T_visual_catalog_literal_guard_v6_probe)
  - gate result: exact `0.125`, executable `0.0`, delta exact vs no-directive `+0.125`
  - interpretation: negative composition result. Adding the literal guard on top of the visual catalog does not fix literal drift and loses the catalog-only executable visual rescue, so it should not spend live replay or H1 budget.
- Exact-probe replay packet:
  - brief: [`docs/continuity/exact-probe-replay.md`](./exact-probe-replay.md)
  - script: [`scripts/build_tool_probe_replay_packet.py`](../../scripts/build_tool_probe_replay_packet.py)
  - packet: [`results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_v1`](../../results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_v1)
  - executed packet: [`results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1`](../../results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1)
  - contracted replay packet: [`results/tool_probe_replay_packets/20260507T_contracted_exact_probe_replay_execute_v1`](../../results/tool_probe_replay_packets/20260507T_contracted_exact_probe_replay_execute_v1)
  - replay comparison: [`results/tool_probe_replay_comparisons/20260507T_contracted_vs_no_directive_exact_replay_v1`](../../results/tool_probe_replay_comparisons/20260507T_contracted_vs_no_directive_exact_replay_v1)
  - source: no-directive probe [`results/tool_directive_probe/20260507T_mlx_no_directive_probe_v1`](../../results/tool_directive_probe/20260507T_mlx_no_directive_probe_v1)
  - shape: `8` failed exact-call probe cases, `4` argument mismatches and `4` no-tool-call cases
  - next-action split: `4` canonical argument replays, `3` visual-state replay executor cases, `1` parallel-array replay/workflow case
  - execution result: `0 / 8` exact, with the same `4` argument mismatches and `4` no-tool-call failures reproduced
  - contracted replay result: `7 / 8` exact, with the remaining visual paraphrase executable
  - comparison result: no-directive exact-rate delta `-0.875` versus contracted on the same cases
  - live operator dry run: [`results/tool_probe_replay_live/20260507T_parallel_array_replay_live_dry_run_v1`](../../results/tool_probe_replay_live/20260507T_parallel_array_replay_live_dry_run_v1)
  - live no-directive parallel execution: [`results/tool_probe_replay_live/20260507T_parallel_array_no_directive_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_parallel_array_no_directive_live_execute_v1), exact `0 / 1`, expected calls `2`, actual calls `0`, replay failure `no_tool_call`
  - live contracted parallel execution: [`results/tool_probe_replay_live/20260507T_parallel_array_contracted_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_parallel_array_contracted_live_execute_v1), exact `1 / 1`, expected calls `2`, actual calls `2`
  - live comparison: [`results/tool_probe_replay_live_comparisons/20260507T_parallel_array_contracted_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260507T_parallel_array_contracted_vs_no_directive_live_v1), contracted exact `1.0`, no-directive exact `0.0`, delta `-1.0`, actual-call delta `-2`
  - live no-directive visual execution: [`results/tool_probe_replay_live/20260507T_visual_state_no_directive_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_visual_state_no_directive_live_execute_v1), exact `0 / 3`, all replay failures `no_tool_call`
  - live contracted visual execution: [`results/tool_probe_replay_live/20260507T_visual_state_contracted_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_visual_state_contracted_live_execute_v1), exact `2 / 3`, remaining visual form case executable through paraphrase
  - live visual comparison: [`results/tool_probe_replay_live_comparisons/20260507T_visual_state_contracted_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260507T_visual_state_contracted_vs_no_directive_live_v1), contracted exact `0.6666666666666666`, no-directive exact `0.0`, delta `-0.6666666666666666`
  - live no-directive canonical-argument execution: [`results/tool_probe_replay_live/20260507T_canonical_argument_no_directive_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_canonical_argument_no_directive_live_execute_v1), exact `0 / 4`, all replay failures `argument_mismatch`
  - live contracted canonical-argument execution: [`results/tool_probe_replay_live/20260507T_canonical_argument_contracted_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_canonical_argument_contracted_live_execute_v1), exact `4 / 4`
  - live canonical-argument comparison: [`results/tool_probe_replay_live_comparisons/20260507T_canonical_argument_contracted_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260507T_canonical_argument_contracted_vs_no_directive_live_v1), delta exact `-1.0`, actual-call delta `0` on all four cases
  - wave-three canonical candidate live packet: [`results/tool_probe_replay_live/20260507T_canonical_argument_canonical_json_copy_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_canonical_argument_canonical_json_copy_live_execute_v1), exact `0 / 4`
  - wave-three visual candidate live packet: [`results/tool_probe_replay_live/20260507T_visual_state_visual_tool_initiation_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_visual_state_visual_tool_initiation_live_execute_v1), exact `1 / 3`, visual executable `1 / 1`
  - wave-three live candidate summary: [`results/reports/mlx_tool_contract_harnessing/tables/wave3_live_candidate_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/wave3_live_candidate_replay_summary.csv)
  - interpretation update: `canonical_json_copy_v3` should not be promoted because live exact stays `0 / 4` and two cases regress to no-call. `visual_tool_initiation_v3` is a real partial live gain over no-directive, but it still misses one visual referent case with the wrong tool and remains below contracted MLX.
  - wave-four visual candidate live packet: [`results/tool_probe_replay_live/20260508T_visual_state_tool_selection_live_execute_v1`](../../results/tool_probe_replay_live/20260508T_visual_state_tool_selection_live_execute_v1), exact `1 / 3`, visual executable `0 / 1`
  - wave-four live comparison vs no-directive: [`results/tool_probe_replay_live_comparisons/20260508T_visual_state_tool_selection_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_state_tool_selection_vs_no_directive_live_v1), exact `+0.3333333333333333`
  - wave-four live comparison vs contracted: [`results/tool_probe_replay_live_comparisons/20260508T_visual_state_contracted_vs_tool_selection_live_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_state_contracted_vs_tool_selection_live_v1), exact `-0.3333333333333333`, executable `-1.0`
  - interpretation update: `visual_state_tool_selection_v4` did not beat `visual_tool_initiation_v3`; `visual_latest_filter_literal` remains `wrong_tool`, and `visual_form_target_literal` regressed to `no_tool_call`.
  - visual tool-choice diagnostic: [`results/tool_probe_replay_live_diagnostics/20260508T_visual_tool_choice_wave3_wave4_catalog_v1`](../../results/tool_probe_replay_live_diagnostics/20260508T_visual_tool_choice_wave3_wave4_catalog_v1)
  - diagnostic read: `visual_latest_filter_literal` expected `refine_selection`; wave three and wave four emitted `extract_layout`, while the catalog profile emitted `refine_selection` with `filter_query` drift. That transition is what motivated the argument-hints candidate.
  - replay-shaped visual hard-slice live matrix:
    - source packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_no_directive_replay_dry_run_v1`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_no_directive_replay_dry_run_v1)
    - summary table: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_live_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_live_replay_summary.csv)
    - case table: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_live_replay_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_live_replay_case_deltas.csv)
    - no-directive baseline: strict/executable/executor-equivalent `0 / 2`
    - contracted MLX: strict/executable/executor-equivalent `2 / 2`
    - role catalog v1 and argument hints v2: strict/executable/executor-equivalent `1 / 2`
    - schema-field hints v4: strict `1 / 2`, executable/executor-equivalent `2 / 2`
    - schema target literals v5: strict `0 / 2`, executable/executor-equivalent `1 / 2`, with a wrong-tool stale-selection miss
  - replay-shaped visual stress live matrix:
    - source packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_dry_run_v1`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_dry_run_v1)
    - summary table: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_stress_live_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_stress_live_replay_summary.csv)
    - no-directive MLX: strict `2 / 4`, executor-equivalent `3 / 4`
    - contracted MLX: strict/executor-equivalent `4 / 4`
    - role catalog v1: strict `1 / 4`, executor-equivalent `2 / 4`
    - argument hints v2: strict `2 / 4`, executor-equivalent `3 / 4`
    - schema-field hints v4 and schema target literals v5: strict `2 / 4`, executor-equivalent `4 / 4`
  - interpretation: this is not a packaged live workflow yet; it is the stable raw-contract replay artifact that should drive the next live discriminator. The new `moonie-agent replay-live` command is the first CLI bridge for watching these exact cases without converting them into easier staged workflows.
- Focused canonical-argument replay:
  - no-directive packet: [`results/tool_probe_replay_packets/20260507T_canonical_argument_exact_replay_no_directive_v1`](../../results/tool_probe_replay_packets/20260507T_canonical_argument_exact_replay_no_directive_v1)
  - contracted packet: [`results/tool_probe_replay_packets/20260507T_canonical_argument_exact_replay_contracted_v1`](../../results/tool_probe_replay_packets/20260507T_canonical_argument_exact_replay_contracted_v1)
  - comparison: [`results/tool_probe_replay_comparisons/20260507T_canonical_argument_contracted_vs_no_directive_v1`](../../results/tool_probe_replay_comparisons/20260507T_canonical_argument_contracted_vs_no_directive_v1)
  - result: no-directive exact `0 / 4`; contracted exact `4 / 4`; exact-rate delta `-1.0`
  - interpretation: the final tool-turn directive is doing concrete work on exact CLI/API argument canonicalization, not just visual or parallel no-call behavior
- Focused visual-state replay:
  - no-directive packet: [`results/tool_probe_replay_packets/20260507T_visual_state_exact_replay_no_directive_v1`](../../results/tool_probe_replay_packets/20260507T_visual_state_exact_replay_no_directive_v1)
  - contracted packet: [`results/tool_probe_replay_packets/20260507T_visual_state_exact_replay_contracted_v1`](../../results/tool_probe_replay_packets/20260507T_visual_state_exact_replay_contracted_v1)
  - comparison: [`results/tool_probe_replay_comparisons/20260507T_visual_state_contracted_vs_no_directive_v1`](../../results/tool_probe_replay_comparisons/20260507T_visual_state_contracted_vs_no_directive_v1)
  - result: no-directive exact `0 / 3`; contracted exact `2 / 3` plus one executable visual paraphrase
- Focused parallel-array replay:
  - no-directive packet: [`results/tool_probe_replay_packets/20260507T_parallel_array_exact_replay_no_directive_v1`](../../results/tool_probe_replay_packets/20260507T_parallel_array_exact_replay_no_directive_v1)
  - contracted packet: [`results/tool_probe_replay_packets/20260507T_parallel_array_exact_replay_contracted_v1`](../../results/tool_probe_replay_packets/20260507T_parallel_array_exact_replay_contracted_v1)
  - comparison: [`results/tool_probe_replay_comparisons/20260507T_parallel_array_contracted_vs_no_directive_v1`](../../results/tool_probe_replay_comparisons/20260507T_parallel_array_contracted_vs_no_directive_v1)
  - result: no-directive exact `0 / 1`; contracted exact `1 / 1`

Current generated research report:

- human report: [`docs/reports/mlx-tool-contract-harnessing.md`](../reports/mlx-tool-contract-harnessing.md)
- generated report: [`results/reports/mlx_tool_contract_harnessing/report.md`](../../results/reports/mlx_tool_contract_harnessing/report.md)
- packet summary table: [`results/reports/mlx_tool_contract_harnessing/tables/packet_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/packet_summary.csv)
- prompt-contract candidate table: [`results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_candidates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_candidates.csv)
- prompt-contract probe gate table: [`results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_probe_gates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_probe_gates.csv)
- prompt-contract wave-two probe gate table: [`results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_wave2_probe_gates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_wave2_probe_gates.csv)
- prompt-contract wave-three probe gate table: [`results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_wave3_probe_gates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_wave3_probe_gates.csv)
- prompt-contract wave-four probe gate table: [`results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_wave4_probe_gates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_wave4_probe_gates.csv)
- prompt-contract wave-five probe gate table: [`results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_wave5_probe_gates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_wave5_probe_gates.csv)
- prompt-contract promotion decisions: [`results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_promotion_decisions.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_promotion_decisions.csv)
- H1i prompt-contract candidate metrics: [`results/reports/mlx_tool_contract_harnessing/tables/h1i_prompt_contract_candidate_metrics.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/h1i_prompt_contract_candidate_metrics.csv)
- H1i prompt-contract repeat3 metrics: [`results/reports/mlx_tool_contract_harnessing/tables/h1i_prompt_contract_repeat3_metrics.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/h1i_prompt_contract_repeat3_metrics.csv)
- H1j probe-derived candidate metrics: [`results/reports/mlx_tool_contract_harnessing/tables/h1j_probe_derived_candidate_metrics.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/h1j_probe_derived_candidate_metrics.csv)
- H1j probe-derived helper metrics: [`results/reports/mlx_tool_contract_harnessing/tables/h1j_probe_derived_helper_metrics.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/h1j_probe_derived_helper_metrics.csv)
- exact probe replay case deltas: [`results/reports/mlx_tool_contract_harnessing/tables/exact_probe_replay_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/exact_probe_replay_case_deltas.csv)
- exact probe replay family deltas: [`results/reports/mlx_tool_contract_harnessing/tables/exact_probe_replay_family_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/exact_probe_replay_family_deltas.csv)
- exact probe replay focus summary: [`results/reports/mlx_tool_contract_harnessing/tables/exact_probe_replay_focus_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/exact_probe_replay_focus_summary.csv)
- wave-three live candidate replay summary: [`results/reports/mlx_tool_contract_harnessing/tables/wave3_live_candidate_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/wave3_live_candidate_replay_summary.csv)
- wave-four live candidate replay summary: [`results/reports/mlx_tool_contract_harnessing/tables/wave4_live_candidate_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/wave4_live_candidate_replay_summary.csv)
- visual catalog argument-hints live summary: [`results/reports/mlx_tool_contract_harnessing/tables/visual_catalog_argument_hints_live_candidate_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_catalog_argument_hints_live_candidate_replay_summary.csv)
- visual catalog argument-hints case deltas: [`results/reports/mlx_tool_contract_harnessing/tables/visual_catalog_argument_hints_live_candidate_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_catalog_argument_hints_live_candidate_case_deltas.csv)
- visual split-selector probe deltas vs v2: [`results/reports/mlx_tool_contract_harnessing/tables/tool_catalog_split_selector_vs_argument_hints_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/tool_catalog_split_selector_vs_argument_hints_case_deltas.csv)
- visual split-selector live decision: [`results/reports/mlx_tool_contract_harnessing/tables/tool_catalog_split_selector_live_replay_decision.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/tool_catalog_split_selector_live_replay_decision.csv)
- visual schema-field probe deltas vs v2: [`results/reports/mlx_tool_contract_harnessing/tables/tool_catalog_schema_field_hints_vs_argument_hints_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/tool_catalog_schema_field_hints_vs_argument_hints_case_deltas.csv)
- visual schema-field probe deltas vs v3: [`results/reports/mlx_tool_contract_harnessing/tables/tool_catalog_schema_field_hints_vs_split_selector_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/tool_catalog_schema_field_hints_vs_split_selector_case_deltas.csv)
- visual schema-field probe deltas vs v1: [`results/reports/mlx_tool_contract_harnessing/tables/tool_catalog_schema_field_hints_vs_role_catalog_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/tool_catalog_schema_field_hints_vs_role_catalog_case_deltas.csv)
- visual schema-field live decision: [`results/reports/mlx_tool_contract_harnessing/tables/tool_catalog_schema_field_hints_live_replay_decision.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/tool_catalog_schema_field_hints_live_replay_decision.csv)
- visual hard-slice latest executed packet: [`results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1)
- visual hard-slice v5-vs-v4 comparison: [`results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1/schema_literal_targets_vs_schema_field_hints`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1/schema_literal_targets_vs_schema_field_hints)
- visual hard-slice exactness diagnostic: [`results/reports/visual_hard_slice_exactness_diagnostic`](../../results/reports/visual_hard_slice_exactness_diagnostic)
- visual hard-slice probe gates: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_probe_gates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_probe_gates.csv)
- visual hard-slice family summary: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_family_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_family_summary.csv)
- visual hard-slice failure modes: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_failure_modes.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_failure_modes.csv)
- visual hard-slice case deltas vs no-directive: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_case_deltas_vs_no_directive.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_case_deltas_vs_no_directive.csv)
- visual hard-slice case deltas vs contracted: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_case_deltas_vs_contracted.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_case_deltas_vs_contracted.csv)
- visual hard-slice probe gate figure: [`results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_probe_gate.svg`](../../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_probe_gate.svg)
- visual hard-slice live replay gate figure: [`results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_live_replay_gate.svg`](../../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_live_replay_gate.svg)
- visual hard-slice stress live replay gate figure: [`results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_stress_live_replay_gate.svg`](../../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_stress_live_replay_gate.svg)
- visual hard-slice H1o control-factorial summary: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_h1o_live_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_h1o_live_replay_summary.csv)
- visual hard-slice H1o control-factorial gate figure: [`results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_h1o_live_replay_gate.svg`](../../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_h1o_live_replay_gate.svg)
- visual hard-slice H1p component-value summary: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_h1p_live_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_h1p_live_replay_summary.csv)
- visual hard-slice H1p component-value gate figure: [`results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_h1p_live_replay_gate.svg`](../../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_h1p_live_replay_gate.svg)
- H1q component-label guard transfer synthesis: [`results/reports/h1q_component_label_guard_transfer_synthesis/report.md`](../../results/reports/h1q_component_label_guard_transfer_synthesis/report.md)
- H1q aggregate table in the main MLX report: [`results/reports/mlx_tool_contract_harnessing/tables/h1q_component_label_guard_aggregate_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/h1q_component_label_guard_aggregate_summary.csv)
- H1q transfer gate figure: [`results/reports/mlx_tool_contract_harnessing/figures/h1q_component_label_guard_transfer_gate.svg`](../../results/reports/mlx_tool_contract_harnessing/figures/h1q_component_label_guard_transfer_gate.svg)
- publication evidence ledger: [`results/reports/publication_evidence_ledger/ledger.md`](../../results/reports/publication_evidence_ledger/ledger.md)
- publication readiness audit: [`results/reports/publication_readiness_audit/publication_readiness_audit.md`](../../results/reports/publication_readiness_audit/publication_readiness_audit.md)
- visual hard-slice design: [`results/reports/visual_hard_slice_design/design.md`](../../results/reports/visual_hard_slice_design/design.md)
- figures: [`results/reports/mlx_tool_contract_harnessing/figures`](../../results/reports/mlx_tool_contract_harnessing/figures)
- current manifest count: `87` tables and `40` figures
- publication evidence ledger: `36` claims, `184` evidence sources, `0` missing sources
- publication readiness audit: `133` checks, `131` blocking checks, `0` blocking failures, status `paper_draft_ready`
- regeneration command:

```bash
uv run python scripts/build_mlx_tool_contract_report.py
uv run python scripts/build_publication_evidence_ledger.py
uv run python scripts/audit_publication_readiness.py
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

## Latest H1r Residual Finding

H1r is now executed. It is the replay-shaped residual packet after H1q, designed to test whether a narrow v12 wording can fix the remaining v11 miss families without reviving the broad v9 component-value regressions.

Evidence:

- profile: `visual_role_catalog_component_residual_guard_v12`
- registry system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard`
- dry-run packet: [`results/tool_probe_replay_packets/20260510T_h1r_component_label_residual_oracle_dry_run_v1`](../../results/tool_probe_replay_packets/20260510T_h1r_component_label_residual_oracle_dry_run_v1)
- no-directive replay: [`results/tool_probe_replay_live/20260510T_h1r_component_label_residual_no_directive_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1r_component_label_residual_no_directive_execute_v1)
- v11 replay: [`results/tool_probe_replay_live/20260510T_h1r_component_label_residual_component_label_guard_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1r_component_label_residual_component_label_guard_execute_v1)
- v12 replay: [`results/tool_probe_replay_live/20260510T_h1r_component_label_residual_component_residual_guard_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1r_component_label_residual_component_residual_guard_execute_v1)
- synthesis: [`results/reports/h1r_component_residual_synthesis/report.md`](../../results/reports/h1r_component_residual_synthesis/report.md)
- packet shape: `6` oracle-valid replay cases across stale-selection component labels, nonstandard component classes (`tag`, `toggle`), and code-label exactness (`alert s92`, `badge c08`)
- result: no-directive `0 / 6` exact and `1 / 6` executor-equivalent; v11 `5 / 6`; v12 `6 / 6`
- interpretation: v12 fixes the v11 `alert s92` residual on H1r, but it is not yet a global default until transferred back across H1n/H1o/H1p

## Latest H1s Component-Residual Transfer Finding

H1s is now executed. It transfer-tested the H1r v12 residual prompt back across the three active visual transfer surfaces before any promotion decision. This is the current best example of why local prompt wins need transfer gates: v12 improves strict exactness but weakens executor-equivalent completion on the broader transfer surface.

Evidence:

- profile: `visual_role_catalog_component_residual_guard_v12`
- registry system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_residual_guard`
- H1n v12 replay: [`results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1n_component_value_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1n_component_value_execute_v1)
- H1o v12 replay: [`results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1o_control_factorial_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1o_control_factorial_execute_v1)
- H1p v12 replay: [`results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1p_component_value_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1s_component_residual_guard_on_h1p_component_value_execute_v1)
- synthesis: [`results/reports/h1s_component_residual_transfer_synthesis/report.md`](../../results/reports/h1s_component_residual_transfer_synthesis/report.md)
- comparisons:
  - H1n v12 versus v11: [`results/tool_probe_replay_live_comparisons/20260510T_h1s_component_residual_guard_h1n_vs_component_label_guard_v1`](../../results/tool_probe_replay_live_comparisons/20260510T_h1s_component_residual_guard_h1n_vs_component_label_guard_v1)
  - H1o v12 versus v11: [`results/tool_probe_replay_live_comparisons/20260510T_h1s_component_residual_guard_h1o_vs_component_label_guard_v1`](../../results/tool_probe_replay_live_comparisons/20260510T_h1s_component_residual_guard_h1o_vs_component_label_guard_v1)
  - H1p v12 versus v11: [`results/tool_probe_replay_live_comparisons/20260510T_h1s_component_residual_guard_h1p_vs_component_label_guard_v1`](../../results/tool_probe_replay_live_comparisons/20260510T_h1s_component_residual_guard_h1p_vs_component_label_guard_v1)

Transfer rates:

- H1n component-value: v12 `5 / 8` exact and executor-equivalent, worse than v11's `6 / 8` exact and `7 / 8` executor-equivalent
- H1o control-factorial: v12 `11 / 12` exact and executor-equivalent, improving strict exactness over v11's `10 / 12` but losing v11's `12 / 12` executor-equivalence ceiling
- H1p component-value: v12 `11 / 12` exact and executor-equivalent, improving over v11's `10 / 12` exact and executor-equivalent
- aggregate across H1n/H1o/H1p: v12 `27 / 32` exact and `27 / 32` executor-equivalent, versus v11 at `26 / 32` exact and `29 / 32` executor-equivalent

Interpretation:

- v12 is a real targeted residual patch: it saturates H1r and improves H1p
- v12 is not the global default: it introduces enough H1n/H1o executor-equivalence loss to be worse than v11 for robust transfer
- the next slice should test conditional routing or prompt-factor isolation: v11 general component-label guard by default, v12 residual wording only when code labels or nonstandard component classes are present

## H1t Conditional Residual-Route Finding

H1t is now live-executed and rejected by an early-stop gate. It turned the H1s verdict into a testable prompt-factor hypothesis:

- profile: `visual_role_catalog_conditional_residual_route_v13`
- registry system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_conditional_residual_route`
- default behavior: preserve v11's narrow component-label guard
- conditional behavior: add v12-style residual handling only when the requested target has a code suffix, a nonstandard component class (`tag`, `toggle`, `switch`), or a field target with stale/old/previous selection text nearby
- explicit anti-overfit rule: do not add residual handling for ordinary `pill`, `badge`, `chip`, or `tile` targets unless a route condition is present
- v13 H1r replay: [`results/tool_probe_replay_live/20260510T_h1t_conditional_residual_route_on_h1r_component_residual_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1t_conditional_residual_route_on_h1r_component_residual_execute_v1)
- synthesis: [`results/reports/h1t_conditional_residual_route_synthesis/report.md`](../../results/reports/h1t_conditional_residual_route_synthesis/report.md)
- result: v13 reaches `3 / 6` exact and executor-equivalent on H1r, below v11's `5 / 6` and v12's `6 / 6`
- failures: `state tag`, `mode toggle`, and `alert s92` all remain argument mismatches
- decision: reject before broader H1n/H1o/H1p transfer because the conditional route failed to preserve the local H1r win
- verification: `uv run pytest tests/test_prompt_contracts.py tests/test_knowledge_work_h1.py::test_h1t_conditional_residual_route_registry_row_preserves_catalog_profile -q`

Next execution step:

- do not run v13 on H1n/H1o/H1p
- next prompt-factor attempt should be more explicit and/or split into independent route bits, because compact conditional prose did not trigger the nonstandard class and code-label behavior

H1u gate:

- `visual_role_catalog_nonstandard_component_class_guard_v14`: targets tag/toggle/switch cases that collapsed into displayed values under v13
- `visual_role_catalog_code_label_exact_guard_v15`: targets code-label exactness and negated neighboring controls such as the `alert s92` miss
- registry systems:
  - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_nonstandard_component_class_guard`
  - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_code_label_exact_guard`
- H1r v14 replay: [`results/tool_probe_replay_live/20260510T_h1u_nonstandard_component_class_guard_on_h1r_component_residual_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1u_nonstandard_component_class_guard_on_h1r_component_residual_execute_v1)
- H1r v15 replay: [`results/tool_probe_replay_live/20260510T_h1u_code_label_exact_guard_on_h1r_component_residual_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1u_code_label_exact_guard_on_h1r_component_residual_execute_v1)
- synthesis: [`results/reports/h1u_split_factor_synthesis/report.md`](../../results/reports/h1u_split_factor_synthesis/report.md)
- result: v14 reaches `5 / 6`, fixing tag/toggle value collapse but still missing `alert s92`; v15 reaches `6 / 6`, tying v12 with narrower code-label exactness wording
- H1v transfer synthesis: [`results/reports/h1v_code_label_exact_transfer_synthesis/report.md`](../../results/reports/h1v_code_label_exact_transfer_synthesis/report.md)
- H1v H1n v15 replay: [`results/tool_probe_replay_live/20260510T_h1v_code_label_exact_guard_on_h1n_component_value_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1v_code_label_exact_guard_on_h1n_component_value_execute_v1)
- H1v H1o v15 replay: [`results/tool_probe_replay_live/20260510T_h1v_code_label_exact_guard_on_h1o_control_factorial_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1v_code_label_exact_guard_on_h1o_control_factorial_execute_v1)
- H1v H1p v15 replay: [`results/tool_probe_replay_live/20260510T_h1v_code_label_exact_guard_on_h1p_component_value_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1v_code_label_exact_guard_on_h1p_component_value_execute_v1)
- H1v verdict: reject v15 as a global promotion; it transfers at `25 / 32` exact and `25 / 32` executor-equivalent, below v11's `29 / 32` executor-equivalent and v12's `27 / 32` exact totals
- next execution: keep v11 as the transfer-stable default, treat v15 as a local code-label repair, and build H1w around the remaining v15/v11 shared residuals

H1w scaffold:

- packet: [`results/tool_probe_replay_packets/20260510T_h1w_residual_overlap_oracle_dry_run_v1`](../../results/tool_probe_replay_packets/20260510T_h1w_residual_overlap_oracle_dry_run_v1)
- suite: `h1w_residual_overlap_v13`
- shape: `8` oracle-valid cases, balanced across stale field-routing, nonstandard component classes, surface component-value collapse, and activation/no-call residuals
- purpose: break the now-saturated top-line transfer story by concentrating on the residual mechanisms that survive v11/v12/v15
- live replays:
  - no-directive: [`results/tool_probe_replay_live/20260510T_h1w_residual_overlap_no_directive_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1w_residual_overlap_no_directive_execute_v1)
  - v11 component-label guard: [`results/tool_probe_replay_live/20260510T_h1w_residual_overlap_component_label_guard_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1w_residual_overlap_component_label_guard_execute_v1)
  - v12 component-residual guard: [`results/tool_probe_replay_live/20260510T_h1w_residual_overlap_component_residual_guard_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1w_residual_overlap_component_residual_guard_execute_v1)
  - v15 code-label exact guard: [`results/tool_probe_replay_live/20260510T_h1w_residual_overlap_code_label_exact_guard_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1w_residual_overlap_code_label_exact_guard_execute_v1)
- synthesis: [`results/reports/h1w_residual_overlap_synthesis/report.md`](../../results/reports/h1w_residual_overlap_synthesis/report.md)
- result: no-directive is `0 / 8`, v11 is `8 / 8`, v12 is `7 / 8`, and v15 is `6 / 8`
- interpretation: H1w is a strong controller-dependence probe but not a v11 breaker; the next hard slice must combine oblique labels, stale selections, and repeated values in the same case to stress v11 directly
- H1x packet: [`results/tool_probe_replay_packets/20260510T_h1x_v11_breaker_oracle_dry_run_v1`](../../results/tool_probe_replay_packets/20260510T_h1x_v11_breaker_oracle_dry_run_v1)
- H1x shape: `8` oracle-valid oblique-label cases over stale fields, surface values, nonstandard classes, and activation/no-call contexts
- live replays:
  - no-directive: [`results/tool_probe_replay_live/20260510T_h1x_v11_breaker_no_directive_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1x_v11_breaker_no_directive_execute_v1)
  - v11 component-label guard: [`results/tool_probe_replay_live/20260510T_h1x_v11_breaker_component_label_guard_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1x_v11_breaker_component_label_guard_execute_v1)
  - v12 component-residual guard: [`results/tool_probe_replay_live/20260510T_h1x_v11_breaker_component_residual_guard_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1x_v11_breaker_component_residual_guard_execute_v1)
  - v15 code-label exact guard: [`results/tool_probe_replay_live/20260510T_h1x_v11_breaker_code_label_exact_guard_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1x_v11_breaker_code_label_exact_guard_execute_v1)
- synthesis: [`results/reports/h1x_v11_breaker_synthesis/report.md`](../../results/reports/h1x_v11_breaker_synthesis/report.md)
- result: no-directive `2 / 8`, v11 `7 / 8`, v12 `8 / 8`, v15 `6 / 8` exact and `7 / 8` executor-equivalent
- interpretation: H1x is the first focused post-H1w packet that breaks v11 saturation; v12 is the local winner, but H1s still blocks global v12 promotion because of broader executor-equivalence loss
- completed follow-up: H1y/H1z prompt-only routed residual tests did not beat v12, and H2a controller-side stale-selection mediation is now the current local winner at `8 / 10`
- next execution: transfer-test H2a across H1n/H1o/H1p/H1x, then isolate the remaining H1y argument-alias/code-label residuals

## Latest H1q Component-Label Guard Transfer Finding

H1q is the current sharpest controller-profile transfer result. It was built to resolve the H1p/H1o/H1n tension: broad v9 component-value guidance wins locally on H1p, ties on H1o, and regresses on the older H1n component-value slice. The new `visual_role_catalog_component_label_guard_v11` narrows the wording to copying requested role-plus-component labels instead of selecting displayed values.

Evidence:

- profile: `visual_role_catalog_component_label_guard_v11`
- registry system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_component_label_guard`
- H1n replay: [`results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1n_component_value_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1n_component_value_execute_v1)
- H1o replay: [`results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1o_control_factorial_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1o_control_factorial_execute_v1)
- H1p replay: [`results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1p_component_value_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1q_component_label_guard_on_h1p_component_value_execute_v1)
- synthesis: [`results/reports/h1q_component_label_guard_transfer_synthesis/report.md`](../../results/reports/h1q_component_label_guard_transfer_synthesis/report.md)

Transfer rates:

- H1n component-value: v11 `6 / 8` exact and `7 / 8` executor-equivalent, versus v9 at `4 / 8` and `4 / 8`
- H1o control-factorial: v11 `10 / 12` exact and `12 / 12` executor-equivalent, a new H1o executor-equivalence ceiling
- H1p component-value: v11 `10 / 12` exact and `10 / 12` executor-equivalent, tying v9 strict exactness but trailing v9 executor-equivalence (`11 / 12`)
- aggregate across H1n/H1o/H1p: v11 `26 / 32` exact and `29 / 32` executor-equivalent, above v9 at `23 / 32` and `25 / 32`

Interpretation:

- v11 is the strongest transfer candidate so far, not a global default
- it repairs the H1n broad-v9 regression pattern while preserving the H1p strict win
- it still introduces or preserves misses on `component_value_owner_field_stale_selection_decoy`, `h1p_compact_state_tag_log_value_decoy`, and `h1p_surface_mode_toggle_note_value_decoy`
- the next slice should isolate owner-field stale-selection routing and nonstandard component classes (`tag`, `toggle`) without adding broad prose that harms H1p/H1o

## Previous H1p Component-Value Holdout Finding

H1p was the component-only holdout that exposed v9's local activation domain. It was built directly from the H1o conclusion that component label versus displayed value was the remaining residue, but it removes the activation/no-call wording that could confound the mechanism.

Evidence:

- packet: [`results/tool_probe_replay_packets/20260510T_h1p_component_value_holdout_oracle_dry_run_v1`](../../results/tool_probe_replay_packets/20260510T_h1p_component_value_holdout_oracle_dry_run_v1)
- no-directive live baseline: [`results/tool_probe_replay_live/20260510T_h1p_component_value_no_directive_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1p_component_value_no_directive_execute_v1)
- argument-hints live packet: [`results/tool_probe_replay_live/20260510T_h1p_component_value_argument_hints_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1p_component_value_argument_hints_execute_v1)
- hybrid-label live packet: [`results/tool_probe_replay_live/20260510T_h1p_component_value_hybrid_label_guard_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1p_component_value_hybrid_label_guard_execute_v1)
- component-value guard live packet: [`results/tool_probe_replay_live/20260510T_h1p_component_value_component_value_guard_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1p_component_value_component_value_guard_execute_v1)
- diagnostic: [`results/reports/visual_h1p_component_value_diagnostic/diagnostic.md`](../../results/reports/visual_h1p_component_value_diagnostic/diagnostic.md)
- report table: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_h1p_live_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_h1p_live_replay_summary.csv)
- report figure: [`results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_h1p_live_replay_gate.svg`](../../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_h1p_live_replay_gate.svg)

Current H1p live replay rates:

- no-directive MLX: `0 / 12` exact and executor-equivalent
- argument hints v2: `6 / 12` exact and executor-equivalent
- no-call control rescue v10: `6 / 12` exact and executor-equivalent
- hybrid label guard v8: `9 / 12` exact and `10 / 12` executor-equivalent
- component-value guard v9: `10 / 12` exact and `11 / 12` executor-equivalent

Interpretation:

- H1p successfully broke top-line saturation; no-directive collapses when the packet is pure component/value ambiguity
- v9 is no longer simply "bad"; it has a real activation domain on component-only surfaces
- v9 is still not globally promoted because H1n showed the same broad prose can regress passable pill/chip targets, and H1o only tied argument hints on a mixed mechanism packet
- the next research question is transfer and narrowing, not another broad profile: build H1q around component-only guard variants and test them against H1p, H1o, and the earlier H1n component-value cases

## Previous H1o Control-Factorial Finding

H1o was the mechanism-split predecessor to H1p. It was built after the H1n component-value/v10 transfer loop to stop asking whether a single profile "wins" and instead separate the active mechanisms:

- activation/no-call rescue
- code-like label and negation preservation
- component label versus component value disambiguation

Evidence:

- packet: [`results/tool_probe_replay_packets/20260510T_h1o_control_factorial_oracle_dry_run_v1`](../../results/tool_probe_replay_packets/20260510T_h1o_control_factorial_oracle_dry_run_v1)
- no-directive live baseline: [`results/tool_probe_replay_live/20260510T_h1o_control_factorial_no_directive_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1o_control_factorial_no_directive_execute_v1)
- argument-hints live packet: [`results/tool_probe_replay_live/20260510T_h1o_control_factorial_argument_hints_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1o_control_factorial_argument_hints_execute_v1)
- component-value guard live packet: [`results/tool_probe_replay_live/20260510T_h1o_control_factorial_component_value_guard_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1o_control_factorial_component_value_guard_execute_v1)
- diagnostic: [`results/reports/visual_h1o_control_factorial_diagnostic/diagnostic.md`](../../results/reports/visual_h1o_control_factorial_diagnostic/diagnostic.md)
- synthesis: [`results/reports/h1o_control_factorial_synthesis/report.md`](../../results/reports/h1o_control_factorial_synthesis/report.md)
- report table: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_h1o_live_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_h1o_live_replay_summary.csv)

Current H1o live replay rates:

- no-directive MLX: `5 / 12` exact and `6 / 12` executor-equivalent
- argument hints v2: `9 / 12` exact and `10 / 12` executor-equivalent
- component-value guard v9: `9 / 12` exact and `10 / 12` executor-equivalent
- hybrid label guard v8: `8 / 12` exact and `10 / 12` executor-equivalent
- oblique code guard v7: `8 / 12` exact and `9 / 12` executor-equivalent
- no-call control rescue v10: `7 / 12` exact and `8 / 12` executor-equivalent

Mechanism split:

- activation/no-call is not the remaining bottleneck: no-directive is already `4 / 4` exact on that family
- no-call rescue v10 is not a global fix: it regresses `h1o_activation_error_banner_previous_region_decoy`
- code/negation is repairable: best rows reach `3 / 4` exact and `4 / 4` executor-equivalent
- component/value remains the hard residue: best rows reach only `2 / 4` exact and executor-equivalent
- argument hints is still the conservative default on mixed mechanisms, while H1p now shows component-value guard has a local component-only domain

Next research move:

- run H1q as a transfer/narrowing slice over H1p, H1o, and H1n component-value cases
- avoid global v9 promotion until narrower component-only wording survives the H1n/H1o counterevidence
- preserve argument hints as the conservative mixed-mechanism default

## Previous H1n Component-Value Finding

The current H1n visual replay frontier is now the component-role/value holdout that followed the residual `state pill` miss.

- component-value packet: [`results/tool_probe_replay_packets/20260510T_visual_hard_slice_component_value_oracle_dry_run_v1`](../../results/tool_probe_replay_packets/20260510T_visual_hard_slice_component_value_oracle_dry_run_v1)
- v10 live packet: [`results/tool_probe_replay_live/20260510T_h1n_component_value_no_call_control_rescue_execute_v1`](../../results/tool_probe_replay_live/20260510T_h1n_component_value_no_call_control_rescue_execute_v1)
- v10 transfer synthesis: [`results/reports/h1n_no_call_rescue_transfer_synthesis/report.md`](../../results/reports/h1n_no_call_rescue_transfer_synthesis/report.md)
- diagnostic: [`results/reports/visual_component_value_diagnostic/diagnostic.md`](../../results/reports/visual_component_value_diagnostic/diagnostic.md)
- report table: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_component_value_live_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_component_value_live_replay_summary.csv)
- predecessor packet: [`results/tool_probe_replay_packets/20260510T_visual_hard_slice_live_stress_alias_transfer_residual_oracle_dry_run_v1`](../../results/tool_probe_replay_packets/20260510T_visual_hard_slice_live_stress_alias_transfer_residual_oracle_dry_run_v1)

Current component-value live replay rates:

- contracted/default MLX: `1 / 8` exact and executor-equivalent
- no-directive MLX: `5 / 8` exact and `6 / 8` executor-equivalent
- no-call control rescue v10: `7 / 8` exact and `8 / 8` executor-equivalent
- argument hints v2: `6 / 8` exact and `7 / 8` executor-equivalent
- hybrid label guard v8: `6 / 8` exact and `7 / 8` executor-equivalent
- oblique code guard v7: `5 / 8` exact and executor-equivalent
- component-value guard v9: `4 / 8` exact and executor-equivalent
- schema-field hints v4: `3 / 8` exact and `4 / 8` executor-equivalent
- oblique code hints v6: `2 / 8` exact and executor-equivalent

Interpretation:

- v9 component-value guard is negative evidence, not a profile to promote
- v10 no-call control rescue is the current component-value upper bound
- the useful gains are no-call rescues on `status badge` and `owner field` without broad component-value prose
- transfer synthesis says v10 is a scoped activation improvement: `22 / 30` exact and `25 / 30` executor-equivalent across four H1n packets, versus no-directive at `11 / 30` and `12 / 30`, but behind incumbents at `25 / 30` and `26 / 30`
- the harmful pattern is broad component-value prose causing argument mismatches on already-passable pill/chip targets
- H1o completed that split and showed component/value, not activation/no-call, is now the residual mechanism to stress

## Previous H1n Residual Finding

The current H1n visual replay frontier is now the residual hybrid-label holdout, not the older post-repair packet alone.

- residual packet: [`results/tool_probe_replay_packets/20260510T_visual_hard_slice_live_stress_alias_transfer_residual_oracle_dry_run_v1`](../../results/tool_probe_replay_packets/20260510T_visual_hard_slice_live_stress_alias_transfer_residual_oracle_dry_run_v1)
- diagnostic: [`results/reports/visual_alias_transfer_residual_diagnostic/diagnostic.md`](../../results/reports/visual_alias_transfer_residual_diagnostic/diagnostic.md)
- report table: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_residual_live_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_residual_live_replay_summary.csv)
- predecessor packet: [`results/tool_probe_replay_packets/20260510T_visual_hard_slice_live_stress_alias_transfer_post_repair_oracle_dry_run_v1`](../../results/tool_probe_replay_packets/20260510T_visual_hard_slice_live_stress_alias_transfer_post_repair_oracle_dry_run_v1)

Current residual live replay rates:

- contracted/default MLX: `2 / 8` exact and executor-equivalent
- no-directive MLX: `4 / 8`
- argument hints v2: `5 / 8` exact and `7 / 8` executor-equivalent
- oblique code hints v6: `6 / 8`
- oblique code guard v7: `6 / 8` exact and `7 / 8` executor-equivalent
- hybrid label guard v8: `7 / 8` exact and executor-equivalent

Interpretation:

- v8 hybrid label guard is the current strict-selector upper bound on the residual holdout
- the v8 gain over argument hints and v7 is mostly exactness, not broader executor-equivalence
- the remaining miss is `state pill`, where the model still confuses the component label with the state/content value
- next research move should build a component-role/value ambiguity micro-slice before any broad packaged-workflow promotion
