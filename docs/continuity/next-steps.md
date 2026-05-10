# Next Steps

## Immediate

### 1. Pivot the main live-testing surface to a CLI-first sandboxed operator harness

Current state:

- the runtime/session/approval/artifact substrate already exists
- the repo can already run live local sessions
- the React harness exists, but it is no longer the main next workstream
- the higher-value need is a research-grade terminal operator surface with real sandboxing
- first CLI sandbox scaffold exists:
  - sessions/traces carry sandbox metadata
  - runtime outputs and native artifacts write under per-session sandbox roots
  - `moonie-agent live` and `moonie-agent attach` exist as Rich terminal entrypoints
  - live-web dry-run holds are recorded as `sandbox_policy_block` events and session/trace metadata
  - `moonie-agent attach --action approve|deny|resume|retry|quit` can operate on a session from the terminal
  - `moonie-agent inspect` can inspect sandbox roots, artifacts, policy blocks, and summary paths
  - a real `mlx_gemma4_e2b_reasoner_only` CLI smoke completed on `executive_visual_dashboard_review`
  - a second real `mlx_gemma4_e2b_reasoner_only` CLI smoke completed on `20260506T220039380037Z_executive_visual_dashboard_review`
  - `moonie-agent inspect --target scorecard` now exposes scorecard metrics and per-task controller repair findings
  - `scripts/run_runtime_live_smoke_packet.py` now writes compact tracked packet summaries for packaged-workflow runtime smokes
  - runtime live-smoke packets support `--repeat` and write workflow-level summaries plus repeat-indexed findings/policy sidecars
  - first tracked packet: [`results/runtime_live_smoke_packets/20260506T_runtime_live_smoke_mlx_v2_runtime_live_smoke_packet`](../../results/runtime_live_smoke_packets/20260506T_runtime_live_smoke_mlx_v2_runtime_live_smoke_packet)
  - approval/smoke trio packet: [`results/runtime_live_smoke_packets/20260506T_runtime_live_smoke_mlx_trio_v2_runtime_live_smoke_packet`](../../results/runtime_live_smoke_packets/20260506T_runtime_live_smoke_mlx_trio_v2_runtime_live_smoke_packet)
  - live-web policy packet: [`results/runtime_live_smoke_packets/20260506T_runtime_live_web_policy_mlx_v2_runtime_live_smoke_packet`](../../results/runtime_live_smoke_packets/20260506T_runtime_live_web_policy_mlx_v2_runtime_live_smoke_packet)
  - repeated live-web H1c-overlap packet: [`results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet`](../../results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet)
  - [`scripts/analyze_runtime_live_smoke_packet.py`](../../scripts/analyze_runtime_live_smoke_packet.py) now writes repair-family, policy-family, and workflow-stability summaries for runtime packets
  - latest analyzer result: `stable_repair_family_count = 4`, `stable_policy_block_family_count = 7`
  - H1 primary runs now pass `--pipeline-name monolith` for `local_reasoner` systems so `mlx_gemma4_e2b_reasoner_only` matches `moonie-agent live`
  - corrected H1c MLX monolith primary packet: [`results/knowledge_work_h1_slice/20260506T_h1c_mlx_live_primary_monolith_v1_knowledge_work_h1c_live_policy_controller_dependence_v1`](../../results/knowledge_work_h1_slice/20260506T_h1c_mlx_live_primary_monolith_v1_knowledge_work_h1c_live_policy_controller_dependence_v1)
  - local MLX monolith helper-ablation profiles exist for `no_controller_repair`, `no_controller_fallback`, and `no_argument_repair`
  - H1c MLX monolith helper ablation: [`results/knowledge_work_h1_slice/20260506T_h1c_mlx_monolith_helpers_v1_knowledge_work_h1c_live_policy_controller_dependence_v1`](../../results/knowledge_work_h1_slice/20260506T_h1c_mlx_monolith_helpers_v1_knowledge_work_h1c_live_policy_controller_dependence_v1)
  - H1d candidate brief: [`docs/continuity/h1d-candidates.md`](./h1d-candidates.md)
  - H1d config scaffold: [`configs/knowledge_work_h1d_slice.yaml`](../../configs/knowledge_work_h1d_slice.yaml)
  - first named H1d packet: [`results/knowledge_work_h1_slice/20260506T_h1d_mlx_controller_stress_v1_knowledge_work_h1d_mlx_monolith_controller_stress_v1`](../../results/knowledge_work_h1_slice/20260506T_h1d_mlx_controller_stress_v1_knowledge_work_h1d_mlx_monolith_controller_stress_v1)
  - H1c live-policy scaffold exists:
    - [`configs/knowledge_work_h1c_slice.yaml`](../../configs/knowledge_work_h1c_slice.yaml)
    - [`docs/continuity/h1c-slice.md`](./h1c-slice.md)
  - `moonie-agent gemini-baseline` can prepare a dry-run Gemini CLI baseline packet for packaged workflows

Next implementation moves:

- treat H1h as the current causal restart point for MLX harnessing:
  - contracted MLX is controller-clean on all `10` live workflow families
  - no-directive MLX only stays top-line clean through controller help
  - disabling controller repair under no-directive drops readiness to `0.73801`
- use H1h workflow-family attribution before changing the controller again
- keep H1g as a negative result: visual rescue, intent priority, and deterministic visual follow-on do not carry the compact live slice under the directive
- use the completed Gemini CLI dry-run packet as the external-reference baseline for the H1h workflow family set; rerun with real execution only when the binary/run environment is intentionally part of the comparison
- keep using live CLI scorecard and policy inspection as the active operator proof path
- treat replay-shaped CLI-live packets as the active discriminator when packaged workflows are saturated
- use [`scripts/build_visual_hard_slice_replay_packet.py`](../../scripts/build_visual_hard_slice_replay_packet.py) to preserve hard-slice visual cases for `moonie-agent replay-live`
- current replay-shaped matrix: contracted MLX is `2 / 2` strict/executor-equivalent; role catalog v1 and argument hints v2 are `1 / 2`; schema-field hints v4 is `1 / 2` strict and `2 / 2` executor-equivalent; schema target literals v5 is `0 / 2` strict and `1 / 2` executor-equivalent with a wrong-tool stale-selection miss
- current replay-shaped stress matrix: no-directive MLX is `2 / 4` strict and `3 / 4` executor-equivalent; contracted MLX is `4 / 4`; schema-field hints v4 and schema target literals v5 are `2 / 4` strict and `4 / 4` executor-equivalent
- alias-repeat matrix: no-directive MLX is `2 / 8` strict and `5 / 8` executor-equivalent; contracted MLX is `7 / 8` strict and `8 / 8` executor-equivalent; schema-field hints v4 is `2 / 8` strict and `7 / 8` executor-equivalent; schema target literals v5 is `3 / 8` strict and `8 / 8` executor-equivalent
- H1m packaged promotion was executed and saturated:
  - config: [`configs/knowledge_work_h1m_slice.yaml`](../../configs/knowledge_work_h1m_slice.yaml)
  - brief: [`docs/continuity/h1m-slice.md`](./h1m-slice.md)
  - candidate packet: `mlx_visual_alias_repeat_packaged_candidates`
  - executed result: all six rows tie at readiness `0.87783`, strict `0.75`, recovered `0.667`, raw clean `1.0`, and zero repair/fallback/argument-repair burden
  - guardrail: do not run H1m helper ablation until a live surface separates rows
- packaged replay gap diagnostic:
  - diagnostic: [`results/reports/packaged_replay_gap_diagnostic/diagnostic.md`](../../results/reports/packaged_replay_gap_diagnostic/diagnostic.md)
  - result: H1l and H1m both show positive replay gains followed by zero packaged readiness/strict span
  - implication: packaging is part of the benchmark contract; a packaged workflow can be valid live scaffolding while still being too staged for a mechanism-level visual claim
- H1n alias-transfer replay result:
  - brief: [`docs/continuity/h1n-slice.md`](./h1n-slice.md)
  - packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_dry_run_v1`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_dry_run_v1)
  - oracle v2 packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_dry_run_v2`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_oracle_dry_run_v2)
  - report table: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_alias_transfer_live_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_alias_transfer_live_replay_summary.csv)
  - oracle report table: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_alias_transfer_oracle_live_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_alias_transfer_oracle_live_replay_summary.csv)
  - contract-split diagnostic: [`results/reports/h1n_alias_transfer_contract_split/diagnostic.md`](../../results/reports/h1n_alias_transfer_contract_split/diagnostic.md)
  - oracle diagnostic: [`results/reports/visual_alias_transfer_oracle_diagnostic/diagnostic.md`](../../results/reports/visual_alias_transfer_oracle_diagnostic/diagnostic.md)
  - legacy v1 result: argument hints v2 is the transfer executor-equivalence winner at `6 / 6`; contracted MLX is the strict winner at `5 / 6`; no-directive is `0 / 6` strict and `2 / 6` executor-equivalent
- H1n contract finding:
  - `5 / 6` generated expected calls fail the packet's own expected-execution oracle
  - contracted has `4` exact-but-not-executor rows
  - strict H1n v1 exactness should be treated as heuristic planner-call fidelity, not target success
- H1n oracle v2 result:
  - replay-live now preserves serialized packet expected calls
  - no-directive is `2 / 6`; contracted is `1 / 6`; role catalog v1 is `3 / 6`; argument hints v2 is `5 / 6` strict and `6 / 6` executor-equivalent; schema-field hints v4 is `2 / 6`; schema target literals v5 is `4 / 6`
  - interpretation: argument hints v2 is the clean H1n winner; schema target literals are second; contracted prompting is not an oracle-transfer upper bound
- H1n helper-ablation setup:
  - registry rows now exist for argument hints with controller repair disabled, controller fallback disabled, and argument repair disabled
  - executed result: all three rows preserve argument hints at `5 / 6` strict and `6 / 6` executor-equivalent
  - diagnostic: [`results/reports/h1n_oracle_helper_ablation/diagnostic.md`](../../results/reports/h1n_oracle_helper_ablation/diagnostic.md)
  - interpretation: this H1n argument-hints gain is not explained by controller repair, controller fallback, or argument repair on the oracle replay slice
- H1n repeat setup:
  - repeat packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_repeat_oracle_dry_run_v1`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_repeat_oracle_dry_run_v1)
  - suite: `alias_transfer_repeat_v4`
  - design: six fresh labels/decoys with oracle expected calls derived from target region labels
  - executed result: no-directive `2 / 6`; contracted `0 / 6`; role catalog v1 and schema-field hints v4 `4 / 6`; argument hints v2 and schema target literals v5 `5 / 6` strict and `6 / 6` executor-equivalent
  - diagnostic: [`results/reports/visual_alias_transfer_repeat_diagnostic/diagnostic.md`](../../results/reports/visual_alias_transfer_repeat_diagnostic/diagnostic.md)
- H1n oracle transfer synthesis:
  - report: [`results/reports/h1n_oracle_transfer_synthesis/report.md`](../../results/reports/h1n_oracle_transfer_synthesis/report.md)
  - script: [`scripts/build_h1n_oracle_transfer_synthesis.py`](../../scripts/build_h1n_oracle_transfer_synthesis.py)
  - result: across two oracle packets, argument hints is executor-equivalent in both packets at `6 / 6` and `6 / 6`; schema target literals rises from `4 / 6` to `6 / 6`; contracted is `1 / 6` then `0 / 6`; helper ablation preserves argument hints with zero exact/executor-equivalence deltas
  - interpretation: the next visual question is no longer whether catalog-profile transfer exists. It is whether argument hints or schema target literals is more robust under less staged visual operation.
- H1n oblique-label oracle packet:
  - packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_oblique_oracle_dry_run_v1`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_oblique_oracle_dry_run_v1)
  - suite: `alias_transfer_oblique_v5`
  - design: six held-out labels use code-like visible target tokens such as `node q17`, `badge m88`, `chip z33`, `field e19`, and `alert p55`, with semantic decoys nearby
  - result: no-directive `0 / 6`; contracted `1 / 6`; role catalog v1 `2 / 6`; argument hints v2 `4 / 6`; schema-field hints v4 `3 / 6`; schema target literals v5 `0 / 6`
  - diagnostic: [`results/reports/visual_alias_transfer_oblique_diagnostic/diagnostic.md`](../../results/reports/visual_alias_transfer_oblique_diagnostic/diagnostic.md)
  - miss analysis: [`results/reports/h1n_oblique_miss_analysis/diagnostic.md`](../../results/reports/h1n_oblique_miss_analysis/diagnostic.md)
  - interpretation: the oblique packet breaks the argument-hints/schema-literal tie in favor of argument hints, with schema-field hints second; target-literal wording is brittle when visible labels are code-like and decoys repeat the semantic content
- next candidate ready to execute:
  - system: `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_oblique_code_hints`
  - profile: `visual_role_catalog_oblique_code_hints_v6`
  - purpose: test a narrow code-suffix/negated-decoy repair while preserving the four argument-hints wins
- next replay-shaped target: execute the oblique-code profile against the oblique packet, compare it to argument hints, and reject it unless it improves the two misses without losing existing wins
- later, consider a true keyboard TUI after the command-driven operator loop is useful
- keep hardening sandbox policies around file writes and external process/network actions
- keep packaged workflows as the only live entrypoint in v1
- preserve benchmark-backed traces, artifacts, and scorecards for every live run

Success condition:

- a person can safely launch and watch a real local Gemma MLX run from CLI, approve or resume when needed, and inspect the run live without leaving the terminal

### 2. Use H1h as the current MLX no-directive restart point

The H1f expansion is complete. H1h is now the highest-value benchmark packet for local Gemma tool-contract work.

Current evidence:

- current cross-packet report:
  - [`docs/reports/mlx-tool-contract-harnessing.md`](../reports/mlx-tool-contract-harnessing.md)
  - [`results/reports/mlx_tool_contract_harnessing/report.md`](../../results/reports/mlx_tool_contract_harnessing/report.md)
- H1f compact live packet:
  - [`results/knowledge_work_h1_slice/20260506T_h1f_mlx_no_directive_v1_knowledge_work_h1f_mlx_tool_contract_ablation_v1`](../../results/knowledge_work_h1_slice/20260506T_h1f_mlx_no_directive_v1_knowledge_work_h1f_mlx_tool_contract_ablation_v1)
- H1h full live packet:
  - [`results/knowledge_work_h1_slice/20260507T_h1h_mlx_full_no_directive_v1_knowledge_work_h1h_mlx_full_tool_contract_ablation_v1`](../../results/knowledge_work_h1_slice/20260507T_h1h_mlx_full_no_directive_v1_knowledge_work_h1h_mlx_full_tool_contract_ablation_v1)
- H1h contracted MLX row:
  - readiness `0.96891`
  - strict/recovered `1.0 / 1.0`
  - repair/fallback/argument repair `0.0 / 0.0 / 0.0`
  - raw clean `1.0`
- H1h no-directive MLX with helpers:
  - readiness `0.96891`
  - repair/fallback/argument repair `0.70 / 0.25 / 0.45`
  - raw clean `0.30`
- H1h no-directive helper removals:
  - `no_controller_repair = 0.73801`
  - `no_controller_fallback = 0.89598`
  - `no_argument_repair = 0.83016`
- H1h workflow-family attribution:
  - [`workflow_family_summary.json`](../../results/knowledge_work_h1_slice/20260507T_h1h_mlx_full_no_directive_v1_knowledge_work_h1h_mlx_full_tool_contract_ablation_v1/workflow_family_summary.json)
  - [`workflow_family_failures.csv`](../../results/knowledge_work_h1_slice/20260507T_h1h_mlx_full_no_directive_v1_knowledge_work_h1h_mlx_full_tool_contract_ablation_v1/workflow_family_failures.csv)
- H1h Gemini CLI dry-run baseline:
  - [`results/gemini_cli/20260507T_h1h_gemini_cli_dry_run_baseline_v1`](../../results/gemini_cli/20260507T_h1h_gemini_cli_dry_run_baseline_v1)
- MLX no-directive tool probe:
  - [`results/tool_directive_probe/20260507T_mlx_no_directive_probe_v1`](../../results/tool_directive_probe/20260507T_mlx_no_directive_probe_v1)
  - no-directive exact match `0 / 8`
  - no-directive executable visual match `0 / 1`
  - delta against contracted MLX probe: exact-rate `-0.875`, executable-rate `-1.0`
- H1i compact worst-family packet:
  - [`configs/knowledge_work_h1i_slice.yaml`](../../configs/knowledge_work_h1i_slice.yaml)
  - [`results/knowledge_work_h1_slice/20260507T_h1i_mlx_worst_no_directive_v1_knowledge_work_h1i_mlx_worst_family_tool_contract_v1`](../../results/knowledge_work_h1_slice/20260507T_h1i_mlx_worst_no_directive_v1_knowledge_work_h1i_mlx_worst_family_tool_contract_v1)
  - no-directive with helpers stays top-line clean but uses repair/fallback/argument repair `1.00 / 0.50 / 0.50` and raw clean `0.00`
  - no-directive + no controller repair drops to readiness `0.64697`
  - no-directive + no controller fallback drops to readiness `0.83125`
- prompt-contract candidate queue:
  - registry systems:
    - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_anchor`
    - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_literal_guard`
    - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_tool_required`
  - generated table: [`results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_candidates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_candidates.csv)
  - dry-run probe packet: [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_dry_run_v2`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_dry_run_v2)
  - v2 packet schema writes `no_directive_probe_dir`, `delta_exact_vs_no_directive`, and `probe_gate` fields for executed candidate comparisons
  - executed probe packet: [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1)
  - executed summary: [`candidate_gate_summary.md`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1/candidate_gate_summary.md)
  - first-gate result:
    - `schema_anchor_v1`: weak exact gain, exact `0.125`, delta exact vs no-directive `+0.125`
    - `literal_argument_guard_v1`: visual executable gain only, exact `0.0`, executable `1.0`
    - `tool_required_parallel_v1`: visual executable gain only, exact `0.0`, executable `1.0`, still dominated by `no_tool_call`
  - H1i graduation packet id: `mlx_prompt_contract_candidates`
  - H1i mechanism packet: [`results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_v1_knowledge_work_ablation_packet)
  - H1i result: all five rows are clean at readiness `0.97710`, repair/fallback/argument repair `0.0 / 0.0 / 0.0`, raw clean `1.0`, with `0` failure candidates
  - H1/H1i packet runners now support `--repeat`, so second-stage saturation checks can run the same packaged workflow families multiple times per row with repeat count written into manifests and summaries
  - H1i repeat3 packet: [`results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_repeat3_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_repeat3_v1_knowledge_work_ablation_packet)
  - H1i repeat3 result: `60` traces, all five rows still readiness `0.97710`, strict/recovered `1.0 / 1.0`, repair/fallback/argument repair `0.0 / 0.0 / 0.0`, raw clean `1.0`, with `0` notes and `0` failure candidates
  - design guardrail: candidates add generic interface contract wording only; they must not embed the exact next planned call

What remains:

- treat the H1i candidate packet as saturated/non-discriminating after the probe gate
- treat repeated H1i as a completed negative result; it confirms this packet is too deterministic for candidate validation
- define a harder probe-derived live packet before spending another full H1h run; force the visual/parallel no-call and argument-mismatch probe families into live execution
- H1j now scaffolds the packaged-workflow-only version of that packet:
  - config: [`configs/knowledge_work_h1j_slice.yaml`](../../configs/knowledge_work_h1j_slice.yaml)
  - brief: [`docs/continuity/h1j-slice.md`](./h1j-slice.md)
  - candidate packet id: `mlx_probe_derived_tool_contract_candidates`
  - helper-ablation packet id: `mlx_probe_derived_helper_ablation`
  - parallel no-call is explicitly deferred because there is no faithful live packaged workflow for `parallel_audit_array_literal` yet
- H1j candidate packet is now executed:
  - packet: [`results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_candidates_v1_knowledge_work_ablation_packet)
  - result: all five rows matched readiness `0.96577`, strict/recovered `1.0 / 1.0`, repair/fallback/argument repair `0.0 / 0.0 / 0.0`, raw clean `1.0`, with `0` notes and `0` failure candidates
- H1j helper-ablation packet is now executed:
  - packet: [`results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_helpers_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_helpers_v1_knowledge_work_ablation_packet)
  - result: contracted, no-directive, no-repair, no-fallback, and no-argument-repair all matched readiness `0.96577`, strict/recovered `1.0 / 1.0`, repair/fallback/argument repair `0.0 / 0.0 / 0.0`, raw clean `1.0`
  - trace analysis found `21` `controller_repair_disabled` markers but `0` failure candidates
- H1k now promotes the deferred parallel no-call replay case into a packaged live workflow:
  - config: [`configs/knowledge_work_h1k_slice.yaml`](../../configs/knowledge_work_h1k_slice.yaml)
  - brief: [`docs/continuity/h1k-slice.md`](./h1k-slice.md)
  - workflow: `ops_parallel_audit_review`
  - replay pressure: `parallel_audit_array_literal`
  - candidate packet id: `mlx_parallel_audit_tool_contract_candidates`
  - helper-ablation packet id: `mlx_parallel_audit_helper_ablation`
  - CLI preflight:

    ```bash
    uv run moonie-agent workflows --lane live_web_stress --workflow-id ops_parallel_audit_review --validate
    ```

  - candidate dry run:

    ```bash
    uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1k_slice.yaml --packet-id mlx_parallel_audit_tool_contract_candidates --run-group-id <timestamp>_h1k_parallel_audit_candidates_v1 --dry-run
    ```

  - executed candidate packet: [`results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_candidates_v1_knowledge_work_ablation_packet)
  - result: all five rows matched readiness `0.91780`, strict/recovered `1.0 / 1.0`, repair/fallback/argument repair `0.0 / 0.0 / 0.0`, raw clean `1.0`, with `0` trace notes and `0` failure candidates
  - executed helper packet: [`results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_helpers_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_helpers_v1_knowledge_work_ablation_packet)
  - helper result: contracted, no-directive, no-repair, no-fallback, and no-argument-repair rows all matched readiness `0.91780`; trace analysis found `3` expected disabled-repair markers and `0` failure candidates
  - interpretation: the packaged parallel workflow is still too staged to reproduce the raw no-directive one-turn parallel no-call failure. The next discriminator should preserve the exact-call replay shape.

- H1l packaged visual executor-equivalence is now a completed negative packaged-workflow result:
  - config: [`configs/knowledge_work_h1l_slice.yaml`](../../configs/knowledge_work_h1l_slice.yaml)
  - result packet: [`results/knowledge_work_h1_slice/20260509T_h1l_visual_executor_equivalence_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260509T_h1l_visual_executor_equivalence_candidates_v1_knowledge_work_ablation_packet)
  - result: contracted, no-directive, role catalog v1, argument hints v2, schema-field hints v4, and schema target literals v5 all tie at readiness `0.90406`, strict `0.85`, recovered `0.8`, raw clean `1.0`, and zero controller burden
  - interpretation: packaged visual workflows are too staged to preserve the hard-slice executor-equivalence split
- replay-shaped visual hard-slice live replay is the new active positive surface:
  - source packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_no_directive_replay_dry_run_v1`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_no_directive_replay_dry_run_v1)
  - matrix summary table: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_live_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_live_replay_summary.csv)
  - case-delta table: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_live_replay_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_live_replay_case_deltas.csv)
  - full matrix result:
    - no-directive baseline: exact/executable/executor-equivalent `0 / 2`
    - contracted MLX: exact/executable/executor-equivalent `2 / 2`
    - role catalog v1: exact/executable/executor-equivalent `1 / 2`
    - argument hints v2: exact/executable/executor-equivalent `1 / 2`
    - schema-field hints v4: exact `1 / 2`, executable/executor-equivalent `2 / 2`
    - schema target literals v5: exact `0 / 2`, executable/executor-equivalent `1 / 2`
  - completed stress packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_dry_run_v1`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_dry_run_v1)
  - stress result table: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_stress_live_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_stress_live_replay_summary.csv)
  - stress interpretation: schema-field hints v4 and schema target literals v5 recover full executor-equivalence on the metric-panel stress case without increasing strict exactness over no-directive. The next best move is more alias/decoy repetition, not H1m yet.
  - alias-repeat follow-up: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_repeat_dry_run_v1`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_repeat_dry_run_v1)
  - alias-repeat summary table: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_alias_repeat_live_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_alias_repeat_live_replay_summary.csv)
  - alias-repeat diagnostic: [`results/reports/visual_alias_repeat_diagnostic/diagnostic.md`](../../results/reports/visual_alias_repeat_diagnostic/diagnostic.md)
  - alias-repeat interpretation: schema-field hints improve executor-equivalence from `5 / 8` to `7 / 8` without strict gain, while schema target literals reach `3 / 8` strict and full `8 / 8` executor-equivalence; repeat or package the surviving mechanisms before H1m.

- second prompt-contract wave now exists:
  - contracts: `schema_literal_tool_required_v2`, `visual_next_call_state_v2`, `parallel_array_required_v2`
  - registry systems:
    - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_literal_tool_required`
    - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_next_call_state`
    - `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_parallel_array_required`
  - dry-run packet: [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_dry_run_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_dry_run_v1)
- second prompt-contract wave is now executed:
  - packet: [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_execute_v1)
  - `schema_literal_tool_required_v2`: exact `0.125`, executable `0.0`, weak exact gain
  - `visual_next_call_state_v2`: exact `0.0`, executable `1.0`, visual executable gain only
  - `parallel_array_required_v2`: exact `0.0`, executable `0.0`, no probe gain
- do not promote wave two back to H1 as a fix; use it as evidence that raw-probe replay or a faithful parallel live workflow is the next needed discriminator
- wave three established the first prompt-contract live movement:
  - dry-run packet: [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave3_dry_run_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave3_dry_run_v1)
  - executed packet: [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave3_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave3_execute_v1)
  - `canonical_json_copy_v3`: exact `0.125`, executable `0.0`, weak exact gain
  - `visual_tool_initiation_v3`: exact `0.125`, executable `1.0`, weak exact gain
  - `parallel_two_call_array_v3`: exact `0.0`, executable `0.0`, no probe gain
  - live canonical replay: [`results/tool_probe_replay_live/20260507T_canonical_argument_canonical_json_copy_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_canonical_argument_canonical_json_copy_live_execute_v1), exact `0 / 4`, no promotion
  - live visual replay: [`results/tool_probe_replay_live/20260507T_visual_state_visual_tool_initiation_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_visual_state_visual_tool_initiation_live_execute_v1), exact `1 / 3`, executable visual target recovered
  - interpretation: `visual_tool_initiation_v3` is useful but incomplete; do not spend H1 on canonical JSON or parallel two-call wording as currently written
- wave four is now executed and should be treated as a negative discriminator:
  - dry-run packet: [`results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_dry_run_v1`](../../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_dry_run_v1)
  - executed packet: [`results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_execute_v1)
  - `visual_state_tool_selection_v4`: exact `0.125`, executable `0.0`, weak exact gain
  - live visual replay: [`results/tool_probe_replay_live/20260508T_visual_state_tool_selection_live_execute_v1`](../../results/tool_probe_replay_live/20260508T_visual_state_tool_selection_live_execute_v1), exact `1 / 3`, executable visual target not recovered
  - comparison vs no-directive: [`results/tool_probe_replay_live_comparisons/20260508T_visual_state_tool_selection_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_state_tool_selection_vs_no_directive_live_v1), delta exact `+0.3333333333333333`
  - comparison vs contracted: [`results/tool_probe_replay_live_comparisons/20260508T_visual_state_contracted_vs_tool_selection_live_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_state_contracted_vs_tool_selection_live_v1), delta exact `-0.3333333333333333`, delta executable `-1.0`
  - interpretation: visual state/tool-selection wording did not fix `visual_latest_filter_literal`; the remaining failure is still `wrong_tool`, and v4 regresses `visual_form_target_literal` to `no_tool_call`
- wave five is now executed and rejected at the raw gate:
  - dry-run packet: [`results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave5_dry_run_v1`](../../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave5_dry_run_v1)
  - executed packet: [`results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave5_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave5_execute_v1)
  - `visual_refine_selection_v5`: exact `0.0`, executable `0.0`, no probe gain
  - live replay was skipped because the raw gate did not move
  - interpretation: standalone wording-only visual refinements have now produced one failed-to-improve live candidate and one raw-gate rejection
- visual role catalog is now the stable routing baseline:
  - isolated catalog probe: [`results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_v1_probe`](../../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_v1_probe)
  - raw result: exact `0.125`, executable `1.0`, delta exact vs no-directive `+0.125`
  - live replay: [`results/tool_probe_replay_live/20260508T_visual_role_catalog_live_execute_v1`](../../results/tool_probe_replay_live/20260508T_visual_role_catalog_live_execute_v1), exact `1 / 3`, executable visual target recovered
  - comparison vs wave four: [`results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_visual_state_tool_selection_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_role_catalog_vs_visual_state_tool_selection_v1)
  - interpretation: the catalog profile fixes the remaining wrong-tool failure class by making `refine_selection` a separable role, but exact literals still drift (`latest` becomes `latest issue`; `validation error` becomes `phone issue`)
- visual role catalog argument hints are now the best focused-replay exact visual no-directive candidate:
  - profile: `visual_role_catalog_argument_hints_v2`
  - isolated probe: [`results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_argument_hints_v2_probe`](../../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_argument_hints_v2_probe)
  - raw result: exact `0.25`, executable `0.0`, delta exact vs no-directive `+0.25`
  - live replay: [`results/tool_probe_replay_live/20260508T_visual_catalog_argument_hints_live_execute_v1`](../../results/tool_probe_replay_live/20260508T_visual_catalog_argument_hints_live_execute_v1), exact `2 / 3`
  - comparison vs no-directive: [`results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_no_directive_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_no_directive_v1), delta exact `+0.6666666666666666`
  - comparison vs contracted: [`results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_contracted_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_contracted_v1), delta exact `0.0`, delta executable `-1.0`
  - comparison vs v1 catalog: [`results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_role_catalog_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_catalog_argument_hints_vs_role_catalog_v1), delta exact `+0.3333333333333333`, delta executable `-1.0`
  - interpretation: v2 fixes `visual_latest_filter_literal` exactly and preserves readback on the original focused visual replay, but it regresses `visual_form_target_literal` from executable paraphrase to non-executable argument mismatch. The hard slice now shows v4 as the stronger executable candidate, so v2 is a focused-replay reference rather than the sole next target.
- visual split-selector hints are now negative evidence:
  - profile: `visual_role_catalog_split_selector_hints_v3`
  - isolated probe: [`results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_split_selector_hints_v3_probe`](../../results/tool_catalog_profile_probe_packets/20260508T_visual_role_catalog_split_selector_hints_v3_probe)
  - comparison vs v2: [`results/tool_catalog_profile_probe_comparisons/20260508T_visual_split_selector_hints_vs_argument_hints_v2`](../../results/tool_catalog_profile_probe_comparisons/20260508T_visual_split_selector_hints_vs_argument_hints_v2), delta exact `-0.125`
  - skipped-live decision: [`results/tool_probe_replay_live/20260508T_visual_split_selector_hints_live_replay_skipped_v1`](../../results/tool_probe_replay_live/20260508T_visual_split_selector_hints_live_replay_skipped_v1)
  - interpretation: adding broader split-selector prose preserved `filter_query="latest"` but broke the readback JSON shape and did not restore form-target executability. It is negative evidence against broad visual selector prose.
- visual schema-field hints are now split evidence rather than a simple negative:
  - profile: `visual_role_catalog_schema_field_hints_v4`
  - isolated probe: [`results/tool_catalog_profile_probe_packets/20260509T_visual_role_catalog_schema_field_hints_v4_probe`](../../results/tool_catalog_profile_probe_packets/20260509T_visual_role_catalog_schema_field_hints_v4_probe)
  - comparison vs v2: [`results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_argument_hints_v2`](../../results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_argument_hints_v2), delta exact `0.0`
  - comparison vs v3: [`results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_split_selector_v3`](../../results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_split_selector_v3), delta exact `+0.125`
  - comparison vs v1: [`results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_role_catalog_v1`](../../results/tool_catalog_profile_probe_comparisons/20260509T_visual_schema_field_hints_vs_role_catalog_v1), delta exact `+0.125`, executable regression vs v1
  - skipped-live decision: [`results/tool_probe_replay_live/20260509T_visual_schema_field_hints_live_replay_skipped_v1`](../../results/tool_probe_replay_live/20260509T_visual_schema_field_hints_live_replay_skipped_v1)
  - focused-slice interpretation: schema-local field hints restored exact readback and tied v2 at `2 / 8`, but did not recover the original form-target executable case.
  - fresh hard-slice result: strict `6 / 8`, executable/executor-equivalent `8 / 8`; this is the strongest no-directive hard-slice candidate, but still below contracted MLX on exact protocol fidelity.
- visual hard-slice execution is now the active discriminator:
  - script: [`scripts/build_visual_hard_slice_design.py`](../../scripts/build_visual_hard_slice_design.py)
  - runner: [`scripts/run_visual_hard_slice_probe_packet.py`](../../scripts/run_visual_hard_slice_probe_packet.py)
  - design packet: [`results/reports/visual_hard_slice_design/design.md`](../../results/reports/visual_hard_slice_design/design.md)
  - dry-run packet: [`results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_dry_run_v1`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_dry_run_v1)
  - first executed packet: [`results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_execute_v1`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_execute_v1)
  - latest executed packet: [`results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1)
  - latest gate summary: [`candidate_gate_summary.md`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1/candidate_gate_summary.md)
  - v5-vs-v4 comparison: [`schema_literal_targets_vs_schema_field_hints`](../../results/visual_hard_slice_probe_packets/20260509T_visual_hard_slice_executor_equivalence_v1/schema_literal_targets_vs_schema_field_hints)
  - exactness-vs-executor diagnostic: [`results/reports/visual_hard_slice_exactness_diagnostic`](../../results/reports/visual_hard_slice_exactness_diagnostic)
  - result: contracted MLX `8 / 8` strict/executable/executor-equivalent; no-directive MLX `1 / 8` strict/executable/executor-equivalent; schema-field hints v4 `6 / 8` strict and `8 / 8` executor-equivalent; schema-target-literal v5 `5 / 8` strict and `7 / 8` executor-equivalent.
  - diagnostic result: v4 has `2` non-exact rows and both are executor-target matches, so the current evidence classifies them as benchmark-label artifact candidates; v5 has those same `2` aliases plus `1` true wrong-tool stale-selection failure.
  - next use: do not write another target-query wording repair yet. Use the first-class executor-equivalence score to design a packaged visual H1 workflow that separates strict protocol fidelity from executor-visible success.
- wave six is now executed and should be treated as negative composition evidence:
  - dry-run packet: [`results/tool_prompt_contract_probe_packets/20260508T_visual_catalog_literal_guard_v6_dry_run`](../../results/tool_prompt_contract_probe_packets/20260508T_visual_catalog_literal_guard_v6_dry_run)
  - executed packet: [`results/tool_prompt_contract_probe_packets/20260508T_visual_catalog_literal_guard_v6_probe`](../../results/tool_prompt_contract_probe_packets/20260508T_visual_catalog_literal_guard_v6_probe)
  - candidate: `literal_argument_guard_v1` + `visual_role_catalog_v1`
  - raw result: exact `0.125`, executable `0.0`, delta exact vs no-directive `+0.125`
  - interpretation: combining broad literal-guard wording with the catalog profile loses the catalog-only executable recovery and introduces no-call regressions; do not promote it
- visual tool-choice diagnostics now exist:
  - script: [`scripts/analyze_visual_tool_choice_diagnostics.py`](../../scripts/analyze_visual_tool_choice_diagnostics.py)
  - packet: [`results/tool_probe_replay_live_diagnostics/20260508T_visual_tool_choice_wave3_wave4_catalog_v1`](../../results/tool_probe_replay_live_diagnostics/20260508T_visual_tool_choice_wave3_wave4_catalog_v1)
  - result: wave-three and wave-four candidates choose `extract_layout` where `visual_latest_filter_literal` expects `refine_selection`; the catalog profile reaches `refine_selection` and only misses the literal selector
- exact-probe replay now exists:
  - brief: [`docs/continuity/exact-probe-replay.md`](./exact-probe-replay.md)
  - packet: [`results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_v1`](../../results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_v1)
  - `8` failed no-directive probe cases
  - failure split: `argument_mismatch = 4`, `no_tool_call = 4`
  - next use: choose between faithful packaged live parallel workflow and replay execution for exact probe cases
  - execution support: add `--execute` to rerun selected cases and write `replay_results.json` / `replay_results.csv`
  - executed replay packet: [`results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1`](../../results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1)
  - executed result: exact `0 / 8`; all source failure modes reproduced
  - contracted replay packet: [`results/tool_probe_replay_packets/20260507T_contracted_exact_probe_replay_execute_v1`](../../results/tool_probe_replay_packets/20260507T_contracted_exact_probe_replay_execute_v1)
  - contracted result: exact `7 / 8`; remaining visual paraphrase is executable
  - replay comparison: [`results/tool_probe_replay_comparisons/20260507T_contracted_vs_no_directive_exact_replay_v1`](../../results/tool_probe_replay_comparisons/20260507T_contracted_vs_no_directive_exact_replay_v1)
  - live operator bridge: `moonie-agent replay-live`
  - first live dry run: [`results/tool_probe_replay_live/20260507T_parallel_array_replay_live_dry_run_v1`](../../results/tool_probe_replay_live/20260507T_parallel_array_replay_live_dry_run_v1)
  - first no-directive live execution: [`results/tool_probe_replay_live/20260507T_parallel_array_no_directive_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_parallel_array_no_directive_live_execute_v1), exact `0 / 1`, failure `no_tool_call`
  - first contracted live execution: [`results/tool_probe_replay_live/20260507T_parallel_array_contracted_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_parallel_array_contracted_live_execute_v1), exact `1 / 1`
  - first live comparison: [`results/tool_probe_replay_live_comparisons/20260507T_parallel_array_contracted_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260507T_parallel_array_contracted_vs_no_directive_live_v1), delta exact `-1.0`, delta actual calls `-2`
  - visual no-directive live execution: [`results/tool_probe_replay_live/20260507T_visual_state_no_directive_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_visual_state_no_directive_live_execute_v1), exact `0 / 3`, all failures `no_tool_call`
  - visual contracted live execution: [`results/tool_probe_replay_live/20260507T_visual_state_contracted_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_visual_state_contracted_live_execute_v1), exact `2 / 3`, remaining case executable
  - visual live comparison: [`results/tool_probe_replay_live_comparisons/20260507T_visual_state_contracted_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260507T_visual_state_contracted_vs_no_directive_live_v1), delta exact `-0.6666666666666666`
  - canonical-argument no-directive live execution: [`results/tool_probe_replay_live/20260507T_canonical_argument_no_directive_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_canonical_argument_no_directive_live_execute_v1), exact `0 / 4`, all failures `argument_mismatch`
  - canonical-argument contracted live execution: [`results/tool_probe_replay_live/20260507T_canonical_argument_contracted_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_canonical_argument_contracted_live_execute_v1), exact `4 / 4`
  - canonical-argument live comparison: [`results/tool_probe_replay_live_comparisons/20260507T_canonical_argument_contracted_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260507T_canonical_argument_contracted_vs_no_directive_live_v1), delta exact `-1.0`, actual-call delta `0`
  - next use: keep exact-probe replay as the controller-dependence anchor, but use the executed visual hard slice as the current visual prompt-contract restart point before spending H1 budget
- promote a candidate beyond H1i only if it moves raw-clean or controller-burden metrics for the right reason
- regenerate the MLX tool-contract report after any H1i, H1h, probe, or Gemini baseline packet changes
- when a real Gemini CLI binary is available, rerun the same packet with `--execute`; keep the dry-run packet as the no-side-effects prompt manifest
- keep the H1h comparison commands close:

```bash
uv run python scripts/build_mlx_tool_contract_report.py
uv run python scripts/analyze_knowledge_work_h1_traces.py <packet_dir>
uv run python scripts/summarize_h1_tool_contract.py <packet_dir>
uv run python scripts/summarize_h1_workflow_families.py <packet_dir> --config configs/knowledge_work_h1h_slice.yaml
uv run python scripts/compare_tool_directive_probes.py results/tool_directive_probe/20260506T_mlx_tool_directive_probe_v4 results/tool_directive_probe/20260507T_mlx_no_directive_probe_v1
uv run python scripts/run_tool_prompt_contract_probe_packet.py --run-group-id <timestamp>_prompt_contract_probe_candidates --execute
uv run python scripts/run_tool_prompt_contract_probe_packet.py --candidate-wave v3 --run-group-id <timestamp>_prompt_contract_wave3_execute_v1 --execute
uv run python scripts/run_tool_prompt_contract_probe_packet.py --candidate-wave v4 --run-group-id <timestamp>_prompt_contract_wave4_execute_v1 --execute
uv run python scripts/run_tool_prompt_contract_probe_packet.py --candidate-wave v5 --run-group-id <timestamp>_prompt_contract_wave5_execute_v1 --execute
uv run python scripts/run_tool_catalog_profile_probe_packet.py --run-group-id <timestamp>_visual_role_catalog_probe --execute
uv run python scripts/run_tool_catalog_profile_probe_packet.py --candidate-wave v2 --run-group-id <timestamp>_visual_catalog_argument_hints_probe --execute
uv run python scripts/run_tool_catalog_profile_probe_packet.py --candidate-wave v4 --run-group-id <timestamp>_visual_schema_field_hints_probe --execute
uv run python scripts/build_visual_hard_slice_design.py
uv run python scripts/run_visual_hard_slice_probe_packet.py --run-group-id <timestamp>_visual_hard_slice_probe --execute
uv run python scripts/compare_tool_directive_probes.py results/visual_hard_slice_probe_packets/<packet_id>/<baseline_system_id> results/visual_hard_slice_probe_packets/<packet_id>/<candidate_system_id> --output-dir results/visual_hard_slice_probe_packets/<packet_id>/<comparison_id>
uv run python scripts/analyze_visual_hard_slice_exactness.py --packet-dir results/visual_hard_slice_probe_packets/<packet_id> --json
uv run moonie-agent packet --kind visual-hard-slice-probe --packet-id <packet_id> --json
uv run python scripts/build_publication_evidence_ledger.py
uv run python scripts/audit_publication_readiness.py
uv run python scripts/run_tool_prompt_contract_probe_packet.py --candidate-wave v6 --run-group-id <timestamp>_visual_catalog_literal_guard_probe --execute
uv run python scripts/summarize_tool_prompt_contract_probe_packet.py results/tool_prompt_contract_probe_packets/<packet_id>
uv run python scripts/build_tool_probe_replay_packet.py --run-group-id <timestamp>_no_directive_exact_probe_replay
uv run moonie-agent replay-live --packet-dir results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1 --case-id parallel_audit_array_literal --execute
uv run moonie-agent replay-live --packet-dir results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_tool_initiation --case-id visual_form_target_literal --case-id visual_latest_filter_literal --case-id visual_readback_region_literal --execute
uv run moonie-agent replay-live --packet-dir results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1 --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_state_tool_selection --case-id visual_form_target_literal --case-id visual_latest_filter_literal --case-id visual_readback_region_literal --execute
uv run python scripts/compare_tool_probe_replay_live_packets.py results/tool_probe_replay_live/<contracted_packet> results/tool_probe_replay_live/<no_directive_packet>
uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1i_slice.yaml --packet-id mlx_prompt_contract_candidates --run-group-id <timestamp>_h1i_prompt_contract_candidates
uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1i_slice.yaml --packet-id mlx_prompt_contract_candidates --run-group-id 20260507T_h1i_prompt_contract_candidates_repeat3_v1 --repeat 3
uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1j_slice.yaml --packet-id mlx_probe_derived_tool_contract_candidates --run-group-id <timestamp>_h1j_probe_derived_candidates_v1
uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1j_slice.yaml --packet-id mlx_probe_derived_helper_ablation --run-group-id <timestamp>_h1j_probe_derived_helpers_v1
uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1l_slice.yaml --packet-id mlx_visual_executor_equivalence_candidates --run-group-id <timestamp>_h1l_visual_executor_equivalence_candidates_v1 --dry-run
uv run python scripts/run_knowledge_work_h1_ablation_packet.py --config configs/knowledge_work_h1l_slice.yaml --packet-id mlx_visual_executor_equivalence_candidates --run-group-id <timestamp>_h1l_visual_executor_equivalence_candidates_v1
uv run python scripts/summarize_h1_tool_contract.py results/knowledge_work_h1_slice/<packet_id>
```

Success condition:

- every controller or prompt-contract change is evaluated against H1h or a smaller slice derived from its worst workflow families
- Gemini CLI baseline artifacts are attributable to the same workflow family IDs
- prompt-contract changes show raw probe improvement before they are allowed into H1i
- prompt-contract changes beat the current live replay ceiling, not merely tie wave three's `1 / 3` visual exact rate
- visual catalog changes preserve the strict-vs-executor-equivalence distinction when promoted into H1l packaged workflows
- broad aligned `32 / 26` reruns stay paused until this harder packet produces a specific mechanism-level change

### 3. Keep using focused packets before any broader rerun

Current packet:

- [`results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v4_knowledge_work_ablation_packet_knowledge_work_ablation_packet`](../../results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v4_knowledge_work_ablation_packet_knowledge_work_ablation_packet)

Operational rule:

- make controller/planner changes against the 9-episode packet first
- rerun the aligned `32 / 26` surface only after the packet shifts in a real way

Why:

- the packet still shows the causal helper ranking clearly
- it is the cheapest clean instrument for Gemma controller research

### 4. Use H1 before another broad same-surface rerun

Current H1 definition:

- [`configs/knowledge_work_h1_slice.yaml`](../../configs/knowledge_work_h1_slice.yaml)
- [`docs/continuity/h1-slice.md`](./h1-slice.md)
- [`scripts/run_knowledge_work_h1_slice.py`](../../scripts/run_knowledge_work_h1_slice.py)

What it concentrates:

- packaged workflow families only
- replayable/live mirrors
- resume and project-memory pressure
- latest-instruction and stale-override pressure
- CLI/API/function-call choice
- approval-safe stop behavior
- artifact revision after review

Success condition:

- H1 produces a clearer separation than the current saturated aligned `32 / 26` top-line readiness read
- controller repair, fallback, raw planning cleanliness, and approval-safe stop behavior become the primary comparison fields

Smoke command:

```bash
uv run python scripts/run_knowledge_work_h1_slice.py --dry-run --run-set primary --lane replayable_core
```

Ablation dry-run command:

```bash
uv run python scripts/run_knowledge_work_h1_slice.py --dry-run --run-set ablation --lane replayable_core
```

Completed primary run:

```bash
uv run python scripts/run_knowledge_work_h1_slice.py --run-set primary --lane replayable_core --system-id mlx_gemma4_e2b_reasoner_only --run-group-id 20260506T_h1_mlx_gemma_primary_v1
```

Current empirical status:

- the H1 primary replayable MLX Gemma run completed with `5 / 5` episodes
- `real_world_readiness_avg = 0.9749800000000001`
- `controller_repair_avg = 0.0`
- `controller_fallback_avg = 0.0`
- `raw_planning_clean_rate_avg = 1.0`

Completed HF service-backed ablation run:

```bash
uv run python scripts/run_knowledge_work_h1_ablation_packet.py --lane replayable_core --run-group-id 20260506T_h1_hf_service_prompt_patch_ablation_v1
```

Current empirical status:

- concrete-hint full ablation output: [`results/knowledge_work_h1_slice/20260506T_h1_hf_service_concrete_hint_ablation_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_concrete_hint_ablation_v1_knowledge_work_ablation_packet)
- visual-filter-repair full ablation output: [`results/knowledge_work_h1_slice/20260506T_h1_hf_service_visual_filter_repair_ablation_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_visual_filter_repair_ablation_v1_knowledge_work_ablation_packet)
- baseline `hf_service_gemma4_specialists_cpu`: `real_world_readiness_avg = 0.9749800000000001`
- baseline controller burden is now clean after concrete FunctionGemma hints: `controller_repair_avg = 0.0`, `controller_fallback_avg = 0.0`, `raw_planning_clean_rate_avg = 1.0`
- `no_controller_repair`: `real_world_readiness_avg = 0.8874599999999999`
- `no_controller_fallback`: `real_world_readiness_avg = 0.9749800000000001`
- `no_visual_rescue`: unchanged at `0.9749800000000001`
- `no_intent_priority`: unchanged at `0.9749800000000001`
- `no_argument_repair`: unchanged at `0.9749800000000001`
- `no_deterministic_visual_follow_on`: restored to `real_world_readiness_avg = 0.9749800000000001` after the pending-filter semantic repair
- trace failure candidates dropped from `10` to `6` after concrete hints, then to `3` after visual-filter repair; all remaining failures are in `no_controller_repair`
- the old aggregate `generic_tool_name` mode is gone

Use the H1 ablation packet wrapper instead of the generic H1 runner for this wave; it shares one HF service-backed reasoner plus in-process HF specialist adapters across the ablation rows.

Completed empirical move:

- use the completed visual canaries:
  - [`results/knowledge_work_h1_slice/20260506T_h1_visual_sequence_hint_canary_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_visual_sequence_hint_canary_v1_knowledge_work_ablation_packet)
  - [`results/knowledge_work_h1_slice/20260506T_h1_visual_filter_repair_canary_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_visual_filter_repair_canary_v1_knowledge_work_ablation_packet)
- treat fallback-disabled as solved on this H1 slice: `no_controller_fallback` now matches baseline
- full H1 ablation after visual filter repair verifies that `no_deterministic_visual_follow_on` moves from `0.88748` back to baseline
- compact visual semantics packet output: [`results/knowledge_work_h1_slice/20260506T_h1_visual_semantics_no_repair_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_visual_semantics_no_repair_v1_knowledge_work_ablation_packet)
- compact packet reproduces the residual gap:
  - baseline readiness `0.9715666666666666`
  - `no_controller_repair = 0.8257`
  - `no_deterministic_visual_follow_on = 0.9715666666666666`
  - all `3` failure candidates are in `no_controller_repair`
- visual prompt-contract candidate output: [`results/knowledge_work_h1_slice/20260506T_h1_visual_semantics_prompt_contract_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_visual_semantics_prompt_contract_v1_knowledge_work_ablation_packet)
  - stronger system-level wording did not improve `no_controller_repair`
  - `no_controller_repair` stayed at readiness `0.8257`, strict/recovered `0.625 / 0.5`
  - this is a useful negative result: the exact-next-call hint likely needs to be closer to the final generation turn or encoded differently
- visual turn-directive candidate output: [`results/knowledge_work_h1_slice/20260506T_h1_visual_semantics_turn_directive_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_visual_semantics_turn_directive_v1_knowledge_work_ablation_packet)
  - final turn-level exact-call directive restored all three packet rows to readiness `0.9715666666666666`, strict/recovered `1.0 / 1.0`, raw clean `1.0`
  - trace mining found `0` failure candidates
  - this is now the strongest model-side harnessing improvement on H1 visual semantics
- full H1 turn-directive ablation output: [`results/knowledge_work_h1_slice/20260506T_h1_hf_service_turn_directive_ablation_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1_hf_service_turn_directive_ablation_v1_knowledge_work_ablation_packet)
  - all seven H1 ablation rows now match baseline
  - every row has readiness `0.9749800000000001`, strict/recovered `1.0 / 1.0`, raw clean `1.0`, repair/fallback `0.0 / 0.0`
  - trace mining found `0` failure candidates
  - remaining notes are only ablation markers (`controller_repair_disabled`, `intent_priority_disabled`)

Next empirical move:

- H1 is saturated again after the visual turn directive; define H1b before another broad same-slice rerun
- H1b scaffold now exists:
  - [`configs/knowledge_work_h1b_slice.yaml`](../../configs/knowledge_work_h1b_slice.yaml)
  - [`docs/continuity/h1b-slice.md`](./h1b-slice.md)
- H1b should stress:
  - longer visual chains where the next filter is not directly lexical in the user request
  - mixed visual + API/CLI dependencies after the readback
  - refusal/defer/clarify decisions after partial progress
  - approval-safe stop after a correct visual readback
  - latest-instruction override when the stale visual chain looks easier than the current instruction
- also run a live CLI packaged-workflow smoke with the same FunctionGemma turn directive to validate live operator behavior, not just replayable traces
- next verification command:

```bash
uv run python scripts/run_knowledge_work_h1_ablation_packet.py \
  --config configs/knowledge_work_h1b_slice.yaml \
  --packet-id visual_policy_no_controller_repair \
  --run-group-id <timestamp>_h1b_visual_policy_packet
```

- completed H1b compact packet: [`results/knowledge_work_h1_slice/20260506T_h1b_visual_policy_packet_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1b_visual_policy_packet_v1_knowledge_work_ablation_packet)
  - all three rows matched at readiness `0.9472999999999999`, strict/recovered `1.0 / 1.0`, raw clean `1.0`, repair/fallback `0.0 / 0.0`
  - trace mining found `0` failure candidates
- completed H1b full ablation: [`results/knowledge_work_h1_slice/20260506T_h1b_hf_service_ablation_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260506T_h1b_hf_service_ablation_v1_knowledge_work_ablation_packet)
  - all seven rows matched at readiness `0.9581199999999999`, strict/recovered `1.0 / 1.0`, repair/fallback `0.0 / 0.0`, raw clean `1.0`
  - trace mining found `0` failure candidates; notes are only disabled-helper markers
  - H1b is harder on artifact/readiness than H1, but it still does not restore controller dependence after the final FunctionGemma turn directive

- rerun the compact packet after each candidate patch before returning to full H1:

```bash
uv run python scripts/run_knowledge_work_h1_ablation_packet.py \
  --packet-id visual_semantics_no_controller_repair \
  --run-group-id <timestamp>_h1_visual_semantics_candidate
```

- avoid another broad aligned rerun until a new mechanism or harder slice changes the controller/failure picture

Current next move:

- quantify the discrepancy between the clean H1c MLX benchmark runner and the earlier CLI live-smoke repair/fallback packets
- add repeat support to `scripts/run_runtime_live_smoke_packet.py` so the same packaged workflow can be run several times with attributable session ids
- run repeated CLI live smoke over the H1c-overlapping workflows on `mlx_gemma4_e2b_reasoner_only`
- summarize repair/fallback frequency, policy-block frequency, approval stops, and raw-clean rate by workflow family
- only define H1d after that repeatability check shows which live CLI failure family is stable enough to stress

### 5. Install the local Gemma `31B` `GGUF` artifact and run the first real `llama.cpp` posture row

Current blocker:

- `GEMMA4_31B_GGUF_PATH` unset
- no local Gemma `31B` `GGUF` under `/Users/cheickdiakite/models`

Why it matters:

- runtime posture is now clearly part of the capability story
- a real `31B` row is still missing from the research package

## Near Term

### 6. Add Gemini CLI as a design reference and external baseline

Current state:

- `moonie-agent gemini-baseline` exists as a dry-run-first packaged-workflow adapter
- `/usr/local/bin/gemini` is detected on this machine

Use Gemini CLI for:

- CLI ergonomics reference
- trust and sandbox design reference
- wrapped external baseline on selected Moonie workflows

Do not use Gemini CLI as a replacement for Moonie or as proof that local MLX Gemma harnessing is solved.

### 7. Expand the failure taxonomy, not just the leaderboard

Keep pushing on:

- tool-family choice
- argument fidelity
- direction-following under conflicting instructions
- approval-safe stop behavior
- clarify vs defer vs refuse judgment quality
- artifact revision quality after feedback

Current second-wave ablation toggles:

- `disable_intent_priority`
- `disable_argument_repair`
- `disable_deterministic_visual_follow_on`

### 8. Keep the live harness benchmark-backed

The runtime, CLI, API, operator console, and mobile companion should keep sharing the same execution semantics as the benchmark.

No parallel orchestration path should be introduced.

### 9. Decide the next Gemma posture bet explicitly

Decision rule:

- if the question is still controller dependence, stay on HF Gemma specialists + packet work
- if the question is runtime posture, prioritize Gemma `31B` local artifact work next
- only add specialist-backed MLX Gemma after the current reasoner-only posture has paid off analytically

## Ongoing Discipline

- keep claims tied to reproduced same-surface rows
- keep public benchmark context separate from Moonie rows
- treat community discourse as hypotheses, not evidence
- rebuild history exports after each meaningful benchmark pass
- keep README and continuity docs aligned with the current actual numbers
