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
  - `moonie-agent gemini-baseline` can prepare a dry-run Gemini CLI baseline packet for packaged workflows

Next implementation moves:

- use the completed visual-filter repair full H1 ablation rerun as the new causal snapshot
- isolate the remaining `no_controller_repair` failures, which are now mostly valid-but-semantically-wrong visual refinements rather than malformed calls
- use the compact `visual_semantics_no_controller_repair` packet before changing more harness code
- later, consider a true keyboard TUI after the command-driven operator loop is useful
- keep hardening sandbox policies around file writes and external process/network actions
- keep packaged workflows as the only live entrypoint in v1
- preserve benchmark-backed traces, artifacts, and scorecards for every live run

Success condition:

- a person can safely launch and watch a real local Gemma MLX run from CLI, approve or resume when needed, and inspect the run live without leaving the terminal

### 2. Reduce the remaining HF Gemma specialist controller burden on the aligned `32 / 26` surface

This is still the highest-value move.

What changed:

- the deterministic visual follow-on patch already removed the old `feedback_prior:refine_selection` and `feedback_prior:read_region_text` families
- HF Gemma replayable `controller_repair_avg` is now `0.71875`
- HF Gemma live `controller_repair_avg` is now `0.8076923076923077`

What remains:

- `controller_fallback_planner`
- `repaired_arguments:extract_layout`
- `intent_prior:record_or_update`
- `intent_prior:inspect_or_lookup`

Success condition:

- keep replayable `real_world_readiness_avg = 0.976853125`
- keep live `real_world_readiness_avg = 0.9791653846153847`
- lower controller repair and fallback further
- improve raw planning cleanliness if possible

### 3. Keep using the focused replayable packet before any broader rerun

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

- run a real packaged-workflow CLI live smoke on `mlx_gemma4_e2b_reasoner_only`
- inspect sandbox roots, event timelines, artifacts, and trace attribution from the terminal
- then define H1c with new episodes that mix visual readback, API/CLI dependencies, approval/defer/refuse choices, and partial-progress recovery
- keep H1c replayable/live mirrors so live CLI behavior remains benchmark-backed

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
