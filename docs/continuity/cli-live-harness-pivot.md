# CLI Live Harness Pivot

## Purpose

This file is the current restart point for Moonie.

If a new chat needs to resume work quickly, start here instead of reconstructing intent from older React-oriented continuity notes.

The repo direction has changed:

- frontend polish is no longer the priority
- the primary goal is research and harnessing
- the main new product surface should be a CLI-first live operator harness for local Gemma on MLX

## Why We Are Pivoting

The current repo already proved the first important thing:

- on the aligned exploratory `32 / 26` surface, `oracle_gemma4_e2b`, `hf_gemma4_e2b_specialists_cpu`, `mlx_qwen3_8b_reasoner_only`, and `mlx_gemma4_e2b_reasoner_only` all reach the same top-line readiness tier

That means the next research question is no longer:

- can Gemma finish the current lane?

It is now:

- what exact harness interventions are doing the work?
- what survives under harder realism?
- how do we test and observe local Gemma live without drifting into product/UI churn?

The React shell is real and useful as prior work, but it is no longer the highest-value next move.

## Current Repo Truth

### Benchmark surface

Current generated corpus on disk:

- atomic tasks: `91`
- variants: `396`
- replayable KWA episodes: `33`
- live KWA episodes: `27`

Current source-of-truth comparison surface:

- aligned exploratory `32 / 26`
- board export:
  - [`results/history/knowledge_work_board_latest.csv`](../../results/history/knowledge_work_board_latest.csv)

### Headline comparison read

Replayable `32`:

- `oracle_gemma4_e2b`
  - `real_world_readiness_avg = 0.976853125`
  - `controller_repair_avg = 0.578125`
  - `controller_fallback_avg = 0.0`
- `hf_gemma4_e2b_specialists_cpu`
  - `real_world_readiness_avg = 0.976853125`
  - `controller_repair_avg = 0.71875`
  - `controller_fallback_avg = 0.28125`
  - `raw_planning_clean_rate_avg = 0.46875`
- `mlx_qwen3_8b_reasoner_only`
  - `real_world_readiness_avg = 0.976853125`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`
- `mlx_gemma4_e2b_reasoner_only`
  - `real_world_readiness_avg = 0.976853125`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
  - `raw_planning_clean_rate_avg = 1.0`

Live `26`:

- `oracle_gemma4_e2b`
  - `real_world_readiness_avg = 0.9791653846153847`
  - `controller_repair_avg = 0.7115384615384616`
- `hf_gemma4_e2b_specialists_cpu`
  - `real_world_readiness_avg = 0.9791653846153847`
  - `controller_repair_avg = 0.8076923076923077`
  - `controller_fallback_avg = 0.23076923076923078`
- `mlx_qwen3_8b_reasoner_only`
  - `real_world_readiness_avg = 0.9791653846153847`
  - `controller_repair_avg = 0.0`
- `mlx_gemma4_e2b_reasoner_only`
  - `real_world_readiness_avg = 0.9791653846153847`
  - `controller_repair_avg = 0.0`

### Focused Gemma packet

Current focused replayable research harness:

- [`results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v4_knowledge_work_ablation_packet_knowledge_work_ablation_packet`](../../results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v4_knowledge_work_ablation_packet_knowledge_work_ablation_packet)

Packet baseline:

- `real_world_readiness_avg = 0.9627777777777777`
- `controller_repair_avg = 0.8888888888888888`
- `controller_fallback_avg = 0.4444444444444444`

Ablation rows:

- `no_controller_repair = 0.6551777777777779`
- `no_controller_fallback = 0.8182333333333333`
- `no_visual_rescue = 0.9627777777777777`

Interpretation:

- controller repair is causal
- controller fallback is causal
- visual rescue is not doing useful work on this focused slice

### Strongest current research finding

Moonie can carry Gemma to the same final readiness tier on the current aligned surface, but the autonomy story is still different.

That means the real remaining signal is:

- controller dependence
- raw planning cleanliness
- harder realism that breaks current parity

The freshest MLX evidence is now H1h/H1i/H1g:

- current generated report:
  - [`docs/reports/mlx-tool-contract-harnessing.md`](../reports/mlx-tool-contract-harnessing.md)
  - [`results/reports/mlx_tool_contract_harnessing/report.md`](../../results/reports/mlx_tool_contract_harnessing/report.md)
- H1h shows the tool-turn directive is causal on the full ten-workflow live family set:
  - contracted MLX is controller-clean at readiness `0.96891`
  - no-directive MLX keeps readiness `0.96891` only with repair/fallback/argument repair `0.70 / 0.25 / 0.45`
  - no-directive + no controller repair drops to `0.73801`
  - no-directive + no controller fallback drops to `0.89598`
  - no-directive + no argument repair drops to `0.83016`
  - workflow-family attribution shows the worst no-repair cases are executive latest-action resume, jobs phone patch resume, jobs visual form hold, and executive stale brief packet
- the H1h Gemini CLI dry-run packet now exists for the same ten workflow families:
  - [`results/gemini_cli/20260507T_h1h_gemini_cli_dry_run_baseline_v1`](../../results/gemini_cli/20260507T_h1h_gemini_cli_dry_run_baseline_v1)
  - this is a prompt/command manifest only; it intentionally did not execute a real Gemini CLI binary
- the no-directive MLX tool probe makes the prompt-contract gap stark:
  - no-directive exact copy `0 / 8`
  - no-directive executable visual target `0 / 1`
  - contracted-vs-no-directive probe comparison exact-rate delta `-0.875`
- H1i is now the compact fast loop derived from the worst H1h no-repair workflow families:
  - [`configs/knowledge_work_h1i_slice.yaml`](../../configs/knowledge_work_h1i_slice.yaml)
  - [`results/knowledge_work_h1_slice/20260507T_h1i_mlx_worst_no_directive_v1_knowledge_work_h1i_mlx_worst_family_tool_contract_v1`](../../results/knowledge_work_h1_slice/20260507T_h1i_mlx_worst_no_directive_v1_knowledge_work_h1i_mlx_worst_family_tool_contract_v1)
  - no-directive + no controller repair drops to readiness `0.64697`, strict/recovered `0.297 / 0.000`
  - no-directive with helpers still matches top-line readiness only with raw clean `0.00`
- H1g shows the remaining helpers are not causal under the directive on the compact live slice:
  - `no_visual_rescue`, `no_intent_priority`, and `no_deterministic_visual_follow_on` all match baseline with `0` failure candidates
- the latest directive probe separates exact-copy from executable readiness:
  - exact JSON copy `7 / 8`
  - visual selector paraphrase executable target `1 / 1`

The freshest replay-shaped visual controller evidence is now H2p:

- H2j target-query normalization proved the repair can be controller-attributable rather than prompt-prose-attributable:
  - H2f: `10 / 10`
  - H2b: `5 / 5`
  - H1x: `8 / 8`
- H2k showed the post-H2j target/decoy-overlap win is target-normalization evidence, not stale-selection rescue:
  - H2j: `8 / 8`
  - H2j without stale-selection: `8 / 8`
  - H2e: `3 / 8` strict
  - both H2j rows record `5` target-query-normalization interventions and `0` stale-selection interventions
- H2l showed direct target-is wording does not trigger over-stripping:
  - H2j and H2j-no-stale: `8 / 8`
  - H2e: `7 / 8`
- H2m removed that direct wording and exposed H2j's boundary:
  - H2j and H2j-no-stale: `3 / 8` strict and executor-equivalent
  - H2e: `1 / 8` strict and `3 / 8` executor-equivalent
  - H2j records `3` value-bearing over-strip rows
- H2n is the scoped blocking candidate:
  - H2m: `3 / 8` strict and `5 / 8` executor-equivalent
  - H2m H2n-vs-H2j: `0.0` strict delta and `+0.25` executor-equivalence delta
  - H2k: `8 / 8`
  - H2l: `8 / 8`
  - H2f: `10 / 10`
  - synthesis: [`results/reports/h2n_scoped_target_normalization_synthesis/report.md`](../../results/reports/h2n_scoped_target_normalization_synthesis/report.md)
- H2o is the strict value-bearing synthesis candidate:
  - H2m: `7 / 8` strict and executor-equivalent
  - H2m H2o-vs-H2n: `+0.50` strict delta and `+0.25` executor-equivalence delta
  - H2m H2o-vs-H2j: `+0.50` strict delta and `+0.50` executor-equivalence delta
  - H2k: `8 / 8`
  - H2l: `8 / 8`
  - H2f: `10 / 10`
  - synthesis: [`results/reports/h2o_value_bearing_target_synthesis/report.md`](../../results/reports/h2o_value_bearing_target_synthesis/report.md)
- H2p is the contextual surface-alias routing candidate:
  - H2m: `8 / 8` strict and executor-equivalent
  - H2m H2p-vs-H2o: `+0.125` strict delta and `+0.125` executor-equivalence delta
  - H2m H2p-vs-H2n: `+0.625` strict delta and `+0.375` executor-equivalence delta
  - H2m H2p-vs-H2j: `+0.625` strict delta and `+0.625` executor-equivalence delta
  - H2m H2p-vs-H2e: `+0.875` strict delta and `+0.625` executor-equivalence delta
  - H2k: `8 / 8`
  - H2l: `8 / 8`
  - H2f: `10 / 10`
  - synthesis: [`results/reports/h2p_contextual_surface_alias_routing_synthesis/report.md`](../../results/reports/h2p_contextual_surface_alias_routing_synthesis/report.md)
- H2q is the next composed dry-run packet, not yet a live result:
  - packet: [`results/tool_probe_replay_packets/20260512T_h2q_composed_surface_value_stale_dry_run_v1`](../../results/tool_probe_replay_packets/20260512T_h2q_composed_surface_value_stale_dry_run_v1)
  - suite: `h2q_composed_surface_value_stale_v20`
  - shape: `8` cases mixing surface aliases, value-bearing labels, stale-selection hints, and decoy overlap
  - control plan: execute H2q on H2p, H2o-without-H2p, H2n, and H2e before adding another helper
- next controller question:
  - test whether H2p survives H2q's composed pressure rather than another isolated H2m repair
  - require value-bearing construction plus surface-alias routing under stale/decoy pressure, with H2o-only/H2n/H2e ablation rows

## Direction Reset

### Primary goal

Build a CLI-first live harness for local Gemma on MLX that lets us:

- sandbox runs safely
- watch live execution
- approve or deny when needed
- resume interrupted sessions
- inspect tool use, controller help, and artifact revisions
- keep live testing tied to benchmark-backed workflows

### Explicit non-goals right now

- no more React refinement unless required for runtime/API support
- no Streamlit product work
- no broad comparator expansion
- no additional same-surface reruns without a harder slice or clearer ablation target

## Next Research Priorities

1. Build a CLI-first live operator harness over the existing runtime.
2. Add proper sandboxing for live runs.
3. Use packaged workflows as the only live entrypoint in v1.
4. Add a Gemini CLI adapter as an external baseline and design reference.
5. Use the completed Gemini CLI dry-run as the attributable H1h workflow-family baseline until a real external run is intentionally requested.
6. Use H1i as the current smaller no-directive stress packet derived from H1h workflow attribution.
7. Use H1g as the second-wave helper baseline: visual rescue, intent priority, and deterministic visual follow-on are negative on the compact live slice.
8. Regenerate the MLX tool-contract report after every H1i/H1h/probe/Gemini packet change.
9. Only after that, revisit Gemma `31B` `GGUF` runtime posture.

## Implementation Order

### Phase 0: Re-ground and tighten narrative

- treat this file as the active restart point
- keep older React-oriented docs as historical context, not current priority
- ensure continuity docs stop implying that UI is the next main workstream

Status: underway.

### Phase 1: Runtime sandbox model

Add a real sandbox execution layer to the runtime.

Target behavior:

- every live run gets explicit sandbox metadata
- default mode is per-session ephemeral copy
- all file writes stay inside sandbox root
- path escape attempts are blocked and recorded
- live-web workflows stay dry-run / non-destructive

Important session/runtime additions:

- `sandbox_mode`
- `sandbox_root`
- `sandbox_source`
- `sandbox_policy_id`
- `sandbox_manifest_path`

Status: first scaffold implemented.

- new sessions default to `ephemeral_copy`
- packaged workflow and episode inputs are copied into `sandbox/input`
- native artifacts, trace JSON, summary JSON, and runtime manifest write under `sandbox/output`
- session and runtime trace records carry the sandbox fields above

### Phase 2: CLI live entrypoint

Add:

- `moonie-agent live`
- `moonie-agent attach <session_id>`

V1 live entrypoint rules:

- packaged workflows only
- default profile is `mlx_gemma4_e2b_reasoner_only`
- runs launch in background under the shared runtime
- operator immediately attaches in terminal

Status: first scaffold implemented.

- `moonie-agent live --workflow-id <id>` launches and attaches
- `moonie-agent attach <session_id>` watches an existing session
- both use the same persistent runtime/session/event substrate

### Phase 3: Rich operator harness

Build a proper terminal operator surface with:

- top status bar
- event timeline
- summary / approvals / artifacts / sandbox side panel
- operator controls for approve, deny, resume, retry, expand, inspect, quit

Execution model:

- continuous execution by default
- pause only on approval gates, hard failures, or policy blocks

Status: command-driven operator scaffold implemented.

- `moonie-agent attach <session_id> --action approve|deny|resume|retry|quit`
- `moonie-agent inspect <session_id> --target sandbox|artifacts|policy|summary|scorecard`
- the live side panel shows completion metrics, including readiness and repair/raw-clean

The first replayable live-smoke packet wrapper now exists:

- [`scripts/run_runtime_live_smoke_packet.py`](../../scripts/run_runtime_live_smoke_packet.py)
- [`results/runtime_live_smoke_packets/20260506T_runtime_live_smoke_mlx_v2_runtime_live_smoke_packet`](../../results/runtime_live_smoke_packets/20260506T_runtime_live_smoke_mlx_v2_runtime_live_smoke_packet)
- [`results/runtime_live_smoke_packets/20260506T_runtime_live_smoke_mlx_trio_v2_runtime_live_smoke_packet`](../../results/runtime_live_smoke_packets/20260506T_runtime_live_smoke_mlx_trio_v2_runtime_live_smoke_packet)
- [`results/runtime_live_smoke_packets/20260506T_runtime_live_web_policy_mlx_v2_runtime_live_smoke_packet`](../../results/runtime_live_smoke_packets/20260506T_runtime_live_web_policy_mlx_v2_runtime_live_smoke_packet)
- [`results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet`](../../results/runtime_live_smoke_packets/20260506T_runtime_live_repeat_mlx_h1c_overlap_v2_runtime_live_smoke_packet)

The repeated live-web packet adds a stable CLI signal:

- `3` workflows x `3` repeats on `mlx_gemma4_e2b_reasoner_only`
- `failed_sessions = 0`
- `status_counts.completed = 3`
- `status_counts.awaiting_approval = 6`
- `controller_repair_avg = 0.6666666666666666`
- `argument_repair_avg = 0.5`
- `controller_fallback_avg = 0.16666666666666666`
- `raw_planning_clean_rate_avg = 0.3333333333333333`
- repeated repair families: dashboard `extract_layout`, finance `cli_search_logs`, jobs `cli_apply_patch`, and jobs live visual fallback
- analyzer outputs now summarize the same packet as `4` stable repair families and `7` stable policy-block families

The H1c comparison found a benchmark/runtime posture mismatch:

- the earlier clean H1c MLX primary row used a modular benchmark bundle with a heuristic router
- the CLI live profile uses monolith/reasoner-only execution
- after adding `--pipeline-name monolith` for `local_reasoner` H1 rows, the corrected H1c MLX primary run shows the same controller-dependence family as CLI live

Next step: add local MLX helper-ablation profiles and run a compact H1c monolith ablation.

### Phase 4: Gemini CLI adapter

Use Gemini CLI as:

- a reference for CLI ergonomics
- a reference for sandbox and trust concepts
- an external baseline on selected Moonie workflows

Do not treat Gemini CLI as a replacement for Moonie.

### Phase 5: Harder `H1` slice

Design a new slice that stresses:

- latest-instruction override under conflict
- clarify vs defer vs refuse judgment
- approval-safe stop behavior
- multi-tool-family decisions across CLI, API, and browser
- artifact revision after resume
- long-horizon constraint carryover

Status: `H1 v1` is now defined as a packaged-workflow-first slice.

- config: [`configs/knowledge_work_h1_slice.yaml`](../../configs/knowledge_work_h1_slice.yaml)
- note: [`docs/continuity/h1-slice.md`](./h1-slice.md)
- replayable episodes: `5`
- live episodes: `5`
- workflow families: executive dashboard review, executive stale brief packet, jobs visual form hold, finance billing patch hold, finance visual invoice review

Runner status:

- [`scripts/run_knowledge_work_h1_slice.py`](../../scripts/run_knowledge_work_h1_slice.py) validates the H1 config, emits a manifest, and delegates real runs to `scripts/run_knowledge_work_arena.py` with explicit episode filters
- focused tests cover H1 config validity, run-spec construction, ablation flag preservation, and arena command generation

Example smoke:

```bash
uv run python scripts/run_knowledge_work_h1_slice.py --dry-run --run-set primary --lane replayable_core
```

### Phase 6: Second ablation wave

Expand beyond the current three toggles.

Next likely targets:

- fallback planner path
- argument repair, especially `extract_layout`
- intent-priority overrides
- deterministic visual follow-on logic as its own isolated toggle
- specialist routing
- clarify/defer approval judgment fallback

Status: second-wave ablation controls are scaffolded for the middle three targets above.

- `disable_intent_priority`
- `disable_argument_repair`
- `disable_deterministic_visual_follow_on`

They are exposed through `ResearchControls`, model-registry rows, `run_knowledge_work_arena.py`, `run_knowledge_work_matrix.py`, and the H1 runner. Specialist routing and clarify/defer judgment fallback remain future targets.

## Acceptance Criteria

- we can launch `mlx_gemma4_e2b_reasoner_only` from CLI into a temp sandbox
- we can watch the run live in terminal
- we can approve, deny, and resume safely
- all writes stay inside the sandbox root
- live runs still emit traces, artifacts, and scorecards
- Gemini CLI can be run as a wrapped external baseline on selected tasks
- the harder `H1` slice produces non-saturated differences

## Files To Inspect First In A New Chat

- [`README.md`](../../README.md)
- [`docs/continuity/current-state.md`](./current-state.md)
- [`docs/continuity/next-steps.md`](./next-steps.md)
- [`docs/research-log.md`](../research-log.md)
- [`results/history/knowledge_work_board_latest.csv`](../../results/history/knowledge_work_board_latest.csv)
- [`src/gemma4_capability_map/runtime`](../../src/gemma4_capability_map/runtime)
- [`src/gemma4_capability_map/api`](../../src/gemma4_capability_map/api)
- [`src/gemma4_capability_map/research_controls.py`](../../src/gemma4_capability_map/research_controls.py)
- [`configs/knowledge_work_matrix_ablation_32_replayable.yaml`](../../configs/knowledge_work_matrix_ablation_32_replayable.yaml)

## Copy-Paste Prompt For A New Chat

```text
We are continuing work in `/Users/cheickdiakite/Codex/moonie`.

Please pick up from the latest repo state and continue systematically.

Important direction reset:
- Deprioritize all frontend/UI/UX work for now.
- Do not spend time refining React or Streamlit surfaces.
- Focus on research, benchmarking, harnessing, and live CLI testing.
- We still need the ability to watch the agent execute live, but CLI is the right surface right now.

Current understanding you should start from:
- The aligned exploratory `32 / 26` surface is partially saturated on top-line readiness.
- The real remaining signal is controller dependence, not just readiness.
- HF Gemma specialists still need materially more controller help than the clean MLX rows.
- The focused replayable ablation packet already showed:
  - controller repair is causal
  - controller fallback is causal
  - visual rescue is not doing useful work on that slice
- We want to understand the best ways of harnessing Gemma, especially local Gemma on MLX.
- Gemini CLI is relevant as a design reference and possible external baseline, but not a replacement for Moonie.

Your job:
1. Re-ground yourself in the current repo and docs.
2. Turn the repo direction into a CLI-first live harness plan.
3. Start implementing the next best move systematically.

Primary next phase:
- Build a CLI-first live operator harness for `mlx_gemma4_e2b_reasoner_only`
- Add proper sandboxing for live runs
- Use packaged workflows as the only live entrypoint in v1
- Add a Gemini CLI adapter as an external baseline/reference
- Then define a harder `H1` slice that breaks current top-line saturation
- Then run a second ablation wave on remaining controller helpers

Constraints:
- No new frontend work unless absolutely required for backend/runtime support.
- Prefer using the existing runtime/session/event/approval substrate.
- Keep live runs benchmark-backed and attributable to workflow families.
- Use Rich for terminal rendering, not a new frontend framework.
- Keep changes incremental and verified slice by slice.

Please begin by:
1. Reading the current source-of-truth docs and configs
2. Summarizing the repo’s current actual state
3. Writing a short execution plan for this CLI-first pivot
4. Implementing the first slice: runtime sandbox model + CLI live entrypoint scaffolding

Files to inspect first:
- `/Users/cheickdiakite/Codex/moonie/README.md`
- `/Users/cheickdiakite/Codex/moonie/docs/continuity/current-state.md`
- `/Users/cheickdiakite/Codex/moonie/docs/continuity/next-steps.md`
- `/Users/cheickdiakite/Codex/moonie/docs/research-log.md`
- `/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_board_latest.csv`
- `/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/runtime/`
- `/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/api/`
- `/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/app/`
- `/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/research_controls.py`
- `/Users/cheickdiakite/Codex/moonie/configs/knowledge_work_matrix_ablation_32_replayable.yaml`

Important note:
The docs currently still emphasize the React harness in places. Treat that as stale relative to this new direction and update the narrative as part of the pivot.

Work autonomously, verify as you go, and keep the focus on original research goals: understanding what actually improves Gemma harnessing, tool use, direction following, approvals, recovery, and live local execution.
```

## Working Rule For The Next Chat

Do not start by expanding product scope.

Start by making live CLI observation, sandbox safety, and causal harness attribution easier.
