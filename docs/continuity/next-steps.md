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

Next implementation moves:

- run the first real MLX live smoke through the CLI harness
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

### 4. Install the local Gemma `31B` `GGUF` artifact and run the first real `llama.cpp` posture row

Current blocker:

- `GEMMA4_31B_GGUF_PATH` unset
- no local Gemma `31B` `GGUF` under `/Users/cheickdiakite/models`

Why it matters:

- runtime posture is now clearly part of the capability story
- a real `31B` row is still missing from the research package

## Near Term

### 5. Add Gemini CLI as a design reference and external baseline

Use Gemini CLI for:

- CLI ergonomics reference
- trust and sandbox design reference
- wrapped external baseline on selected Moonie workflows

Do not use Gemini CLI as a replacement for Moonie or as proof that local MLX Gemma harnessing is solved.

### 6. Expand the failure taxonomy, not just the leaderboard

Keep pushing on:

- tool-family choice
- argument fidelity
- direction-following under conflicting instructions
- approval-safe stop behavior
- clarify vs defer vs refuse judgment quality
- artifact revision quality after feedback

### 7. Keep the live harness benchmark-backed

The runtime, CLI, API, operator console, and mobile companion should keep sharing the same execution semantics as the benchmark.

No parallel orchestration path should be introduced.

### 8. Decide the next Gemma posture bet explicitly

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
