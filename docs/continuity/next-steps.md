# Next Steps

## Immediate

### 1. Reduce HF Gemma specialist controller dependence on the clean aligned surface

This is the highest-value next move.

Why:

- the aligned `32 / 26` board now ties oracle, HF Gemma specialists, MLX Qwen, and MLX Gemma on top-line readiness
- the remaining meaningful difference is HF controller burden
- top-line reruns without controller cleanup would mostly restate the same result

Primary targets:

- `feedback_prior:refine_selection`
- `feedback_prior:read_region_text`
- remaining fallback-planner follow-ons after the latest patch

Success condition:

- keep the current aligned readiness tier
- lower `controller_repair_avg`
- lower `controller_fallback_avg`
- improve `raw_planning_clean_rate_avg`

### 2. Use the focused replayable ablation packet as the main Gemma research harness

Packet:

- [`results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v3_knowledge_work_ablation_packet`](../../results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v3_knowledge_work_ablation_packet)

Why this packet matters:

- it concentrates the remaining HF Gemma burden into a tractable slice
- it already shows the clean helper ranking:
  - baseline `0.9627777777777777`
  - `no_controller_repair = 0.6551777777777779`
  - `no_controller_fallback = 0.8182333333333333`
  - `no_visual_rescue = 0.9627777777777777`

Operational rule:

- do focused packet reruns first
- only do full aligned reruns after a controller change actually shifts the packet meaningfully

### 3. Install the local Gemma `31B` `GGUF` artifact and run the first real `llama.cpp` posture row

Current blocker:

- `GEMMA4_31B_GGUF_PATH` is unset
- there is no local Gemma `31B` `GGUF` bundle under `/Users/cheickdiakite/models`

Repo state:

- the code path and registry support already exist
- the missing piece is the actual local model artifact

Why this matters:

- runtime posture is now clearly part of capability research in this repo
- a real `31B` local posture row could change the Gemma story materially

### 4. Keep the product surface work benchmark-backed

Immediate rule:

- no parallel orchestration path
- CLI, API, operator console, and mobile companion should keep using the same runtime semantics the benchmark exercises

Why:

- the repo’s main thesis now depends on the benchmark and harness sharing one substrate

## Near Term

### 5. Extend harder realism where the current aligned surface is already clean

Bias additions toward:

- latest-instruction preservation
- ambiguous clarify-vs-defer judgment
- approval-safe stop behavior
- resume after changed direction
- revision after feedback
- visual referent follow-on pressure

Do not widen with easy completion-only tasks.

### 6. Expand tool-family and direction-following analysis, not just rows

Focus:

- function-call vs CLI vs API tool-family behavior
- wrong tool-family selection
- wrong args
- stale-instruction drift
- bad follow-on sequencing
- over-action when the right move is to stop or clarify

### 7. Decide whether the next Gemma posture step is MLX specialists or `31B`

Decision rule:

- if the next real research question is controller dependence, stay on HF specialists and the packet
- if the next real research question is runtime posture, prioritize the Gemma `31B` local artifact
- only add specialist-backed MLX Gemma after the current reasoner-only posture has paid off analytically

## Ongoing Discipline

- keep claims tied to same-surface reproduced runs
- keep external benchmark context separate from Moonie rows
- treat community signals as hypotheses, not evidence
- rescore old runs after scoring or planner-gap metric changes
- refresh history and continuity docs after every substantial benchmark pass

## Current Best Research Statement

Right now the repo supports this statement:

- Moonie materially improved Gemma 4 as a local full-stack agent
- oracle, HF Gemma specialists, MLX Qwen, and MLX Gemma now tie on top-line readiness on the aligned exploratory `32 / 26` surface
- that tie hides a real planner/controller difference:
  - HF Gemma specialists still need materially more controller help than the clean MLX rows

That is the seam to attack next.
