# Next Steps

## Immediate

### 1. Harden the shared local-agent runtime as the common substrate

Goal:

- keep the benchmark and product surfaces on one execution model instead of drifting into separate orchestration paths

Why:

- the repo now has a real local runtime, CLI, local API, and Streamlit operator/mobile shells
- those surfaces are only worth keeping if they remain benchmark-backed and trace-compatible
- the next engineering risk is substrate divergence, not missing benchmark breadth

### 2. Push the board/reporting layer toward richer public-style cuts

Focus:

- extend the registry-backed board into a more publishable benchmark + product surface with role, category, track, latency, cost, and packaged-workflow cuts
- extend the new published external benchmark context layer with more official or clearly third-party rows, while keeping provenance labels explicit

Why:

- the board now has role/category/track exports plus runtime-facing columns like `warmup_load_ms` and `last_request_elapsed_ms`
- the board now also has a clean separate published external context layer for frontier-model benchmark references
- the new operator-console and mobile-companion views exist, but they still need richer shared view models and stronger product-level polish
- the next gap is comparison quality and workflow/product visibility, not raw data extraction

### 3. Broaden system coverage on the same full-lane surface

Focus:

- keep the current publishable-default Gemma specialist row as the local headline baseline
- compare more systems against the same `26 / 20` KWA surface:
  - oracle
  - reasoner-only local
  - specialist-backed local
  - the first real non-Gemma local stack that can run end to end on this machine

Why:

- the current direct in-process Gemma specialist row now matches the oracle row on the publishable-default full-lane board surface
- that gives the repo a strong “we made Gemma better” claim
- the next missing evidence is not more Gemma self-comparison; it is a real non-Gemma comparator on the same matrix
- external GPT/Gemini rows are now useful context, but they are not substitutes for a same-harness reproduced comparator
- do not claim a Qwen comparison until there is a completed full-lane Qwen run in the board/history layer
- the plumbing work is now done:
  - `Qwen/Qwen3-8B` is registered
  - the HF runner has a tokenizer-based text path for non-Gemma models
  - the experimental matrix includes the Qwen row
- Qwen should still be the first target once a real local checkpoint is available, because it is the clearest non-Gemma open-weight comparator for the publishable claim set

### 4. Deepen softer-realism scoring and harder episode design

Focus:

- inspect clean runs that still have bounded role-readiness loss from artifact or revision quality
- current examples:
  - bounded visual KWA slices with `artifact_quality_avg < 1.0`
  - future visual referent-carryover or stale-selection tasks
  - any regression in revision-heavy finance or jobs artifacts
  - broader visual memo/note artifacts that still rely on overly generic section synthesis

Why:

- the benchmark is now strong enough that soft-readiness deltas are often more informative than fail/pass transitions

## Near-Term

### 5. Deepen visual-tool realism after the next widening

Focus:

- add harder referent-carryover, stale-selection, and ambiguous-filter visual tasks
- push more visual episodes into job-shaped KWA slices instead of keeping them bounded
- keep extending native artifact grading instead of relaxing it
- the newest atomic realism additions are now:
  - dashboard backlog -> enablement-ops refinement
  - latest-issue -> email refinement
- count-heavy visual tasks are now also stricter at scoring time:
  - a wrong tool-side count no longer gets rescued by a lucky final-answer number mention
- focus especially on contradictions where a model must preserve the latest human visual constraint after an earlier valid selection
- use the new direct-HF full-lane comparison result to prioritize:
  - visual KWA revision loops
  - visual referent carryover
  - visual artifact quality under approval-sensitive holds
  - broader visual note/memo quality after review feedback rather than the now-closed invoice/form referent-repair path

## Ongoing Discipline

- keep claims honest:
  - strong current claim:
    - we improved Gemma 4 materially as a local full-stack agent on our own benchmark
  - non-claim until new evidence exists:
    - we have not yet shown Gemma beating Qwen on the same local full-lane surface
- keep CLI/API/operator/mobile surfaces honest about current capability:
  - packaged workflows are benchmark-backed bounded flows, not unbounded general autonomy
- use the runtime event contract consistently:
  - `created`
  - `warming`
  - `running`
  - `approval_required`
  - `completed`
  - `approved`
  - `denied`
  - `failed`
- keep canonical lane pointers clean
- checkpoint long model-backed runs
- rerun canonical oracle lanes after expanding the generated KWA corpus so continuity docs do not drift from the data roots
- rescore saved KWA traces after scoring-logic changes:
  - `uv run python scripts/rescore_knowledge_work_runs.py ...`
- append new findings to the research log
- refresh continuity files after every major benchmark pass
