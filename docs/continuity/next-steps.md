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

Why:

- the board now has role/category/track exports plus runtime-facing columns like `warmup_load_ms` and `last_request_elapsed_ms`
- the new operator-console and mobile-companion views exist, but they still need richer shared view models and stronger product-level polish
- the next gap is comparison quality and workflow/product visibility, not raw data extraction

### 3. Broaden system coverage on the same full-lane surface

Focus:

- keep the full-lane specialist-backed references as the local baseline
- compare more systems against the same `24 / 18` KWA surface:
  - oracle
  - reasoner-only local
  - specialist-backed local
  - any future alternative local stacks

Why:

- the current `hf_service` specialist-backed stack now survives the full generated corpus cleanly
- the board now resolves legacy run ids to the registry and prefers `full_lane` exploratory comparisons, so adding new systems will land on a cleaner public-style comparison surface
- the next high-value benchmark question is comparative, not just wider volume
- the direct in-process specialist-backed rows are now repaired enough that future misses should be more informative than the closed referent-repair bug family

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
- focus especially on contradictions where a model must preserve the latest human visual constraint after an earlier valid selection
- use the new direct-HF full-lane comparison result to prioritize:
  - visual KWA revision loops
  - visual referent carryover
  - visual artifact quality under approval-sensitive holds
  - broader visual note/memo quality after review feedback rather than the now-closed invoice/form referent-repair path

## Ongoing Discipline

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
