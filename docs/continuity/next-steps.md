# Next Steps

## Immediate

### 1. Push the board/reporting layer toward richer public-style cuts

Goal:

- extend the registry-backed board into a more publishable benchmark surface with role, category, track, latency, and cost cuts

Why:

- the board now has role/category/track exports plus runtime-facing columns like `warmup_load_ms` and `last_request_elapsed_ms`
- the next gap is comparison quality and wider system coverage, not basic data extraction

### 2. Broaden system coverage on the same full-lane surface

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
- the new direct in-process HF specialist-backed system is now on the board and partially recovers the direct-HF reasoner-only drop, which sharpens the next comparative questions:
  - which remaining in-process misses are visual-controller issues versus visual artifact-generation issues?
  - do those misses stay concentrated in the same bounded visual KWA subset as we add more local/open-weight systems?

### 2a. Immediate comparative follow-up

Focus:

- debug the concentrated direct-HF specialist misses before widening to more systems blindly
- current worst rows:
  - replayable:
    - `kwa_finance_visual_invoice_hold`
  - live:
    - `kwa_finance_live_visual_invoice_hold`
    - `kwa_jobs_live_visual_form_hold`

Why:

- the new full-lane comparison already answered the first-order question:
  - real specialists help the direct-HF path materially
  - but they do not fully close the gap to the `hf_service` specialist-backed baseline
- the next high-value work is to understand the remaining visual KWA losses, not to immediately pile on more systems with the same unresolved blind spot
- `mlx` is still blocked locally, so it should not be the next execution target until the runtime exists on this machine

### 3. Deepen softer-realism scoring and harder episode design

Focus:

- inspect clean runs that still have bounded role-readiness loss from artifact or revision quality
- current examples:
  - bounded visual KWA slices with `artifact_quality_avg < 1.0`
  - future visual referent-carryover or stale-selection tasks
  - any regression in revision-heavy finance or jobs artifacts

Why:

- the benchmark is now strong enough that soft-readiness deltas are often more informative than fail/pass transitions

## Near-Term

### 4. Deepen visual-tool realism after the next widening

Focus:

- add harder referent-carryover, stale-selection, and ambiguous-filter visual tasks
- push more visual episodes into job-shaped KWA slices instead of keeping them bounded
- keep extending native artifact grading instead of relaxing it
- focus especially on contradictions where a model must preserve the latest human visual constraint after an earlier valid selection
- use the new direct-HF full-lane comparison result to prioritize:
  - visual KWA revision loops
  - visual referent carryover
  - visual artifact quality under approval-sensitive holds
  - visual invoice/layout tasks where specialists still lose strict or recovered execution

## Ongoing Discipline

- keep canonical lane pointers clean
- checkpoint long model-backed runs
- rerun canonical oracle lanes after expanding the generated KWA corpus so continuity docs do not drift from the data roots
- rescore saved KWA traces after scoring-logic changes:
  - `uv run python scripts/rescore_knowledge_work_runs.py ...`
- append new findings to the research log
- refresh continuity files after every major benchmark pass
