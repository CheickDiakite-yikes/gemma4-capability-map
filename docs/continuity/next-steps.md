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
- the new harnessability wave should be validated through the same `function_call`, CLI, and API tool families before the product surface claims get broader

### 3. Broaden system coverage on the same full-lane surface

Focus:

- keep the current publishable-default Gemma specialist row as the local headline baseline
- compare more systems against the same widened `29 / 23` KWA surface:
  - oracle
  - reasoner-only local
  - specialist-backed local
  - the next real non-Gemma local stack after the first reproduced Qwen row

Why:

- the refreshed replayable `29`-episode Gemma specialist row is now strict/recovered clean on the widened surface
- that gives the repo a stronger “we made Gemma better” claim on the new harnessability-aware replayable lane
- the first real non-Gemma comparator now exists:
  - `mlx_qwen3_8b_reasoner_only` on the widened `29 / 23` lane
- that row already tells a useful story:
  - it beats the direct in-process Gemma reasoner-only control
  - it improved materially after the shared rescue/planner fixes
  - on the current widened `29 / 23` surface it now matches oracle and the Gemma specialist stack
- the next missing evidence is no longer “can we run Qwen at all”; it is whether the current widened surface is already saturated for the best harnessed rows and where Gemma still needs more controller help than Qwen
- the widened generated corpora now read `91 / 396 / 32 / 26`, and oracle, Gemma specialist, and Qwen now all have widened board rows on the current reproduced `29 / 23` surface
- external GPT/Gemini rows are now useful context, but they are not substitutes for a same-harness reproduced comparator
- the plumbing and first reproduced run are now done:
  - `hf_qwen3_8b_reasoner_only` exists as the direct-HF appendix path
  - `mlx_qwen3_8b_reasoner_only` exists as the Apple-Silicon-native benchmark row
  - the HF reasoner now forces deterministic decode and explicitly disables Qwen thinking-mode defaults in benchmarked text runs
  - the Gemma 4 `31B` `GGUF` / `llama.cpp` posture is still only an experimental support path until a local model/runtime is installed and reproduced
- the next concrete widening target is now explicit:
  - add harder direction-following and tool-use pressure where the current widened rows are now clean
  - add the Gemma 4 `31B` `GGUF` / `llama.cpp` posture row
  - widen beyond this single reproduced Qwen row to the next non-Gemma family or stronger Qwen posture
  - use the new planner-gap exports to guide the next Gemma-improvement pass:
    - reduce `controller_fallback_avg`
    - reduce `argument_repair_avg`
    - improve `raw_planning_clean_rate_avg`
    - focus first on the visual refine/readback tasks where Gemma specialists currently need controller repair even when the final episode score is clean

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
- the harder `v5` replayable smoke episodes currently saturate oracle, Gemma specialists, and Qwen MLX with identical tool traces
- that means the next discriminating move is not another blind full-lane rerun first
- it is either:
  - a more model-judgment-sensitive slice
  - or a Gemma-specific controller-cleanup pass followed by the widened `32 / 26` reruns

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
- stronger current same-surface comparative claim:
  - the Gemma specialist stack and the first reproduced local Qwen3 8B MLX row now both match the oracle row on the widened `29 / 23` board surface
- non-claim until new evidence exists:
  - we have not yet shown Gemma beating broader Qwen families or frontier closed models on the same local full-lane surface
- non-claim until the other rows are rerun:
  - the widened board now exists for oracle, Gemma specialist, and Qwen, but the direct Gemma reasoner-only control still sits on the older `26 / 20` surface and should not be merged into a fake same-surface parity claim
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
