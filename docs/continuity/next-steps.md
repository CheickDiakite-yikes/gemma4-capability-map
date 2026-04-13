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
- keep the aligned exploratory `32 / 26` board surface honest
- now work the residual differences inside that aligned surface:
  - specialist-backed local Gemma controller dependence
  - MLX Gemma readiness gap
  - Gemma `31B` runtime-posture expansion

Why:

- the aligned exploratory `32 / 26` matrix now exists for:
  - `oracle_gemma4_e2b`
  - `hf_gemma4_e2b_specialists_cpu`
  - `mlx_qwen3_8b_reasoner_only`
  - `mlx_gemma4_e2b_reasoner_only`
- oracle, HF Gemma specialists, and MLX Qwen now tie on top-line replayable and live readiness on that aligned surface
- HF Gemma specialists still rely on materially more controller repair and fallback than MLX Qwen
- MLX Gemma is now aligned on the same surface and stays controller-clean, but still lands slightly lower readiness
- the widened generated corpora now read `91 / 396 / 32 / 26`
- external GPT/Gemini rows are now useful context, but they are not substitutes for a same-harness reproduced comparator
- the plumbing and first reproduced run are now done:
  - `hf_qwen3_8b_reasoner_only` exists as the direct-HF appendix path
  - `mlx_qwen3_8b_reasoner_only` exists as the Apple-Silicon-native benchmark row
  - `mlx_gemma4_e2b_reasoner_only` now exists as the Apple-Silicon-native Gemma posture row
  - the HF reasoner now forces deterministic decode and explicitly disables Qwen thinking-mode defaults in benchmarked text runs
  - the Gemma 4 `31B` `GGUF` / `llama.cpp` posture is still only an experimental support path until a local model/runtime is installed and reproduced
- the next concrete target is now explicit:
  - reduce HF Gemma specialist `controller_fallback_avg`
  - reduce HF Gemma specialist `controller_repair_avg`
  - improve HF Gemma specialist `raw_planning_clean_rate_avg`
  - inspect the residual MLX Gemma readiness gap now that the replayable miss is closed and the row is aligned
  - add harder direction-following and tool-use pressure where the current aligned `32 / 26` rows are now clean
  - add the Gemma 4 `31B` `GGUF` / `llama.cpp` posture row

Immediate research execution order:

1. Run the focused replayable Gemma ablation packet, not a blind full-lane rerun.
   The current 9-episode packet already explains `35.5 / 65.5` replayable controller repairs and `18.5 / 33.0` replayable controller fallbacks on the HF Gemma specialist row.
2. Patch the residual MLX Gemma executive-assistant judgment miss.
   The remaining aligned readiness gap is concentrated in `kwa_exec_travel_conflict_resolution` and `kwa_exec_vendor_access_hold`, and both are escalation-correctness misses rather than visual or tool-execution misses.
3. Install the local Gemma `31B` `GGUF` artifact and wire `GEMMA4_31B_GGUF_PATH`.
   The runtime path already exists; the current blocker is the missing local model.
4. Only after those three steps, rerun the aligned comparison surface again.
   Another full-lane rerun before the ablations and judgment patch would mostly restate the same finding.

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
  - or a Gemma-specific controller-cleanup pass followed by another aligned `32 / 26` rerun

More precise target now:

- bias the next harder-realism additions toward executive-assistant ambiguity, clarify-vs-defer judgment, and approval-language precision
- the residual MLX Gemma gap is already telling us that these seams are more discriminating than another generic visual difficulty increase

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
  - on the aligned exploratory `32 / 26` board surface, oracle, HF Gemma specialists, and MLX Qwen now tie on top-line replayable and live readiness
  - MLX Gemma is controller-clean on that same surface, but still lands slightly lower readiness
- non-claim until new evidence exists:
  - we have not yet shown Gemma beating broader Qwen families or frontier closed models on the same local full-lane surface
- non-claim until the other rows are rerun:
  - the direct Gemma reasoner-only control still sits on the older `26 / 20` surface and should not be merged into a fake same-surface parity claim
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
