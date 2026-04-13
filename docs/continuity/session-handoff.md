# Session Handoff

## Resume Here

The current research seam is no longer “make the rows tie.”

That part is done on the aligned exploratory `32 / 26` surface.

The current seam is:

- reduce HF Gemma specialist controller burden further
- without losing the current aligned readiness tier

## Current Source Runs

Aligned comparison surface:

- HF Gemma controller-burden rerun:
  - [`results/knowledge_work_matrix/20260413Taligned_controller_burden_patch_v2_knowledge_work_alignment_32_26`](../../results/knowledge_work_matrix/20260413Taligned_controller_burden_patch_v2_knowledge_work_alignment_32_26)
- oracle + MLX Gemma aligned reference:
  - [`results/knowledge_work_matrix/20260413Toracle_mlx_gemma_judgment_patch_v1_knowledge_work_alignment_32_26`](../../results/knowledge_work_matrix/20260413Toracle_mlx_gemma_judgment_patch_v1_knowledge_work_alignment_32_26)
- MLX Qwen aligned reference:
  - [`results/knowledge_work_matrix/20260412T235251Z_knowledge_work_alignment_32_26`](../../results/knowledge_work_matrix/20260412T235251Z_knowledge_work_alignment_32_26)

Focused replayable Gemma packet:

- [`results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v4_knowledge_work_ablation_packet_knowledge_work_ablation_packet`](../../results/knowledge_work_matrix/20260413Tresearch_ablation_focus_v4_knowledge_work_ablation_packet_knowledge_work_ablation_packet)

## Latest Headline Readout

Replayable `32`:

- oracle:
  - `readiness = 0.976853125`
  - `controller_repair_avg = 0.578125`
  - `controller_fallback_avg = 0.0`
- HF Gemma specialists:
  - `readiness = 0.976853125`
  - `controller_repair_avg = 0.71875`
  - `controller_fallback_avg = 0.28125`
  - `raw_planning_clean_rate_avg = 0.46875`
- MLX Qwen:
  - `readiness = 0.976853125`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`
- MLX Gemma:
  - `readiness = 0.976853125`
  - `controller_repair_avg = 0.0`
  - `controller_fallback_avg = 0.0`

Live `26`:

- oracle:
  - `readiness = 0.9791653846153847`
  - `controller_repair_avg = 0.7115384615384616`
- HF Gemma specialists:
  - `readiness = 0.9791653846153847`
  - `controller_repair_avg = 0.8076923076923077`
  - `controller_fallback_avg = 0.23076923076923078`
- MLX Qwen:
  - `readiness = 0.9791653846153847`
  - `controller_repair_avg = 0.0`
- MLX Gemma:
  - `readiness = 0.9791653846153847`
  - `controller_repair_avg = 0.0`

## What Just Changed

The latest pass added deterministic runtime execution for obvious visual follow-ons.

Code path:

- [`src/gemma4_capability_map/runtime/core.py`](../../src/gemma4_capability_map/runtime/core.py)
- [`src/gemma4_capability_map/tools/planner.py`](../../src/gemma4_capability_map/tools/planner.py)
- [`tests/test_tool_planner.py`](../../tests/test_tool_planner.py)
- [`tests/test_smoke_eval.py`](../../tests/test_smoke_eval.py)
- [`tests/test_trace_metrics.py`](../../tests/test_trace_metrics.py)

What that means:

- after a successful `extract_layout` or `refine_selection`, the runtime now auto-executes deterministic `refine_selection` / `read_region_text` follow-ons
- the runtime no longer asks the model again for those same obvious visual steps

## Measured Effect

Focused packet delta versus the prior packet:

- readiness unchanged at `0.9627777777777777`
- `controller_repair_avg` improved from `2.3333333333333335` to `0.8888888888888888`
- `feedback_prior:refine_selection` dropped from `16` to `0`
- `feedback_prior:read_region_text` dropped from `10` to `0`
- `controller_fallback_planner` stayed at `8`

Aligned full-lane delta for HF Gemma specialists:

- replayable:
  - `controller_repair_avg` improved from `1.296875` to `0.71875`
  - `controller_fallback_avg` stayed `0.28125`
  - readiness stayed `0.976853125`
- live:
  - `controller_repair_avg` improved from `1.5192307692307692` to `0.8076923076923077`
  - `controller_fallback_avg` stayed `0.23076923076923078`
  - readiness stayed `0.9791653846153847`

Interpretation:

- the old visual follow-on repairs were inflating controller burden
- removing them did not reduce the actual causal value of repair/fallback
- the remaining burden is now more honestly concentrated in fallback planner and non-visual repair families

## What Not To Re-Learn

Do not spend time re-proving:

- aligned top-line readiness parity exists
- MLX Gemma’s earlier executive-assistant judgment miss is closed
- MLX Qwen is a real same-surface comparator
- the direct in-process Gemma reasoner-only control is still materially weaker on the older reproduced surface

## Next Best Move

1. Attack the remaining HF Gemma specialist note families directly.
Primary targets:
   - `controller_fallback_planner`
   - `repaired_arguments:extract_layout`
   - `intent_prior:record_or_update`
   - `intent_prior:inspect_or_lookup`

2. Keep using the focused replayable packet first.
Only rerun the aligned `32 / 26` surface after the packet shifts again.

3. If the next question becomes runtime posture instead of controller dependence, switch to installing the Gemma `31B` local `GGUF` artifact and run the first real `llama.cpp` row.

## Verification State

Current code-side verification from the latest patch:

- targeted suite: `61 passed`

Benchmark outputs rebuilt:

- [`results/history/knowledge_work_board_latest.csv`](../../results/history/knowledge_work_board_latest.csv)
- [`results/history/knowledge_work_history.md`](../../results/history/knowledge_work_history.md)

## Operational Notes

- `output/` and `tmp/` remain untracked scratch dirs
- the Gemma `31B` lane is still blocked by local artifact availability:
  - `GEMMA4_31B_GGUF_PATH` unset
  - no local bundle under `/Users/cheickdiakite/models`
