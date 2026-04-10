# Research Log

## 2026-04-09

### Scope advanced

- Strengthened multilingual multimodal coverage by replacing benchmark-critical French prompt variants with exact translations in [`language.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/stressors/language.py).
- Hardened multilingual answer scoring in [`answer_match.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/metrics/answer_match.py) so accented French action phrases and weekday/time phrases map cleanly to benchmark expectations.
- Added two new screenshot-centric thinking tasks in [`make_gold.py`](/Users/cheickdiakite/Codex/moonie/scripts/make_gold.py): `think_011_incident_screenshot_toggle` and `think_012_billing_invoice_lock`.
- Replaced heuristic-only specialist placeholders with real HF-capable `FunctionGemmaRunner` and `EmbeddingGemmaRetriever` implementations in [`functiongemma_runner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/models/functiongemma_runner.py) and [`embeddinggemma_runner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/models/embeddinggemma_runner.py).
- Expanded the alpha corpus to `52` gold tasks and regenerated `232` explicit variants, keeping the benchmark balanced at `13` tasks per track.
- Added two new routing tasks, two new retrieval tasks, and two new retrieval-bearing full-stack tasks in [`make_gold.py`](/Users/cheickdiakite/Codex/moonie/scripts/make_gold.py) so broader specialist-backed comparisons are possible without changing the benchmark contract.
- Added a first-class real-world autonomy layer:
  - task-level `benchmark_tags`
  - task-level `real_world_profile`
  - trace-level propagation of those fields
  - real-world metrics such as `state_integrity_score`, `collateral_damage_free`, `intervention_free_success`, and `real_world_readiness_score`
- Tagged `16` current tasks as real-world job-like probes across `release_ops`, `billing_ops`, `calendar_ops`, `incident_ops`, `finance_ops`, and `access_ops`.
- Added the first dedicated real-world matrix at [`alpha_real_world_matrix.yaml`](/Users/cheickdiakite/Codex/moonie/configs/alpha_real_world_matrix.yaml) plus design notes in [`real-world-benchmarking.md`](/Users/cheickdiakite/Codex/moonie/docs/real-world-benchmarking.md).
- The real-world matrix is now service-backed and canonical on this machine:
  - the previous in-process real-world run stalled during repeated HF warmup and should not be treated as authoritative
  - [`20260409T210500Z_alpha_real_world`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T210500Z_alpha_real_world) completed cleanly with subprocess-isolated experiments and `hf_service` as the Gemma 4 reasoner path
- `KnowledgeWorkArena` has now been hardened beyond simple markdown-contract episodes:
  - replayable-core grew to `15` episodes and live-web stress grew to `9`
  - new partial-progress hold episodes now require the correct move to be `defer`, `escalate`, or `refuse to send` after useful work has already been completed
  - browser traces now capture validation rules, state updates, approval gates, blocked reasons, sandbox endpoints for dry-run submissions, and explicit state-machine transitions
  - finance and job artifacts now materialize as real `.xlsx`, `.pptx`, and `.docx` work products before grading
  - artifact graders now check formulas, deck section structure, revision diffs, and application-packet consistency instead of only generic fragment presence
  - long `KnowledgeWorkArena` runs now checkpoint after each episode through `progress.json`, partial traces, and partial summaries instead of only writing at the end
  - the current canonical `KnowledgeWorkArena` summaries are [`results/knowledge_work/replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/replayable_core/summary.json) and [`results/knowledge_work/live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/live_web_stress/summary.json)
  - the first finished non-oracle episode baseline now exists at [`results/knowledge_work/model_backed_hf_exec_hold/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_exec_hold/summary.json)
  - a broader multi-episode HF reasoner pilot exists at [`results/knowledge_work/model_backed_hf_reasoner_pilot/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/model_backed_hf_reasoner_pilot/summary.json), currently stopped after two completed episodes so the repo has a clean finished model-backed baseline plus a separate partial pilot artifact

### Backend findings

- Backend preflight now has a dedicated artifact at [`backend_preflight.json`](/Users/cheickdiakite/Codex/moonie/results/tables/backend_preflight.json) and [`backend_preflight.md`](/Users/cheickdiakite/Codex/moonie/results/tables/backend_preflight.md).
- Backend posture is session-sensitive. Earlier preflight snapshots showed MLX crashing with a Metal initialization exception; the latest preflight at [`backend_preflight.json`](/Users/cheickdiakite/Codex/moonie/results/tables/backend_preflight.json) shows MLX healthy again and currently recommends `mlx` for the local default path.
- HF auth is present and detected via `HF_TOKEN`; local backend posture should now be decided from preflight rather than fixed prose.
- A reusable local `hf_service` reasoner path now exists so repeated benchmark runs can reuse a warmed HF model instead of paying full warmup on each experiment.
- Service observability is now first-class through `state.json`, `events.jsonl`, `service.log`, and `requests.jsonl` under [`results/runtime/hf_reasoner`](/Users/cheickdiakite/Codex/moonie/results/runtime/hf_reasoner).
- The first canonical real-world run justified `hf_service` as a research-execution primitive on this Mac even though preflight may still recommend `mlx` for the general local default. The issue was not inference speed alone; it was repeated HF warmup stability across a matrix.
- The reusable worker is not the current default on this Mac. Cold fresh-process HF startup can be dominated by import cost before model loading; the live worker probe showed `torch` import around `75.6s` and `torch + transformers` around `214.4s`, while the warmed standalone import probe later measured `1.9s` and `4.1s`. That gap is now a tracked observability variable instead of an assumption.
- The first finished model-backed `KnowledgeWorkArena` executive episode re-confirmed that cold service startup remains a first-order cost on this machine:
  - fresh `hf_service` boot to ready on `google/gemma-4-E2B-it` took about `345s`
  - once ready, the episode itself completed cleanly with `artifact_quality = 1.0`, `strict_interface = 1.0`, `recovered_execution = 1.0`, and `role_readiness = 0.9056`
- The stopped two-episode HF reasoner pilot also produced usable partial evidence:
  - completed `kwa_jobs_tailored_packet`
  - completed `kwa_finance_three_statement_model`
  - partial `role_readiness_avg = 0.9074`
- Backend preflight now marks dead service workers as `stale` instead of treating their last `state.json` as a live `loading` service.
- A dedicated import-timing artifact now exists at [`hf_import_probe.json`](/Users/cheickdiakite/Codex/moonie/results/tables/hf_import_probe.json) and [`hf_import_probe.md`](/Users/cheickdiakite/Codex/moonie/results/tables/hf_import_probe.md).

### Benchmark posture

- Added a dedicated specialist probe matrix at [`alpha_specialist_matrix.yaml`](/Users/cheickdiakite/Codex/moonie/configs/alpha_specialist_matrix.yaml) for:
  - multilingual multimodal thinking verification
  - real EmbeddingGemma retrieval probes
  - real FunctionGemma routing probes
- The focused specialist rerun at [`20260409T120000Z_alpha_specialist_probe`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T120000Z_alpha_specialist_probe) completed on the real HF path with `4/4` success. That closes the last known French screenshot drift miss on the authoritative HF multimodal slice.
- The subsequent specialist replacement pass clarified the remaining blockers:
  - [`20260409T160700Z_alpha_specialist_probe`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T160700Z_alpha_specialist_probe) showed `EmbeddingGemma` is blocked by Hugging Face manual gating, not by the local harness.
  - [`20260409T161500Z_alpha_specialist_probe`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T161500Z_alpha_specialist_probe) showed the original FunctionGemma repo id was wrong in config, and after correcting it to `google/functiongemma-270m-it`, the real blocker is also Hugging Face manual gating.
- Specialist access is now recorded proactively in [`specialist_access_probe.json`](/Users/cheickdiakite/Codex/moonie/results/tables/specialist_access_probe.json) and [`specialist_access_probe.md`](/Users/cheickdiakite/Codex/moonie/results/tables/specialist_access_probe.md) so these failures surface before a matrix run starts.
- Real specialist replacement is now materially validated rather than partially aspirational:
  - [`20260409T163000Z_alpha_specialist_probe`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T163000Z_alpha_specialist_probe) established that real `EmbeddingGemma` retrieval is strong (`1.0` success on the bounded retrieval slice), while real `FunctionGemma` routing was weaker under renamed-field schema drift (`0.75` success).
  - The first repair pass in [`planner.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/planner.py) fixed controller fallback so renamed schema keys are preserved when the router output collapses to pads.
  - The second repair pass in [`executor.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/tools/executor.py) fixed the runtime adapter so renamed schema keys are translated back to canonical handler arguments without losing the original variant arguments in the trace.
  - Tool-track scoring in [`tool_eval.py`](/Users/cheickdiakite/Codex/moonie/src/gemma4_capability_map/evals/tool_eval.py) now requires validator success, closing a benchmark-integrity gap where malformed executions could still appear as successful if tool choice and argument strings matched the gold event.
- The corrected specialist routing rerun at [`20260409T172000Z_alpha_specialist_probe`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T172000Z_alpha_specialist_probe) is now the authoritative FunctionGemma routing snapshot: `8/8` success, `recovery_correct = 1.0`, `malformed_call_rate = 0.0`, and no failing variants.
- The integrated clean matrix at [`20260409T180500Z_alpha_integrated_specialists`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T180500Z_alpha_integrated_specialists) is now the authoritative expanded clean snapshot:
  - local heuristic clean baselines remain strong on retrieval and full-stack (`mlx` retrieval `1.0`, `mlx` modular full-stack `1.0`)
  - the expanded clean thinking track is weaker than the earlier narrow slice (`mlx` thinking `0.8333`, `hf` thinking-off `0.9167`)
  - `hf` thinking-on materially regressed on the expanded image-heavy clean slice (`0.75`) with `thinking_overflow`, `generation_truncated`, and image-grounding misses
  - real `EmbeddingGemma` clean retrieval on the full `12`-task retrieval track remained `1.0`
  - real `FunctionGemma` clean routing on the full `12`-task routing track fell to `0.8333`, with both failures concentrated on direct patch-record intents: `tool_008_patch_record_clean` and `tool_012_billing_patch_record_clean`
  - real specialist-backed modular full-stack remained `1.0` on the narrow retrieval-bearing clean slice
- The specialist-backed drift matrix at [`20260409T190500Z_alpha_specialist_drift`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T190500Z_alpha_specialist_drift) is now the authoritative variant snapshot for real specialist lanes:
  - real `EmbeddingGemma` retrieval variants scored `0.9` task success while keeping `Recall@k = 1.0` and `evidence_hit_rate = 1.0`; both misses were answer-side, not retrieval-side
  - the failing retrieval variants were `retr_011_approval_policy_language_fr` and `retr_012_rollout_toggle_multimodal_context_long_history`
  - real `FunctionGemma` routing variants scored `0.75`; every failure came from `tool_012_billing_patch_record` across clean, code-switched, schema-renamed, and context-noised variants, all with the same `wrong_tool` + `arg_mismatch` signature
  - real specialist-backed modular full-stack variants scored `0.9375`; the only user-visible failing variant was `agent_011_runbook_guided_patch_language_fr` with `answer_mismatch`, while strict interface metrics remained lower because two tool-selection mismatches were recovered downstream
- The first canonical real-world autonomy snapshot at [`20260409T210500Z_alpha_real_world`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T210500Z_alpha_real_world) established a sharper boundary between bounded task execution and true job-like autonomy:
  - `hf_e2b_real_world_thinking_variants`: `0.0` success, `0.0` readiness
  - `hf_e2b_real_world_retrieval_variants`: `0.875` success, `1.0` strict interface, `0.88125` readiness
  - `hf_e2b_real_world_routing_variants`: `0.5` success, `0.5` strict interface, `0.6375` readiness
  - `hf_e2b_real_world_full_stack_variants`: `0.75` strict success, `1.0` recovered execution, `0.8696` readiness
  - the biggest real-world failure families are now explicit:
    - no-tool escalation judgment on `think_013_prod_approval_escalation`
    - French answer-surface misses on retrieval and full-stack variants
    - billing-patch and unsafe-billing-disable routing/refusal failures under real FunctionGemma routing

### Architecture findings

- Real retrieval is stronger than real routing under drift on the current benchmark. `EmbeddingGemma` kept perfect retrieval evidence metrics under variants, while `FunctionGemma` exposed a stable intent-class weakness around direct patch-record requests.
- The current bottleneck for retrieval-backed tasks is answer synthesis, not document finding. This is the strongest current argument for splitting retrieval quality claims from final-answer quality claims in published reporting.
- HF thinking-on is not currently the default reasoning path for this machine or benchmark slice. On the expanded clean thinking track, it is slower and less reliable than thinking-off because of overflow and truncation behavior.
- Patch-oriented routing needs a stronger intent prior than the current specialist stack provides. The repeated `tool_012_billing_patch_record` failures suggest a real router-side ambiguity between "inspect/lookup" and "propose/update patch record" tool classes.
- Real specialist-backed modular full-stack is operationally viable. Even under drift, it stayed above `0.93` success on the current narrow slice, which is enough to justify scaling that lane rather than treating it as experimental-only.
- Real-world autonomy is materially weaker than bounded task execution.
  - The model can preserve state and complete many operational tasks once tools are in motion.
  - It is still weak at deciding when not to act, when to escalate, and when a high-cost billing intent should be refused or redirected.
- Answer-surface multilinguality is still an end-to-end bottleneck.
  - Retrieval evidence and final state can both be correct while the real-world benchmark still fails because the answer layer misses the required French action phrasing.

### Verification

- Narrow regression suite passes after the scoring and runtime-preflight changes:
  - `tests/test_answer_match.py`
  - `tests/test_stressors.py`
  - `tests/test_runtime_utils.py`
- Additional routing/runtime integrity regressions now pass:
  - `tests/test_tool_planner.py`
  - `tests/test_executor.py`
  - `tests/test_tool_eval.py`
  - `tests/test_gemma4_runner.py`
- Real-world execution/reporting integrity also passes after the new service-backed matrix work:
  - `tests/test_alpha_matrix_script.py`
  - `tests/test_benchmark_module.py`
  - `tests/test_runtime_utils.py`
  - `tests/test_real_world_metrics.py`
  - `tests/test_replay_summary.py`
