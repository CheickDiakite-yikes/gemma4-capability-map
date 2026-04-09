# Architecture Improvements

This document tracks architecture changes that the benchmark has already justified,
plus the next improvements that are most likely to move the benchmark materially.

## Changes the benchmark already justified

- Split planning and final-synthesis token budgets.
  Full-stack traces showed correct tool execution followed by truncated final answers. The benchmark is cleaner and the agent is more reliable when planning stays cheap and the final answer gets its own larger budget.

- Add schema-aware repair at the planner boundary.
  Real FunctionGemma routing under renamed-field drift needed controller fallback to preserve variant schema keys instead of silently snapping back to canonical field names.

- Add schema-aware adaptation at the executor boundary.
  Variant schemas are benchmark-facing contracts, but runtime tool handlers still operate on canonical argument names. A thin adapter layer is required so renamed-field benchmark variants execute cleanly without corrupting trace arguments.

- Make tool success depend on validator success.
  Exact tool-name and argument-string matches are not enough if the call still fails validation or runtime execution. Tool-track success must remain operational, not just structural.

- Record backend posture before runs.
  Local runs on this machine are sensitive to MLX health, HF startup cost, and gated-model availability. Preflight artifacts are required context for interpreting results honestly.

- Prefer immutable run-group snapshots plus append-only history.
  The benchmark has already produced false confidence from partial reruns. Canonical run groups plus experiment history are required for trustworthy research claims.

## Improvements to implement next

- Separate thought-budget and answer-budget decoding in the HF thinking path.
  The integrated clean matrix is showing image-heavy failures and thought overflow in the HF thinking-on lane. A stronger implementation should cap or summarize thought tokens before final answer generation, or run answer synthesis as a second decode.

- Add image-task-specific prompting or specialist synthesis for multimodal reasoning.
  The expanded clean thinking track shows that image-grounding failures cluster on a few screenshot tasks. The likely fix is a stricter multimodal answer template or a dedicated screenshot-answer synthesis pass, not just more tokens.

- Add router-side intent priors for patch-record tasks.
  The integrated clean matrix and specialist drift matrix both exposed the same repeated failure family: real `FunctionGemma` routes `tool_008_patch_record` and especially `tool_012_billing_patch_record` to the wrong tool. The next architecture change should explicitly separate "inspect", "lookup", and "record/update patch" intents before schema repair runs.

- Add escalation/defer/refuse control prompts for no-tool judgment tasks.
  The canonical real-world matrix scored `0.0` on `think_013_prod_approval_escalation`. The model is not just slow there; it is failing the core job-like judgment of when to escalate instead of acting.

- Add multilingual answer normalization or second-pass synthesis for retrieval/full-stack outputs.
  Real `EmbeddingGemma` drift results and real specialist-backed modular drift results both lost accuracy on French answer variants while keeping the underlying retrieval or final state correct. That is an answer-surface problem, not a retrieval problem.

- Add refusal-aware billing routing priors.
  The real-world routing matrix fell to `0.5`, and the worst failures were not general routing noise. They clustered around high-cost billing patch and unsafe-billing-disable intents where the system should choose a safer tool or refuse action outright.

- Add reusable warmed worker processes for specialist lanes.
  Partially validated. A reusable warmed HF reasoner service is now required for clean real-world matrix execution on this Mac. The next step is to extend the same pattern to specialist lanes only if router/retriever warmup becomes the next dominant bottleneck.

- Add retrieval-aware planning context for modular full-stack runs.
  Right now retrieval hits are visible to model prompts, but heuristic planning logic does not consume them directly. A better planner interface would expose retrieved evidence in structured form so planning and repair can use it deterministically when needed.

- Add strict and recovered scorecards side by side for tool-bearing tasks.
  Current metrics now preserve operational correctness, but the benchmark would benefit from explicitly separating:
  - strict interface correctness
  - recovered execution correctness
  - final user-visible task success

- Add direct local-path and offline-first specialist bundles.
  The benchmark is local-first at runtime, but still sensitive to initial Hub setup and cache state. Local-path model bundles plus `GEMMA4_OFFLINE=1` should become the standard for stable repeat runs.

## Improvements to validate, not assume

- Real FunctionGemma on broader clean routing slices.
  Validated. The bounded specialist pass is clean, but the broader clean routing track regressed to `0.8333` because direct patch-record tasks still confuse the router. The remaining work is now architectural, not observational.

- Real EmbeddingGemma on broader drift slices.
  Validated with an important caveat. The specialist-backed drift matrix shows `Recall@k = 1.0` and `evidence_hit_rate = 1.0`, but final task success dropped to `0.9` because answer generation failed on two variants. The benchmark should treat the retriever as strong and the synthesis layer as the current bottleneck.

- Real specialist-backed modular full-stack.
  Validated on the current narrow slice. The clean specialist-backed modular lane is `1.0`, and the drift lane is `0.9375`. The remaining misses are one answer-language miss plus strict interface penalties caused by recovered routing mismatches.
