# Key Learnings

These are the durable findings worth carrying forward.

## 1. Interface failures appear before raw reasoning failures

Schema drift, validator retries, truncated synthesis, answer-surface mismatches, and browser-state mistakes show up sooner and more consistently than pure “the model cannot think” failures.

## 2. Final completion is not enough

The benchmark must keep `strict_interface`, `recovered_execution`, `artifact_quality`, `browser_workflow`, and `role_readiness` separate. Systems can reach a correct end state while still being operationally untrustworthy.

## 3. Domain-native grading materially improves benchmark quality

Schedules need fields. Models need formulas. Decks need structure and revision diffs. Application packets need consistency. Generic text grading is too weak for knowledge-work claims.

## 4. Browser realism is about state and guardrails, not just clicks

Validation failures, approval gates, blocked submissions, and sandbox-only endpoints are central to realistic autonomy measurement. Many real tasks should stop safely rather than complete aggressively.

## 5. Specialist models help most on interface-heavy surfaces

Real `EmbeddingGemma` helped retrieval evidence quality. Real `FunctionGemma` helped routing once schema-aware repair and intent priors were in place. The biggest wins came where interfaces were ambiguous, not where free-form wording was hard.

## 6. Cold-start behavior is part of capability reality

On this Apple Silicon machine, service warmup and runtime posture change what is practically benchmarkable. The first finished KWA model-backed executive run paid about `345s` of cold startup before task execution.

## 7. Judgment is harder than bounded execution

The hardest real-world surfaces remain:

- escalate
- defer
- clarify
- refuse

especially after useful progress has already been made.

## 8. Benchmark engineering changes the truth you can see

Checkpointed runs, canonical lane pointers, native artifact generation, browser state machines, and history reporting all changed the quality of the conclusions. Benchmark infrastructure is part of the research object.
