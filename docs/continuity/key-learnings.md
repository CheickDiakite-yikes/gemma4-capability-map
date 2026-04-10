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

The newer bounded specialist-backed finance pilot warmed much faster, about `37.9s`, which means the service-backed path is improving. The current blocker has shifted from “can the stack load?” to “can the stack stay interface-clean while executing?”

## 7. Judgment is harder than bounded execution

The hardest real-world surfaces remain:

- escalate
- defer
- clarify
- refuse

especially after useful progress has already been made.

## 8. Benchmark engineering changes the truth you can see

Checkpointed runs, canonical lane pointers, native artifact generation, browser state machines, and history reporting all changed the quality of the conclusions. Benchmark infrastructure is part of the research object.

## 9. Native-file grading catches a better class of mistakes

Once grading moved onto real `.xlsx`, `.pptx`, and `.docx` outputs, the benchmark started checking workbook rows/formulas, deck structure, and document field consistency directly. That is the right direction if the claim is job readiness rather than text similarity.

## 10. Specialist-backed composition exposes the next real weakness

The first fully specialist-backed finance KWA pilot completed with perfect artifacts and recovered execution, but only `0.7` strict interface. That means the next frontier is controller/router exactness under composition, not loading stability or final artifact generation.

That specific leak is now partially closed. Once the planner respected feedback-prior next steps after successful tool execution, the bounded cross-role specialist-backed run reached `1.0` strict interface across executive, jobs, and finance. The next weaknesses are likely to be subtler than repeated-tool controller drift.

## 11. Canonical benchmark state and historical “latest” state can diverge

Once exploratory policy-only runs share the same lane labels as canonical runs, generic “latest by lane” views can become misleading. The durable source of truth must be the canonical summary pointers, not whichever run most recently wrote a lane-tagged summary.

## 12. Partial-progress policy judgment can now stay interface-clean in the oracle lane

The new replayable and live policy-hardening subsets kept `strict_interface = 1.0` while forcing the system to recover from validation failures and then defer, escalate, or refuse after useful progress. That means the next benchmark question is no longer whether the lane can represent these branches cleanly; it is whether real model-backed specialists can keep those scores under the same conditions.

## 13. Judgment-mode evaluation needs its own contract

Replayable policy hardening exposed that generic fragment matching is too brittle for judgment tasks. `action: refuse` plus the right safety basis should not fail just because the model says `high-risk` or `critical safety control` instead of a legacy fragment like `cannot`.

The benchmark is better when judgment-mode tasks score against:

- expected action class
- required safety / approval / ambiguity basis
- fallback semantic aliases for operational phrasing

## 14. Replayable policy judgment moved from model weakness to benchmark-contract weakness

The first real specialist-backed replayable policy runs looked weak, but the decisive fix was in the benchmark contract: judgment-aware scoring, judgment-aware rescue acceptance, and broader semantic aliases for policy-safety language.

Once that landed, the replayable specialist-backed policy subset matched the live subset on the bounded three-episode slice:

- `strict_interface = 1.0`
- `recovered_execution = 1.0`
- `escalation_correctness = 1.0`

That means the previous replayable weakness was real as a benchmark finding, but the root cause was evaluation rigidity rather than a stable inability of the model stack to refuse or clarify correctly.

## 15. Clarify vs defer needs an explicit precedence rule

The live `agent_013_ambiguous_vendor_defer` miss showed a separate judgment failure mode: once an episode already contains approval-gate framing, the model can over-index on `defer` even when the target itself is still ambiguous.

The benchmark got better once the judgment contract stated the precedence directly:

- if the exact target is ambiguous, choose `clarify`
- only choose `defer` once the target is understood and the blocker is truly approval or prerequisite state

## 16. Broader specialist-backed policy slices are now stable

After the judgment-scoring hardening and the clarify-precedence prompt fix, both broader 6-episode specialist-backed policy slices are clean on:

- `strict_interface = 1.0`
- `recovered_execution = 1.0`
- `escalation_correctness = 1.0`

That is enough evidence to move the next frontier away from this policy bug family and toward broader model-backed episode volume.

## 17. Parallel evidence tasks need controller-enforced completeness

The broader replayable cross-role specialist-backed `v1` run exposed a different benchmark truth: schema-valid tool calls are not good enough when the task contract requires multiple evidence sources before mutation.

In `agent_010_parallel_audit_patch`, the model produced:

- `inspect_image`
- then a schema-valid but semantically wrong `propose_patch`
- only afterward `read_repo_file`

That failure was not fixed by more prompt pressure. The decisive fix was controller-side:

- force the full pending parallel evidence batch before accepting a partial plan
- block `propose_patch` until both image and repo evidence have passed
- repair patch arguments from combined successful feedback instead of trusting the latest single tool call

That is a durable benchmark lesson: when the workflow contract requires multiple evidence sources, the controller has to enforce that contract explicitly.

## 18. The broader specialist-backed cross-role slice is now stable in both replayable and live lanes

The current 9-episode replayable and 9-episode live cross-role specialist-backed slices are both clean on:

- `strict_interface = 1.0`
- `recovered_execution = 1.0`
- `escalation_correctness = 1.0`

That matters because it shifts the frontier again. The next benchmark question is not whether the current balanced specialist-backed slice works. It is how quickly performance degrades as we widen episode volume, increase mixed-evidence composition, and push harder revision-heavy or approval-gated scenarios.

## 19. Human-work evals get better when they explicitly forbid the wrong “reasonable” answer

The new stale-context, constraint-preservation, and stale-assumption episodes improved because the benchmark now says not only what must be present, but what must not survive:

- stale draft versions
- recruiter-overridden candidate constraints
- stale finance assumptions

`forbidden_fragments` turned out to be a useful benchmark primitive. In real work, many failures are not omission failures. They are “the system kept a plausible but superseded thing alive” failures.

## 20. Canonical benchmark runners must default to full-lane execution

The first attempt to refresh the harder canonical KWA lanes surfaced a benchmark-infrastructure bug: `run_knowledge_work_arena.py` still defaulted to `--limit 12`, which silently truncated the canonical lane.

That is a durable lesson. Canonical paths should never have surprise truncation defaults. A benchmark can look stable while silently measuring the wrong surface.

## 21. Harder human-nuance episodes are clean in isolation on the specialist-backed stack

The new replayable and live three-episode slices for:

- stale-context repair
- original-constraint preservation
- stale-assumption repair

are now clean on the fully specialist-backed stack:

- `artifact_quality = 1.0`
- `strict_interface = 1.0`
- `recovered_execution = 1.0`
- `escalation_correctness = 1.0`

That matters because these harder human-style failure modes are now validated on the model-backed stack, not just in oracle form.

## 22. The next frontier is composition of human nuances, not isolated handling

Once the hard-human slices are clean in isolation, the next benchmark question changes again:

- can the stack handle stale context plus approval pressure plus revision comments in the same episode?
- can it preserve the human’s original constraint after multiple rounds of external pressure?
- can it repair to the latest assumption set and still avoid unnecessary action?

The next useful benchmark gains will come from composing these pressures together, not just adding more isolated episodes.

## 23. Internal leaderboard-quality reporting needs normalized system metadata

Once KWA started producing multiple canonical and exploratory system-backed runs, raw summary files stopped being enough for ranking and scatter views.

The useful reporting primitive is now:

- registry-backed system metadata
- explicit `system_id`
- `lane`
- `run_intent`

That is what lets the repo produce stable board exports without conflating oracle, heuristic, and specialist-backed local systems.

## 24. Mixed-pressure widening exposed a real replayable policy bug that isolated slices missed

The first replayable `hardmix` specialist-backed run was the right kind of failure:

- live stayed clean
- replayable failed only once
- the miss was concentrated in `kwa_finance_billing_patch_hold`

That meant the next step was not “the stack is bad at mixed pressure.” It was “find the exact refusal-versus-escalate contract leak under composition.” The corrected replayable reference is now `hardmix_replayable_v2`.

## 25. Memory-retention scoring should track preserved facts, not verbatim reasoning blocks

Revision-heavy artifacts often keep the right patch, control, or state change while dropping explanatory prose from the earlier step.

Exact-string memory scoring was therefore too brittle. The improved scorer now gives credit when the final artifact preserves:

- exact text when available
- salient quoted or backticked fragments
- high-overlap clauses with the same operational fact

That is why `kwa_finance_partner_deck_revision` now shows the more truthful split:

- weak `revision_responsiveness`
- correct `memory_retention_score`
