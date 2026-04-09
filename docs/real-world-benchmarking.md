# Real-World Benchmarking

This benchmark started as a controlled white-box evaluation of reasoning, routing,
retrieval, and full-stack execution. That remains useful, but it is not enough for
the harder question:

When can a local agent reliably do human operational work without a human in the loop?

## What "real-world" means here

A real-world benchmark lane should not reward only correct final text. It should
stress whether the system can act like a bounded autonomous worker:

- preserve state correctly
- choose the right tool before repair logic rescues it
- recover from validator feedback without human intervention
- use multilingual and multimodal evidence when the job requires it
- avoid collateral damage while still finishing the task
- stay grounded when stale context, distractors, or noisy UI cues appear

The repo now encodes this at the task level through:

- `benchmark_tags`
- `real_world_profile`
- real-world metrics such as:
  - `state_integrity_score`
  - `collateral_damage_free`
  - `intervention_free_success`
  - `real_world_readiness_score`
  - `human_time_ratio` when a human baseline is provided

## Current real-world task classes

The first real-world slice is intentionally operational rather than academic:

- `release_ops`
  - rollout policy lookup
  - screenshot + repo audit before patching
  - runbook-guided config patching
- `billing_ops`
  - invoice lock screenshot and patch tasks
  - direct billing patch-record routing
- `calendar_ops`
  - rescheduling and validator-safe meeting creation
- `incident_ops`
  - multimodal incident diagnosis from notes plus screenshots
- `finance_ops`
  - latest-file discovery and budget deltas

These are bounded, deterministic, and replayable, but closer to job-shaped work than
generic QA.

## Why this matters

The latest runs already show why a real-world layer is necessary:

- retrieval can remain perfect while the final answer still fails
- routing can recover execution sometimes while still being unsafe at the interface
- thinking mode can look attractive in principle but regress badly under image-heavy
  operational tasks because of overflow and truncation

Without explicit real-world scoring, those distinctions blur together.

## Current benchmark stance

The first real-world matrix is defined in:

- [`alpha_real_world_matrix.yaml`](/Users/cheickdiakite/Codex/moonie/configs/alpha_real_world_matrix.yaml)

It is designed to answer three practical questions:

1. Does real retrieval stay strong when the task resembles operational work?
2. Does real routing stay safe on high-cost tool intents such as billing patch records?
3. Can a true specialist stack complete bounded operational tasks with high state integrity and low collateral damage?

## Current canonical snapshot

The first canonical real-world autonomy run is:

- [`20260409T210500Z_alpha_real_world`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T210500Z_alpha_real_world)

That run is important operationally because it is the first one executed with:

- subprocess-isolated experiments
- `hf_service` for the Gemma 4 reasoner
- real FunctionGemma and EmbeddingGemma specialist lanes where applicable

The older in-process real-world attempt stalled during repeated HF warmup and should not be treated as authoritative.

Current findings from the canonical run:

- escalation judgment is the weakest lane right now
  - the no-tool approval-escalation slice scored `0.0`
- retrieval remains strong as an evidence system
  - retrieval scored `0.875` task success with `1.0` strict interface correctness
  - the remaining misses are French answer-surface misses, not retrieval misses
- routing is the main operational specialist weakness
  - real-world routing scored `0.5`
  - the dominant failure family is billing patch-record and unsafe-billing-disable intent handling
- full-stack bounded autonomy is much stronger than pure judgment
  - strict full-stack success scored `0.75`
  - recovered execution scored `1.0`
  - real-world readiness averaged about `0.87`

This is exactly why the benchmark now publishes three layers:

- strict interface correctness
- recovered execution
- real-world readiness

## Next real-world expansions

The next task classes worth authoring are:

- inbox and queue triage with conflicting priorities
- customer-support resolution where a wrong action has downstream cost
- multi-day scheduling with preference conflicts and stale history
- incident-command tasks that require deciding whether to escalate, rollback, or defer
- repo change review where the system must refuse unsafe edits, not just propose plausible ones

The next architectural fixes are also clearer now:

- add a stronger escalation/defer/refuse prompt contract for no-tool judgment tasks
- add router-side intent priors for billing patch and refusal flows
- add multilingual answer-surface normalization or second-pass synthesis for French operational answers

## Research principle

The benchmark should not claim AGI-like autonomy from polished demos.

A stronger standard is:

- job-shaped tasks
- replayable state
- explicit side-effect risk
- strict interface scoring
- recovered end-to-end scoring
- durable historical tracking across runs

That is the direction this repo is now moving toward.
