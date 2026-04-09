# Methodology

The benchmark core is deterministic by design:

- gold tasks are human-authored JSONL records
- perturbations are explicit, inspectable variant overrides
- tool environments are simulated and replayable
- traces record raw outputs, normalized calls, retrieval hits, state transitions, and metrics
- matrix runs are logged historically so benchmark claims can be tied to specific run groups

The primary comparison axis is architecture under drift:

1. `monolith`
2. `hybrid`
3. `modular`

The intended top-line figure is task success versus latency under drift, with Interface Reliability Score as the tool-centric diagnostic metric.

The current alpha corpus is intentionally balanced:

- `52` gold tasks total
- `13` tasks per track
- `232` explicit factorized variants

That balance matters because the broader integrated runs surfaced weaknesses that the earlier narrow specialist probes did not.

## Run discipline

Every matrix run produces an immutable run-group folder and also appends experiment-level records to history logs. This gives us two complementary views:

- snapshot view: one folder per run group
- longitudinal view: one append-only record per experiment execution

The benchmark should be interpreted from both:

- the run-group snapshot is the exact state of a reported comparison
- the history log shows whether a change was a durable improvement, a regression, or a harness artifact
- the canonical matrix snapshot for a matrix name should come from the latest complete run group, not from an arbitrary partial rerun

For the current alpha stage, the authoritative snapshots are:

- expanded clean integrated matrix:
  [`20260409T180500Z_alpha_integrated_specialists`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T180500Z_alpha_integrated_specialists)
- specialist-backed drift matrix:
  [`20260409T190500Z_alpha_specialist_drift`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T190500Z_alpha_specialist_drift)
- real-world autonomy matrix:
  [`20260409T210500Z_alpha_real_world`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T210500Z_alpha_real_world)

Backend posture should also be recorded, not assumed. Before making claims from a local run on this machine:

- run `scripts/preflight_backends.py`
- when fresh-process HF startup cost matters, also record `scripts/probe_hf_import_runtime.py`
- record the recommended local reasoner backend for that session
- note whether the run used online Hub resolution or `GEMMA4_OFFLINE=1`
- if `hf_service` is used, retain the service state and request logs under `results/runtime/hf_reasoner`

## Failure analysis

Failure tags are generated from traces, not added manually. They are intended to distinguish:

- model capability misses
- interface reliability misses
- harness-induced failures such as answer truncation or thought-only output
- multimodal grounding misses

That makes the benchmark usable for research iteration instead of only headline scoring.

Recent specialist-backed runs justify one more analytical split:

- retrieval misses must be separated from answer-generation misses

In the current real `EmbeddingGemma` drift lane, retrieval metrics remained perfect while task success dropped. Without this split, the benchmark would incorrectly attribute answer-language and multimodal synthesis misses to the retriever.

## Runtime observability

Benchmark traces now carry backend/runtime metadata for planning turns and final turns. Matrix snapshots also include the latest backend preflight payload. This is important because:

- a failure can come from model behavior, parser behavior, or backend warmup/runtime posture
- local Apple Silicon backend health can change independently of benchmark code changes
- specialist probe results are only interpretable when the runtime path is explicit
- fresh-process HF startup cost can dominate bounded slices unless the execution mode is made explicit

The real-world matrix now makes that operational split explicit:

- experiments run in isolated subprocesses
- the Gemma 4 reasoner runs through `hf_service`
- canonical autonomy claims should be taken from this service-backed path on this Mac, not from older in-process partial runs that stalled during repeated HF warmup

## Strict vs recovered performance

Tool-routing and full-stack slices deliberately keep a strict notion of success:

- `success_rate` is based on exact tool and argument correctness for tool-routing tasks
- repair layers and validator recovery are still logged so controller improvements remain visible

That means a run can show perfect final execution while still losing strict interface score under schema drift. Those cases should be read through:

- `interface_reliability_score`
- `controller_repair_count`
- failure tags and failing variants in the experiment summary

The current real specialist runs justify publishing both strict and recovered views:

- real `FunctionGemma` routing under drift dropped to `0.75` because one task family repeatedly chose the wrong tool
- real specialist-backed modular full-stack under drift still achieved `0.9375` because some interface misses were recovered before final task completion

The real-world autonomy matrix reinforces this split more strongly:

- routing success dropped to `0.5` on job-shaped routing tasks even though some repair logic still fired
- full-stack strict success dropped to `0.75`, but recovered execution stayed `1.0`
- escalation judgment failed completely on the current no-tool approval-escalation slice, so real-world readiness cannot be inferred from tool-bearing task success alone
