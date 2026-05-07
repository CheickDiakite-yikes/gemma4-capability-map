# Replay To Live Packaged Workflows

## Purpose

The exact-probe replay packets now isolate the remaining no-directive MLX failures better than the saturated H1i/H1j live workflow packets. The next live-harness move is to promote those replay findings into packaged workflows without losing the discriminating mechanism.

This brief is the contract for that promotion. It keeps the direction CLI-first and packaged-workflow-only while making sure new live slices remain benchmark-backed and attributable to replay families.

## Current Evidence

| Replay slice | Contracted exact | No-directive exact | Gap | Live status |
| --- | ---: | ---: | ---: | --- |
| all failed exact-probe cases | `7 / 8` | `0 / 8` | `-0.875` | replay artifact exists |
| canonical arguments | `4 / 4` | `0 / 4` | `-1.0` | partly represented by H1j API/CLI families |
| visual no-call | `2 / 3` plus one executable paraphrase | `0 / 3` | `-0.667` | partly represented by H1j visual families |
| parallel array | `1 / 1` | `0 / 1` | `-1.0` | not faithfully represented by any packaged workflow yet |

The important conclusion is that H1i/H1j packaged workflows are easier than the raw exact probe. They are useful for controller-burden attribution, but they currently wash out the strongest raw no-call failure family.

## Promotion Rules

Do not promote a replay case into a live slice unless all of these are true:

- It enters through `configs/packaged_workflows.yaml`; no ad hoc live runner.
- It has both `replayable_core` and `live_web_stress` episode IDs.
- It is tagged with the originating replay pressure, for example `parallel_tool_calling` or `canonical_arguments`.
- Its H1 config records the source replay comparison packet.
- The acceptance metric includes raw tool-interface cleanliness, not only readiness.
- Its CLI path can be listed with `moonie-agent workflows` and launched with `moonie-agent live`.

## First Target: Parallel Audit Array

`parallel_audit_array_literal` is the highest-value promotion target because the raw case requires two independent evidence calls before any answer:

- `inspect_image` over `img-parallel`
- `read_repo_file` over `config/settings.yaml`

The live packaged workflow should preserve that shape. It should not collapse the task into a generic "review the evidence" workflow, and it should not rely on a controller-side planner to substitute the missing second call.

Proposed workflow identity:

- workflow ID: `ops_parallel_audit_review`
- role family: `operations_audit`
- category: `parallel_evidence_review`
- replayable episode ID: `kwa_ops_parallel_audit_review_v1`
- live episode ID: `kwa_ops_live_parallel_audit_review_v1`
- source replay case: `parallel_audit_array_literal`
- expected pressure tags:
  - `parallel_tool_calling`
  - `two_source_evidence`
  - `no_tool_call`
  - `read_repo_file`
  - `inspect_image`

## Acceptance Criteria

The first scaffold is accepted if:

- the workflow loads through `load_packaged_workflows`
- `moonie-agent workflows --lane live_web_stress` lists it
- H1 config validation can resolve both episode IDs
- dry-run or packet construction records the workflow family and source replay comparison
- no frontend changes are needed

The first execution packet is accepted only if it reports:

- readiness
- strict interface
- recovered execution
- raw planning clean rate
- controller repair count
- controller fallback count
- argument repair count
- per-workflow trace findings

If the new live workflow still saturates, the next move is a replay executor surface for exact probe cases rather than another broad H1 packet.

