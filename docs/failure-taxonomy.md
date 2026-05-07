# Failure Taxonomy

Moonie failure tags are meant to separate model capability misses, interface-contract misses, controller-rescue behavior, and harness/runtime artifacts. The taxonomy is used across atomic tasks, `KnowledgeWorkArena`, H1 packets, runtime live-smoke packets, and the tool-directive probe.

## Top-Level Outcomes

| Tag | Meaning |
| --- | --- |
| `failed` | Top-level task failure marker. |
| `answer_missing` | No final answer was produced. |
| `answer_mismatch` | A final answer was produced but did not satisfy the scorer. |
| `wrong_final_state` | Tool execution completed but the resulting environment state was wrong. |
| `retrieval_miss` | Retrieval recall or evidence scoring failed. |

## Generation And Harness Artifacts

| Tag | Meaning |
| --- | --- |
| `generation_truncated` | The decode budget was exhausted before the answer completed. |
| `thinking_overflow` | Thought text consumed the decode budget and the final answer never arrived. |
| `malformed_call` | The raw tool request could not be parsed into the canonical call AST. |
| `hallucinated_tool` | The model invoked a tool that was not present in the declared tool set. |

These tags should not be mixed casually with domain failures. A task can fail because the model misunderstood the work, because the prompt contract was weak, or because the harness let malformed output travel too far. Those are different research findings.

## Tool-Contract Tags

| Tag | Meaning |
| --- | --- |
| `wrong_tool` | The selected tool family or tool name is incorrect. |
| `arg_mismatch` | The selected tool is correct but one or more arguments are wrong or incomplete. |
| `argument_mismatch` | Probe-level name for a canonical argument mismatch. Used in `probe_case_deltas.csv`. |
| `argument_repair` | The controller normalized a near-miss argument into the canonical executable form. |
| `fallback_planner` | The controller substituted a fallback plan after the raw model plan was missing, unusable, or unsafe for the next step. |
| `fallback_disabled` | An ablation row disabled fallback, exposing behavior that would otherwise have recovered. |
| `repair_disabled` | An ablation row disabled controller repair, exposing raw model behavior. |
| `no_tool_call` | Probe-level failure where the model did not emit an executable tool call. |
| `executable_paraphrase` | Probe-level case where the call is not exact-copy canonical but can still execute through accepted semantic aliasing. |

The H1f/H1h/H1i no-directive wave depends on this split. No-directive MLX can keep top-line readiness while still showing heavy `argument_repair` and `fallback_planner` counts. That is controller-mediated completion, not raw model-side contract compliance.

## Visual Workflow Tags

| Tag | Meaning |
| --- | --- |
| `image_grounding_miss` | The model saw a screenshot or image but answered with the wrong grounded setting or action. |
| `visual_stepwise_control` | The workflow requires ordered visual tool use rather than one-shot image answering. |
| `visual_repeated_refinement` | The model repeats a prior visual refinement instead of moving to the required next visual step. |
| `visual_readback_missing` | The model does not complete the expected readback after selecting or refining a visual region. |
| `visual_argument_copying` | Probe family for exact copying of visual target arguments. |
| `visual_referent_carryover` | Probe family for preserving the latest visual referent across turns. |

Visual tags are especially important because exact-copy and executable-readiness can diverge. Contracted MLX currently has one exact visual paraphrase in the probe, but the executor can still resolve it to the intended region. No-directive MLX loses even executable visual behavior on the same probe.

## Workflow And Policy Tags

| Tag | Meaning |
| --- | --- |
| `approval_required` | The correct behavior is to stop for approval before continuing. |
| `sandbox_only` | The live-web or external action is restricted to a sandbox/dry-run endpoint. |
| `policy_block` | A workflow action was blocked by runtime policy. |
| `clarify_expected` | The task requires clarification rather than action. |
| `defer_expected` | The task requires deferral or handoff rather than action. |
| `refuse_expected` | The task requires refusal or safe redirection. |

These tags support the difference between a task-completing assistant and a role-ready operator. Moonie should not reward a workflow that acts when approval, deferral, clarification, or refusal is required.

## Current H1i Failure Modes

The current H1i packet records `12` failure candidates across the no-directive/no-helper rows. The main failure-mode counts are:

| Failure mode | Count | Interpretation |
| --- | ---: | --- |
| `fallback_planner` | 8 | No-directive rows still require fallback planning when helpers are constrained. |
| `visual_stepwise_control` | 8 | The hardest workflow families still stress ordered visual/API/CLI sequencing. |
| `argument_repair` | 4 | Canonical arguments drift without the directive. |
| `fallback_disabled` | 4 | Removing fallback exposes the hidden recovery dependence. |
| `repair_disabled` | 4 | Removing repair exposes raw no-directive weakness. |
| `visual_repeated_refinement` | 3 | Disabled-repair rows can repeat stale visual refinements. |
| `visual_readback_missing` | 2 | Disabled-repair rows can fail to complete the visual readback step. |

Generated table:

- [`results/reports/mlx_tool_contract_harnessing/tables/h1i_failure_modes.csv`](../results/reports/mlx_tool_contract_harnessing/tables/h1i_failure_modes.csv)

## Current Probe Failure Modes

The no-directive probe comparison currently reports:

| Side | Failure mode | Case count |
| --- | --- | ---: |
| candidate | `argument_mismatch` | 4 |
| candidate | `no_tool_call` | 4 |
| baseline non-exact | `executable_paraphrase` | 1 |

Generated table:

- [`results/reports/mlx_tool_contract_harnessing/tables/probe_failure_modes.csv`](../results/reports/mlx_tool_contract_harnessing/tables/probe_failure_modes.csv)

## Reporting Rule

When adding or interpreting a failure tag, record whether it is:

- a direct scorer outcome
- a trace-derived observation
- a controller repair note
- an ablation marker
- a probe comparison mode
- a runtime policy event

This keeps benchmark-harness issues separate from true model capability failures and keeps recovered completion separate from raw tool-contract compliance.
