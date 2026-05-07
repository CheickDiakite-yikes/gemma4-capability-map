# MLX Tool-Contract Harnessing Report

This is the current human-readable report for the local Gemma-on-MLX tool-contract research wave. The generated packet that backs this document lives at:

- [`results/reports/mlx_tool_contract_harnessing/report.md`](../../results/reports/mlx_tool_contract_harnessing/report.md)
- [`results/reports/mlx_tool_contract_harnessing/report.json`](../../results/reports/mlx_tool_contract_harnessing/report.json)
- [`results/reports/mlx_tool_contract_harnessing/manifest.json`](../../results/reports/mlx_tool_contract_harnessing/manifest.json)

Regenerate it with:

```bash
uv run python scripts/build_mlx_tool_contract_report.py
```

## Executive Read

The current Moonie frontier is no longer whether `mlx_gemma4_e2b_reasoner_only` can match top-line readiness on the aligned `32 / 26` comparison surface. That surface is partially saturated. The active research question is now narrower and more useful:

> Which harness interventions make local MLX Gemma stay inside the tool interface without controller repair, fallback, or argument normalization?

The newest evidence says the final tool-turn directive is a causal model-side harness intervention. With the directive present, MLX Gemma is controller-clean on the current H1 live workflow packets. When the directive is removed, no-directive MLX can still match top-line readiness only because Moonie's controller repairs arguments, falls back to a planner, or substitutes the correct call. Raw no-directive tool compliance collapses on the focused probe suite.

That means the next useful work is not broad leaderboard reruns or UI polish. It is a CLI-first, benchmark-backed harness loop around:

- tool-call contract prompts
- controller repair and fallback dependence
- argument fidelity
- approval-safe stop behavior
- attributable packaged workflow families
- local live execution with sandbox evidence

## Current Figures

![H1i readiness, strict interface, and recovered execution](../../results/reports/mlx_tool_contract_harnessing/figures/h1i_readiness_strict_recovered.svg)

![H1h vs H1i no-directive controller burden](../../results/reports/mlx_tool_contract_harnessing/figures/h1h_h1i_controller_burden.svg)

![Tool probe contract gap](../../results/reports/mlx_tool_contract_harnessing/figures/tool_probe_contract_gap.svg)

![H1i failure modes](../../results/reports/mlx_tool_contract_harnessing/figures/h1i_failure_modes.svg)

![Prompt contract candidate targets](../../results/reports/mlx_tool_contract_harnessing/figures/prompt_contract_candidate_targets.svg)

![Executed prompt contract probe gate](../../results/reports/mlx_tool_contract_harnessing/figures/prompt_contract_probe_gate.svg)

## Evidence Sources

| Artifact | Purpose |
| --- | --- |
| [`H1f compact packet`](../../results/knowledge_work_h1_slice/20260506T_h1f_mlx_no_directive_v1_knowledge_work_h1f_mlx_tool_contract_ablation_v1) | First compact no-directive causal test on five live workflow families. |
| [`H1h full packet`](../../results/knowledge_work_h1_slice/20260507T_h1h_mlx_full_no_directive_v1_knowledge_work_h1h_mlx_full_tool_contract_ablation_v1) | Full ten-workflow no-directive replication. |
| [`H1i worst-family packet`](../../results/knowledge_work_h1_slice/20260507T_h1i_mlx_worst_no_directive_v1_knowledge_work_h1i_mlx_worst_family_tool_contract_v1) | Smaller fast loop derived from the worst H1h workflow families. |
| [`contracted tool probe`](../../results/tool_directive_probe/20260506T_mlx_tool_directive_probe_v4) | Exact-call probe for MLX with the tool-turn directive. |
| [`no-directive tool probe`](../../results/tool_directive_probe/20260507T_mlx_no_directive_probe_v1) | Exact-call probe after removing the directive. |
| [`executed prompt-contract probe packet`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1) | Three generic no-directive prompt-contract candidates compared against both contracted and no-directive probe baselines. |
| [`Gemini CLI dry-run baseline`](../../results/gemini_cli/20260507T_h1h_gemini_cli_dry_run_baseline_v1) | External-reference prompt and command manifest over the H1h workflow families. |

## Packet Summary

| Packet | Episodes | Contracted readiness | No-directive readiness | No-directive repair | No-directive fallback | No-directive arg repair | No-directive raw clean | No repair readiness | No fallback readiness | No arg repair readiness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H1f compact | 5 | `0.97936` | `0.97936` | `0.70` | `0.20` | `0.50` | `0.30` | `0.73818` | `0.92104` | `0.82036` |
| H1h full | 10 | `0.96891` | `0.96891` | `0.70` | `0.25` | `0.45` | `0.30` | `0.73801` | `0.89598` | `0.83016` |
| H1i worst-family | 4 | `0.97710` | `0.97710` | `1.00` | `0.50` | `0.50` | `0.00` | `0.64697` | `0.83125` | `0.81220` |

## H1i System Read

H1i is now the best fast loop for local MLX prompt-contract work because it keeps the same causal ordering as H1h while making the gaps larger.

| System | Readiness | Strict interface | Recovered execution | Controller repair | Controller fallback | Arg repair | Raw clean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| contracted | `0.97710` | `1.00000` | `1.00000` | `0.00` | `0.00` | `0.00` | `1.00` |
| no directive | `0.97710` | `1.00000` | `1.00000` | `1.00` | `0.50` | `0.50` | `0.00` |
| no directive + no repair | `0.64697` | `0.29688` | `0.00000` | `1.25` | `1.25` | `0.00` | `0.725` |
| no directive + no fallback | `0.83125` | `0.62500` | `0.50000` | `0.50` | `0.00` | `0.50` | `0.50` |
| no directive + no arg repair | `0.81220` | `0.71875` | `0.50000` | `0.50` | `0.50` | `0.00` | `0.50` |

The no-directive row is the important diagnostic: it is top-line clean but has zero raw-clean rate. That is a strong sign that readiness parity is controller-mediated, not raw model compliance.

## Probe Result

The focused tool probe explains the H1h/H1i behavior.

| Probe row | Exact match | Executable visual match |
| --- | ---: | ---: |
| contracted MLX | `7 / 8` | `1 / 1` |
| no-directive MLX | `0 / 8` | `0 / 1` |

Failure-mode split:

| Side | Failure mode | Case count | Interpretation |
| --- | --- | ---: | --- |
| candidate | `argument_mismatch` | 4 | The right tool family often appears, but canonical CLI/API arguments drift. |
| candidate | `no_tool_call` | 4 | Visual referent and parallel-tool cases collapse out of the tool protocol. |
| baseline non-exact | `executable_paraphrase` | 1 | Contracted MLX paraphrases one visual selector, but the executor can still resolve it. |

This is the strongest current reason to keep exact interface metrics, executable recovery metrics, and readiness metrics separate. A row can look solved at the task level while being fragile at the contract level.

## Prompt-Contract Candidate Queue

The next empirical wave has three generic no-directive prompt-contract candidates. They deliberately do not include the exact planned tool call. That keeps the probe honest: a candidate should improve raw tool protocol behavior without simply leaking the oracle next call.

| Candidate system | Contract | Target |
| --- | --- | --- |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_anchor` | `schema_anchor_v1` | Generic JSON/schema obedience for CLI/API canonicalization. |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_literal_guard` | `literal_argument_guard_v1` | Literal argument copying for path, query, record ids, visual selectors, and filters. |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_tool_required` | `tool_required_parallel_v1` | No-tool-call and parallel/visual protocol collapse. |

Generated candidate table:

- [`results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_candidates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_candidates.csv)

Dry-run candidate probe packet command:

```bash
uv run python scripts/run_tool_prompt_contract_probe_packet.py \
  --run-group-id <timestamp>_prompt_contract_probe_candidates
```

Real candidate probe packet command:

```bash
uv run python scripts/run_tool_prompt_contract_probe_packet.py \
  --run-group-id <timestamp>_prompt_contract_probe_candidates \
  --execute
```

## Executed Prompt-Contract Probe Gate

The first executed candidate packet is now recorded at:

- [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1)
- [`candidate_gate_summary.md`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1/candidate_gate_summary.md)
- generated report table: [`prompt_contract_probe_gates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_probe_gates.csv)
- generated failure table: [`prompt_contract_probe_failure_modes.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_probe_failure_modes.csv)

| Contract | Exact | Executable | Delta exact vs no-directive | Dominant failure | Recommendation |
| --- | ---: | ---: | ---: | --- | --- |
| `schema_anchor_v1` | `0.125` | `0.0` | `+0.125` | `argument_mismatch` | weak exact gain |
| `literal_argument_guard_v1` | `0.0` | `1.0` | `0.0` | `no_tool_call` | visual executable gain only |
| `tool_required_parallel_v1` | `0.0` | `1.0` | `0.0` | `no_tool_call` | visual executable gain only |

The result is useful but not victory-shaped. All three candidates improve over the raw no-directive probe on one case, but none approaches the contracted MLX row's `7 / 8` exact-call rate. The main practical interpretation is:

- `schema_anchor_v1` is the only exact-copy gain, but it is a weak one-case visual-readback recovery.
- `literal_argument_guard_v1` and `tool_required_parallel_v1` recover the executable visual target, but they do not improve exact JSON copy rate.
- `tool_required_parallel_v1` still has the most `no_tool_call` failures, so its wording is not yet solving the no-call family it was meant to target.
- H1i can use these candidates as mechanism probes, but they are not replacements for the final tool-turn directive.

## H1i Candidate Packet Result

The H1i mechanism-probe packet is now recorded at:

- [`results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_v1_knowledge_work_ablation_packet)
- [`tool_contract_summary.md`](../../results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_v1_knowledge_work_ablation_packet/tool_contract_summary.md)
- generated report table: [`h1i_prompt_contract_candidate_metrics.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/h1i_prompt_contract_candidate_metrics.csv)

All five rows matched:

| Row | Readiness | Repair | Fallback | Arg repair | Raw clean |
| --- | ---: | ---: | ---: | ---: | ---: |
| contracted | `0.97710` | `0.00` | `0.00` | `0.00` | `1.00` |
| no directive | `0.97710` | `0.00` | `0.00` | `0.00` | `1.00` |
| schema anchor | `0.97710` | `0.00` | `0.00` | `0.00` | `1.00` |
| literal guard | `0.97710` | `0.00` | `0.00` | `0.00` | `1.00` |
| tool required | `0.97710` | `0.00` | `0.00` | `0.00` | `1.00` |

This means H1i did not discriminate after the probe gate. The earlier H1i no-directive packet showed controller burden; this candidate packet sampled clean no-directive tool calls across the same four workflow families. The practical next step is not to declare the prompt contracts solved. It is to define a harder second-stage packet with repeated no-directive trials or probe-derived live cases where exact protocol failure is stable.

## Gemini CLI Baseline Status

The Gemini CLI adapter is currently a dry-run external baseline, not a replacement for Moonie. The packet uses the same H1h workflow families and records prompt/command artifacts without external side effects:

- [`results/gemini_cli/20260507T_h1h_gemini_cli_dry_run_baseline_v1`](../../results/gemini_cli/20260507T_h1h_gemini_cli_dry_run_baseline_v1)
- `workflow_count = 10`
- `dry_run_count = 10`
- `available_count = 0` for the intentionally missing binary in that packet

Use it as:

- a CLI ergonomics reference
- a trust and sandboxing reference
- an external-baseline packet once a real binary run is explicitly desired

Do not use it as:

- a substitute for local Gemma harnessing
- evidence that Moonie's local MLX tool contract is solved
- a merged same-harness leaderboard row

## Interpretation

The current result has three layers:

1. Contracted MLX is strong on the H1 live workflow packets.
2. No-directive MLX can still be made to finish the work.
3. The ability to finish without the directive is mostly controller-mediated.

That distinction matters because Moonie's research goal is not merely to produce good final artifacts. It is to understand what makes Gemma harnessable as a local operator. If final readiness hides controller dependence, then the harness is doing the work and the model-side contract remains weak.

The H1h -> H1i narrowing is also useful methodologically. H1h proves the phenomenon across the full ten-workflow live set. H1i turns the worst H1h workflow-family attribution into a cheap, repeatable packet for prompt-contract experiments.

## Next Experiments

Use this order before broad `32 / 26` reruns:

1. Design a harder second-stage packet because the H1i candidate packet saturated after the probe gate.
2. Design a second prompt-contract wave that combines schema anchoring with visual executable recovery while reducing no-call failures.
3. H1h only after H1i moves for the right reason.
4. Gemini CLI real execution only when the binary/run environment is explicitly meant to be part of the comparison.
5. Runtime live-smoke packets after benchmark movement, to confirm the CLI operator path sees the same repair/fallback pattern.

Acceptance criteria for a useful candidate:

- improves no-directive exact-call or executable-call probe behavior
- improves H1i no-directive raw-clean rate or reduces repair/fallback burden
- does not hide failures by broadening controller rescue
- preserves approval-safe stop behavior and sandbox policy events
- leaves generated report artifacts attributable to workflow families

## Reporting Discipline

Generated reporting now has three layers:

- packet-local summaries under each H1/probe/Gemini output directory
- generated cross-packet report artifacts under [`results/reports/mlx_tool_contract_harnessing`](../../results/reports/mlx_tool_contract_harnessing)
- this curated narrative report under [`docs/reports`](.)
- dry-run or executed prompt-contract probe packets under [`results/tool_prompt_contract_probe_packets`](../../results/tool_prompt_contract_probe_packets)

When a new H1i/H1h/probe wave runs, update the report by rerunning:

```bash
uv run python scripts/build_mlx_tool_contract_report.py
uv run pytest tests/test_mlx_tool_contract_report.py -q
```

Then update this document only if the interpretation changes.
