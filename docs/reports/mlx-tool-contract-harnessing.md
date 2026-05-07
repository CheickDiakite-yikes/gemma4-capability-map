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

![Prompt contract wave two probe gate](../../results/reports/mlx_tool_contract_harnessing/figures/prompt_contract_wave2_probe_gate.svg)

![H1i prompt-contract repeat3 burden](../../results/reports/mlx_tool_contract_harnessing/figures/h1i_prompt_contract_repeat3_burden.svg)

![H1j probe-derived candidate burden](../../results/reports/mlx_tool_contract_harnessing/figures/h1j_probe_derived_burden.svg)

![H1j probe-derived helper burden](../../results/reports/mlx_tool_contract_harnessing/figures/h1j_probe_derived_helper_burden.svg)

![H1k parallel-audit candidate burden](../../results/reports/mlx_tool_contract_harnessing/figures/h1k_parallel_audit_burden.svg)

![H1k parallel-audit helper burden](../../results/reports/mlx_tool_contract_harnessing/figures/h1k_parallel_audit_helper_burden.svg)

![Exact probe replay gap](../../results/reports/mlx_tool_contract_harnessing/figures/exact_probe_replay_gap.svg)

![Focused exact replay gaps](../../results/reports/mlx_tool_contract_harnessing/figures/exact_probe_replay_focus_gap.svg)

## Evidence Sources

| Artifact | Purpose |
| --- | --- |
| [`H1f compact packet`](../../results/knowledge_work_h1_slice/20260506T_h1f_mlx_no_directive_v1_knowledge_work_h1f_mlx_tool_contract_ablation_v1) | First compact no-directive causal test on five live workflow families. |
| [`H1h full packet`](../../results/knowledge_work_h1_slice/20260507T_h1h_mlx_full_no_directive_v1_knowledge_work_h1h_mlx_full_tool_contract_ablation_v1) | Full ten-workflow no-directive replication. |
| [`H1i worst-family packet`](../../results/knowledge_work_h1_slice/20260507T_h1i_mlx_worst_no_directive_v1_knowledge_work_h1i_mlx_worst_family_tool_contract_v1) | Smaller fast loop derived from the worst H1h workflow families. |
| [`contracted tool probe`](../../results/tool_directive_probe/20260506T_mlx_tool_directive_probe_v4) | Exact-call probe for MLX with the tool-turn directive. |
| [`no-directive tool probe`](../../results/tool_directive_probe/20260507T_mlx_no_directive_probe_v1) | Exact-call probe after removing the directive. |
| [`executed prompt-contract probe packet`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_candidates_execute_v1) | Three generic no-directive prompt-contract candidates compared against both contracted and no-directive probe baselines. |
| [`executed prompt-contract wave-two packet`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_execute_v1) | Three second-wave prompt-contract candidates tested on the raw probe before any further H1 promotion. |
| [`H1i prompt-contract repeat3 packet`](../../results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_repeat3_v1_knowledge_work_ablation_packet) | Repeated second-stage candidate packet: three attempts per H1i workflow family per row. |
| [`H1j probe-derived candidate packet`](../../results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_candidates_v1_knowledge_work_ablation_packet) | Six packaged live workflows selected from exact no-directive probe failure families. |
| [`H1j probe-derived helper packet`](../../results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_helpers_v1_knowledge_work_ablation_packet) | Controller-helper ablation on the same H1j probe-derived packaged workflow set. |
| [`H1k parallel-audit candidate packet`](../../results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_candidates_v1_knowledge_work_ablation_packet) | Packaged live promotion of the deferred `parallel_audit_array_literal` replay case. |
| [`H1k parallel-audit helper packet`](../../results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_helpers_v1_knowledge_work_ablation_packet) | Controller-helper ablation on the packaged parallel-audit workflow. |
| [`exact-probe replay packet`](../../results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_v1) | Dry-run replay artifacts for the eight failed no-directive exact-call probe cases. |
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

The prompt-contract queue now has two waves. They deliberately do not include the exact planned tool call. That keeps the probe honest: a candidate should improve raw tool protocol behavior without simply leaking the oracle next call.

| Candidate system | Contract | Target |
| --- | --- | --- |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_anchor` | `schema_anchor_v1` | Generic JSON/schema obedience for CLI/API canonicalization. |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_literal_guard` | `literal_argument_guard_v1` | Literal argument copying for path, query, record ids, visual selectors, and filters. |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_tool_required` | `tool_required_parallel_v1` | No-tool-call and parallel/visual protocol collapse. |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_schema_literal_tool_required` | `schema_literal_tool_required_v2` | Combined schema, literal-copy, and tool-required pressure. |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_next_call_state` | `visual_next_call_state_v2` | Stateful visual next-call behavior after a visual result exists. |
| `mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_parallel_array_required` | `parallel_array_required_v2` | JSON-array shape for independent multi-source/parallel checks. |

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

Second-wave dry-run packet:

- [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_dry_run_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_dry_run_v1)

Second-wave execution command:

```bash
uv run python scripts/run_tool_prompt_contract_probe_packet.py \
  --run-group-id <timestamp>_prompt_contract_wave2_execute_v1 \
  --candidate-wave v2 \
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

## Prompt-Contract Wave Two Probe Gate

The second executed candidate packet is now recorded at:

- [`results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_execute_v1)
- [`candidate_gate_summary.md`](../../results/tool_prompt_contract_probe_packets/20260507T_prompt_contract_wave2_execute_v1/candidate_gate_summary.md)
- generated report table: [`prompt_contract_wave2_probe_gates.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_wave2_probe_gates.csv)
- generated failure table: [`prompt_contract_wave2_probe_failure_modes.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_wave2_probe_failure_modes.csv)
- generated promotion table: [`prompt_contract_promotion_decisions.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_promotion_decisions.csv)

| Contract | Exact | Executable | Delta exact vs no-directive | Dominant failure | Recommendation |
| --- | ---: | ---: | ---: | --- | --- |
| `schema_literal_tool_required_v2` | `0.125` | `0.0` | `+0.125` | `argument_mismatch` | weak exact gain |
| `visual_next_call_state_v2` | `0.0` | `1.0` | `0.0` | `no_tool_call` | visual executable gain only |
| `parallel_array_required_v2` | `0.0` | `0.0` | `0.0` | `no_tool_call` | no probe gain |

Wave two does not change the research direction. Combining schema anchoring, literal copying, and tool-required language still recovers only one exact case. The visual-state contract again repairs executable visual behavior without exact JSON fidelity. The parallel-array contract does not fix the no-call family. None of these candidates should be promoted as a replacement for the final tool-turn directive.

## Prompt-Contract Promotion Decisions

The generated report now writes a machine-readable promotion table:

- [`results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_promotion_decisions.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/prompt_contract_promotion_decisions.csv)

The current decisions are intentionally conservative:

- weak one-case exact gains are held for exact-probe live replay
- visual executable-only gains are held for visual replay, not general H1 promotion
- no-gain parallel/no-call candidates are rejected for H1 promotion

That table is the guardrail for the next loop: a prompt-contract candidate should not graduate because it sounds plausible; it should graduate only after it moves exact probe behavior or reduces controller burden on a harder live discriminator.

## Exact-Probe Replay Packet

The first replay packet is now recorded at:

- [`results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_v1`](../../results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_v1)
- executed packet: [`results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1`](../../results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1)
- contracted replay packet: [`results/tool_probe_replay_packets/20260507T_contracted_exact_probe_replay_execute_v1`](../../results/tool_probe_replay_packets/20260507T_contracted_exact_probe_replay_execute_v1)
- replay A/B comparison: [`results/tool_probe_replay_comparisons/20260507T_contracted_vs_no_directive_exact_replay_v1`](../../results/tool_probe_replay_comparisons/20260507T_contracted_vs_no_directive_exact_replay_v1)
- brief: [`docs/continuity/exact-probe-replay.md`](../continuity/exact-probe-replay.md)

It contains all `8` failed no-directive exact-call cases:

| Failure mode | Count |
| --- | ---: |
| `argument_mismatch` | `4` |
| `no_tool_call` | `4` |

The generated next-action table splits the implementation backlog into:

| Next action | Count |
| --- | ---: |
| `build_canonical_argument_replay` | `4` |
| `build_visual_state_replay_executor` | `3` |
| `build_parallel_array_replay_or_workflow` | `1` |

This packet is deliberately a dry-run replay artifact, not a packaged live workflow. It preserves each case's messages, media, allowed tool specs, expected calls, no-directive actual calls, and contracted baseline context. The next implementation should either turn the deferred parallel case into a faithful packaged workflow or make these exact probe cases operator-visible in a replay executor.

The executed replay packet reproduced the raw failure set exactly: exact match stayed `0 / 8`, the four argument-mismatch cases remained argument mismatches, and the four no-tool-call cases remained no-tool-call cases. That makes this a stable mechanism target rather than a one-off probe artifact.

The contracted replay packet restores `7 / 8` exact on the same cases, with the only non-exact case remaining executable through visual selector aliasing. This is now the cleanest current A/B artifact for the model-side tool-turn directive.

The generated replay comparison records a `-0.875` exact-rate delta for no-directive versus contracted MLX on the same replay case set.

Focused replay slices now isolate each `next_action` family:

| Slice | Contracted exact | No-directive exact | Delta | Interpretation |
| --- | ---: | ---: | ---: | --- |
| canonical arguments | `4 / 4` | `0 / 4` | `-1.0` | Final directive is carrying exact CLI/API argument canonicalization. |
| visual no-call | `2 / 3` plus one executable paraphrase | `0 / 3` | `-0.667` | Visual cases still need a stateful/executable replay path before another broad H1 run. |
| parallel array | `1 / 1` | `0 / 1` | `-1.0` | The missing faithful live workflow should preserve independent two-call array shape. |

Generated replay tables:

- [`exact_probe_replay_case_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/exact_probe_replay_case_deltas.csv)
- [`exact_probe_replay_family_deltas.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/exact_probe_replay_family_deltas.csv)
- [`exact_probe_replay_focus_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/exact_probe_replay_focus_summary.csv)

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

The first concrete second-stage lever has now been executed:

```bash
uv run python scripts/run_knowledge_work_h1_ablation_packet.py \
  --config configs/knowledge_work_h1i_slice.yaml \
  --packet-id mlx_prompt_contract_candidates \
  --run-group-id 20260507T_h1i_prompt_contract_candidates_repeat3_v1 \
  --repeat 3
```

Result:

- packet: [`results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_repeat3_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1i_prompt_contract_candidates_repeat3_v1_knowledge_work_ablation_packet)
- runs: `60` episode traces, `5` rows x `4` workflow families x `3` repeats
- all five rows matched readiness `0.97710`, strict/recovered `1.0 / 1.0`, repair/fallback/argument repair `0.0 / 0.0 / 0.0`, raw clean `1.0`
- trace analysis found `0` note events and `0` failure candidates
- generated report table: [`h1i_prompt_contract_repeat3_metrics.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/h1i_prompt_contract_repeat3_metrics.csv)

This is a useful negative result. Repeating the saturated H1i candidate packet did not recover the earlier H1i no-directive controller-burden signal. The current H1i packaged workflows are too deterministic to validate prompt-contract candidates after the probe gate. The next harder packet should be probe-derived live cases, especially visual/parallel no-call cases.

## H1j Probe-Derived Packet Result

H1j moved from worst-family attribution to exact probe-family attribution:

- config: [`configs/knowledge_work_h1j_slice.yaml`](../../configs/knowledge_work_h1j_slice.yaml)
- packet: [`results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_candidates_v1_knowledge_work_ablation_packet)
- generated table: [`h1j_probe_derived_candidate_metrics.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/h1j_probe_derived_candidate_metrics.csv)
- runs: `30` traces, `5` rows x `6` packaged live workflow families

Result:

| Row | Readiness | Repair | Fallback | Arg repair | Raw clean |
| --- | ---: | ---: | ---: | ---: | ---: |
| contracted | `0.96577` | `0.00` | `0.00` | `0.00` | `1.00` |
| no directive | `0.96577` | `0.00` | `0.00` | `0.00` | `1.00` |
| schema anchor | `0.96577` | `0.00` | `0.00` | `0.00` | `1.00` |
| literal guard | `0.96577` | `0.00` | `0.00` | `0.00` | `1.00` |
| tool required | `0.96577` | `0.00` | `0.00` | `0.00` | `1.00` |

Trace analysis again found `0` note events and `0` failure candidates. This makes the current research picture sharper: the raw probe remains a better discriminator than benchmark-style packaged workflow execution, even when the packaged workflow set is selected from the same probe failure families.

The paired helper-ablation packet is now recorded at:

- packet: [`results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_helpers_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1j_probe_derived_helpers_v1_knowledge_work_ablation_packet)
- generated table: [`h1j_probe_derived_helper_metrics.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/h1j_probe_derived_helper_metrics.csv)
- runs: `30` traces, `5` rows x `6` packaged live workflow families

It also saturated. Contracted, no-directive, no-controller-repair, no-controller-fallback, and no-argument-repair rows all matched readiness `0.96577`, strict/recovered `1.0 / 1.0`, repair/fallback/argument repair `0.0 / 0.0 / 0.0`, and raw clean `1.0`. Trace mining recorded `21` `controller_repair_disabled` markers on the disabled-repair row, but `0` failure candidates.

That matters: controller repair remains causal on H1h/H1i, but not on H1j. H1j is currently evidence that the packaged workflow route can wash out raw probe failures, not evidence that the model-side contract is fixed.

## H1k Parallel-Audit Packet Result

H1k closed the last packaged-workflow gap left by H1j: the deferred parallel no-call probe case.

- config: [`configs/knowledge_work_h1k_slice.yaml`](../../configs/knowledge_work_h1k_slice.yaml)
- workflow: `ops_parallel_audit_review`
- replay pressure: `parallel_audit_array_literal`
- candidate packet: [`results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_candidates_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_candidates_v1_knowledge_work_ablation_packet)
- helper packet: [`results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_helpers_v1_knowledge_work_ablation_packet`](../../results/knowledge_work_h1_slice/20260507T_h1k_parallel_audit_helpers_v1_knowledge_work_ablation_packet)
- generated candidate table: [`h1k_parallel_audit_candidate_metrics.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/h1k_parallel_audit_candidate_metrics.csv)
- generated helper table: [`h1k_parallel_audit_helper_metrics.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/h1k_parallel_audit_helper_metrics.csv)

Both H1k packets saturated:

| Packet | Rows | Readiness | Strict/recovered | Repair/fallback/arg repair | Raw clean | Failure candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| candidate | `5` | `0.91780` | `1.0 / 1.0` | `0.0 / 0.0 / 0.0` | `1.0` | `0` |
| helper | `5` | `0.91780` | `1.0 / 1.0` | `0.0 / 0.0 / 0.0` | `1.0` | `0` |

The negative result is important. H1k proves Moonie now has a safe packaged live scaffold for the parallel-audit family, but that staged workflow is easier than the raw one-turn replay case. It decomposes the pressure into known task steps, and no-directive MLX stays controller-clean even when controller repair, controller fallback, or argument repair are removed.

So the next move is not another packaged H1 derivative. The next discriminator should preserve exact-call replay shape, especially the independent two-call array behavior in `parallel_audit_array_literal`.

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

1. Convert the raw probe result into a stricter exact-probe live replay packet, because H1i repeat3, H1j, and wave two all failed to break the same packaged-workflow saturation.
2. Make the replay mode explicitly cover `parallel_audit_array_literal`; H1k already showed the packaged live workflow version is too staged to discriminate.
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
