# CLI-Live Exact Replay Results

## Why This Matters

H1j and H1k showed that packaged live workflows can saturate even when raw no-directive MLX fails exact tool calls. The CLI-live exact replay path keeps the raw probe shape intact while making the run operator-visible from the terminal.

Use this brief as the restart point for live replay evidence.

## Entrypoints

Dry run or execute exact replay cases:

```bash
uv run moonie-agent replay-live \
  --packet-dir results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1 \
  --case-id parallel_audit_array_literal \
  --execute
```

Inspect live replay packets:

```bash
uv run moonie-agent packet \
  --kind tool-probe-replay-live \
  --packet-id 20260507T_parallel_array_no_directive_live_execute_v1
```

Inspect live replay comparisons:

```bash
uv run moonie-agent packet \
  --kind tool-probe-replay-live-comparison \
  --packet-id 20260507T_canonical_argument_contracted_vs_no_directive_live_v1
```

## Result Matrix

| Family | Cases | Contracted exact | No-directive exact | Delta exact | No-directive failure | Call-count signal |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| canonical CLI/API arguments | `4` | `4 / 4` | `0 / 4` | `-1.0` | `argument_mismatch` | actual-call delta `0`; tool calls happen, arguments drift |
| visual no-call | `3` | `2 / 3` plus one executable paraphrase | `0 / 3` | `-0.6666666666666666` | `no_tool_call` | no-directive emits `0` calls in all three cases |
| parallel array | `1` | `1 / 1` | `0 / 1` | `-1.0` | `no_tool_call` | no-directive misses both expected calls |

## Source Packets

Canonical arguments:

- contracted: [`results/tool_probe_replay_live/20260507T_canonical_argument_contracted_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_canonical_argument_contracted_live_execute_v1)
- no directive: [`results/tool_probe_replay_live/20260507T_canonical_argument_no_directive_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_canonical_argument_no_directive_live_execute_v1)
- comparison: [`results/tool_probe_replay_live_comparisons/20260507T_canonical_argument_contracted_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260507T_canonical_argument_contracted_vs_no_directive_live_v1)

Visual no-call:

- contracted: [`results/tool_probe_replay_live/20260507T_visual_state_contracted_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_visual_state_contracted_live_execute_v1)
- no directive: [`results/tool_probe_replay_live/20260507T_visual_state_no_directive_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_visual_state_no_directive_live_execute_v1)
- comparison: [`results/tool_probe_replay_live_comparisons/20260507T_visual_state_contracted_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260507T_visual_state_contracted_vs_no_directive_live_v1)

Parallel array:

- contracted: [`results/tool_probe_replay_live/20260507T_parallel_array_contracted_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_parallel_array_contracted_live_execute_v1)
- no directive: [`results/tool_probe_replay_live/20260507T_parallel_array_no_directive_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_parallel_array_no_directive_live_execute_v1)
- comparison: [`results/tool_probe_replay_live_comparisons/20260507T_parallel_array_contracted_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260507T_parallel_array_contracted_vs_no_directive_live_v1)

## Interpretation

The final tool-turn directive is doing three different kinds of work:

- canonical argument fidelity for CLI/API calls
- staying inside the tool protocol for visual follow-on calls
- preserving independent two-call array shape for parallel evidence gathering

This explains why H1k can be useful but non-discriminating. H1k proves the packaged `ops_parallel_audit_review` workflow is safe and runnable, but it decomposes the raw parallel pressure enough that no-directive MLX stays clean. The CLI-live exact replay path keeps the raw contract pressure intact.

## Candidate Replay Update

Wave three was the first prompt-contract wave gated through CLI-live replay; wave four tested the next visual-state hypothesis and did not improve the live ceiling:

| Candidate | Family | Candidate exact | No-directive exact | Contracted exact | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `canonical_json_copy_v3` | canonical CLI/API arguments | `0 / 4` | `0 / 4` | `4 / 4` | no live gain; two cases regress to no-call |
| `visual_tool_initiation_v3` | visual no-call | `1 / 3` plus one executable paraphrase | `0 / 3` | `2 / 3` plus one executable paraphrase | first live candidate movement, but still one wrong-tool visual referent miss |
| `visual_state_tool_selection_v4` | visual state/tool selection | `1 / 3` and no executable visual-form recovery | `0 / 3` | `2 / 3` plus one executable paraphrase | preserves one exact visual recovery, but does not fix `visual_latest_filter_literal` and regresses form targeting to no-call |

Source comparisons:

- [`20260507T_canonical_argument_canonical_json_copy_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260507T_canonical_argument_canonical_json_copy_vs_no_directive_live_v1)
- [`20260507T_visual_state_visual_tool_initiation_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260507T_visual_state_visual_tool_initiation_vs_no_directive_live_v1)
- [`20260508T_visual_state_tool_selection_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_state_tool_selection_vs_no_directive_live_v1)
- [`20260508T_visual_state_contracted_vs_tool_selection_live_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_state_contracted_vs_tool_selection_live_v1)

## Next Use

Use live replay before promoting any new prompt-contract candidate back into H1:

1. Probe candidate with `scripts/run_tool_prompt_contract_probe_packet.py`.
2. If the probe improves, run the relevant `moonie-agent replay-live --execute` family.
3. Compare contracted, no-directive, and candidate live replay packets.
4. Only then spend H1i/H1h cycles.
