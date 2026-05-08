# Wave Four Visual State Candidate

## Status

Executed on May 8, 2026. Result: useful negative evidence, not a promotion candidate.

- raw probe packet: [`results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_execute_v1`](../../results/tool_prompt_contract_probe_packets/20260508T_prompt_contract_wave4_execute_v1)
- live visual packet: [`results/tool_probe_replay_live/20260508T_visual_state_tool_selection_live_execute_v1`](../../results/tool_probe_replay_live/20260508T_visual_state_tool_selection_live_execute_v1)
- comparison vs no-directive: [`results/tool_probe_replay_live_comparisons/20260508T_visual_state_tool_selection_vs_no_directive_live_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_state_tool_selection_vs_no_directive_live_v1)
- comparison vs contracted: [`results/tool_probe_replay_live_comparisons/20260508T_visual_state_contracted_vs_tool_selection_live_v1`](../../results/tool_probe_replay_live_comparisons/20260508T_visual_state_contracted_vs_tool_selection_live_v1)

Key result:

| Surface | Result |
| --- | --- |
| raw probe | exact `1 / 8`, executable `0 / 1`, weak exact gain |
| live visual vs no-directive | exact improves from `0 / 3` to `1 / 3` |
| live visual vs contracted | exact falls from `2 / 3` to `1 / 3`; executable falls from `1.0` to `0.0` |
| targeted case | `visual_latest_filter_literal` still fails as `wrong_tool` |
| regression | `visual_form_target_literal` falls back to `no_tool_call` |

Interpretation: the broad visual state/tool-selection wording did not solve the remaining wrong-tool referent failure. Wave three's useful behavior was visual tool initiation; v4 did not preserve that enough to recover the executable form target.

## Why This Exists

Wave three produced the first prompt-contract candidate with live replay movement:

- `visual_tool_initiation_v3` improved visual live exact replay from `0 / 3` to `1 / 3`
- it recovered the executable visual-form target
- it emitted one tool call in all three visual replay cases

It still failed one core case:

- `visual_latest_filter_literal`
- failure mode: `wrong_tool`

So the next candidate did not broadly repeat schema, literal-copy, or parallel wording. The useful target was narrower:

> Preserve visual tool initiation while improving visual state and tool selection.

## Candidate Hypothesis

`visual_state_tool_selection_v4` tested whether no-directive MLX can choose the correct visual tool from the latest visual state when multiple visual tools are available.

Expected mechanism:

- if latest state has an existing `selection_id` and the user asks to filter, narrow, latest-only, or constrain that selection, call `refine_selection`
- if latest state has a `region_id` and the user asks to read or report text, call `read_region_text`
- if no selection or region exists and the task asks to locate a target, call the locating visual tool first
- preserve `image_id`, `selection_id`, `region_id`, `target_query`, and `filter_query` literally

Non-goals:

- do not reintroduce the final exact tool-turn directive
- do not leak the planned next call for a specific probe case
- do not spend H1i/H1h until raw probe or CLI-live replay moves
- do not treat packaged-workflow saturation as a pass

## Executed Gate

The gate was run in this order:

```bash
uv run python scripts/run_tool_prompt_contract_probe_packet.py \
  --candidate-wave v4 \
  --run-group-id <timestamp>_prompt_contract_wave4_execute_v1 \
  --execute
```

The raw probe produced a weak exact gain, so the visual live replay was run:

```bash
uv run moonie-agent replay-live \
  --packet-dir results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1 \
  --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_state_tool_selection \
  --case-id visual_form_target_literal \
  --case-id visual_latest_filter_literal \
  --case-id visual_readback_region_literal \
  --output-dir results/tool_probe_replay_live/<timestamp>_visual_state_tool_selection_live_execute_v1 \
  --execute
```

Compare against both no-directive and contracted visual replay:

```bash
uv run python scripts/compare_tool_probe_replay_live_packets.py \
  results/tool_probe_replay_live/20260507T_visual_state_no_directive_live_execute_v1 \
  results/tool_probe_replay_live/<candidate_packet> \
  --output-dir results/tool_probe_replay_live_comparisons/<timestamp>_visual_state_tool_selection_vs_no_directive_live_v1

uv run python scripts/compare_tool_probe_replay_live_packets.py \
  results/tool_probe_replay_live/20260507T_visual_state_contracted_live_execute_v1 \
  results/tool_probe_replay_live/<candidate_packet> \
  --output-dir results/tool_probe_replay_live_comparisons/<timestamp>_visual_state_contracted_vs_tool_selection_live_v1
```

## Promotion Criteria Outcome

Minimum useful signal:

- visual live exact rate improves beyond `1 / 3`, or
- `visual_latest_filter_literal` stops failing as `wrong_tool`, or
- executable visual recovery stays at `1.0` while exact visual replay improves

Reject if:

- raw probe exact/executable behavior does not improve
- live visual exact stays `1 / 3` with the same wrong-tool case
- candidate improves only packaged workflow readiness
- candidate increases no-call failures in canonical or visual replay

Outcome: reject for H1 promotion. Live visual exact stayed `1 / 3`, `visual_latest_filter_literal` remained `wrong_tool`, and executable visual-form recovery did not stay at `1.0`.

## Current Baseline To Beat

| Row | Visual exact | Visual executable | Notes |
| --- | ---: | ---: | --- |
| no directive | `0 / 3` | `0 / 1` | all visual cases no-call |
| `visual_tool_initiation_v3` | `1 / 3` | `1 / 1` | one wrong-tool visual referent miss |
| contracted | `2 / 3` | `1 / 1` | remaining non-exact case is executable |

The target was not a new leaderboard row. The target was evidence that a weaker generic contract can reduce model-side visual tool-selection fragility before the controller has to rescue it. The current answer is no: this wording did not reduce the remaining visual tool-selection fragility.
