# H1n Visual Alias-Transfer Replay Slice

H1n is the next visual discriminator after H1l/H1m packaged saturation.

The goal is to preserve the replay-live shape that still separates rows, but move beyond the exact alias-repeat examples. The suite uses fresh labels and decoys so we can test transfer rather than memorized wording:

- review tile vs notice/table decoys
- status pill vs chart/table decoys
- error banner vs note/table decoys
- queue badge vs person/table decoys
- current form validation error vs stale selection/status chip
- signature warning text vs checkbox/table decoys

Source packet:

- packet: [`results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_dry_run_v1`](../../results/tool_probe_replay_packets/20260509T_visual_hard_slice_live_stress_alias_transfer_dry_run_v1)
- suite flag: `--suite alias_transfer_v3`
- entrypoint: `moonie-agent replay-live`
- cases: `6`
- families: `visual_argument_transfer = 4`, `visual_tool_routing_transfer = 2`

Build command:

```bash
uv run python scripts/build_visual_hard_slice_live_stress_packet.py \
  --run-group-id 20260509T_visual_hard_slice_live_stress_alias_transfer_dry_run_v1 \
  --suite alias_transfer_v3
```

Candidate execution order:

```bash
uv run moonie-agent replay-live \
  --packet-id 20260509T_visual_hard_slice_live_stress_alias_transfer_dry_run_v1 \
  --output-dir results/tool_probe_replay_live/20260509T_visual_hard_slice_live_stress_alias_transfer_no_directive_execute_v1 \
  --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive \
  --execute --json

uv run moonie-agent replay-live \
  --packet-id 20260509T_visual_hard_slice_live_stress_alias_transfer_dry_run_v1 \
  --output-dir results/tool_probe_replay_live/20260509T_visual_hard_slice_live_stress_alias_transfer_schema_field_hints_execute_v1 \
  --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_visual_role_catalog_schema_field_hints \
  --execute --json
```

## Executed Result

The full matrix has now been executed:

- no-directive: strict `0 / 6`, executor-equivalent `2 / 6`
- contracted: strict `5 / 6`, executor-equivalent `1 / 6`
- role catalog v1: strict `1 / 6`, executor-equivalent `3 / 6`
- argument hints v2: strict `1 / 6`, executor-equivalent `6 / 6`
- schema-field hints v4: strict `1 / 6`, executor-equivalent `2 / 6`
- schema target literals v5: strict `1 / 6`, executor-equivalent `4 / 6`

Generated evidence:

- diagnostic: [`results/reports/visual_alias_transfer_diagnostic/diagnostic.md`](../../results/reports/visual_alias_transfer_diagnostic/diagnostic.md)
- report table: [`results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_alias_transfer_live_replay_summary.csv`](../../results/reports/mlx_tool_contract_harnessing/tables/visual_hard_slice_alias_transfer_live_replay_summary.csv)
- report figure: [`results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_alias_transfer_live_replay_gate.svg`](../../results/reports/mlx_tool_contract_harnessing/figures/visual_hard_slice_alias_transfer_live_replay_gate.svg)

Interpretation:

- Argument hints v2 is the strongest current no-directive executor-grounding profile on fresh transfer cases.
- Schema target literals v5 still help executor-equivalence, but less than argument hints on this packet.
- Contracted MLX is still the strict-fidelity upper bound, but its exact rows need scorer-level inspection because the current executor-target scorer marks only `1 / 6` executor-equivalent.
- The next move is not packaged H1; it is scorer inspection plus a repeat or helper-ablation around the non-packaged replay-live surface.
