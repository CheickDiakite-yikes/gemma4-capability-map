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

Only run broader rows after the two-row read:

- if schema-field hints separates from no-directive, add contracted and schema-target-literal rows
- if both rows saturate, redesign the transfer cases before spending more execution budget
- if no-directive fails but schema-field also fails, inspect raw traces before adding another prompt contract
