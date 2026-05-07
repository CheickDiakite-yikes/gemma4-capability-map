# Exact Probe Replay Packet

## Why This Exists

H1i repeat3, H1j candidate, and H1j helper-ablation packets all saturated even though the raw no-directive probe still fails every exact-call case. The packaged workflow surface is therefore washing out the failure mechanism we care about.

The exact-probe replay packet is the next bridge. It preserves raw probe cases as CLI-inspectable artifacts before they are promoted into any live workflow or H1 packet.

## Current Packet

- packet: [`results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_v1`](../../results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_v1)
- source probe: [`results/tool_directive_probe/20260507T_mlx_no_directive_probe_v1`](../../results/tool_directive_probe/20260507T_mlx_no_directive_probe_v1)
- contracted baseline: [`results/tool_directive_probe/20260506T_mlx_tool_directive_probe_v4`](../../results/tool_directive_probe/20260506T_mlx_tool_directive_probe_v4)
- case count: `8`
- failure modes: `argument_mismatch = 4`, `no_tool_call = 4`
- next actions:
  - `build_canonical_argument_replay = 4`
  - `build_visual_state_replay_executor = 3`
  - `build_parallel_array_replay_or_workflow = 1`
- dry run: `true`

## Packet Contents

- `manifest.json`: source probes, replay system id, selected case ids, and failure counts
- `summary.json`: compact packet summary
- `commands.json`: one runnable `run_tool_directive_probe.py --case-id <case>` command per case
- `replay_cases.csv`: one row per failed no-directive probe case
- `replay_cases.json`: full replay payloads
- `cases/<case_id>.json`: messages, media, tool specs, expected calls, source actual calls, and baseline comparison context
- `replay_next_actions.csv`: per-case implementation backlog for the next replay/live discriminator

## Method Guardrail

This packet is not a packaged live workflow and should not be counted as live workflow execution. It is a raw exact-call replay artifact for deciding what must be represented next.

Use it to choose the next live discriminator:

- CLI/API argument mismatch needs canonical argument replay under the same allowed tool schemas.
- Visual no-call cases need operator-visible visual-state replay, not only packaged workflow completion.
- `parallel_audit_array_literal` needs either a faithful packaged workflow or a replay execution path that preserves the expected two-call array.

## Regeneration

```bash
uv run python scripts/build_tool_probe_replay_packet.py \
  --run-group-id 20260507T_no_directive_exact_probe_replay_v1
```

Execution mode:

```bash
uv run python scripts/build_tool_probe_replay_packet.py \
  --run-group-id <timestamp>_no_directive_exact_probe_replay_execute \
  --execute
```

Execution mode writes per-case runs under `runs/<case_id>/` plus `replay_results.json` and `replay_results.csv`.

Focused verification:

```bash
uv run pytest tests/test_tool_probe_replay_packet.py tests/test_tool_directive_probe.py -q
```

CLI inspection:

```bash
uv run moonie-agent packet \
  --kind tool-probe-replay \
  --packet-id 20260507T_no_directive_exact_probe_replay_v1
```
