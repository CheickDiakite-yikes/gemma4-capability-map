# Exact Probe Replay Packet

## Why This Exists

H1i repeat3, H1j candidate, and H1j helper-ablation packets all saturated even though the raw no-directive probe still fails every exact-call case. The packaged workflow surface is therefore washing out the failure mechanism we care about.

The exact-probe replay packet is the next bridge. It preserves raw probe cases as CLI-inspectable artifacts before they are promoted into any live workflow or H1 packet.

The live operator bridge now starts with `moonie-agent replay-live`. It is intentionally separate from packaged workflow sessions: it replays exact probe cases as raw tool-contract checks, renders progress with Rich, and writes a small live replay packet under `results/tool_probe_replay_live/`.

## Current Packet

- packet: [`results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_v1`](../../results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_v1)
- executed packet: [`results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1`](../../results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1)
- contracted replay packet: [`results/tool_probe_replay_packets/20260507T_contracted_exact_probe_replay_execute_v1`](../../results/tool_probe_replay_packets/20260507T_contracted_exact_probe_replay_execute_v1)
- replay A/B comparison: [`results/tool_probe_replay_comparisons/20260507T_contracted_vs_no_directive_exact_replay_v1`](../../results/tool_probe_replay_comparisons/20260507T_contracted_vs_no_directive_exact_replay_v1)
- source probe: [`results/tool_directive_probe/20260507T_mlx_no_directive_probe_v1`](../../results/tool_directive_probe/20260507T_mlx_no_directive_probe_v1)
- contracted baseline: [`results/tool_directive_probe/20260506T_mlx_tool_directive_probe_v4`](../../results/tool_directive_probe/20260506T_mlx_tool_directive_probe_v4)
- case count: `8`
- failure modes: `argument_mismatch = 4`, `no_tool_call = 4`
- next actions:
  - `build_canonical_argument_replay = 4`
  - `build_visual_state_replay_executor = 3`
  - `build_parallel_array_replay_or_workflow = 1`
- dry run: `true`
- executed replay: `8` cases, exact `0 / 8`, same failure split reproduced
- contracted replay: `7 / 8` exact, with the remaining visual selector paraphrase executable
- A/B delta: no-directive exact rate is `-0.875` versus contracted on the same eight cases
- first live operator dry run: [`results/tool_probe_replay_live/20260507T_parallel_array_replay_live_dry_run_v1`](../../results/tool_probe_replay_live/20260507T_parallel_array_replay_live_dry_run_v1)
- first no-directive live execution: [`results/tool_probe_replay_live/20260507T_parallel_array_no_directive_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_parallel_array_no_directive_live_execute_v1)
- first contracted live execution: [`results/tool_probe_replay_live/20260507T_parallel_array_contracted_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_parallel_array_contracted_live_execute_v1)

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

The `replay-live` CLI follows the same rule. It is an operator-visible exact replay harness, not a workflow-family leaderboard row. Use it to test whether raw no-directive or prompt-contract systems can reproduce exact calls under the same probe messages, media, and allowed tool schemas.

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

The builder also supports focused filters:

```bash
uv run python scripts/build_tool_probe_replay_packet.py \
  --run-group-id <timestamp>_visual_state_replay \
  --next-action build_visual_state_replay_executor \
  --execute
```

Available filters: `--case-id`, `--family`, `--failure-mode`, and `--next-action`.

## Live Operator Replay

Dry-run one exact replay case through the Rich CLI surface:

```bash
uv run moonie-agent replay-live \
  --packet-dir results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1 \
  --case-id parallel_audit_array_literal \
  --output-dir results/tool_probe_replay_live/<timestamp>_parallel_array_replay_live_dry_run_v1
```

JSON dry run for scripted checks:

```bash
uv run moonie-agent replay-live \
  --packet-dir results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1 \
  --case-id parallel_audit_array_literal \
  --output-dir results/tool_probe_replay_live/<timestamp>_parallel_array_replay_live_dry_run_v1 \
  --json
```

Execution mode:

```bash
uv run moonie-agent replay-live \
  --packet-dir results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1 \
  --case-id parallel_audit_array_literal \
  --system-id mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive \
  --output-dir results/tool_probe_replay_live/<timestamp>_parallel_array_replay_live_execute_v1 \
  --execute
```

The first tracked live dry run is:

- [`results/tool_probe_replay_live/20260507T_parallel_array_replay_live_dry_run_v1`](../../results/tool_probe_replay_live/20260507T_parallel_array_replay_live_dry_run_v1)

It records the selected source packet, system id, exact replay case id, source failure mode, and the exact command needed to execute the same case.

The first tracked no-directive live execution is:

- [`results/tool_probe_replay_live/20260507T_parallel_array_no_directive_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_parallel_array_no_directive_live_execute_v1)

Result: expected `2` tool calls, actual `0`, exact `false`, replay failure mode `no_tool_call`. The raw model output asks the operator to provide the screenshot and `config/settings.yaml`, which is precisely the protocol collapse the final tool-turn directive was preventing.

The paired contracted live execution is:

- [`results/tool_probe_replay_live/20260507T_parallel_array_contracted_live_execute_v1`](../../results/tool_probe_replay_live/20260507T_parallel_array_contracted_live_execute_v1)

Result: expected `2` tool calls, actual `2`, exact `true`. The actual calls are `inspect_image({"image_id": "img-parallel"})` and `read_repo_file({"path": "config/settings.yaml"})`, in the same independent-call shape as the exact replay oracle.

CLI inspection:

```bash
uv run moonie-agent packet \
  --kind tool-probe-replay-live \
  --packet-id 20260507T_parallel_array_replay_live_dry_run_v1
```

Executed packet:

```bash
uv run python scripts/build_tool_probe_replay_packet.py \
  --run-group-id 20260507T_no_directive_exact_probe_replay_execute_v1 \
  --execute
```

Result: `0 / 8` exact. All four source argument mismatches replayed as argument mismatches, and all four source no-tool-call cases replayed as no-tool-call cases.

Contracted replay baseline:

```bash
uv run python scripts/build_tool_probe_replay_packet.py \
  --run-group-id 20260507T_contracted_exact_probe_replay_execute_v1 \
  --system-id mlx_gemma4_e2b_reasoner_only \
  --execute
```

Result: `7 / 8` exact. The only non-exact case is `visual_form_target_literal`, which remains executable through visual selector aliasing.

Comparison:

```bash
uv run python scripts/compare_tool_probe_replay_packets.py \
  results/tool_probe_replay_packets/20260507T_contracted_exact_probe_replay_execute_v1 \
  results/tool_probe_replay_packets/20260507T_no_directive_exact_probe_replay_execute_v1 \
  --output-dir results/tool_probe_replay_comparisons/20260507T_contracted_vs_no_directive_exact_replay_v1
```

Result: no-directive exact replay rate is `0.0` versus contracted `0.875`, for a delta of `-0.875`.

## Focused Visual Replay

No-directive visual-state packet:

- [`results/tool_probe_replay_packets/20260507T_visual_state_exact_replay_no_directive_v1`](../../results/tool_probe_replay_packets/20260507T_visual_state_exact_replay_no_directive_v1)
- result: exact `0 / 3`; all three cases remain `no_tool_call`

Contracted visual-state packet:

- [`results/tool_probe_replay_packets/20260507T_visual_state_exact_replay_contracted_v1`](../../results/tool_probe_replay_packets/20260507T_visual_state_exact_replay_contracted_v1)
- result: exact `2 / 3`; the remaining case is executable through visual selector aliasing

Comparison:

- [`results/tool_probe_replay_comparisons/20260507T_visual_state_contracted_vs_no_directive_v1`](../../results/tool_probe_replay_comparisons/20260507T_visual_state_contracted_vs_no_directive_v1)
- no-directive delta exact rate: `-0.6666666666666666`
- visual-form target case also drops from executable paraphrase to no tool call

## Focused Parallel Replay

No-directive parallel-array packet:

- [`results/tool_probe_replay_packets/20260507T_parallel_array_exact_replay_no_directive_v1`](../../results/tool_probe_replay_packets/20260507T_parallel_array_exact_replay_no_directive_v1)
- result: exact `0 / 1`; the case remains `no_tool_call`

Contracted parallel-array packet:

- [`results/tool_probe_replay_packets/20260507T_parallel_array_exact_replay_contracted_v1`](../../results/tool_probe_replay_packets/20260507T_parallel_array_exact_replay_contracted_v1)
- result: exact `1 / 1`

Comparison:

- [`results/tool_probe_replay_comparisons/20260507T_parallel_array_contracted_vs_no_directive_v1`](../../results/tool_probe_replay_comparisons/20260507T_parallel_array_contracted_vs_no_directive_v1)
- no-directive delta exact rate: `-1.0`

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
