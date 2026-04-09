# Research Ops

This repo now maintains experiment history as a first-class research artifact rather than a side effect of individual runs.

## History files

- [`results/history/experiment_runs.jsonl`](/Users/cheickdiakite/Codex/moonie/results/history/experiment_runs.jsonl)
  One append-only row per matrix experiment execution, including failed attempts.
- [`results/history/improvements.jsonl`](/Users/cheickdiakite/Codex/moonie/results/history/improvements.jsonl)
  One append-only delta record when a newly completed experiment has a previous completed baseline to compare against.

## Canonical run artifacts

Each matrix run writes a dedicated directory under [`results/alpha_matrix`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix):

- `manifest.json`
- `summary.json`
- `summary.md`
- `combined_traces.jsonl`
- `combined_leaderboard.csv`
- per-experiment subdirectories with their own summaries and trace files

Treat the matrix folder as the immutable snapshot. History logs are the cross-run index.

Two distinctions matter:

- a complete matrix snapshot finished every configured experiment in that manifest
- a partial rerun can still contain valid experiment-level results, but it should not replace the canonical matrix snapshot for that matrix name

## Build the historical digest

Use:

```bash
uv run python scripts/build_history_report.py
```

This writes:

- [`results/history/history_report.json`](/Users/cheickdiakite/Codex/moonie/results/history/history_report.json)
- [`results/history/history_report.md`](/Users/cheickdiakite/Codex/moonie/results/history/history_report.md)
- [`results/history/latest_completed_experiments.csv`](/Users/cheickdiakite/Codex/moonie/results/history/latest_completed_experiments.csv)
- [`results/history/run_groups.csv`](/Users/cheickdiakite/Codex/moonie/results/history/run_groups.csv)
- [`results/history/canonical_run_groups.csv`](/Users/cheickdiakite/Codex/moonie/results/history/canonical_run_groups.csv)

The history digest now distinguishes:

- latest experiment-level completions
- best experiment-level completions
- latest run group per matrix
- latest complete run group per matrix

## Recommended research loop

0. Run `scripts/preflight_backends.py` and keep the resulting backend-health snapshot with the run notes.
0.1. If the recommended local reasoner backend is `hf`, decide whether the slice should use direct `hf` or the reusable `hf_service` path.
0.2. If the slice is bounded and specialist-heavy, prefer direct in-process `hf` execution first; use `hf_service` only when the worker probe is healthy enough to justify the extra process layer.
0.3. Run `scripts/probe_specialist_access.py` before any real FunctionGemma or EmbeddingGemma claim. If access is `gated_denied`, do not treat heuristic specialist stand-ins as real-specialist evidence.
1. Run a matrix config.
2. Inspect the latest matrix `summary.json` and `combined_traces.jsonl`.
3. Confirm whether the run is a complete matrix snapshot or only a partial rerun.
4. Build the history report.
5. Record the conclusion in a short methods or findings note if the run changes a benchmark claim.
6. Only then expand coverage or change prompts/backends.

## What counts as a meaningful improvement

- Higher `success_rate` on the same experiment id and task surface.
- Lower `avg_latency_ms` without materially reducing success.
- Better `failure_breakdown`, especially if a category disappears entirely.
- Fewer harness-induced failures, such as truncation or parser leakage.
- Better recovered reliability on tool-bearing tasks, even when strict schema-exact success is unchanged.

## Summary discipline

- Use `success_rate` as the strict top-line metric.
- Use `metric_averages.interface_reliability_score` to track repaired-but-not-perfect tool execution.
- Use `failing_variants` from experiment summaries as the short list for the next debugging pass.

## Current benchmark posture

- Clean alpha lanes have validated local reference paths.
- Apple Silicon default is `MLX + E2B` only when the current backend preflight says MLX is healthy.
- Native HF thinking remains a separate comparison lane and should be tracked independently because its token/latency profile is fundamentally different.
- Real specialist backend probes should use [`configs/alpha_specialist_matrix.yaml`](/Users/cheickdiakite/Codex/moonie/configs/alpha_specialist_matrix.yaml) so FunctionGemma and EmbeddingGemma claims are grounded in actual specialist-model runs, not heuristic stand-ins.
- Reusable local HF reasoner state now lives under [`results/runtime/hf_reasoner`](/Users/cheickdiakite/Codex/moonie/results/runtime/hf_reasoner), with `state.json`, `events.jsonl`, `service.log`, and `requests.jsonl` per service id.
- Fresh-process HF startup cost should be tracked explicitly through [`hf_import_probe.json`](/Users/cheickdiakite/Codex/moonie/results/tables/hf_import_probe.json) before promoting the worker path as the operational default.
- Specialist availability should be tracked explicitly through [`specialist_access_probe.json`](/Users/cheickdiakite/Codex/moonie/results/tables/specialist_access_probe.json) so gated Hugging Face models fail in preflight, not halfway through matrix interpretation.
