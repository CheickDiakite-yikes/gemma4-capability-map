# gemma4-capability-map

`gemma4-capability-map` is a local-first benchmark scaffold for comparing Gemma-native agent architectures under reasoning, tool use, retrieval, long-context, and efficiency drift.

The repo is built around three architecture families:

- `monolith`: Gemma 4 plans, routes, retrieves, and answers.
- `hybrid`: EmbeddingGemma retrieves; Gemma 4 plans, routes, and answers.
- `modular`: EmbeddingGemma retrieves; FunctionGemma proposes single-step or parallel tool calls; Gemma 4 plans chained work and synthesizes answers.

The repo now also includes `KnowledgeWorkArena`, a role-based benchmark layer for executive assistant, job-application operations, and finance work across replayable-core and live-web stress lanes.

## What ships in this scaffold

- Versioned task, variant, and trace schemas
- Deterministic simulated tool environments for files, calendar, repo, and screenshot/doc tasks
- Adapter-ready runners for Gemma 4, FunctionGemma, and EmbeddingGemma
- Factorized stressor generation with optional crossed variants
- Leaderboard export, trace export, and a Streamlit replay app
- Fifty-two gold tasks across four tracks, plus factorized variants, smoke tests, and a first-class real-world autonomy layer
- A new `KnowledgeWorkArena` episode layer with replayable-core and live-web stress lanes for executive assistant, job application ops, and finance work

## Tracks

- `thinking`: text and screenshot tasks with thinking on/off comparisons
- `tool_routing`: single-tool, parallel-tool, distractor, schema-drift, and validator-retry tasks
- `retrieval`: stuffing versus retrieve-then-synthesize
- `full_stack`: deterministic 2-4 step execution tasks

## Quickstart

Create an environment and install the package:

```bash
uv sync --extra dev
```

Optional local credentials can live in `.env.local` or `.env`. The repo auto-loads those files on package import and does not override values already exported in your shell.

```bash
cp .env.example .env.local
```

The same env file can also point the runtime at local model directories and force cache-only operation once weights are on disk:

```bash
GEMMA4_E2B_PATH=/absolute/path/to/gemma-4-E2B-it
GEMMA4_E4B_PATH=/absolute/path/to/gemma-4-E4B-it
GEMMA4_OFFLINE=1
```

Generate factorized variants:

```bash
uv run python scripts/make_variants.py
```

Run the deterministic smoke bundle:

```bash
uv run python scripts/run_eval.py --pipeline monolith --backend oracle --limit 12
```

Run the real Hugging Face backend smoke checks:

```bash
uv sync --extra dev --extra hf
uv run python scripts/smoke_hf_backend.py --backend hf --model google/gemma-4-E2B-it --device mps --skip-image
```

Start or inspect the reusable local HF reasoner service:

```bash
uv run python scripts/hf_reasoner_service.py start --model google/gemma-4-E2B-it --device mps
uv run python scripts/hf_reasoner_service.py status --model google/gemma-4-E2B-it --device mps
```

Run the MLX smoke path on an Apple Silicon-compatible converted model:

```bash
uv sync --extra dev --extra mlx
uv run python scripts/smoke_hf_backend.py --backend mlx --model mlx-community/gemma-4-e2b-it-4bit --skip-image
```

Build the local backend comparison report from current harness and alpha-slice artifacts:

```bash
uv run python scripts/build_backend_report.py
```

Run backend preflight before local model work:

```bash
uv run python scripts/preflight_backends.py
```

This writes a backend health snapshot under [`results/tables`](/Users/cheickdiakite/Codex/moonie/results/tables), including the current MLX runtime probe, the recommended local reasoner backend for this machine/session, the latest HF import probe when present, the latest specialist-access probe when present, and the default HF service state paths.

Record specialist model availability before claiming real FunctionGemma or EmbeddingGemma results:

```bash
uv run python scripts/probe_specialist_access.py
```

Build the longitudinal research/history report:

```bash
uv run python scripts/build_history_report.py
```

This refreshes canonical matrix pointers, latest experiment completions, and run-group history under [`results/history`](/Users/cheickdiakite/Codex/moonie/results/history).

Run the targeted drift probe matrix:

```bash
uv run python scripts/run_alpha_matrix.py --config configs/alpha_drift_matrix.yaml
```

Run the real specialist-backend probe matrix:

```bash
uv run python scripts/run_alpha_matrix.py --config configs/alpha_specialist_matrix.yaml
```

Run the real-world autonomy matrix:

```bash
uv run python scripts/run_alpha_matrix.py --config configs/alpha_real_world_matrix.yaml
```

Current canonical real-world snapshot:

- [`20260409T210500Z_alpha_real_world`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T210500Z_alpha_real_world)
- `hf_e2b_real_world_thinking_variants`: `0.0` success
- `hf_e2b_real_world_retrieval_variants`: `0.875` success, `1.0` strict interface
- `hf_e2b_real_world_routing_variants`: `0.5` success
- `hf_e2b_real_world_full_stack_variants`: `0.75` strict success, `1.0` recovered execution

Generate the `KnowledgeWorkArena` seed episodes and fixtures:

```bash
uv run python scripts/make_knowledge_work_arena.py
```

Run the replayable-core `KnowledgeWorkArena` lane:

```bash
uv run python scripts/run_knowledge_work_arena.py --lane replayable_core --backend oracle
```

Run the live-web stress lane separately:

```bash
uv run python scripts/run_knowledge_work_arena.py --lane live_web_stress --backend oracle
```

Build the `KnowledgeWorkArena` history and canonical lane snapshot report:

```bash
uv run python scripts/build_knowledge_work_history.py
```

Current canonical `KnowledgeWorkArena` lane summaries:

- replayable core: [`results/knowledge_work/replayable_core/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/replayable_core/summary.json)
- live web stress: [`results/knowledge_work/live_web_stress/summary.json`](/Users/cheickdiakite/Codex/moonie/results/knowledge_work/live_web_stress/summary.json)

`KnowledgeWorkArena` now reports four useful realism layers together:

- `artifact_quality_avg`
- `browser_workflow_avg`
- `strict_interface_avg`
- `recovered_execution_avg`

Those feed the role-level readiness score while keeping interface correctness and recovered end-state completion visibly separate.

Launch the replay UI:

```bash
uv run streamlit run src/gemma4_capability_map/app/streamlit_app.py
```

## Hugging Face model backend

The benchmark core is runnable without model weights by using the `oracle` or `heuristic` backends. Real Gemma evaluation is adapter-ready through optional dependencies:

```bash
uv sync --extra dev --extra hf
uv run python scripts/run_eval.py \
  --pipeline hybrid \
  --reasoner-backend hf \
  --retriever-backend hf \
  --reasoner google/gemma-4-E4B-it
```

This repository does not bundle Gemma weights. Large-model execution depends on local hardware, Hugging Face access, optional package installation, and in practice works best when the reasoner, router, and retriever backends can be configured independently.

For Hugging Face access, the smoke and probe scripts automatically use `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` when either is present in the shell or in `.env.local` / `.env`. For MLX, use an MLX-converted model id; the smoke script auto-resolves `google/gemma-4-E2B-it` and `google/gemma-4-E4B-it` to the matching `mlx-community/...-4bit` repos as a convenience inference.

For fully local runs after the first download, you can either pass a local model path directly anywhere a model id is accepted or set `GEMMA4_E2B_PATH`, `GEMMA4_E4B_PATH`, `FUNCTIONGEMMA_PATH`, or `EMBEDDINGGEMMA_PATH` in `.env.local`. Setting `GEMMA4_OFFLINE=1` also forces local-files-only loading on the Hugging Face backend, which is useful once the cache or mirrored directories are already populated.

The MLX path currently targets local Apple Silicon throughput. Native Gemma thinking mode remains best validated on the Hugging Face backend because `enable_thinking=True` is an HF chat-template feature; the MLX backend approximates thinking-mode prompts but does not expose the same native toggle today.

For repeated local HF runs on this Mac, use two different modes deliberately:

- for bounded single-shot specialist slices, direct in-process `hf` is still fine
- for multi-experiment HF matrices on this Mac, prefer `--reasoner-backend hf_service` plus subprocess experiment execution so the heavy Gemma 4 reasoner loads once and is reused across the matrix

Current local recommendation on this Apple Silicon machine:

- Run `scripts/preflight_backends.py` first and trust the recorded recommendation for the current session.
- Use `google/gemma-4-E2B-it` with `--backend mlx` only when the MLX runtime probe is healthy.
- Use `google/gemma-4-E2B-it` or `google/gemma-4-E4B-it` with `--backend hf` when validating native thinking behavior or running higher-fidelity comparison probes.
- Check [`results/tables/hf_import_probe.json`](/Users/cheickdiakite/Codex/moonie/results/tables/hf_import_probe.json) before assuming fresh-process HF startup is cheap on this machine.
- Use `google/gemma-4-E2B-it` with `--backend hf_service` only when the HF reasoner service is healthy and the import probe says the worker path is worth it for the slice you want to run.
- Prefer `GEMMA4_OFFLINE=1` once weights are cached locally so benchmark runs do not depend on Hub availability.
- Keep `google/gemma-4-E4B-it` off the default interactive path on this Mac. The local HF probe loads successfully, but the multimodal alpha slice is too slow to treat as an everyday inner-loop backend here.

## Research Tracking

- matrix snapshots live in [`results/alpha_matrix`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix)
- longitudinal history lives in [`results/history`](/Users/cheickdiakite/Codex/moonie/results/history)
- canonical run-group pointers live in [`results/history/canonical_run_groups.csv`](/Users/cheickdiakite/Codex/moonie/results/history/canonical_run_groups.csv)
- current canonical real-world autonomy snapshot lives at [`20260409T210500Z_alpha_real_world`](/Users/cheickdiakite/Codex/moonie/results/alpha_matrix/20260409T210500Z_alpha_real_world)
- operational guidance for research runs lives in [`docs/research-ops.md`](/Users/cheickdiakite/Codex/moonie/docs/research-ops.md)
- dated findings and notable posture changes live in [`docs/research-log.md`](/Users/cheickdiakite/Codex/moonie/docs/research-log.md)
- real-world autonomy design notes live in [`docs/real-world-benchmarking.md`](/Users/cheickdiakite/Codex/moonie/docs/real-world-benchmarking.md)
- `KnowledgeWorkArena` design notes live in [`docs/knowledge-work-arena.md`](/Users/cheickdiakite/Codex/moonie/docs/knowledge-work-arena.md)
- `KnowledgeWorkArena` history lives in [`results/history/knowledge_work_history.md`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_history.md) and [`results/history/knowledge_work_history.json`](/Users/cheickdiakite/Codex/moonie/results/history/knowledge_work_history.json)
- HF reasoner service runtime artifacts live under [`results/runtime/hf_reasoner`](/Users/cheickdiakite/Codex/moonie/results/runtime/hf_reasoner)
- HF import timing artifacts live at [`results/tables/hf_import_probe.json`](/Users/cheickdiakite/Codex/moonie/results/tables/hf_import_probe.json) and [`results/tables/hf_import_probe.md`](/Users/cheickdiakite/Codex/moonie/results/tables/hf_import_probe.md)
- Specialist availability artifacts live at [`results/tables/specialist_access_probe.json`](/Users/cheickdiakite/Codex/moonie/results/tables/specialist_access_probe.json) and [`results/tables/specialist_access_probe.md`](/Users/cheickdiakite/Codex/moonie/results/tables/specialist_access_probe.md)

## References

- [Gemma 4 launch](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [Thinking mode](https://ai.google.dev/gemma/docs/capabilities/thinking)
- [Function calling](https://ai.google.dev/gemma/docs/capabilities/function-calling)
- [FunctionGemma](https://ai.google.dev/gemma/docs/functiongemma)
- [EmbeddingGemma](https://ai.google.dev/gemma/docs/embeddinggemma)
- [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
