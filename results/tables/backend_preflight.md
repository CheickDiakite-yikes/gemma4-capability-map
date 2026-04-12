# Backend Preflight

- Generated at: `2026-04-12T18:12:54.005087+00:00`
- Recommended local reasoner backend: `mlx`
- HF token present: `True`
- HF token source: `HF_TOKEN`
- Offline mode enabled: `False`

## MLX runtime probe

- ok: `True`
- elapsed_ms: `221`
- returncode: `0`

## HF import probe

- ok: `True`
- elapsed_ms: `4441`
- torch_ms: `1865`
- transformers_ms: `4093`

## Specialist access probe

- `google/gemma-4-E2B-it`: access=`available` api=`200` config=`200` gated=`False`
- `google/functiongemma-270m-it`: access=`available` api=`200` config=`200` gated=`manual`
- `google/embeddinggemma-300m`: access=`available` api=`200` config=`200` gated=`manual`

## HF reasoner service

- status: `ready`
- socket_path: `/Users/cheickdiakite/Codex/moonie/results/runtime/hf_reasoner/google__gemma_4_E2B_it_mps/service.sock`
- state_path: `/Users/cheickdiakite/Codex/moonie/results/runtime/hf_reasoner/google__gemma_4_E2B_it_mps/state.json`

## Resolved model sources

- `Qwen/Qwen3-8B` -> `/Users/cheickdiakite/models/Qwen3-8B`
- `Qwen/Qwen3-8B-MLX-4bit` -> `/Users/cheickdiakite/models/Qwen3-8B-MLX-4bit`
- `google/embeddinggemma-300m` -> `google/embeddinggemma-300m`
- `google/functiongemma-270m-it` -> `google/functiongemma-270m-it`
- `google/gemma-4-31b-it` -> `google/gemma-4-31b-it`
- `google/gemma-4-E2B-it` -> `google/gemma-4-E2B-it`
- `google/gemma-4-E4B-it` -> `google/gemma-4-E4B-it`
- `mlx-community/gemma-4-e2b-it-4bit` -> `mlx-community/gemma-4-e2b-it-4bit`
