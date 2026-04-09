# Backend Preflight

- Generated at: `2026-04-09T16:29:51.430409+00:00`
- Recommended local reasoner backend: `mlx`
- HF token present: `True`
- HF token source: `HF_TOKEN`
- Offline mode enabled: `False`

## MLX runtime probe

- ok: `True`
- elapsed_ms: `147`
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

- status: `stale`
- socket_path: `/Users/cheickdiakite/Codex/moonie/results/runtime/hf_reasoner/google__gemma_4_E2B_it_mps/service.sock`
- state_path: `/Users/cheickdiakite/Codex/moonie/results/runtime/hf_reasoner/google__gemma_4_E2B_it_mps/state.json`

## Resolved model sources

- `google/gemma-4-E2B-it` -> `google/gemma-4-E2B-it`
- `google/gemma-4-E4B-it` -> `google/gemma-4-E4B-it`
- `google/functiongemma-270m-it` -> `google/functiongemma-270m-it`
- `google/embeddinggemma-300m` -> `google/embeddinggemma-300m`
