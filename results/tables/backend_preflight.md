# Backend Preflight

- Generated at: `2026-04-10T21:21:05.687593+00:00`
- Recommended local reasoner backend: `hf`
- HF token present: `True`
- HF token source: `HF_TOKEN`
- Offline mode enabled: `False`

## MLX runtime probe

- ok: `False`
- elapsed_ms: `20`
- returncode: `1`
- stderr: `Traceback (most recent call last):
  File "<string>", line 2, in <module>
    import mlx.core as mx
ModuleNotFoundError: No module named 'mlx'`

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

- `google/gemma-4-E2B-it` -> `google/gemma-4-E2B-it`
- `google/gemma-4-E4B-it` -> `google/gemma-4-E4B-it`
- `google/functiongemma-270m-it` -> `google/functiongemma-270m-it`
- `google/embeddinggemma-300m` -> `google/embeddinggemma-300m`
