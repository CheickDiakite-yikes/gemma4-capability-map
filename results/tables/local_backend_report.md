# Local Backend Report

Default Apple Silicon local reasoner backend: `mlx`

Reason: MLX E2B matches the validated multimodal alpha slice on this Mac with materially lower load and warm inference latency than HF.

HF role: native thinking validation and larger or slower experimental runs

## E2B comparison

| Backend | Load ms | Warm text ms | Warm image ms | Alpha success | Alpha avg latency ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF | 43401 | 747.0 | 4050.0 | 1.0 | 58777.5 |
| MLX | 16749 | 431.0 | 1374.0 | 1.0 | 9231.5 |

## E4B local probe

- Status: `completed`
- Load elapsed ms: `1597752`
- Runtime device: `mps`
- Load mode: `vision`