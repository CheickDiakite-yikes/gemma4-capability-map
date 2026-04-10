# Decision Log

## Active Decisions

### 1. `KnowledgeWorkArena` stays inside this repo

Reason:

- the episode layer is built directly on the atomic task substrate
- keeping both layers together preserves comparability and reduces duplicated infrastructure

### 2. `replayable_core` is the canonical job-readiness lane

Reason:

- it is deterministic and publishable
- `live_web_stress` is useful realism pressure, but not stable enough for top-line claims

### 3. Memory is episode-scoped in v1

Reason:

- enough realism for multi-stage work
- avoids conflating benchmark quality with cross-episode memory persistence before the episode layer is fully mature

### 4. Primary scoring truth is artifacts plus traces together

Reason:

- artifacts alone hide workflow mistakes
- traces alone hide output usefulness

### 5. Published KWA results must separate strict, recovered, and readiness layers

Reason:

- these scores diverge materially in practice
- collapsing them into one score destroys the benchmark signal

### 6. Ad hoc pilots must not overwrite canonical lane pointers

Operational rule:

- use `--no-update-latest`
- assign an explicit `--output-dir`

Reason:

- preserves the published replayable/live snapshots
- keeps exploratory evidence and canonical evidence distinct

### 7. Native file-backed artifacts are required for finance and job-application episodes

Reason:

- markdown-only stand-ins are too weak for role-readiness claims
- `.xlsx`, `.pptx`, and `.docx` better reflect actual work products
