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

### 8. KWA history must distinguish canonical and exploratory runs

Operational rule:

- canonical lane pointers come from dedicated canonical summaries
- exploratory pilots use `--no-update-latest --run-intent exploratory`
- generated history must show at least:
  - latest canonical by lane
  - latest exploratory by lane
  - best historical by lane

Reason:

- exploratory KWA runs materially inform research, but they should not shadow published benchmark state
- future threads need a durable way to resume without re-deriving which run is authoritative

### 9. Board and chart exports must be registry-backed

Operational rule:

- add new systems to [`configs/model_registry.yaml`](../../configs/model_registry.yaml)
- keep board exports derived from:
  - `system_id`
  - `lane`
  - `run_intent`

Reason:

- board-style comparisons break down when system metadata is inferred ad hoc from run directories
- public-style reporting needs normalized params, provider, local/remote, and cost fields

### 10. Scoring changes should be propagated by rescoring traces, not rerunning models by default

Operational rule:

- when scoring logic changes, use:
  - `uv run python scripts/rescore_knowledge_work_runs.py <run_dir> ...`
- rerun models only when the change affects execution rather than grading

Reason:

- trace-backed rescoring is faster and cheaper
- it keeps leaderboard deltas attributable to scoring logic instead of runtime variance

### 11. `visual_tool_orchestration` is a first-class atomic track

Operational rule:

- use the dedicated visual tool family instead of baking visual selection logic into ad hoc KWA helpers
- keep visual categories in `benchmark_tags` for v1:
  - `visual_ui_ops`
  - `visual_document`
  - `visual_aerial`
  - `visual_map`

Reason:

- multimodal tool orchestration is a distinct benchmark surface
- it should be comparable to routing, retrieval, and full-stack execution rather than hidden inside one-off episodes

### 12. Seeded visual replay is canonical; local visual execution is supplementary

Operational rule:

- publish visual-tool claims from the replayable seeded lane
- use the local visual executor path as live stress, not as the only source of truth

Reason:

- seeded visual execution is deterministic and reproducible
- local visual execution is valuable realism pressure, but it is not stable enough to replace the canonical lane
