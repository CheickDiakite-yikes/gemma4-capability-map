# Next Steps

## Immediate

### 1. Push the board/reporting layer toward public-style benchmark cuts

Goal:

- turn the new registry-backed board into a real benchmark surface with per-role, per-category, latency, and cost cuts

Why:

- we can already generate internal leaderboard and scatter outputs
- the next gap is presentation quality and richer metadata, not basic exportability

### 2. Widen the fully specialist-backed mixed-pressure matrix again

Focus:

- move beyond the current `12` replayable + `12` live mixed-pressure slices
- keep `hf_service` + real HF `FunctionGemma` + real HF `EmbeddingGemma`
- add more revision-heavy, mixed-evidence, and approval-gated episodes per role

Why:

- the bounded mixed-pressure references are now clean in both lanes
- the next question is how quickly performance degrades as pressure composition keeps widening

### 3. Investigate softer realism signals instead of only binary failures

Focus:

- inspect episodes that are clean on `strict_interface` and `recovered_execution` but weaker on:
  - `revision_responsiveness`
  - `memory_retention_score`
  - `role_readiness_score`
- current concrete target:
  - `kwa_finance_partner_deck_revision`

Why:

- the benchmark is now good enough to surface softer job-readiness weaknesses that binary pass/fail views would miss

## Near-Term

### 4. Add even tougher conflict and revision loops after the next mixed-pressure widening

Focus:

- include more mixed evidence, revision, and browser-heavy episodes per role
- bias new additions toward stages where the agent must recover, then still choose whether to continue, revise, or stop
- add more explicit contradiction handling between stale context, new approvals, and external stakeholder pressure
- keep extending native artifact grading instead of relaxing it

## Ongoing Discipline

- keep canonical lane pointers clean
- checkpoint long model-backed runs
- rescore saved KWA traces after scoring-logic changes:
  - `uv run python scripts/rescore_knowledge_work_runs.py ...`
- append new findings to the research log
- refresh continuity files after every major benchmark pass
