# Agent Context

This file is the repo entrypoint for future agent and human continuation work.

## Read Order

1. [`docs/continuity/session-handoff.md`](docs/continuity/session-handoff.md)
2. [`docs/continuity/current-state.md`](docs/continuity/current-state.md)
3. [`docs/continuity/next-steps.md`](docs/continuity/next-steps.md)
4. [`docs/continuity/key-learnings.md`](docs/continuity/key-learnings.md)
5. [`docs/continuity/decision-log.md`](docs/continuity/decision-log.md)
6. [`docs/research-log.md`](docs/research-log.md)
7. [`results/history/knowledge_work_history.md`](results/history/knowledge_work_history.md)
8. [`results/history/history_report.md`](results/history/history_report.md)

## Purpose

Use this system to:

- preserve current technical state
- keep durable learnings separate from transient notes
- make interrupted work resumable across threads
- prevent canonical benchmark pointers from being overwritten by ad hoc smoke or pilot runs

## Update Contract

After each meaningful milestone:

- append new findings to [`docs/research-log.md`](docs/research-log.md)
- update [`docs/continuity/current-state.md`](docs/continuity/current-state.md) if benchmark posture changed
- update [`docs/continuity/next-steps.md`](docs/continuity/next-steps.md) to reflect the real priority stack
- overwrite [`docs/continuity/session-handoff.md`](docs/continuity/session-handoff.md) with the latest resumable state
- append or extend [`docs/continuity/decision-log.md`](docs/continuity/decision-log.md) only when a durable architectural/product decision changes

## Resume Checklist

When picking up in a new thread:

1. Read the files above in order.
2. Check `git status --short`.
3. Confirm the canonical benchmark pointers still match:
   - [`results/knowledge_work/replayable_core/summary.json`](results/knowledge_work/replayable_core/summary.json)
   - [`results/knowledge_work/live_web_stress/summary.json`](results/knowledge_work/live_web_stress/summary.json)
4. Rebuild history if new run outputs were added:
   - `uv run python scripts/build_knowledge_work_history.py`
5. Run verification before claiming a new canonical result:
   - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q`

## Special Rule For Pilots

Model-backed pilots and smoke runs must use:

- `--output-dir ...`
- `--no-update-latest`

This preserves the canonical `replayable_core` and `live_web_stress` lane pointers for published results.
