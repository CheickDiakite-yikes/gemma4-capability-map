# Session Handoff

## Resume Here

The repo is in a good state after a major realism and continuity pass.

What just landed:

- native `.xlsx`, `.pptx`, and `.docx` artifact generation for finance and job episodes
- browser state machines with validation failures, approval gates, and blocked submissions
- per-episode checkpointing for long `KnowledgeWorkArena` runs
- first finished non-oracle `KnowledgeWorkArena` baseline
- continuity files for future agent and human resume flow

## Current Canonical Pointers

- replayable core:
  - [`results/knowledge_work/replayable_core/summary.json`](../../results/knowledge_work/replayable_core/summary.json)
- live-web stress:
  - [`results/knowledge_work/live_web_stress/summary.json`](../../results/knowledge_work/live_web_stress/summary.json)
- knowledge-work history:
  - [`results/history/knowledge_work_history.md`](../../results/history/knowledge_work_history.md)

## Finished Model-Backed Evidence

Primary finished baseline:

- [`results/knowledge_work/model_backed_hf_exec_hold/summary.json`](../../results/knowledge_work/model_backed_hf_exec_hold/summary.json)

Exploratory stopped pilot:

- [`results/knowledge_work/model_backed_hf_reasoner_pilot/summary.json`](../../results/knowledge_work/model_backed_hf_reasoner_pilot/summary.json)

## What To Do First In The Next Thread

1. Read [`../../AGENT_CONTEXT.md`](../../AGENT_CONTEXT.md).
2. Check `git status --short`.
3. If new result dirs were added, run:
   - `uv run python scripts/build_knowledge_work_history.py`
4. If making new benchmark claims, run:
   - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q`

## Highest-Value Next Work

1. Run a fully specialist-backed model pilot for `KnowledgeWorkArena`.
2. Deepen native artifact grading beyond structural checks.
3. Add more recovery-heavy browser branches and approval-gated episodes.

## Important Operational Notes

- Use `--no-update-latest` for all ad hoc pilots.
- Do not let smoke runs overwrite `results/knowledge_work/replayable_core`.
- The `hf_service` reasoner is useful but cold startup is expensive on this Mac.
- The current branch is expected to remain `main` unless a feature branch is explicitly needed.
