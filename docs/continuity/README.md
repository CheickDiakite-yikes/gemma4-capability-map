# Continuity System

This folder is the repo’s long-horizon memory system for research, engineering, and cross-thread continuation.

## Files

- [`current-state.md`](current-state.md)
  - authoritative snapshot of what the system currently is
- [`key-learnings.md`](key-learnings.md)
  - stable takeaways that should influence future design and evaluation
- [`decision-log.md`](decision-log.md)
  - durable architecture and benchmark decisions
- [`next-steps.md`](next-steps.md)
  - prioritized work queue
- [`session-handoff.md`](session-handoff.md)
  - short, practical resume packet for the next thread

## Division of Labor

- use [`../research-log.md`](../research-log.md) for chronological research notes
- use [`current-state.md`](current-state.md) for the distilled answer to “where are we now?”
- use [`key-learnings.md`](key-learnings.md) for the durable answer to “what have we learned?”
- use [`decision-log.md`](decision-log.md) for “what did we lock in and why?”
- use [`next-steps.md`](next-steps.md) for “what should we do next?”
- use [`session-handoff.md`](session-handoff.md) for “how do we resume if this thread dies right now?”

## Operating Rules

- `session-handoff.md` is overwrite-friendly and should reflect the latest practical resume point.
- `current-state.md` should change only when benchmark posture or runtime posture meaningfully changes.
- `key-learnings.md` should be curated, not noisy.
- `decision-log.md` is append-only unless an old decision is explicitly superseded.
- smoke runs and exploratory pilots should not overwrite canonical lane pointers.
