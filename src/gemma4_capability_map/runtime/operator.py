from __future__ import annotations

import time

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from gemma4_capability_map.runtime.core import LocalAgentRuntime
from gemma4_capability_map.runtime.schemas import AgentSession, RuntimeEvent, SessionStatus


SETTLED_STATUSES = {
    SessionStatus.AWAITING_APPROVAL,
    SessionStatus.COMPLETED,
    SessionStatus.DENIED,
    SessionStatus.FAILED,
    SessionStatus.INTERRUPTED,
}


def attach_to_session(
    runtime: LocalAgentRuntime,
    session_id: str,
    *,
    refresh_s: float = 0.5,
    timeout_s: float = 15.0,
    once: bool = False,
    console: Console | None = None,
) -> AgentSession:
    console = console or Console()
    events = runtime.get_events(session_id)
    session = runtime.get_session(session_id)
    if once:
        console.print(_render(session, events))
        return session

    after_sequence = max((event.sequence for event in events), default=0)
    with Live(_render(session, events), console=console, refresh_per_second=max(1, int(1 / max(refresh_s, 0.1)))) as live:
        while session.status not in SETTLED_STATUSES:
            payload = runtime.stream_session(session_id, after_sequence=after_sequence, timeout_s=min(timeout_s, refresh_s))
            session = payload["session"]
            new_events: list[RuntimeEvent] = payload["events"]
            if new_events:
                events.extend(new_events)
                after_sequence = max(event.sequence for event in new_events)
            live.update(_render(session, events))
            time.sleep(refresh_s)
        live.update(_render(session, events))
    return session


def apply_operator_action(
    runtime: LocalAgentRuntime,
    session_id: str,
    *,
    action: str,
    note: str = "",
    resume: bool = True,
    background: bool = True,
) -> AgentSession:
    if action == "approve":
        return runtime.resolve_approval(session_id, decision="approve", note=note, resume=resume)
    if action == "deny":
        return runtime.resolve_approval(session_id, decision="deny", note=note, resume=False)
    if action == "resume":
        return runtime.resume_session(session_id, note=note, background=background)
    if action == "retry":
        return runtime.retry_session(session_id, note=note, background=background)
    if action == "quit":
        return runtime.get_session(session_id)
    raise ValueError(f"Unsupported operator action `{action}`.")


def _render(session: AgentSession, events: list[RuntimeEvent]) -> Group:
    return Group(
        _status_panel(session),
        Columns([_events_panel(events), _side_panel(session)], equal=False, expand=True),
    )


def _status_panel(session: AgentSession) -> Panel:
    table = Table.grid(expand=True)
    table.add_column(ratio=1)
    table.add_column(ratio=1)
    table.add_column(ratio=1)
    table.add_row(
        f"[bold]{session.title}[/bold]",
        f"status: [bold]{session.status.value}[/bold]",
        f"profile: {session.system_id}",
    )
    table.add_row(
        f"workflow: {session.workflow_id}",
        f"lane: {session.lane}",
        f"attempt: {session.attempt}",
    )
    return Panel(table, title="Moonie Live Operator", border_style="cyan")


def _events_panel(events: list[RuntimeEvent]) -> Panel:
    table = Table(expand=True)
    table.add_column("#", justify="right", width=4)
    table.add_column("kind", width=20)
    table.add_column("message", overflow="fold")
    for event in events[-14:]:
        table.add_row(str(event.sequence), event.kind, event.message)
    return Panel(table, title="Event Timeline", border_style="blue")


def _side_panel(session: AgentSession) -> Panel:
    table = Table.grid(expand=True)
    table.add_column(ratio=1)
    table.add_column(ratio=2)
    table.add_row("session", session.session_id)
    table.add_row("sandbox", session.sandbox_root or "not prepared")
    table.add_row("policy", session.sandbox_policy_id or "unset")
    table.add_row("source", session.sandbox_source or "unset")
    table.add_row("artifact", session.latest_artifact_title or "none")
    table.add_row("approval", session.active_approval_id or "none")
    table.add_row("message", session.latest_message or "")
    if session.status == SessionStatus.AWAITING_APPROVAL:
        table.add_row("approve", f"moonie-agent attach {session.session_id} --action approve")
        table.add_row("deny", f"moonie-agent attach {session.session_id} --action deny")
    elif session.status == SessionStatus.INTERRUPTED:
        table.add_row("resume", f"moonie-agent attach {session.session_id} --action resume")
        table.add_row("retry", f"moonie-agent attach {session.session_id} --action retry")
    else:
        table.add_row("watch", f"moonie-agent attach {session.session_id}")
    return Panel(table, title="Run Context", border_style="green")
