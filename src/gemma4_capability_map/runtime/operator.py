from __future__ import annotations

import time
from pathlib import Path
from typing import Any

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


def session_inspection_payload(runtime: LocalAgentRuntime, session_id: str, *, target: str = "all") -> dict[str, Any]:
    session = runtime.get_session(session_id)
    runtime_trace = session.runtime_trace
    artifacts = [
        {
            "path": path,
            "exists": Path(path).exists(),
            "name": Path(path).name,
        }
        for path in session.artifact_paths
    ]
    payload: dict[str, Any] = {
        "session_id": session.session_id,
        "status": session.status.value,
        "workflow_id": session.workflow_id,
        "system_id": session.system_id,
        "lane": session.lane,
    }
    if target in {"all", "sandbox"}:
        payload["sandbox"] = {
            "mode": session.sandbox_mode,
            "root": session.sandbox_root,
            "source": session.sandbox_source,
            "policy_id": session.sandbox_policy_id,
            "manifest_path": session.sandbox_manifest_path,
            "manifest_exists": Path(session.sandbox_manifest_path).exists() if session.sandbox_manifest_path else False,
            "output_dir": runtime_trace.output_dir if runtime_trace else "",
        }
    if target in {"all", "artifacts"}:
        payload["artifacts"] = artifacts
    if target in {"all", "policy"}:
        payload["policy_blocks"] = list(session.sandbox_policy_blocks)
    if target in {"all", "summary"}:
        payload["summary"] = {
            "summary_path": runtime_trace.summary_path if runtime_trace else "",
            "episode_trace_path": runtime_trace.episode_trace_path if runtime_trace else "",
            "manifest_path": runtime_trace.manifest_path if runtime_trace else "",
        }
    return payload


def print_session_inspection(runtime: LocalAgentRuntime, session_id: str, *, target: str = "all", console: Console | None = None) -> None:
    console = console or Console()
    payload = session_inspection_payload(runtime, session_id, target=target)
    console.print(_inspection_renderable(payload, target=target))


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
    table.add_row("inspect", f"moonie-agent inspect {session.session_id}")
    return Panel(table, title="Run Context", border_style="green")


def _inspection_renderable(payload: dict[str, Any], *, target: str) -> Group:
    panels = [_inspection_header(payload, target=target)]
    if "sandbox" in payload:
        panels.append(_key_value_panel("Sandbox", payload["sandbox"], border_style="green"))
    if "policy_blocks" in payload:
        panels.append(_policy_panel(payload["policy_blocks"]))
    if "artifacts" in payload:
        panels.append(_artifacts_panel(payload["artifacts"]))
    if "summary" in payload:
        panels.append(_key_value_panel("Summary", payload["summary"], border_style="blue"))
    return Group(*panels)


def _inspection_header(payload: dict[str, Any], *, target: str) -> Panel:
    table = Table.grid(expand=True)
    table.add_column(ratio=1)
    table.add_column(ratio=1)
    table.add_column(ratio=1)
    table.add_row(f"session: {payload['session_id']}", f"status: {payload['status']}", f"target: {target}")
    table.add_row(f"workflow: {payload['workflow_id']}", f"profile: {payload['system_id']}", f"lane: {payload['lane']}")
    return Panel(table, title="Moonie Inspect", border_style="cyan")


def _key_value_panel(title: str, values: dict[str, Any], *, border_style: str) -> Panel:
    table = Table.grid(expand=True)
    table.add_column(ratio=1)
    table.add_column(ratio=3)
    for key, value in values.items():
        table.add_row(str(key), str(value))
    return Panel(table, title=title, border_style=border_style)


def _artifacts_panel(artifacts: list[dict[str, Any]]) -> Panel:
    table = Table(expand=True)
    table.add_column("exists", width=8)
    table.add_column("name", width=28)
    table.add_column("path", overflow="fold")
    for artifact in artifacts:
        table.add_row("yes" if artifact["exists"] else "no", artifact["name"], artifact["path"])
    return Panel(table, title="Artifacts", border_style="magenta")


def _policy_panel(blocks: list[dict[str, Any]]) -> Panel:
    table = Table(expand=True)
    table.add_column("severity", width=10)
    table.add_column("action", width=20)
    table.add_column("reason", overflow="fold")
    for block in blocks:
        table.add_row(str(block.get("severity", "")), str(block.get("action", "")), str(block.get("reason", "")))
    if not blocks:
        table.add_row("none", "", "No sandbox policy blocks recorded.")
    return Panel(table, title="Policy Blocks", border_style="yellow")
