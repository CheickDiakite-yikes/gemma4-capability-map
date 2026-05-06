from __future__ import annotations

import json
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
    if target in {"all", "scorecard"}:
        payload["scorecard"] = {
            "metrics": dict(runtime_trace.scorecard if runtime_trace else session.metrics),
            "controller_findings": _controller_findings(runtime_trace.episode_trace_path if runtime_trace else None),
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
    if session.sandbox_policy_blocks:
        table.add_row("policy blocks", str(len(session.sandbox_policy_blocks)))
    table.add_row("message", session.latest_message or "")
    if session.metrics:
        table.add_row("readiness", _format_metric(session.metrics.get("role_readiness_score")))
        table.add_row(
            "repair/raw",
            f"{_format_metric(session.metrics.get('controller_repair_count'))} / {_format_metric(session.metrics.get('raw_planning_clean_rate'))}",
        )
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
    if "scorecard" in payload:
        panels.append(_scorecard_panel(payload["scorecard"]))
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
    table.add_column("sev", width=7)
    table.add_column("gate", width=14)
    table.add_column("action", width=20)
    table.add_column("detail", overflow="fold")
    for block in blocks:
        details = [
            str(block.get("target", "")),
            str(block.get("sandbox_endpoint", "")),
            str(block.get("reason", "")),
        ]
        table.add_row(
            str(block.get("severity", "")),
            str(block.get("submission_gate", "")),
            str(block.get("action", "")),
            "\n".join(detail for detail in details if detail),
        )
    if not blocks:
        table.add_row("none", "", "", "No sandbox policy blocks recorded.")
    return Panel(table, title="Policy Blocks", border_style="yellow")


def _scorecard_panel(payload: dict[str, Any]) -> Panel:
    table = Table(expand=True)
    table.add_column("metric", width=28)
    table.add_column("value", width=12)
    table.add_column("detail", overflow="fold")
    metrics = payload.get("metrics", {})
    for metric in (
        "role_readiness_score",
        "strict_interface_score",
        "recovered_execution_score",
        "controller_repair_count",
        "argument_repair_count",
        "controller_fallback_count",
        "raw_planning_clean_rate",
    ):
        if metric in metrics:
            table.add_row(metric, _format_metric(metrics.get(metric)), "")
    findings = payload.get("controller_findings", [])
    if findings:
        table.add_section()
    for finding in findings:
        notes = ", ".join(str(note) for note in finding.get("repair_notes", [])) or "none"
        raw_outputs = finding.get("raw_outputs", [])
        detail = str(raw_outputs[0]) if raw_outputs else ""
        table.add_row(str(finding.get("task_id", "")), notes, detail)
    if not metrics and not findings:
        table.add_row("none", "", "No scorecard recorded yet.")
    return Panel(table, title="Scorecard And Controller Signal", border_style="cyan")


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _controller_findings(trace_path: str | None) -> list[dict[str, Any]]:
    if not trace_path:
        return []
    path = Path(trace_path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    findings: list[dict[str, Any]] = []
    for stage in data.get("stage_traces", []):
        for task in stage.get("task_traces", []):
            prompt_artifacts = task.get("prompt_artifacts", {})
            repair_notes = [
                str(note)
                for group in prompt_artifacts.get("planning_repair_notes", [])
                for note in (group if isinstance(group, list) else [group])
            ]
            metrics = task.get("metrics", {})
            controller_signal = any(
                _positive_metric(metrics.get(metric, 0.0))
                for metric in (
                    "controller_repair_count",
                    "argument_repair_count",
                    "controller_fallback_count",
                    "intent_override_count",
                )
            )
            if not repair_notes and not controller_signal:
                continue
            findings.append(
                {
                    "stage_id": task.get("stage_id") or stage.get("stage_id", ""),
                    "task_id": task.get("task_id", ""),
                    "repair_notes": repair_notes,
                    "raw_outputs": prompt_artifacts.get("planning_raw_outputs", []),
                    "metrics": {
                        key: metrics.get(key)
                        for key in (
                            "controller_repair_count",
                            "argument_repair_count",
                            "controller_fallback_count",
                            "intent_override_count",
                            "raw_planning_clean_rate",
                        )
                        if key in metrics
                    },
                }
            )
    return findings


def _positive_metric(value: Any) -> bool:
    try:
        return float(value or 0.0) > 0.0
    except (TypeError, ValueError):
        return False
