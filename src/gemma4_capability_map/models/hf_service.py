from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = ROOT / "results" / "runtime" / "hf_reasoner"


def service_id_for(model_id: str, device: str) -> str:
    safe_model = model_id.replace("/", "__").replace("-", "_").replace(".", "_")
    safe_device = device.replace("/", "_").replace("-", "_")
    return f"{safe_model}_{safe_device}"


def service_paths_for(model_id: str, device: str) -> dict[str, str]:
    service_id = service_id_for(model_id, device)
    root = SERVICE_ROOT / service_id
    return {
        "service_id": service_id,
        "root": str(root),
        "socket_path": str(root / "service.sock"),
        "state_path": str(root / "state.json"),
        "event_log_path": str(root / "events.jsonl"),
        "request_log_path": str(root / "requests.jsonl"),
        "stdout_log_path": str(root / "service.log"),
    }


def read_service_state(state_path: str | Path) -> dict[str, Any] | None:
    path = Path(state_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def request_service(
    socket_path: str | Path,
    payload: dict[str, Any],
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout_seconds)
    try:
        sock.connect(str(socket_path))
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
        sock.sendall(body)
        response = _recv_json_line(sock)
    finally:
        sock.close()
    return response


def ensure_hf_reasoner_service(
    model_id: str,
    device: str,
    max_new_tokens: int,
    startup_timeout_seconds: float = 900.0,
) -> dict[str, Any]:
    paths = service_paths_for(model_id, device)
    ready = _service_ready(paths)
    if ready is not None:
        return ready

    existing_state = read_service_state(paths["state_path"]) or {}
    existing_status = str(existing_state.get("status", ""))
    existing_pid_running = _pid_is_running(existing_state.get("pid"))
    if existing_status == "ready" and existing_pid_running:
        waited = _wait_for_existing_ready(paths, timeout_seconds=min(60.0, startup_timeout_seconds))
        if waited is not None:
            return waited
        raise TimeoutError(
            f"Existing HF reasoner service for {model_id} on {device} is running but unreachable; refusing to launch a duplicate process."
        )
    waiting_for_existing = (
        existing_status in {"starting", "loading"}
        and existing_pid_running
    )

    root = Path(paths["root"])
    root.mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen[str] | None = None
    if not waiting_for_existing:
        _reset_service_artifacts(paths)
        stdout_handle = Path(paths["stdout_log_path"]).open("a", encoding="utf-8", buffering=1)
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "scripts" / "hf_reasoner_service.py"),
                "serve",
                "--model",
                model_id,
                "--device",
                device,
                "--max-new-tokens",
                str(max_new_tokens),
                "--socket-path",
                paths["socket_path"],
                "--state-path",
                paths["state_path"],
                "--event-log-path",
                paths["event_log_path"],
                "--request-log-path",
                paths["request_log_path"],
            ],
            cwd=ROOT,
            stdout=stdout_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    deadline = time.time() + startup_timeout_seconds
    last_state: dict[str, Any] | None = existing_state if waiting_for_existing else None
    while time.time() < deadline:
        last_state = read_service_state(paths["state_path"])
        if last_state is not None:
            status = str(last_state.get("status"))
            if status == "ready":
                enriched = dict(last_state)
                enriched["paths"] = paths
                return enriched
            if status == "failed":
                raise RuntimeError(last_state.get("error", "HF reasoner service failed to start."))
        if process is not None and process.poll() is not None:
            state = last_state or {}
            raise RuntimeError(
                state.get("error") or f"HF reasoner service exited early with code {process.returncode}."
            )
        if process is None and last_state is not None:
            pid = last_state.get("pid")
            status = str(last_state.get("status", ""))
            if status not in {"ready", "failed"} and not _pid_is_running(pid):
                raise RuntimeError(f"HF reasoner service pid {pid} exited before becoming ready.")
        time.sleep(1.0)

    raise TimeoutError(f"Timed out while starting HF reasoner service for {model_id} on {device}.")


def stop_hf_reasoner_service(model_id: str, device: str, timeout_seconds: float = 15.0) -> dict[str, Any]:
    paths = service_paths_for(model_id, device)
    state = read_service_state(paths["state_path"]) or {}
    response: dict[str, Any]
    try:
        response = request_service(paths["socket_path"], {"type": "shutdown"}, timeout_seconds=timeout_seconds)
    except OSError:
        response = {"ok": False, "error": "service_unreachable"}
    response["state"] = state
    response["paths"] = paths
    return response


def _service_ready(paths: dict[str, str]) -> dict[str, Any] | None:
    state = read_service_state(paths["state_path"])
    if not state or state.get("status") != "ready":
        return None
    try:
        response = request_service(paths["socket_path"], {"type": "status"}, timeout_seconds=5.0)
    except OSError:
        return None
    if not response.get("ok"):
        return None
    enriched = dict(response)
    enriched["paths"] = paths
    return enriched


def _wait_for_existing_ready(paths: dict[str, str], timeout_seconds: float) -> dict[str, Any] | None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        state = read_service_state(paths["state_path"]) or {}
        pid = state.get("pid")
        if state.get("status") == "failed":
            return None
        if pid and not _pid_is_running(pid):
            return None
        ready = _service_ready(paths)
        if ready is not None:
            return ready
        time.sleep(1.0)
    return None


def _reset_service_artifacts(paths: dict[str, str]) -> None:
    for key in ("socket_path", "state_path", "event_log_path", "request_log_path"):
        artifact = Path(paths[key])
        if artifact.exists():
            artifact.unlink()


def _pid_is_running(pid: Any) -> bool:
    try:
        numeric_pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        os.kill(numeric_pid, 0)
    except OSError:
        return False
    return True


def serve_hf_reasoner_service(
    model_id: str,
    device: str,
    max_new_tokens: int,
    socket_path: str,
    state_path: str,
    event_log_path: str,
    request_log_path: str,
) -> None:
    from gemma4_capability_map.models.gemma4_runner import Gemma4Runner
    from gemma4_capability_map.schemas import Message, ToolSpec

    state_file = Path(state_path)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    event_log_file = Path(event_log_path)
    event_log_file.parent.mkdir(parents=True, exist_ok=True)
    request_log_file = Path(request_log_path)
    request_log_file.parent.mkdir(parents=True, exist_ok=True)
    socket_file = Path(socket_path)
    if socket_file.exists():
        socket_file.unlink()

    service_id = service_id_for(model_id, device)
    state: dict[str, Any] = {
        "service_id": service_id,
        "model_id": model_id,
        "device": device,
        "max_new_tokens": max_new_tokens,
        "pid": os.getpid(),
        "status": "starting",
        "phase": "starting",
        "created_at": _utc_now(),
        "requests_completed": 0,
        "requests_failed": 0,
        "recent_events": [],
    }
    _write_state(state_file, state)
    _record_event(
        state=state,
        state_file=state_file,
        event_log_file=event_log_file,
        event="service_boot",
        detail="HF reasoner service process started.",
        socket_path=socket_path,
    )
    _record_event(
        state=state,
        state_file=state_file,
        event_log_file=event_log_file,
        event="runner_init",
        detail="Constructing Gemma4Runner for HF service.",
        backend="hf",
    )

    runner = Gemma4Runner(
        model_id,
        backend="hf",
        max_new_tokens=max_new_tokens,
        device=device,
        load_event_hook=lambda payload: _record_event(
            state=state,
            state_file=state_file,
            event_log_file=event_log_file,
            event=str(payload.get("event", "load_event")),
            detail=str(payload.get("detail", "")),
            **{key: value for key, value in payload.items() if key not in {"event", "detail"}},
        ),
    )
    load_started = time.perf_counter()
    try:
        state["status"] = "loading"
        state["phase"] = "loading"
        _write_state(state_file, state)
        runtime_info = runner.ensure_loaded()
    except Exception as exc:
        state["status"] = "failed"
        state["phase"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["failed_at"] = _utc_now()
        _write_state(state_file, state)
        _record_event(
            state=state,
            state_file=state_file,
            event_log_file=event_log_file,
            event="service_failed",
            detail="HF reasoner service failed during startup.",
            error=state["error"],
        )
        raise
    state["status"] = "ready"
    state["phase"] = "ready"
    state["load_elapsed_ms"] = int((time.perf_counter() - load_started) * 1000)
    state["ready_at"] = _utc_now()
    state["runtime_info"] = runtime_info
    _write_state(state_file, state)
    _record_event(
        state=state,
        state_file=state_file,
        event_log_file=event_log_file,
        event="service_ready",
        detail="HF reasoner service is ready to accept requests.",
        load_elapsed_ms=state["load_elapsed_ms"],
    )

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_file))
    server.listen()
    try:
        while True:
            connection, _ = server.accept()
            with connection:
                try:
                    request = _recv_json_line(connection)
                    response = _handle_request(
                        request=request,
                        runner=runner,
                        state=state,
                        state_file=state_file,
                        event_log_file=event_log_file,
                        request_log_file=request_log_file,
                    )
                except Exception as exc:
                    state["requests_failed"] = int(state.get("requests_failed", 0)) + 1
                    state["last_error"] = f"{type(exc).__name__}: {exc}"
                    state["last_request_at"] = _utc_now()
                    state["active_request"] = None
                    state["phase"] = "request_failed"
                    _write_state(state_file, state)
                    response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                connection.sendall(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")
                if response.get("shutdown"):
                    break
    finally:
        server.close()
        if socket_file.exists():
            socket_file.unlink()


def _handle_request(
    request: dict[str, Any],
    runner: Any,
    state: dict[str, Any],
    state_file: Path,
    event_log_file: Path,
    request_log_file: Path,
) -> dict[str, Any]:
    request_type = request.get("type")
    if request_type == "status":
        return {"ok": True, **state}
    if request_type == "shutdown":
        state["status"] = "stopping"
        state["phase"] = "stopping"
        state["stopped_at"] = _utc_now()
        _write_state(state_file, state)
        return {"ok": True, "shutdown": True}
    if request_type != "generate":
        raise ValueError(f"Unsupported request type: {request_type}")

    from gemma4_capability_map.schemas import Message, ToolSpec

    started = time.perf_counter()
    request_id = f"{state['service_id']}_{int(time.time() * 1000)}"
    messages = [Message.model_validate(item) for item in request.get("messages", [])]
    tool_specs = [ToolSpec.model_validate(item) for item in request.get("tool_specs", [])]
    state["last_request_type"] = request_type
    state["last_request_started_at"] = _utc_now()
    state["active_request"] = {
        "request_id": request_id,
        "request_type": request_type,
        "message_count": len(messages),
        "media_count": len(request.get("media", [])),
        "tool_spec_count": len(tool_specs),
        "thinking": bool(request.get("thinking", False)),
        "max_new_tokens": request.get("max_new_tokens"),
    }
    state["phase"] = "generating"
    _write_state(state_file, state)
    _record_event(
        state=state,
        state_file=state_file,
        event_log_file=event_log_file,
        event="request_started",
        detail="HF reasoner service started a generation request.",
        request_id=request_id,
        message_count=len(messages),
        media_count=len(request.get("media", [])),
        tool_spec_count=len(tool_specs),
        thinking=bool(request.get("thinking", False)),
        max_new_tokens=request.get("max_new_tokens"),
    )
    turn = runner.generate(
        messages=messages,
        media=list(request.get("media", [])),
        tool_specs=tool_specs,
        thinking=bool(request.get("thinking", False)),
        max_new_tokens=request.get("max_new_tokens"),
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    state["requests_completed"] = int(state.get("requests_completed", 0)) + 1
    state["last_request_at"] = _utc_now()
    state["last_request_id"] = request_id
    state["last_request_elapsed_ms"] = elapsed_ms
    state["active_request"] = None
    state["phase"] = "request_complete"
    _write_state(state_file, state)
    _record_event(
        state=state,
        state_file=state_file,
        event_log_file=event_log_file,
        event="request_completed",
        detail="HF reasoner service completed a generation request.",
        request_id=request_id,
        elapsed_ms=elapsed_ms,
    )
    _append_request_log(
        request_log_file,
        {
            "request_id": request_id,
            "request_type": request_type,
            "created_at": _utc_now(),
            "elapsed_ms": elapsed_ms,
            "message_count": len(messages),
            "media_count": len(request.get("media", [])),
            "tool_spec_count": len(tool_specs),
            "thinking": bool(request.get("thinking", False)),
            "max_new_tokens": request.get("max_new_tokens"),
            "turn": turn.model_dump(mode="json"),
        },
    )
    return {
        "ok": True,
        "request_id": request_id,
        "service_id": state["service_id"],
        "elapsed_ms": elapsed_ms,
        "turn": turn.model_dump(mode="json"),
        "runtime_info": runner.runtime_info(),
    }


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _record_event(
    state: dict[str, Any],
    state_file: Path,
    event_log_file: Path,
    event: str,
    detail: str = "",
    **extra: Any,
) -> None:
    created_at = _utc_now()
    entry: dict[str, Any] = {"created_at": created_at, "event": event}
    if detail:
        entry["detail"] = detail
    entry.update(extra)
    recent_events = list(state.get("recent_events", []))
    recent_events.append(entry)
    state["recent_events"] = recent_events[-25:]
    state["last_event_at"] = created_at
    state["phase"] = event
    state["heartbeat_at"] = created_at
    _write_state(state_file, state)
    _append_request_log(event_log_file, entry)
    print(json.dumps(entry, ensure_ascii=False), flush=True)


def _append_request_log(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _recv_json_line(sock: socket.socket) -> dict[str, Any]:
    buffer = b""
    while not buffer.endswith(b"\n"):
        chunk = sock.recv(65536)
        if not chunk:
            break
        buffer += chunk
    if not buffer:
        raise RuntimeError("Received empty response from HF reasoner service.")
    return json.loads(buffer.decode("utf-8").strip())


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
