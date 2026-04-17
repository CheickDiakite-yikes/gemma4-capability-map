from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from gemma4_capability_map.api.app import serve
from gemma4_capability_map.runtime.core import LocalAgentRuntime


def _json_request(base_url: str, path: str, method: str = "GET", payload: dict | None = None) -> dict:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(f"{base_url}{path}", data=data, method=method, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:  # pragma: no cover - assertion helper
        raise AssertionError(exc.read().decode("utf-8")) from exc


def _raw_request(base_url: str, path: str, method: str = "GET") -> tuple[int, dict[str, str], bytes]:
    request = Request(f"{base_url}{path}", method=method)
    with urlopen(request, timeout=30) as response:  # noqa: S310
        return response.status, dict(response.headers.items()), response.read()


def test_local_agent_api_serves_runtime_sessions(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    server = serve(host="127.0.0.1", port=0, runtime=runtime)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        health = _json_request(base_url, "/health")
        assert health == {"ok": True}

        profiles = _json_request(base_url, "/v1/profiles")
        assert any(profile["system_id"] == "oracle_gemma4_e2b" for profile in profiles["profiles"])

        workflows = _json_request(base_url, f"/v1/workflows?{urlencode({'lane': 'replayable_core'})}")
        assert any(workflow["workflow_id"] == "executive_visual_dashboard_review" for workflow in workflows["workflows"])

        session = _json_request(
            base_url,
            "/v1/sessions",
            method="POST",
            payload={
                "workflow_id": "executive_visual_dashboard_review",
                "system_id": "oracle_gemma4_e2b",
                "lane": "replayable_core",
                "project_id": "research-alpha",
                "background": False,
            },
        )
        assert session["status"] == "completed"
        assert session["project_id"] == "research-alpha"
        assert session["latest_instruction"]

        session_id = session["session_id"]
        detail = _json_request(base_url, f"/v1/sessions/{session_id}")
        assert detail["workflow_id"] == "executive_visual_dashboard_review"
        assert detail["latest_artifact_title"]
        assert detail["latest_instruction"]

        events = _json_request(base_url, f"/v1/sessions/{session_id}/events")
        event_kinds = [event["kind"] for event in events["events"]]
        assert event_kinds[0:3] == ["created", "instruction_updated", "warming"]
        assert "tool_call_attempt" in event_kinds
        assert "tool_call_result" in event_kinds
        assert "artifact_revision" in event_kinds
        assert event_kinds[-2:] == ["artifacts_ready", "completed"]

        artifacts = _json_request(base_url, f"/v1/sessions/{session_id}/artifacts")
        assert artifacts["runtime_trace"]["summary_path"]
        history = _json_request(base_url, f"/v1/sessions/{session_id}/history")
        assert history["session"]["project_id"] == "research-alpha"
        assert history["instruction_history"]
        assert history["artifact_history"]
        project_sessions = _json_request(base_url, f"/v1/sessions?{urlencode({'project': 'research-alpha'})}")
        assert any(item["session_id"] == session_id for item in project_sessions["sessions"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_local_agent_api_exposes_approval_resolution(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    server = serve(host="127.0.0.1", port=0, runtime=runtime)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        session = _json_request(
            base_url,
            "/v1/sessions",
            method="POST",
            payload={
                "workflow_id": "finance_visual_invoice_review",
                "system_id": "oracle_gemma4_e2b",
                "lane": "replayable_core",
                "background": False,
            },
        )
        assert session["status"] == "awaiting_approval"
        approval_id = session["approvals"][0]["approval_id"]

        resolved = _json_request(
            base_url,
            f"/v1/approvals/{approval_id}/resolve",
            method="POST",
            payload={"decision": "approve", "note": "Ship it.", "resume": True},
        )
        assert resolved["status"] == "completed"
        assert resolved["approvals"][0]["status"] == "approved"
        assert resolved["approvals"][0]["note"] == "Ship it."
        assert resolved["latest_instruction"] == "Ship it."
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_local_agent_api_supports_event_tail_and_retry(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    server = serve(host="127.0.0.1", port=0, runtime=runtime)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        session = _json_request(
            base_url,
            "/v1/sessions",
            method="POST",
            payload={
                "workflow_id": "executive_visual_dashboard_review",
                "system_id": "oracle_gemma4_e2b",
                "lane": "replayable_core",
                "background": False,
            },
        )
        tailed = _json_request(base_url, f"/v1/sessions/{session['session_id']}/events/tail?{urlencode({'after': 2, 'timeout_s': 0.1})}")
        assert "running" in [event["kind"] for event in tailed["events"]]

        streamed = _json_request(base_url, f"/v1/sessions/{session['session_id']}/stream?{urlencode({'after': 3, 'timeout_s': 0.1})}")
        assert streamed["session"]["session_id"] == session["session_id"]
        assert streamed["events"]
        assert streamed["pending_approval"] is None

        retried = _json_request(
            base_url,
            f"/v1/sessions/{session['session_id']}/retry",
            method="POST",
            payload={"note": "Retry through API.", "background": False},
        )
        assert retried["retry_of_session_id"] == session["session_id"]
        assert retried["attempt"] == session["attempt"] + 1
        assert retried["status"] == "completed"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_local_agent_api_filters_sessions_and_lists_approvals(tmp_path: Path) -> None:
    runtime = LocalAgentRuntime(results_root=tmp_path / "runtime")
    server = serve(host="127.0.0.1", port=0, runtime=runtime)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        _json_request(
            base_url,
            "/v1/sessions",
            method="POST",
            payload={
                "workflow_id": "executive_visual_dashboard_review",
                "system_id": "oracle_gemma4_e2b",
                "lane": "replayable_core",
                "project_id": "alpha",
                "background": False,
            },
        )
        approval_session = _json_request(
            base_url,
            "/v1/sessions",
            method="POST",
            payload={
                "workflow_id": "finance_visual_invoice_review",
                "system_id": "oracle_gemma4_e2b",
                "lane": "replayable_core",
                "project_id": "beta",
                "background": False,
            },
        )

        completed_sessions = _json_request(base_url, f"/v1/sessions?{urlencode({'status': 'completed'})}")
        alpha_sessions = _json_request(base_url, f"/v1/sessions?{urlencode({'project': 'alpha'})}")
        approvals = _json_request(base_url, "/v1/approvals")
        all_approvals = _json_request(base_url, f"/v1/approvals?{urlencode({'all': 'true'})}")
        approval_stream = _json_request(base_url, f"/v1/sessions/{approval_session['session_id']}/stream?{urlencode({'after': 3, 'timeout_s': 0.1})}")

        assert all(session["status"] == "completed" for session in completed_sessions["sessions"])
        assert all(session["project_id"] == "alpha" for session in alpha_sessions["sessions"])
        assert len(approvals["approvals"]) == 1
        assert approvals["approvals"][0]["session_id"] == approval_session["session_id"]
        assert len(all_approvals["approvals"]) == 1
        assert approval_stream["pending_approval"]["session_id"] == approval_session["session_id"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_local_agent_api_supports_frontend_preflight_and_file_preview(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    runtime = LocalAgentRuntime(results_root=runtime_root)
    preview_file = runtime_root / "preview.txt"
    preview_file.parent.mkdir(parents=True, exist_ok=True)
    preview_file.write_text("frontend preview", encoding="utf-8")

    server = serve(host="127.0.0.1", port=0, runtime=runtime)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status, headers, _ = _raw_request(base_url, "/v1/files", method="OPTIONS")
        assert status == 204
        assert headers["Access-Control-Allow-Origin"] == "*"
        assert "GET" in headers["Access-Control-Allow-Methods"]

        request = Request(f"{base_url}/v1/files?{urlencode({'path': str(preview_file)})}", method="GET")
        with urlopen(request, timeout=30) as response:  # noqa: S310
            body = response.read().decode("utf-8")
            assert response.headers["Access-Control-Allow-Origin"] == "*"
            assert body == "frontend preview"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
