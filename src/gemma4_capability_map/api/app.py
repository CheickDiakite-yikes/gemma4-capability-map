from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from gemma4_capability_map.runtime.core import LocalAgentRuntime
from gemma4_capability_map.runtime.schemas import ApprovalStatus


class LocalAgentAPIHandler(BaseHTTPRequestHandler):
    runtime = LocalAgentRuntime()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        query = parse_qs(parsed.query)
        if path == "/health":
            self._send_json({"ok": True})
            return
        if path == "/v1/profiles":
            self._send_json({"profiles": [profile.model_dump(mode="json") for profile in self.runtime.list_system_profiles()]})
            return
        if path == "/v1/workflows":
            lane = query.get("lane", [None])[0]
            self._send_json({"workflows": self.runtime.list_workflows(lane=lane)})
            return
        if path == "/v1/sessions":
            status = query.get("status", [None])[0]
            project = query.get("project", [None])[0]
            self._send_json({"sessions": [session.model_dump(mode="json") for session in self.runtime.list_sessions(status=status, project_id=project)]})
            return
        if path == "/v1/approvals":
            include_all = str(query.get("all", ["false"])[0]).lower() == "true"
            approvals = self.runtime.list_approvals(status=None if include_all else ApprovalStatus.PENDING)
            self._send_json({"approvals": [approval.model_dump(mode="json") for approval in approvals]})
            return
        if path.startswith("/v1/sessions/"):
            parts = path.split("/")
            session_id = parts[3] if len(parts) > 3 else ""
            if len(parts) == 4:
                self._send_json(self.runtime.get_session(session_id).model_dump(mode="json"))
                return
            if len(parts) == 5 and parts[4] == "history":
                history = self.runtime.get_session_history(session_id)
                self._send_json(
                    {
                        "session": history["session"].model_dump(mode="json"),
                        "instruction_history": [record.model_dump(mode="json") for record in history["instruction_history"]],
                        "artifact_history": [record.model_dump(mode="json") for record in history["artifact_history"]],
                        "events": [event.model_dump(mode="json") for event in history["events"]],
                        "runtime_trace": history["runtime_trace"].model_dump(mode="json") if history["runtime_trace"] else None,
                    }
                )
                return
            if len(parts) == 5 and parts[4] == "events":
                after = int(query.get("after", ["0"])[0])
                limit_raw = query.get("limit", [None])[0]
                limit = int(limit_raw) if limit_raw not in {None, ""} else None
                self._send_json({"events": [event.model_dump(mode="json") for event in self.runtime.get_events(session_id, after_sequence=after, limit=limit)]})
                return
            if len(parts) == 6 and parts[4] == "events" and parts[5] == "tail":
                after = int(query.get("after", ["0"])[0])
                timeout_s = float(query.get("timeout_s", ["15.0"])[0])
                events = self.runtime.wait_for_events(session_id, after_sequence=after, timeout_s=timeout_s)
                self._send_json({"events": [event.model_dump(mode="json") for event in events]})
                return
            if len(parts) == 5 and parts[4] == "stream":
                after = int(query.get("after", ["0"])[0])
                timeout_s = float(query.get("timeout_s", ["15.0"])[0])
                payload = self.runtime.stream_session(session_id, after_sequence=after, timeout_s=timeout_s)
                self._send_json(
                    {
                        "session": payload["session"].model_dump(mode="json"),
                        "events": [event.model_dump(mode="json") for event in payload["events"]],
                        "pending_approval": payload["pending_approval"].model_dump(mode="json") if payload["pending_approval"] else None,
                    }
                )
                return
            if len(parts) == 5 and parts[4] == "artifacts":
                session = self.runtime.get_session(session_id)
                self._send_json({"artifact_paths": session.artifact_paths, "runtime_trace": session.runtime_trace.model_dump(mode="json") if session.runtime_trace else None})
                return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        body = self._json_body()
        if path == "/v1/sessions":
            try:
                session = self.runtime.launch_session(
                    workflow_id=str(body.get("workflow_id", "")),
                    system_id=body.get("system_id"),
                    lane=body.get("lane"),
                    title=body.get("title"),
                    human_request=str(body.get("human_request", "")),
                    project_id=body.get("project_id"),
                    background=bool(body.get("background", True)),
                )
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(session.model_dump(mode="json"), status=201)
            return
        if path.startswith("/v1/sessions/") and path.endswith("/approval"):
            parts = path.split("/")
            session_id = parts[3] if len(parts) > 3 else ""
            decision = str(body.get("decision", "approve"))
            note = str(body.get("note", ""))
            resume = bool(body.get("resume", True))
            try:
                session = self.runtime.resolve_approval(session_id, decision=decision, note=note, resume=resume)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(session.model_dump(mode="json"))
            return
        if path.startswith("/v1/approvals/") and path.endswith("/resolve"):
            parts = path.split("/")
            approval_id = parts[3] if len(parts) > 3 else ""
            decision = str(body.get("decision", "approve"))
            note = str(body.get("note", ""))
            resume = bool(body.get("resume", True))
            if not approval_id:
                self._send_json({"error": "Not found"}, status=404)
                return
            try:
                session = self.runtime.resolve_approval_by_id(approval_id, decision=decision, note=note, resume=resume)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(session.model_dump(mode="json"))
            return
        if path.startswith("/v1/sessions/") and path.endswith("/resume"):
            parts = path.split("/")
            session_id = parts[3] if len(parts) > 3 else ""
            note = str(body.get("note", ""))
            background = bool(body.get("background", True))
            try:
                session = self.runtime.resume_session(session_id, note=note, background=background)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(session.model_dump(mode="json"))
            return
        if path.startswith("/v1/sessions/") and path.endswith("/retry"):
            parts = path.split("/")
            session_id = parts[3] if len(parts) > 3 else ""
            note = str(body.get("note", ""))
            background = bool(body.get("background", True))
            try:
                session = self.runtime.retry_session(session_id, note=note, background=background)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(session.model_dump(mode="json"), status=201)
            return
        self._send_json({"error": "Not found"}, status=404)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw.strip() else {}

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(host: str = "127.0.0.1", port: int = 8765, runtime: LocalAgentRuntime | None = None) -> ThreadingHTTPServer:
    handler = type("ConfiguredLocalAgentAPIHandler", (LocalAgentAPIHandler,), {})
    handler.runtime = runtime or LocalAgentRuntime()
    server = ThreadingHTTPServer((host, port), handler)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Moonie local agent API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = serve(host=args.host, port=args.port)
    try:
        print(json.dumps({"host": args.host, "port": args.port, "status": "serving"}))
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
