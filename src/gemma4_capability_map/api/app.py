from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from gemma4_capability_map.runtime.core import LocalAgentRuntime


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
            self._send_json({"sessions": [session.model_dump(mode="json") for session in self.runtime.list_sessions()]})
            return
        if path.startswith("/v1/sessions/"):
            parts = path.split("/")
            session_id = parts[3] if len(parts) > 3 else ""
            if len(parts) == 4:
                self._send_json(self.runtime.get_session(session_id).model_dump(mode="json"))
                return
            if len(parts) == 5 and parts[4] == "events":
                after = int(query.get("after", ["0"])[0])
                self._send_json({"events": [event.model_dump(mode="json") for event in self.runtime.get_events(session_id, after_sequence=after)]})
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
            try:
                session = self.runtime.resolve_approval(session_id, decision=decision, note=note)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(session.model_dump(mode="json"))
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
