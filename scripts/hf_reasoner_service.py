from __future__ import annotations

import argparse
import json

from gemma4_capability_map.models.hf_service import (
    ensure_hf_reasoner_service,
    read_service_state,
    request_service,
    serve_hf_reasoner_service,
    service_paths_for,
    stop_hf_reasoner_service,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--model", required=True)
    serve_parser.add_argument("--device", default="auto")
    serve_parser.add_argument("--max-new-tokens", type=int, default=256)
    serve_parser.add_argument("--socket-path", required=True)
    serve_parser.add_argument("--state-path", required=True)
    serve_parser.add_argument("--event-log-path", required=True)
    serve_parser.add_argument("--request-log-path", required=True)

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--model", required=True)
    start_parser.add_argument("--device", default="auto")
    start_parser.add_argument("--max-new-tokens", type=int, default=256)
    start_parser.add_argument("--startup-timeout-seconds", type=float, default=900.0)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--model", required=True)
    status_parser.add_argument("--device", default="auto")

    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--model", required=True)
    stop_parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    if args.command == "serve":
        serve_hf_reasoner_service(
            model_id=args.model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            socket_path=args.socket_path,
            state_path=args.state_path,
            event_log_path=args.event_log_path,
            request_log_path=args.request_log_path,
        )
        return

    if args.command == "start":
        payload = ensure_hf_reasoner_service(
            model_id=args.model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            startup_timeout_seconds=args.startup_timeout_seconds,
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "status":
        paths = service_paths_for(args.model, args.device)
        state = read_service_state(paths["state_path"]) or {"status": "missing"}
        try:
            ping = request_service(paths["socket_path"], {"type": "status"}, timeout_seconds=5.0)
        except OSError:
            ping = {"ok": False, "error": "service_unreachable"}
        print(json.dumps({"paths": paths, "state": state, "ping": ping}, indent=2))
        return

    if args.command == "stop":
        payload = stop_hf_reasoner_service(args.model, args.device)
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
