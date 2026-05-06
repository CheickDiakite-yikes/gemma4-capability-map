from __future__ import annotations

import argparse
import json

from gemma4_capability_map.runtime.core import LocalAgentRuntime
from gemma4_capability_map.runtime.operator import apply_operator_action, attach_to_session, print_session_inspection, session_inspection_payload
from gemma4_capability_map.runtime.sandbox import DEFAULT_SANDBOX_POLICY_ID
from gemma4_capability_map.runtime.schemas import ApprovalStatus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Moonie local-agent workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("profiles", help="List available system profiles.")
    workflows_parser = subparsers.add_parser("workflows", help="List packaged workflows.")
    workflows_parser.add_argument("--lane", default=None)

    sessions_parser = subparsers.add_parser("sessions", help="List saved sessions.")
    sessions_parser.add_argument("--status", default=None)

    approvals_parser = subparsers.add_parser("approvals", help="List pending approvals.")
    approvals_parser.add_argument("--all", action="store_true")

    run_parser = subparsers.add_parser("run", help="Launch a packaged workflow.")
    run_parser.add_argument("--workflow-id", required=True)
    run_parser.add_argument("--system-id", default=None)
    run_parser.add_argument("--lane", default=None)
    run_parser.add_argument("--title", default=None)
    run_parser.add_argument("--human-request", default="")
    run_parser.add_argument("--background", action="store_true")
    run_parser.add_argument("--timeout-s", type=float, default=30.0)
    run_parser.add_argument("--sandbox-mode", choices=["ephemeral_copy", "disabled"], default="ephemeral_copy")
    run_parser.add_argument("--sandbox-policy-id", default=DEFAULT_SANDBOX_POLICY_ID)

    live_parser = subparsers.add_parser("live", help="Launch a packaged workflow and attach a Rich live operator view.")
    live_parser.add_argument("--workflow-id", required=True)
    live_parser.add_argument("--system-id", default="mlx_gemma4_e2b_reasoner_only")
    live_parser.add_argument("--lane", default="live_web_stress")
    live_parser.add_argument("--title", default=None)
    live_parser.add_argument("--human-request", default="")
    live_parser.add_argument("--project-id", default=None)
    live_parser.add_argument("--refresh-s", type=float, default=0.5)
    live_parser.add_argument("--timeout-s", type=float, default=15.0)
    live_parser.add_argument("--once", action="store_true")
    live_parser.add_argument("--sandbox-mode", choices=["ephemeral_copy", "disabled"], default="ephemeral_copy")
    live_parser.add_argument("--sandbox-policy-id", default=DEFAULT_SANDBOX_POLICY_ID)

    attach_parser = subparsers.add_parser("attach", help="Attach a Rich live operator view to an existing session.")
    attach_parser.add_argument("session_id")
    attach_parser.add_argument("--refresh-s", type=float, default=0.5)
    attach_parser.add_argument("--timeout-s", type=float, default=15.0)
    attach_parser.add_argument("--once", action="store_true")
    attach_parser.add_argument("--action", choices=["approve", "deny", "resume", "retry", "quit"], default=None)
    attach_parser.add_argument("--note", default="")
    attach_parser.add_argument("--no-resume", action="store_true")
    attach_parser.add_argument("--foreground", action="store_true")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect sandbox, artifacts, policy blocks, and trace paths for a session.")
    inspect_parser.add_argument("session_id")
    inspect_parser.add_argument("--target", choices=["all", "sandbox", "artifacts", "policy", "summary"], default="all")
    inspect_parser.add_argument("--json", action="store_true")

    show_parser = subparsers.add_parser("show", help="Show a session.")
    show_parser.add_argument("session_id")

    events_parser = subparsers.add_parser("events", help="Show session events.")
    events_parser.add_argument("session_id")
    events_parser.add_argument("--after", type=int, default=0)
    events_parser.add_argument("--follow", action="store_true")
    events_parser.add_argument("--timeout-s", type=float, default=15.0)

    watch_parser = subparsers.add_parser("watch", help="Stream session status plus new events.")
    watch_parser.add_argument("session_id")
    watch_parser.add_argument("--after", type=int, default=0)
    watch_parser.add_argument("--timeout-s", type=float, default=15.0)

    approve_parser = subparsers.add_parser("approve", help="Approve a pending session.")
    approve_parser.add_argument("session_id")
    approve_parser.add_argument("--note", default="")
    approve_parser.add_argument("--no-resume", action="store_true")

    deny_parser = subparsers.add_parser("deny", help="Deny a pending session.")
    deny_parser.add_argument("session_id")
    deny_parser.add_argument("--note", default="")

    resume_parser = subparsers.add_parser("resume", help="Resume an interrupted or approval-blocked session.")
    resume_parser.add_argument("session_id")
    resume_parser.add_argument("--note", default="")
    resume_parser.add_argument("--background", action="store_true")
    resume_parser.add_argument("--timeout-s", type=float, default=30.0)

    retry_parser = subparsers.add_parser("retry", help="Retry a session as a new attempt.")
    retry_parser.add_argument("session_id")
    retry_parser.add_argument("--note", default="")
    retry_parser.add_argument("--background", action="store_true")
    retry_parser.add_argument("--timeout-s", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = LocalAgentRuntime()

    if args.command == "profiles":
        print(json.dumps([profile.model_dump(mode="json") for profile in runtime.list_system_profiles()], indent=2, ensure_ascii=False))
        return
    if args.command == "workflows":
        print(json.dumps(runtime.list_workflows(lane=args.lane), indent=2, ensure_ascii=False))
        return
    if args.command == "sessions":
        print(json.dumps([session.model_dump(mode="json") for session in runtime.list_sessions(status=args.status)], indent=2, ensure_ascii=False))
        return
    if args.command == "approvals":
        approvals = runtime.list_approvals(status=None if args.all else ApprovalStatus.PENDING)
        print(json.dumps([approval.model_dump(mode="json") for approval in approvals], indent=2, ensure_ascii=False))
        return
    if args.command == "run":
        session = runtime.launch_session(
            workflow_id=args.workflow_id,
            system_id=args.system_id,
            lane=args.lane,
            title=args.title,
            human_request=args.human_request,
            background=args.background,
            sandbox_mode=args.sandbox_mode,
            sandbox_policy_id=args.sandbox_policy_id,
        )
        if args.background:
            print(json.dumps(session.model_dump(mode="json"), indent=2, ensure_ascii=False))
            return
        settled = runtime.wait_for_session(session.session_id, timeout_s=args.timeout_s)
        print(json.dumps(settled.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return
    if args.command == "live":
        session = runtime.launch_session(
            workflow_id=args.workflow_id,
            system_id=args.system_id,
            lane=args.lane,
            title=args.title,
            human_request=args.human_request,
            project_id=args.project_id,
            background=not args.once,
            sandbox_mode=args.sandbox_mode,
            sandbox_policy_id=args.sandbox_policy_id,
        )
        attach_to_session(
            runtime,
            session.session_id,
            refresh_s=args.refresh_s,
            timeout_s=args.timeout_s,
            once=args.once,
        )
        return
    if args.command == "attach":
        target_session_id = args.session_id
        if args.action:
            acted = apply_operator_action(
                runtime,
                args.session_id,
                action=args.action,
                note=args.note,
                resume=not args.no_resume,
                background=not args.foreground,
            )
            target_session_id = acted.session_id
            if args.action == "quit":
                print(json.dumps(acted.model_dump(mode="json"), indent=2, ensure_ascii=False))
                return
        attach_to_session(
            runtime,
            target_session_id,
            refresh_s=args.refresh_s,
            timeout_s=args.timeout_s,
            once=args.once,
        )
        return
    if args.command == "inspect":
        if args.json:
            print(json.dumps(session_inspection_payload(runtime, args.session_id, target=args.target), indent=2, ensure_ascii=False))
        else:
            print_session_inspection(runtime, args.session_id, target=args.target)
        return
    if args.command == "show":
        print(json.dumps(runtime.get_session(args.session_id).model_dump(mode="json"), indent=2, ensure_ascii=False))
        return
    if args.command == "events":
        if args.follow:
            events = runtime.wait_for_events(args.session_id, after_sequence=args.after, timeout_s=args.timeout_s)
        else:
            events = runtime.get_events(args.session_id, after_sequence=args.after)
        print(json.dumps([event.model_dump(mode="json") for event in events], indent=2, ensure_ascii=False))
        return
    if args.command == "watch":
        payload = runtime.stream_session(args.session_id, after_sequence=args.after, timeout_s=args.timeout_s)
        print(
            json.dumps(
                {
                    "session": payload["session"].model_dump(mode="json"),
                    "events": [event.model_dump(mode="json") for event in payload["events"]],
                    "pending_approval": payload["pending_approval"].model_dump(mode="json") if payload["pending_approval"] else None,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.command == "approve":
        print(
            json.dumps(
                runtime.resolve_approval(args.session_id, decision="approve", note=args.note, resume=not args.no_resume).model_dump(mode="json"),
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.command == "deny":
        print(json.dumps(runtime.resolve_approval(args.session_id, decision="deny", note=args.note).model_dump(mode="json"), indent=2, ensure_ascii=False))
        return
    if args.command == "resume":
        session = runtime.resume_session(args.session_id, note=args.note, background=args.background)
        if args.background:
            print(json.dumps(session.model_dump(mode="json"), indent=2, ensure_ascii=False))
            return
        settled = runtime.wait_for_session(session.session_id, timeout_s=args.timeout_s)
        print(json.dumps(settled.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return
    if args.command == "retry":
        session = runtime.retry_session(args.session_id, note=args.note, background=args.background)
        if args.background:
            print(json.dumps(session.model_dump(mode="json"), indent=2, ensure_ascii=False))
            return
        settled = runtime.wait_for_session(session.session_id, timeout_s=args.timeout_s)
        print(json.dumps(settled.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return


if __name__ == "__main__":
    main()
