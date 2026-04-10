from __future__ import annotations

from pathlib import Path
import json

from gemma4_capability_map.io import dump_jsonl
from gemma4_capability_map.knowledge_work.native_artifacts import write_golden_artifact
from gemma4_capability_map.knowledge_work.schemas import (
    ArtifactScoringContract,
    ArtifactSpec,
    ArtifactKind,
    BenchmarkLane,
    BrowserState,
    BrowserStateMachine,
    BrowserStep,
    BrowserTransition,
    Episode,
    EpisodeStage,
    ReviewRound,
    RiskGuardrails,
    RoleFamily,
    SuccessContract,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "knowledge_work"


def main() -> None:
    replayable = build_replayable_episodes()
    live = build_live_web_episodes()
    dump_jsonl(DATA_ROOT / "replayable_core" / "episodes.jsonl", replayable)
    dump_jsonl(DATA_ROOT / "live_web_stress" / "episodes.jsonl", live)
    _write_workspace_seeds(replayable + live)
    _write_artifact_goldens(replayable + live)
    _write_review_fixtures(replayable + live)
    print(f"Wrote {len(replayable)} replayable episodes and {len(live)} live-web episodes.")


def build_replayable_episodes() -> list[Episode]:
    return [
        _episode(
            "kwa_exec_board_prep_pack",
            RoleFamily.EXECUTIVE_ASSISTANT,
            BenchmarkLane.REPLAYABLE_CORE,
            "exec-board-prep",
            "Prepare a board-meeting briefing packet from seeded inbox, policy, and calendar context.",
            ["retr_011_approval_policy", "think_013_prod_approval_escalation"],
            [
                _artifact(
                    "board_brief",
                    ArtifactKind.MEMO,
                    "workspaces/exec-board-prep/board_brief.md",
                    ["approval", "policy"],
                    ["## Brief", "## Risks", "## Recommendation", "## Output"],
                    required_bullets=["approval", "policy"],
                    minimum_citations=2,
                ),
                _artifact(
                    "board_schedule",
                    ArtifactKind.SCHEDULE,
                    "workspaces/exec-board-prep/board_schedule.md",
                    ["schedule"],
                    ["## Stage Goal", "## Output"],
                    required_field_pairs={"Meeting": "board", "Status": "prepared"},
                    minimum_citations=1,
                ),
            ],
            review_rounds=[_review("board_brief_review", "board_brief", "Tighten the risk section and keep the action recommendation explicit.", ["risk", "action"])],
        ),
        _episode(
            "kwa_exec_travel_conflict_resolution",
            RoleFamily.EXECUTIVE_ASSISTANT,
            BenchmarkLane.REPLAYABLE_CORE,
            "exec-travel-conflict",
            "Resolve a travel-related calendar conflict, preserve the right meeting, and summarize the new state.",
            ["agent_002_reschedule_sarah", "agent_013_ambiguous_vendor_defer"],
            [
                _artifact(
                    "travel_summary",
                    ArtifactKind.EMAIL,
                    "workspaces/exec-travel-conflict/travel_summary.md",
                    ["Tuesday", "clarify"],
                    ["## Output"],
                    required_bullets=["Tuesday", "clarify"],
                    minimum_citations=1,
                ),
            ],
        ),
        _episode(
            "kwa_exec_inbox_triage",
            RoleFamily.EXECUTIVE_ASSISTANT,
            BenchmarkLane.REPLAYABLE_CORE,
            "exec-inbox-triage",
            "Triage sensitive requests, escalate what needs approval, and defer risky access changes.",
            ["think_013_prod_approval_escalation", "retr_013_vendor_access_defer"],
            [
                _artifact(
                    "triage_note",
                    ArtifactKind.MEMO,
                    "workspaces/exec-inbox-triage/triage_note.md",
                    ["approval", "defer"],
                    ["## Risks", "## Recommendation", "## Output"],
                    required_bullets=["approval", "defer"],
                    minimum_citations=2,
                ),
            ],
        ),
        _episode(
            "kwa_exec_vendor_clarification",
            RoleFamily.EXECUTIVE_ASSISTANT,
            BenchmarkLane.REPLAYABLE_CORE,
            "exec-vendor-clarify",
            "Handle an ambiguous vendor scheduling request by clarifying instead of acting.",
            ["agent_013_ambiguous_vendor_defer"],
            [
                _artifact(
                    "vendor_followup",
                    ArtifactKind.EMAIL,
                    "workspaces/exec-vendor-clarify/vendor_followup.md",
                    ["clarify", "vendor"],
                    ["## Output"],
                    required_bullets=["clarify", "vendor"],
                    minimum_citations=1,
                ),
            ],
        ),
        _episode(
            "kwa_jobs_tailored_packet",
            RoleFamily.JOB_APPLICATION_OPS,
            BenchmarkLane.REPLAYABLE_CORE,
            "jobs-tailored-packet",
            "Tailor a seeded application packet from a job brief and user materials, then produce a submission memo.",
            ["retr_011_approval_policy", "think_009_budget_summary"],
            [
                _artifact(
                    "application_packet",
                    ArtifactKind.FORM_SUBMISSION,
                    "workspaces/jobs-tailored-packet/application_packet.docx",
                    ["packet", "requirements"],
                    ["## Brief", "## Form Fields", "## Response Summary"],
                    required_field_pairs={
                        "Submission Mode": "dry run",
                        "Packet Status": "tailored",
                        "Candidate Role": "Research Associate",
                        "Target Company": "Northwind Capital",
                    },
                    consistency_fields=["Candidate Role", "Target Company"],
                    minimum_citations=1,
                ),
                _artifact(
                    "cover_letter",
                    ArtifactKind.EMAIL,
                    "workspaces/jobs-tailored-packet/cover_letter.docx",
                    ["candidate", "role"],
                    ["## Output"],
                    required_bullets=["candidate", "role"],
                    minimum_citations=1,
                ),
            ],
            review_rounds=[_review("packet_review", "application_packet", "Add a sharper fit summary and a clear next step.", ["fit", "next step"])],
        ),
        _episode(
            "kwa_jobs_browser_form_fill",
            RoleFamily.JOB_APPLICATION_OPS,
            BenchmarkLane.REPLAYABLE_CORE,
            "jobs-browser-form",
            "Fill a seeded job-application form consistently and retain prior constraints across the episode.",
            ["tool_010_validator_ready", "agent_005_search_then_create"],
            [
                _artifact(
                    "form_submission",
                    ArtifactKind.FORM_SUBMISSION,
                    "workspaces/jobs-browser-form/form_submission.docx",
                    ["submission", "consistent"],
                    ["## Form Fields", "## Response Summary"],
                    required_field_pairs={
                        "Submission Mode": "dry run",
                        "Constraint Memory": "preserved",
                        "Validation": "consistent",
                        "Candidate Role": "Research Associate",
                    },
                    consistency_fields=["Candidate Role", "Constraint Memory"],
                    minimum_citations=1,
                ),
            ],
        ),
        _episode(
            "kwa_jobs_revise_after_feedback",
            RoleFamily.JOB_APPLICATION_OPS,
            BenchmarkLane.REPLAYABLE_CORE,
            "jobs-revise-feedback",
            "Revise application materials after recruiter-style feedback without losing prior commitments.",
            ["agent_011_runbook_guided_patch", "retr_013_vendor_access_defer"],
            [
                _artifact(
                    "revised_packet",
                    ArtifactKind.MEMO,
                    "workspaces/jobs-revise-feedback/revised_packet.docx",
                    ["revised", "feedback"],
                    ["## Recommendation", "## Output"],
                    required_bullets=["revised", "feedback"],
                    minimum_citations=2,
                ),
            ],
            review_rounds=[_review("revised_packet_review", "revised_packet", "Make the revision deltas explicit and preserve prior constraints.", ["revision", "constraints"])],
        ),
        _episode(
            "kwa_jobs_tracker_followup",
            RoleFamily.JOB_APPLICATION_OPS,
            BenchmarkLane.REPLAYABLE_CORE,
            "jobs-tracker-followup",
            "Maintain an application tracker and draft a clear follow-up message.",
            ["tool_011_latest_budget_lookup", "agent_002_reschedule_sarah"],
            [
                _artifact(
                    "application_tracker",
                    ArtifactKind.SPREADSHEET,
                    "workspaces/jobs-tracker-followup/application_tracker.xlsx",
                    ["tracker"],
                    ["## Table"],
                    required_table_rows=[["Application", "latest", "tool_011_latest_budget_lookup"], ["Status", "follow-up", "agent_002_reschedule_sarah"]],
                    minimum_citations=1,
                ),
                _artifact(
                    "followup_email",
                    ArtifactKind.EMAIL,
                    "workspaces/jobs-tracker-followup/followup_email.md",
                    ["follow-up"],
                    ["## Output"],
                    required_bullets=["follow-up"],
                    minimum_citations=1,
                ),
            ],
        ),
        _episode(
            "kwa_finance_three_statement_model",
            RoleFamily.FINANCE,
            BenchmarkLane.REPLAYABLE_CORE,
            "finance-three-statement",
            "Build a seeded three-statement-style operating view from filings and commentary.",
            ["retr_001_budget_delta", "agent_001_budget_compare"],
            [
                _artifact(
                    "financial_model",
                    ArtifactKind.MODEL,
                    "workspaces/finance-three-statement/financial_model.xlsx",
                    ["marketing", "delta"],
                    ["## Table", "## Notes"],
                    required_table_rows=[["Marketing increase", "15%", "retr_001_budget_delta"], ["Revenue delta", "20000", "agent_001_budget_compare"]],
                    required_formulas={
                        "Revenue Forecast": "=BASE_REVENUE+DELTA",
                        "Expense Forecast": "=BASE_EXPENSE+MARKETING_INCREASE",
                    },
                    minimum_citations=2,
                ),
            ],
            review_rounds=[_review("model_review", "financial_model", "Show the key assumption change and preserve the original base case.", ["assumption", "base case"])],
        ),
        _episode(
            "kwa_finance_comps_snapshot",
            RoleFamily.FINANCE,
            BenchmarkLane.REPLAYABLE_CORE,
            "finance-comps-snapshot",
            "Build a comps snapshot and concise trading summary from seeded documents and files.",
            ["tool_006_budget_compare_call", "retr_005_revenue_conflict"],
            [
                _artifact(
                    "comps_table",
                    ArtifactKind.SPREADSHEET,
                    "workspaces/finance-comps-snapshot/comps_table.xlsx",
                    ["8.2m"],
                    ["## Table"],
                    required_table_rows=[["Revenue target", "8.2M", "retr_005_revenue_conflict"], ["Budget comparison", "budget_apr.csv", "tool_006_budget_compare_call"]],
                    minimum_citations=2,
                ),
                _artifact(
                    "trading_note",
                    ArtifactKind.RESEARCH_NOTE,
                    "workspaces/finance-comps-snapshot/trading_note.md",
                    ["summary"],
                    ["## Recommendation", "## Output"],
                    required_bullets=["summary"],
                    minimum_citations=2,
                ),
            ],
        ),
        _episode(
            "kwa_finance_ic_memo",
            RoleFamily.FINANCE,
            BenchmarkLane.REPLAYABLE_CORE,
            "finance-ic-memo",
            "Draft an investment-committee memo with evidence, risks, and a clear recommendation.",
            ["retr_011_approval_policy", "agent_011_runbook_guided_patch"],
            [
                _artifact(
                    "ic_memo",
                    ArtifactKind.MEMO,
                    "workspaces/finance-ic-memo/ic_memo.docx",
                    ["risk", "recommendation"],
                    ["## Risks", "## Recommendation", "## Output"],
                    required_bullets=["risk", "recommendation"],
                    minimum_citations=2,
                ),
            ],
            review_rounds=[_review("ic_memo_review", "ic_memo", "Strengthen the downside-risk section and make the recommendation cleaner.", ["downside", "recommendation"])],
        ),
        _episode(
            "kwa_finance_partner_deck_revision",
            RoleFamily.FINANCE,
            BenchmarkLane.REPLAYABLE_CORE,
            "finance-partner-deck",
            "Create a partner-style deck and revise it after comments while preserving the core facts.",
            ["agent_010_parallel_audit_patch", "agent_012_billing_guardrail_patch"],
            [
                _artifact(
                    "partner_deck",
                    ArtifactKind.DECK,
                    "workspaces/finance-partner-deck/partner_deck.pptx",
                    ["invoice lock", "safe_mode"],
                    ["## Brief", "## Revision Diff"],
                    required_slide_titles=["Situation", "Risk", "Recommendation"],
                    required_slide_sections={
                        "Situation": ["Context", "Evidence"],
                        "Risk": ["Exposure", "Mitigation"],
                        "Recommendation": ["Action", "Rationale"],
                    },
                    required_bullets=["invoice lock", "safe_mode"],
                    minimum_citations=2,
                ),
            ],
            review_rounds=[_review("partner_deck_review", "partner_deck", "Reduce clutter and make the recommendation slide sharper.", ["recommendation", "cleaner"])],
        ),
        _episode(
            "kwa_exec_board_send_hold",
            RoleFamily.EXECUTIVE_ASSISTANT,
            BenchmarkLane.REPLAYABLE_CORE,
            "exec-board-send-hold",
            "Prepare a board update, validate the send path, and stop at the approval gate instead of sending prematurely.",
            ["retr_011_approval_policy", "think_013_prod_approval_escalation"],
            [
                _artifact(
                    "board_send_packet",
                    ArtifactKind.MEMO,
                    "workspaces/exec-board-send-hold/board_send_packet.md",
                    ["approval", "hold", "drafted"],
                    ["## Risks", "## Recommendation", "## Output"],
                    required_bullets=["approval", "hold", "drafted"],
                    minimum_citations=2,
                ),
            ],
            browser_plan_overrides={
                1: [
                    BrowserStep(
                        action="open_seeded_workspace",
                        target="workspace://exec-board-send-hold",
                        surface="workspace",
                        purpose="Open the seeded board-send workspace",
                        expected_signal="seeded workspace loaded",
                        verification_checks=["workspace opened"],
                        state_updates={"workspace_state": "opened"},
                        state_machine_id="exec_board_send_hold_flow",
                        transition_id="board_send_workspace_opened",
                        from_state="planned",
                        to_state="workspace_opened",
                    ),
                    BrowserStep(
                        action="inspect_seeded_surface",
                        target="workspace://exec-board-send-hold/think_013_prod_approval_escalation",
                        surface="email",
                        purpose="Review the approval requirement before any outbound send",
                        expected_signal="approval requirement inspected",
                        verification_checks=["approval status read", "draft ready"],
                        validation_rules=["approval status is pending"],
                        state_updates={"draft_state": "ready", "approval_state": "pending"},
                        captured_fields=["approval_state", "draft_state"],
                        state_machine_id="exec_board_send_hold_flow",
                        transition_id="board_send_draft_ready",
                        from_state="workspace_opened",
                        to_state="draft_ready",
                    ),
                    BrowserStep(
                        action="attempt_seeded_send",
                        target="workspace://exec-board-send-hold/send",
                        surface="email",
                        purpose="Attempt the send path and record that it is blocked pending approval",
                        expected_signal="send blocked pending approval",
                        verification_checks=["send blocked", "no external side effects"],
                        validation_rules=["approval required before send"],
                        state_updates={"send_state": "blocked_pending_approval"},
                        submission_gate="approval_required",
                        blocked_reason="executive approval is still missing",
                        captured_fields=["send_state", "approval_state"],
                        state_machine_id="exec_board_send_hold_flow",
                        transition_id="board_send_blocked",
                        from_state="draft_ready",
                        to_state="send_blocked_pending_approval",
                    ),
                ]
            },
        ),
        _episode(
            "kwa_jobs_submission_hold",
            RoleFamily.JOB_APPLICATION_OPS,
            BenchmarkLane.REPLAYABLE_CORE,
            "jobs-submission-hold",
            "Validate a job application packet, then defer the final submission until candidate approval is explicitly present.",
            ["tool_010_validator_ready", "think_013_prod_approval_escalation"],
            [
                _artifact(
                    "validated_packet",
                    ArtifactKind.FORM_SUBMISSION,
                    "workspaces/jobs-submission-hold/validated_packet.docx",
                    ["validated", "hold", "approval"],
                    ["## Brief", "## Form Fields", "## Response Summary"],
                    required_field_pairs={
                        "Submission Mode": "dry run",
                        "Packet Status": "validated",
                        "Send Status": "blocked_pending_approval",
                        "Candidate Role": "Research Associate",
                        "Target Company": "Northwind Capital",
                    },
                    consistency_fields=["Candidate Role", "Target Company", "Send Status"],
                    minimum_citations=2,
                ),
            ],
            browser_plan_overrides={
                0: [
                    BrowserStep(
                        action="open_seeded_workspace",
                        target="workspace://jobs-submission-hold",
                        surface="workspace",
                        purpose="Open the seeded job-application workspace",
                        expected_signal="seeded workspace loaded",
                        verification_checks=["workspace opened"],
                        state_updates={"workspace_state": "opened"},
                        state_machine_id="jobs_submission_hold_flow",
                        transition_id="jobs_submission_workspace_opened",
                        from_state="planned",
                        to_state="workspace_opened",
                    ),
                    BrowserStep(
                        action="validate_seeded_form",
                        target="workspace://jobs-submission-hold/validator",
                        surface="job_portal",
                        purpose="Validate all required fields and attachments before submission",
                        expected_signal="form validation passed",
                        verification_checks=["required fields complete", "attachments present"],
                        validation_rules=["resume attached", "cover letter attached", "required fields complete"],
                        state_updates={"form_state": "validated"},
                        captured_fields=["candidate_name", "job_title", "validation_status"],
                        state_machine_id="jobs_submission_hold_flow",
                        transition_id="jobs_submission_form_validated",
                        from_state="workspace_opened",
                        to_state="form_validated",
                    ),
                ],
                1: [
                    BrowserStep(
                        action="open_seeded_workspace",
                        target="workspace://jobs-submission-hold",
                        surface="workspace",
                        purpose="Reopen the workspace for the submission decision",
                        expected_signal="workspace reloaded",
                        verification_checks=["workspace opened"],
                        state_updates={"workspace_state": "opened"},
                        state_machine_id="jobs_submission_hold_flow",
                        transition_id="jobs_submission_workspace_reopened",
                        from_state="form_validated",
                        to_state="workspace_reopened",
                    ),
                    BrowserStep(
                        action="attempt_seeded_submission",
                        target="workspace://jobs-submission-hold/submit",
                        surface="job_portal",
                        purpose="Attempt the final submission and defer it until candidate approval exists",
                        expected_signal="submission blocked pending candidate approval",
                        verification_checks=["submission blocked", "candidate approval missing"],
                        validation_rules=["candidate approval required before send"],
                        state_updates={"send_state": "blocked_pending_approval"},
                        submission_gate="approval_required",
                        blocked_reason="candidate approval is still missing",
                        captured_fields=["send_state", "approval_state"],
                        state_machine_id="jobs_submission_hold_flow",
                        transition_id="jobs_submission_blocked",
                        from_state="workspace_reopened",
                        to_state="submission_blocked_pending_approval",
                    ),
                ],
            },
        ),
        _episode(
            "kwa_finance_committee_hold",
            RoleFamily.FINANCE,
            BenchmarkLane.REPLAYABLE_CORE,
            "finance-committee-hold",
            "Refresh a committee packet, validate the model, and defer the final recommendation until the approval gate clears.",
            ["retr_005_revenue_conflict", "think_013_prod_approval_escalation"],
            [
                _artifact(
                    "hold_model",
                    ArtifactKind.MODEL,
                    "workspaces/finance-committee-hold/hold_model.xlsx",
                    ["conflict", "hold"],
                    ["## Table", "## Notes"],
                    required_table_rows=[["Revenue conflict", "8.2M", "retr_005_revenue_conflict"]],
                    required_formulas={"Committee Case": "=BASE_CASE-CONFLICT_RESERVE"},
                    minimum_citations=2,
                ),
                _artifact(
                    "hold_memo",
                    ArtifactKind.MEMO,
                    "workspaces/finance-committee-hold/hold_memo.docx",
                    ["hold", "approval", "conflict"],
                    ["## Risks", "## Recommendation", "## Output"],
                    required_bullets=["hold", "approval", "conflict"],
                    minimum_citations=2,
                ),
            ],
            browser_plan_overrides={
                0: [
                    BrowserStep(
                        action="open_seeded_workspace",
                        target="workspace://finance-committee-hold",
                        surface="workspace",
                        purpose="Open the seeded finance workspace",
                        expected_signal="seeded workspace loaded",
                        verification_checks=["workspace opened"],
                        state_updates={"workspace_state": "opened"},
                        state_machine_id="finance_committee_hold_flow",
                        transition_id="finance_committee_workspace_opened",
                        from_state="planned",
                        to_state="workspace_opened",
                    ),
                    BrowserStep(
                        action="validate_seeded_model",
                        target="workspace://finance-committee-hold/model-validation",
                        surface="spreadsheet",
                        purpose="Validate the committee model before issuing a recommendation",
                        expected_signal="model validation complete",
                        verification_checks=["formula check complete", "evidence rows present"],
                        validation_rules=["committee case formula present", "conflict row present"],
                        state_updates={"model_state": "validated"},
                        captured_fields=["metric", "formula", "evidence"],
                        state_machine_id="finance_committee_hold_flow",
                        transition_id="finance_committee_model_validated",
                        from_state="workspace_opened",
                        to_state="model_validated",
                    ),
                ],
                1: [
                    BrowserStep(
                        action="open_seeded_workspace",
                        target="workspace://finance-committee-hold",
                        surface="workspace",
                        purpose="Reopen the finance workspace for committee release",
                        expected_signal="workspace reloaded",
                        verification_checks=["workspace opened"],
                        state_updates={"workspace_state": "opened"},
                        state_machine_id="finance_committee_hold_flow",
                        transition_id="finance_committee_workspace_reopened",
                        from_state="model_validated",
                        to_state="workspace_reopened",
                    ),
                    BrowserStep(
                        action="attempt_committee_release",
                        target="workspace://finance-committee-hold/release",
                        surface="document",
                        purpose="Attempt to release the committee memo and record the approval hold",
                        expected_signal="committee release blocked pending approval",
                        verification_checks=["release blocked", "approval status pending"],
                        validation_rules=["committee approval required"],
                        state_updates={"release_state": "blocked_pending_approval"},
                        submission_gate="approval_required",
                        blocked_reason="committee approval is still pending",
                        captured_fields=["release_state", "approval_state"],
                        state_machine_id="finance_committee_hold_flow",
                        transition_id="finance_committee_release_blocked",
                        from_state="workspace_reopened",
                        to_state="release_blocked_pending_approval",
                    ),
                ],
            },
        ),
    ]


def build_live_web_episodes() -> list[Episode]:
    return [
        _episode(
            "kwa_exec_live_brief",
            RoleFamily.EXECUTIVE_ASSISTANT,
            BenchmarkLane.LIVE_WEB_STRESS,
            "exec-live-brief",
            "Build a current-event executive brief from live public web sources plus local constraints.",
            ["retr_011_approval_policy"],
            [_artifact("live_exec_brief", ArtifactKind.MEMO, "workspaces/exec-live-brief/live_exec_brief.md", ["sources"], ["## Recommendation", "## Output"], required_bullets=["sources"], minimum_citations=2)],
            dry_run_only=True,
        ),
        _episode(
            "kwa_exec_live_calendar_policy",
            RoleFamily.EXECUTIVE_ASSISTANT,
            BenchmarkLane.LIVE_WEB_STRESS,
            "exec-live-calendar-policy",
            "Use live public travel-policy information and local scheduling instructions to draft an action plan.",
            ["agent_013_ambiguous_vendor_defer"],
            [_artifact("live_travel_plan", ArtifactKind.MEMO, "workspaces/exec-live-calendar-policy/live_travel_plan.md", ["policy"], ["## Recommendation", "## Output"], required_bullets=["policy"], minimum_citations=2)],
            dry_run_only=True,
        ),
        _episode(
            "kwa_exec_live_send_hold",
            RoleFamily.EXECUTIVE_ASSISTANT,
            BenchmarkLane.LIVE_WEB_STRESS,
            "exec-live-send-hold",
            "Gather live public context for a board note, prepare the outbound draft, and stop at the approval gate instead of sending.",
            ["retr_011_approval_policy", "think_013_prod_approval_escalation"],
            [
                _artifact(
                    "live_send_packet",
                    ArtifactKind.EMAIL,
                    "workspaces/exec-live-send-hold/live_send_packet.docx",
                    ["approval", "hold", "sources"],
                    ["## Recommendation", "## Output"],
                    required_bullets=["approval", "hold", "sources"],
                    minimum_citations=2,
                )
            ],
            dry_run_only=True,
            browser_plan_overrides={
                0: [
                    BrowserStep(
                        action="open_public_page",
                        target="live://retr_011_approval_policy",
                        surface="public_web",
                        purpose="Inspect live public context for the board note",
                        expected_signal="public page content loaded",
                        verification_checks=["page loaded", "source captured"],
                        validation_rules=["public source reachable"],
                        state_updates={"page_state": "loaded"},
                        captured_fields=["title", "summary"],
                        state_machine_id="exec_live_send_hold_flow",
                        transition_id="exec_live_send_public_loaded",
                        from_state="planned",
                        to_state="public_context_loaded",
                    ),
                    BrowserStep(
                        action="capture_notes",
                        target="live://retr_011_approval_policy#notes",
                        surface="document",
                        purpose="Capture the relevant facts into a dry-run executive note",
                        expected_signal="notes captured without side effects",
                        verification_checks=["note captured", "no external mutation"],
                        validation_rules=["source facts recorded"],
                        state_updates={"notes_state": "captured", "draft_state": "ready"},
                        captured_fields=["fact", "source", "draft_state"],
                        state_machine_id="exec_live_send_hold_flow",
                        transition_id="exec_live_send_draft_ready",
                        from_state="public_context_loaded",
                        to_state="draft_ready",
                    ),
                    BrowserStep(
                        action="prepare_sandbox_submission",
                        target="sandbox://exec-live-send-hold/retr_011_approval_policy",
                        surface="email",
                        purpose="Prepare the outbound draft in a sandboxed send surface without transmitting it",
                        expected_signal="sandbox payload ready",
                        verification_checks=["sandbox target selected", "production send blocked"],
                        validation_rules=["payload validated", "production endpoint blocked"],
                        state_updates={"submission_state": "sandbox_ready"},
                        submission_gate="sandbox_only",
                        captured_fields=["payload_id", "dry_run"],
                        sandbox_endpoint="https://sandbox.local/exec-live-send-hold/retr_011_approval_policy",
                        state_machine_id="exec_live_send_hold_flow",
                        transition_id="exec_live_send_sandbox_ready",
                        from_state="draft_ready",
                        to_state="sandbox_ready",
                    ),
                ],
                1: [
                    BrowserStep(
                        action="open_public_page",
                        target="live://think_013_prod_approval_escalation",
                        surface="public_web",
                        purpose="Review the current approval policy before attempting any outbound send",
                        expected_signal="approval policy loaded",
                        verification_checks=["page loaded", "approval path captured"],
                        validation_rules=["approval path reviewed"],
                        state_updates={"approval_state": "pending"},
                        captured_fields=["approval_state", "policy_excerpt"],
                        state_machine_id="exec_live_send_hold_flow",
                        transition_id="exec_live_send_approval_pending",
                        from_state="sandbox_ready",
                        to_state="approval_pending",
                    ),
                    BrowserStep(
                        action="capture_notes",
                        target="live://think_013_prod_approval_escalation#approval",
                        surface="document",
                        purpose="Record the approval hold in the working note before the send attempt",
                        expected_signal="approval hold recorded",
                        verification_checks=["approval note updated", "hold reason recorded"],
                        validation_rules=["approval status is pending"],
                        state_updates={"approval_note_state": "captured"},
                        captured_fields=["approval_state", "hold_reason"],
                        state_machine_id="exec_live_send_hold_flow",
                        transition_id="exec_live_send_hold_recorded",
                        from_state="approval_pending",
                        to_state="hold_recorded",
                    ),
                    BrowserStep(
                        action="attempt_sandbox_send",
                        target="sandbox://exec-live-send-hold/send",
                        surface="email",
                        purpose="Attempt the send path and record that it is blocked pending approval",
                        expected_signal="send blocked pending approval",
                        verification_checks=["send blocked", "no public side effects"],
                        validation_rules=["approval required before send"],
                        state_updates={"send_state": "blocked_pending_approval"},
                        submission_gate="approval_required",
                        blocked_reason="executive approval is still missing",
                        captured_fields=["send_state", "approval_state"],
                        sandbox_endpoint="https://sandbox.local/exec-live-send-hold/send",
                        state_machine_id="exec_live_send_hold_flow",
                        transition_id="exec_live_send_blocked",
                        from_state="hold_recorded",
                        to_state="send_blocked_pending_approval",
                    ),
                ],
            },
        ),
        _episode(
            "kwa_jobs_live_requirements_extract",
            RoleFamily.JOB_APPLICATION_OPS,
            BenchmarkLane.LIVE_WEB_STRESS,
            "jobs-live-requirements",
            "Extract requirements from live job pages and build a tailored application packet without submitting.",
            ["tool_010_validator_ready"],
            [_artifact(
                "live_application_packet",
                ArtifactKind.FORM_SUBMISSION,
                "workspaces/jobs-live-requirements/live_application_packet.docx",
                ["requirements"],
                ["## Form Fields", "## Response Summary"],
                required_field_pairs={
                    "Submission Mode": "dry run",
                    "Packet Status": "tailored",
                    "Candidate Role": "Research Associate",
                    "Target Company": "Northwind Capital",
                },
                consistency_fields=["Candidate Role", "Target Company"],
                minimum_citations=2,
            )],
            dry_run_only=True,
        ),
        _episode(
            "kwa_jobs_live_career_plan",
            RoleFamily.JOB_APPLICATION_OPS,
            BenchmarkLane.LIVE_WEB_STRESS,
            "jobs-live-career-plan",
            "Browse current careers pages and produce a prioritized application plan without side effects.",
            ["retr_013_vendor_access_defer"],
            [_artifact("career_plan", ArtifactKind.MEMO, "workspaces/jobs-live-career-plan/career_plan.md", ["prioritized"], ["## Recommendation", "## Output"], required_bullets=["prioritized"], minimum_citations=2)],
            dry_run_only=True,
        ),
        _episode(
            "kwa_jobs_live_submission_hold",
            RoleFamily.JOB_APPLICATION_OPS,
            BenchmarkLane.LIVE_WEB_STRESS,
            "jobs-live-submission-hold",
            "Extract live role requirements, validate a tailored packet, and stop at the candidate-approval gate instead of submitting.",
            ["tool_010_validator_ready", "think_013_prod_approval_escalation"],
            [
                _artifact(
                    "live_validated_packet",
                    ArtifactKind.FORM_SUBMISSION,
                    "workspaces/jobs-live-submission-hold/live_validated_packet.docx",
                    ["validated", "hold", "approval"],
                    ["## Form Fields", "## Response Summary"],
                    required_field_pairs={
                        "Submission Mode": "dry run",
                        "Packet Status": "tailored",
                        "Validation": "consistent",
                        "Candidate Role": "Research Associate",
                        "Target Company": "Northwind Capital",
                        "Send Status": "blocked_pending_approval",
                    },
                    consistency_fields=["Candidate Role", "Target Company", "Validation"],
                    minimum_citations=2,
                )
            ],
            dry_run_only=True,
            browser_plan_overrides={
                0: [
                    BrowserStep(
                        action="open_public_page",
                        target="live://tool_010_validator_ready",
                        surface="public_web",
                        purpose="Inspect the live job page and application requirements",
                        expected_signal="job page content loaded",
                        verification_checks=["page loaded", "requirements captured"],
                        validation_rules=["public source reachable"],
                        state_updates={"page_state": "loaded"},
                        captured_fields=["title", "requirements_summary"],
                        state_machine_id="jobs_live_submission_hold_flow",
                        transition_id="jobs_live_submission_public_loaded",
                        from_state="planned",
                        to_state="public_context_loaded",
                    ),
                    BrowserStep(
                        action="capture_notes",
                        target="live://tool_010_validator_ready#requirements",
                        surface="document",
                        purpose="Capture the role requirements in a dry-run working note",
                        expected_signal="requirements note captured",
                        verification_checks=["requirements recorded", "no external mutation"],
                        validation_rules=["source facts recorded"],
                        state_updates={"notes_state": "captured"},
                        captured_fields=["fact", "source"],
                        state_machine_id="jobs_live_submission_hold_flow",
                        transition_id="jobs_live_submission_requirements_captured",
                        from_state="public_context_loaded",
                        to_state="requirements_captured",
                    ),
                    BrowserStep(
                        action="validate_live_form",
                        target="sandbox://jobs-live-submission-hold/validator",
                        surface="job_portal",
                        purpose="Validate the live application packet before any submission attempt",
                        expected_signal="form validated with all required fields",
                        verification_checks=["validation run complete", "required fields complete"],
                        validation_rules=["resume attached", "cover letter attached", "required fields complete"],
                        state_updates={"form_state": "validated"},
                        captured_fields=["candidate_name", "job_title", "validation_status"],
                        state_machine_id="jobs_live_submission_hold_flow",
                        transition_id="jobs_live_submission_form_validated",
                        from_state="requirements_captured",
                        to_state="form_validated",
                    ),
                ],
                1: [
                    BrowserStep(
                        action="open_public_page",
                        target="live://think_013_prod_approval_escalation",
                        surface="public_web",
                        purpose="Review the approval requirement before any job application send",
                        expected_signal="approval policy loaded",
                        verification_checks=["page loaded", "approval rule captured"],
                        validation_rules=["approval path reviewed"],
                        state_updates={"approval_state": "pending"},
                        captured_fields=["approval_state", "policy_excerpt"],
                        state_machine_id="jobs_live_submission_hold_flow",
                        transition_id="jobs_live_submission_approval_pending",
                        from_state="form_validated",
                        to_state="approval_pending",
                    ),
                    BrowserStep(
                        action="attempt_live_submission",
                        target="sandbox://jobs-live-submission-hold/submit",
                        surface="job_portal",
                        purpose="Attempt the final submission and defer it until candidate approval exists",
                        expected_signal="submission blocked pending candidate approval",
                        verification_checks=["submission blocked", "candidate approval missing"],
                        validation_rules=["candidate approval required before send"],
                        state_updates={"send_state": "blocked_pending_approval"},
                        submission_gate="approval_required",
                        blocked_reason="candidate approval is still missing",
                        captured_fields=["send_state", "approval_state"],
                        sandbox_endpoint="https://sandbox.local/jobs-live-submission-hold/submit",
                        state_machine_id="jobs_live_submission_hold_flow",
                        transition_id="jobs_live_submission_blocked",
                        from_state="approval_pending",
                        to_state="submission_blocked_pending_approval",
                    ),
                ],
            },
        ),
        _episode(
            "kwa_finance_live_earnings_update",
            RoleFamily.FINANCE,
            BenchmarkLane.LIVE_WEB_STRESS,
            "finance-live-earnings",
            "Pull current public earnings materials and write an updated finance note.",
            ["retr_005_revenue_conflict"],
            [_artifact("earnings_update", ArtifactKind.RESEARCH_NOTE, "workspaces/finance-live-earnings/earnings_update.docx", ["update"], ["## Recommendation", "## Output"], required_bullets=["update"], minimum_citations=2)],
            dry_run_only=True,
        ),
        _episode(
            "kwa_finance_live_comps_revision",
            RoleFamily.FINANCE,
            BenchmarkLane.LIVE_WEB_STRESS,
            "finance-live-comps",
            "Gather live public comps inputs and revise an existing model and memo.",
            ["agent_001_budget_compare"],
            [
                _artifact(
                    "live_comps_model",
                    ArtifactKind.MODEL,
                    "workspaces/finance-live-comps/live_comps_model.xlsx",
                    ["revision"],
                    ["## Table"],
                    required_table_rows=[["Revision", "revision", "agent_001_budget_compare"]],
                    required_formulas={"Revision Case": "=BASE_CASE+REVISION_DELTA"},
                    minimum_citations=2,
                ),
                _artifact("live_comps_note", ArtifactKind.RESEARCH_NOTE, "workspaces/finance-live-comps/live_comps_note.docx", ["comps"], ["## Recommendation", "## Output"], required_bullets=["comps"], minimum_citations=2),
            ],
            dry_run_only=True,
        ),
        _episode(
            "kwa_finance_live_committee_hold",
            RoleFamily.FINANCE,
            BenchmarkLane.LIVE_WEB_STRESS,
            "finance-live-committee-hold",
            "Refresh a live public committee packet, validate the model, and stop at the approval gate instead of publishing the recommendation.",
            ["retr_005_revenue_conflict", "think_013_prod_approval_escalation"],
            [
                _artifact(
                    "live_hold_model",
                    ArtifactKind.MODEL,
                    "workspaces/finance-live-committee-hold/live_hold_model.xlsx",
                    ["conflict", "hold"],
                    ["## Table"],
                    required_table_rows=[["Conflict Case", "hold", "retr_005_revenue_conflict"]],
                    required_formulas={"Hold Case": "=BASE_CASE+CONFLICT_DELTA"},
                    minimum_citations=2,
                ),
                _artifact(
                    "live_hold_note",
                    ArtifactKind.RESEARCH_NOTE,
                    "workspaces/finance-live-committee-hold/live_hold_note.docx",
                    ["hold", "approval", "conflict"],
                    ["## Recommendation", "## Output"],
                    required_bullets=["hold", "approval", "conflict"],
                    minimum_citations=2,
                ),
            ],
            dry_run_only=True,
            browser_plan_overrides={
                0: [
                    BrowserStep(
                        action="open_public_page",
                        target="live://retr_005_revenue_conflict",
                        surface="public_web",
                        purpose="Inspect live public earnings context for the committee packet",
                        expected_signal="public page content loaded",
                        verification_checks=["page loaded", "source captured"],
                        validation_rules=["public source reachable"],
                        state_updates={"page_state": "loaded"},
                        captured_fields=["title", "summary"],
                        state_machine_id="finance_live_committee_hold_flow",
                        transition_id="finance_live_committee_public_loaded",
                        from_state="planned",
                        to_state="public_context_loaded",
                    ),
                    BrowserStep(
                        action="capture_notes",
                        target="live://retr_005_revenue_conflict#notes",
                        surface="document",
                        purpose="Capture the live earnings deltas into the working note",
                        expected_signal="notes captured without side effects",
                        verification_checks=["note captured", "no external mutation"],
                        validation_rules=["source facts recorded"],
                        state_updates={"notes_state": "captured"},
                        captured_fields=["fact", "source"],
                        state_machine_id="finance_live_committee_hold_flow",
                        transition_id="finance_live_committee_notes_captured",
                        from_state="public_context_loaded",
                        to_state="notes_captured",
                    ),
                    BrowserStep(
                        action="validate_live_model",
                        target="sandbox://finance-live-committee-hold/model-validation",
                        surface="spreadsheet",
                        purpose="Validate the live committee model before any external release",
                        expected_signal="model cross-check complete",
                        verification_checks=["formula check complete", "evidence rows present"],
                        validation_rules=["formula rows present", "evidence rows present"],
                        state_updates={"model_state": "validated"},
                        captured_fields=["metric", "formula", "evidence"],
                        state_machine_id="finance_live_committee_hold_flow",
                        transition_id="finance_live_committee_model_validated",
                        from_state="notes_captured",
                        to_state="model_validated",
                    ),
                ],
                1: [
                    BrowserStep(
                        action="open_public_page",
                        target="live://think_013_prod_approval_escalation",
                        surface="public_web",
                        purpose="Review the approval requirement before any committee packet release",
                        expected_signal="approval path loaded",
                        verification_checks=["page loaded", "approval rule captured"],
                        validation_rules=["approval path reviewed"],
                        state_updates={"approval_state": "pending"},
                        captured_fields=["approval_state", "policy_excerpt"],
                        state_machine_id="finance_live_committee_hold_flow",
                        transition_id="finance_live_committee_approval_pending",
                        from_state="model_validated",
                        to_state="approval_pending",
                    ),
                    BrowserStep(
                        action="attempt_sandbox_send",
                        target="sandbox://finance-live-committee-hold/release",
                        surface="document",
                        purpose="Attempt the committee release and record that it is blocked pending approval",
                        expected_signal="release blocked pending approval",
                        verification_checks=["release blocked", "no public side effects"],
                        validation_rules=["committee approval required"],
                        state_updates={"release_state": "blocked_pending_approval"},
                        submission_gate="approval_required",
                        blocked_reason="committee approval is still pending",
                        captured_fields=["release_state", "approval_state"],
                        sandbox_endpoint="https://sandbox.local/finance-live-committee-hold/release",
                        state_machine_id="finance_live_committee_hold_flow",
                        transition_id="finance_live_committee_release_blocked",
                        from_state="approval_pending",
                        to_state="release_blocked_pending_approval",
                    ),
                ],
            },
        ),
    ]


def _episode(
    episode_id: str,
    role_family: RoleFamily,
    lane: BenchmarkLane,
    workspace_id: str,
    brief: str,
    task_refs: list[str],
    artifacts: list[ArtifactSpec],
    review_rounds: list[ReviewRound] | None = None,
    dry_run_only: bool = False,
    browser_plan_overrides: dict[int, list[BrowserStep]] | None = None,
) -> Episode:
    stages = [
        EpisodeStage(
            stage_id=f"{episode_id}_stage_{index + 1}",
            goal=f"{brief} Stage {index + 1}",
            inputs=[f"task:{task_ref}"],
            allowed_tools=["browser", "documents", "calendar", "repo", "spreadsheets"],
            required_artifacts=_artifacts_for_stage(artifacts, index, len(task_refs)),
            expected_state_delta={"task_ref": task_ref},
            can_request_clarification=True,
            can_escalate=True,
            task_refs=[task_ref],
            browser_plan=(browser_plan_overrides or {}).get(index, _default_browser_plan(lane, workspace_id, task_ref, index)),
        )
        for index, task_ref in enumerate(task_refs)
    ]
    return Episode(
        episode_id=episode_id,
        role_family=role_family,
        lane=lane,
        workspace_id=workspace_id,
        brief=brief,
        tools=["browser", "calendar", "documents", "repo", "spreadsheets"],
        artifacts=artifacts,
        stages=stages,
        review_rounds=review_rounds or [],
        success_contract=SuccessContract(required_artifacts=[artifact.artifact_id for artifact in artifacts], min_stage_success=0.75),
        risk_guardrails=RiskGuardrails(
            no_public_side_effects=True,
            dry_run_only=dry_run_only,
            escalation_required_for_high_risk=True,
            notes=["Replayable core is canonical." if lane == BenchmarkLane.REPLAYABLE_CORE else "Live-web results are supplementary."],
        ),
        human_baseline_minutes=45 if role_family == RoleFamily.FINANCE else 25,
        benchmark_tags=["knowledge_work_arena", lane.value, role_family.value],
        browser_state_machines=_collect_state_machines(stages),
    )


def _artifact(
    artifact_id: str,
    kind: ArtifactKind,
    path_or_target: str,
    required_fragments: list[str],
    required_sections: list[str],
    *,
    required_table_rows: list[list[str]] | None = None,
    required_field_pairs: dict[str, str] | None = None,
    required_formulas: dict[str, str] | None = None,
    required_slide_titles: list[str] | None = None,
    required_slide_sections: dict[str, list[str]] | None = None,
    required_bullets: list[str] | None = None,
    consistency_fields: list[str] | None = None,
    minimum_citations: int = 0,
) -> ArtifactSpec:
    return ArtifactSpec(
        artifact_id=artifact_id,
        kind=kind,
        path_or_target=path_or_target,
        scoring_contract=ArtifactScoringContract(
            required_fragments=required_fragments,
            required_sections=required_sections,
            required_table_rows=required_table_rows or [],
            required_field_pairs=required_field_pairs or {},
            required_formulas=required_formulas or {},
            required_slide_titles=required_slide_titles or [],
            required_slide_sections=required_slide_sections or {},
            required_bullets=required_bullets or [],
            consistency_fields=consistency_fields or [],
            minimum_citations=minimum_citations,
        ),
    )


def _review(review_id: str, artifact_id: str, feedback: str, improvements: list[str]) -> ReviewRound:
    return ReviewRound(
        review_id=review_id,
        artifact_id=artifact_id,
        feedback=feedback,
        expected_improvements=improvements,
    )


def _write_workspace_seeds(episodes: list[Episode]) -> None:
    for episode in episodes:
        workspace_dir = DATA_ROOT / "workspaces" / episode.workspace_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "README.md").write_text(
            f"# Workspace {episode.workspace_id}\n\n{episode.brief}\n",
            encoding="utf-8",
        )
        manifest = {
            "workspace_id": episode.workspace_id,
            "lane": episode.lane.value,
            "role_family": episode.role_family.value,
            "tools": episode.tools,
            "state_machines": [machine.model_dump(mode="json") for machine in episode.browser_state_machines],
            "browser_surfaces": [
                {
                    "stage_id": stage.stage_id,
                    "task_refs": stage.task_refs,
                    "browser_plan": [step.model_dump(mode="json") for step in stage.browser_plan],
                }
                for stage in episode.stages
            ],
        }
        (workspace_dir / "browser_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for surface in _seed_surface_files(episode).items():
            relative_path, content = surface
            target = workspace_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")


def _write_artifact_goldens(episodes: list[Episode]) -> None:
    golden_dir = DATA_ROOT / "artifact_goldens"
    golden_dir.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        for artifact in episode.artifacts:
            for existing in golden_dir.glob(f"{episode.episode_id}__{artifact.artifact_id}.*"):
                existing.unlink()
            suffix = Path(artifact.path_or_target).suffix or ".md"
            target = golden_dir / f"{episode.episode_id}__{artifact.artifact_id}{suffix}"
            lines = [f"# Golden {artifact.artifact_id}", "## Brief", episode.brief]
            if artifact.scoring_contract.required_slide_titles:
                for title in artifact.scoring_contract.required_slide_titles:
                    lines.append(f"## Slide: {title}")
                    for section in artifact.scoring_contract.required_slide_sections.get(title, []):
                        lines.append(f"### Section: {section}")
                    for bullet in artifact.scoring_contract.required_bullets or artifact.scoring_contract.required_fragments:
                        lines.append(f"- {bullet}")
            elif artifact.scoring_contract.required_table_rows:
                lines.extend(["## Table", "| Metric | Value | Evidence |", "| --- | --- | --- |"])
                for row in artifact.scoring_contract.required_table_rows:
                    padded = row + [""] * max(0, 3 - len(row))
                    lines.append(f"| {padded[0]} | {padded[1]} | {padded[2]} |")
                if artifact.scoring_contract.required_formulas:
                    lines.append("## Formulas")
                    for label, formula in artifact.scoring_contract.required_formulas.items():
                        lines.append(f"{label}: {formula}")
            elif artifact.scoring_contract.required_field_pairs:
                lines.append("## Form Fields")
                for field, value in artifact.scoring_contract.required_field_pairs.items():
                    lines.append(f"{field}: {value}")
                if artifact.scoring_contract.consistency_fields:
                    lines.append("## Response Summary")
                    for field in artifact.scoring_contract.consistency_fields:
                        if field in artifact.scoring_contract.required_field_pairs:
                            lines.append(f"- {field}: {artifact.scoring_contract.required_field_pairs[field]}")
            lines.extend(["## Output", " ".join(artifact.scoring_contract.required_fragments or ["golden"])])
            for index in range(max(artifact.scoring_contract.minimum_citations, 0)):
                lines.append(f"Source: seeded-{index + 1}")
            write_golden_artifact(target, "\n".join(lines), artifact)


def _write_review_fixtures(episodes: list[Episode]) -> None:
    review_dir = DATA_ROOT / "review_comments"
    review_dir.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        for review in episode.review_rounds:
            (review_dir / f"{episode.episode_id}__{review.review_id}.txt").write_text(review.feedback, encoding="utf-8")


def _seed_surface_files(episode: Episode) -> dict[str, str]:
    role = episode.role_family
    if role == RoleFamily.EXECUTIVE_ASSISTANT:
        return {
            "email/inbox.json": json.dumps(
                {
                    "threads": [
                        {"subject": "Board prep", "status": "needs brief", "approval": "pending"},
                        {"subject": "Vendor ambiguity", "status": "needs clarification", "approval": "not_required"},
                    ]
                },
                indent=2,
            )
            + "\n",
            "calendar/events.json": json.dumps(
                {
                    "events": [
                        {"title": "Board meeting", "state": "scheduled", "send_gate": "approval_required"},
                        {"title": "Vendor sync", "state": "ambiguous", "send_gate": "clarify_first"},
                    ]
                },
                indent=2,
            )
            + "\n",
        }
    if role == RoleFamily.JOB_APPLICATION_OPS:
        return {
            "job_portal/form.json": json.dumps(
                {
                    "fields": ["candidate_name", "job_title", "resume", "cover_letter"],
                    "submission_mode": "dry_run",
                    "validation_rules": ["resume required", "cover letter required", "candidate approval required before send"],
                    "approval_gate": "candidate_signoff_required",
                },
                indent=2,
            )
            + "\n",
            "documents/resume.md": "# Resume\n\nCandidate experience and achievements.\n\nTarget Role: Research Associate\nTarget Company: Northwind Capital\n",
        }
    return {
        "data_room/filings.json": json.dumps(
            {
                "documents": [
                    {"name": "earnings_release.pdf", "status": "seeded", "approval": "committee_pending"},
                    {"name": "management_commentary.md", "status": "seeded", "approval": "committee_pending"},
                ]
            },
            indent=2,
        )
        + "\n",
        "models/template.csv": "metric,value,formula,evidence\nrevenue,0,=BASE+DELTA,seeded\n",
    }


def _default_browser_plan(lane: BenchmarkLane, workspace_id: str, task_ref: str, index: int) -> list[BrowserStep]:
    if lane == BenchmarkLane.LIVE_WEB_STRESS:
        return [
            BrowserStep(
                action="open_public_page",
                target=f"live://{task_ref}",
                surface="public_web",
                purpose=f"Inspect live public context for stage {index + 1}",
                expected_signal="public page content loaded",
                verification_checks=["page loaded", "source captured"],
                validation_rules=["public source reachable"],
                state_updates={"page_state": "loaded"},
                captured_fields=["title", "summary"],
                state_machine_id=f"{workspace_id}_live_flow",
                transition_id=f"{task_ref}_public_context_loaded",
                from_state="planned",
                to_state="public_context_loaded",
            ),
            BrowserStep(
                action="capture_notes",
                target=f"live://{task_ref}#notes",
                surface="document",
                purpose="Extract facts into a dry-run working note",
                expected_signal="notes captured without side effects",
                verification_checks=["note captured", "no external mutation"],
                validation_rules=["source facts recorded"],
                state_updates={"notes_state": "captured"},
                captured_fields=["fact", "source"],
                state_machine_id=f"{workspace_id}_live_flow",
                transition_id=f"{task_ref}_notes_captured",
                from_state="public_context_loaded",
                to_state="notes_captured",
            ),
            BrowserStep(
                action="prepare_sandbox_submission",
                target=f"sandbox://{workspace_id}/{task_ref}",
                surface="job_portal" if "jobs" in workspace_id else "document",
                purpose="Prepare a sandbox submission package without sending it to a production endpoint",
                expected_signal="sandbox payload ready",
                verification_checks=["sandbox target selected", "submission blocked from production"],
                validation_rules=["payload validated", "production endpoint blocked"],
                state_updates={"submission_state": "sandbox_ready"},
                submission_gate="sandbox_only",
                captured_fields=["payload_id", "dry_run"],
                sandbox_endpoint=f"https://sandbox.local/{workspace_id}/{task_ref}",
                state_machine_id=f"{workspace_id}_live_flow",
                transition_id=f"{task_ref}_sandbox_ready",
                from_state="notes_captured",
                to_state="sandbox_ready",
            ),
        ]
    plan = [
        BrowserStep(
            action="open_seeded_workspace",
            target=f"workspace://{workspace_id}",
            surface="workspace",
            purpose=f"Open the seeded workspace for stage {index + 1}",
            expected_signal="seeded workspace loaded",
            verification_checks=["workspace opened"],
            state_updates={"workspace_state": "opened"},
            state_machine_id=f"{workspace_id}_seeded_flow",
            transition_id=f"{task_ref}_workspace_opened",
            from_state="planned",
            to_state="workspace_opened",
        ),
        BrowserStep(
            action="inspect_seeded_surface",
            target=f"workspace://{workspace_id}/{task_ref}",
            surface=_surface_for_task_ref(task_ref, workspace_id),
            purpose="Review the mirrored browser surface or local document state",
            expected_signal="required seeded context inspected",
            verification_checks=["surface inspected", "required fields visible"],
            validation_rules=["seeded fields visible"],
            state_updates={"surface_state": "inspected"},
            captured_fields=_captured_fields_for_surface(task_ref, workspace_id),
            state_machine_id=f"{workspace_id}_seeded_flow",
            transition_id=f"{task_ref}_context_inspected",
            from_state="workspace_opened",
            to_state="seeded_context_inspected",
        ),
    ]
    if "jobs" in workspace_id:
        plan.append(
            BrowserStep(
                action="validate_seeded_form",
                target=f"workspace://{workspace_id}/form-validation/{task_ref}",
                surface="job_portal",
                purpose="Validate the seeded form before any submission attempt",
                expected_signal="form validated with all required fields",
                verification_checks=["validation run complete", "required fields complete"],
                validation_rules=["resume attached", "cover letter attached", "required fields complete"],
                state_updates={"form_state": "validated"},
                captured_fields=["candidate_name", "job_title", "validation_status"],
                state_machine_id=f"{workspace_id}_seeded_flow",
                transition_id=f"{task_ref}_form_validated",
                from_state="seeded_context_inspected",
                to_state="form_validated",
            )
        )
    if "finance" in workspace_id:
        plan.append(
            BrowserStep(
                action="validate_seeded_model",
                target=f"workspace://{workspace_id}/model-validation/{task_ref}",
                surface="spreadsheet",
                purpose="Cross-check the seeded model before recommendation or export",
                expected_signal="model cross-check complete",
                verification_checks=["formula check complete", "evidence rows present"],
                validation_rules=["formula rows present", "evidence rows present"],
                state_updates={"model_state": "validated"},
                captured_fields=["metric", "formula", "evidence"],
                state_machine_id=f"{workspace_id}_seeded_flow",
                transition_id=f"{task_ref}_model_validated",
                from_state="seeded_context_inspected",
                to_state="model_validated",
            )
        )
    if "think_013_prod_approval_escalation" in task_ref or "approval" in task_ref:
        plan.append(
            BrowserStep(
                action="record_approval_gate",
                target=f"workspace://{workspace_id}/approval-gate/{task_ref}",
                surface="document",
                purpose="Record that approval is still required before proceeding",
                expected_signal="approval gate recorded without external side effects",
                verification_checks=["approval status read", "send blocked"],
                validation_rules=["approval status is pending"],
                state_updates={"approval_state": "pending"},
                submission_gate="approval_required",
                blocked_reason="required approval is still missing",
                captured_fields=["approval_state", "decision"],
                state_machine_id=f"{workspace_id}_seeded_flow",
                transition_id=f"{task_ref}_approval_pending",
                from_state="seeded_context_inspected",
                to_state="approval_pending",
            )
        )
    return plan


def _collect_state_machines(stages: list[EpisodeStage]) -> list[BrowserStateMachine]:
    grouped: dict[str, dict[str, object]] = {}
    for stage in stages:
        for step in stage.browser_plan:
            if not step.state_machine_id or not step.transition_id or not step.from_state or not step.to_state:
                continue
            bucket = grouped.setdefault(
                step.state_machine_id,
                {
                    "surface": step.surface,
                    "start_state": step.from_state,
                    "states": {},
                    "transitions": [],
                },
            )
            states = bucket["states"]
            assert isinstance(states, dict)
            states.setdefault(step.from_state, BrowserState(state_id=step.from_state, surface=step.surface, label=step.from_state.replace("_", " ")))
            terminal = step.submission_gate in {"blocked", "approval_required"} or step.to_state.endswith("blocked")
            states.setdefault(
                step.to_state,
                BrowserState(state_id=step.to_state, surface=step.surface, label=step.to_state.replace("_", " "), terminal=terminal),
            )
            transitions = bucket["transitions"]
            assert isinstance(transitions, list)
            transitions.append(
                BrowserTransition(
                    transition_id=step.transition_id,
                    action=step.action,
                    from_state=step.from_state,
                    to_state=step.to_state,
                    outcome="approval_required" if step.submission_gate == "approval_required" else ("blocked" if step.submission_gate == "blocked" else "pass"),
                    notes=step.purpose,
                )
            )
    machines: list[BrowserStateMachine] = []
    for machine_id, payload in grouped.items():
        states = payload["states"]
        transitions = payload["transitions"]
        assert isinstance(states, dict)
        assert isinstance(transitions, list)
        machines.append(
            BrowserStateMachine(
                machine_id=machine_id,
                surface=str(payload["surface"]),
                start_state=str(payload["start_state"]),
                states=list(states.values()),
                transitions=transitions,
            )
        )
    return machines


def _artifacts_for_stage(artifacts: list[ArtifactSpec], index: int, stage_count: int) -> list[str]:
    if stage_count <= 1:
        return [artifact.artifact_id for artifact in artifacts]
    if index < stage_count - 1:
        return [artifacts[index].artifact_id] if index < len(artifacts) else [artifacts[-1].artifact_id]
    remainder = artifacts[index:]
    return [artifact.artifact_id for artifact in remainder] or [artifacts[-1].artifact_id]


def _surface_for_task_ref(task_ref: str, workspace_id: str) -> str:
    if "agent_" in task_ref and "jobs" in workspace_id:
        return "job_portal"
    if "retr_" in task_ref and "finance" in workspace_id:
        return "data_room"
    if "exec" in workspace_id:
        return "email" if "think_" in task_ref or "retr_" in task_ref else "calendar"
    if "finance" in workspace_id:
        return "spreadsheet"
    if "jobs" in workspace_id:
        return "job_portal"
    return "document"


def _captured_fields_for_surface(task_ref: str, workspace_id: str) -> list[str]:
    if "jobs" in workspace_id:
        return ["candidate_name", "job_title", "submission_status"]
    if "finance" in workspace_id:
        return ["metric", "value", "evidence"]
    if "exec" in workspace_id:
        return ["subject", "time", "decision"]
    return [task_ref]


if __name__ == "__main__":
    main()
