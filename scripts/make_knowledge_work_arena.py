from __future__ import annotations

from pathlib import Path
import json

from gemma4_capability_map.io import dump_jsonl
from gemma4_capability_map.knowledge_work.schemas import (
    ArtifactScoringContract,
    ArtifactSpec,
    ArtifactKind,
    BenchmarkLane,
    BrowserStep,
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
                    "workspaces/jobs-tailored-packet/application_packet.md",
                    ["packet", "requirements"],
                    ["## Brief", "## Form Fields", "## Response Summary"],
                    required_field_pairs={"Submission Mode": "dry run", "Packet Status": "tailored"},
                    minimum_citations=1,
                ),
                _artifact(
                    "cover_letter",
                    ArtifactKind.EMAIL,
                    "workspaces/jobs-tailored-packet/cover_letter.md",
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
                    "workspaces/jobs-browser-form/form_submission.md",
                    ["submission", "consistent"],
                    ["## Form Fields", "## Response Summary"],
                    required_field_pairs={"Submission Mode": "dry run", "Constraint Memory": "preserved", "Validation": "consistent"},
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
                    "workspaces/jobs-revise-feedback/revised_packet.md",
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
                    "workspaces/jobs-tracker-followup/application_tracker.md",
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
                    "workspaces/finance-three-statement/financial_model.md",
                    ["marketing", "delta"],
                    ["## Table", "## Notes"],
                    required_table_rows=[["Marketing increase", "15%", "retr_001_budget_delta"], ["Revenue delta", "20000", "agent_001_budget_compare"]],
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
                    "workspaces/finance-comps-snapshot/comps_table.md",
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
                    "workspaces/finance-ic-memo/ic_memo.md",
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
                    "workspaces/finance-partner-deck/partner_deck.md",
                    ["invoice lock", "safe_mode"],
                    ["## Brief"],
                    required_slide_titles=["Situation", "Risk", "Recommendation"],
                    required_bullets=["invoice lock", "safe_mode"],
                    minimum_citations=2,
                ),
            ],
            review_rounds=[_review("partner_deck_review", "partner_deck", "Reduce clutter and make the recommendation slide sharper.", ["recommendation", "cleaner"])],
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
            "kwa_jobs_live_requirements_extract",
            RoleFamily.JOB_APPLICATION_OPS,
            BenchmarkLane.LIVE_WEB_STRESS,
            "jobs-live-requirements",
            "Extract requirements from live job pages and build a tailored application packet without submitting.",
            ["tool_010_validator_ready"],
            [_artifact("live_application_packet", ArtifactKind.FORM_SUBMISSION, "workspaces/jobs-live-requirements/live_application_packet.md", ["requirements"], ["## Form Fields", "## Response Summary"], required_field_pairs={"Submission Mode": "dry run", "Packet Status": "tailored"}, minimum_citations=2)],
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
            "kwa_finance_live_earnings_update",
            RoleFamily.FINANCE,
            BenchmarkLane.LIVE_WEB_STRESS,
            "finance-live-earnings",
            "Pull current public earnings materials and write an updated finance note.",
            ["retr_005_revenue_conflict"],
            [_artifact("earnings_update", ArtifactKind.RESEARCH_NOTE, "workspaces/finance-live-earnings/earnings_update.md", ["update"], ["## Recommendation", "## Output"], required_bullets=["update"], minimum_citations=2)],
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
                _artifact("live_comps_model", ArtifactKind.MODEL, "workspaces/finance-live-comps/live_comps_model.md", ["revision"], ["## Table"], required_table_rows=[["Revision", "revision", "agent_001_budget_compare"]], minimum_citations=2),
                _artifact("live_comps_note", ArtifactKind.RESEARCH_NOTE, "workspaces/finance-live-comps/live_comps_note.md", ["comps"], ["## Recommendation", "## Output"], required_bullets=["comps"], minimum_citations=2),
            ],
            dry_run_only=True,
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
            browser_plan=_default_browser_plan(lane, workspace_id, task_ref, index),
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
    required_slide_titles: list[str] | None = None,
    required_bullets: list[str] | None = None,
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
            required_slide_titles=required_slide_titles or [],
            required_bullets=required_bullets or [],
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
            target = golden_dir / f"{episode.episode_id}__{artifact.artifact_id}.md"
            lines = [f"# Golden {artifact.artifact_id}", "## Brief", episode.brief]
            if artifact.scoring_contract.required_slide_titles:
                for title in artifact.scoring_contract.required_slide_titles:
                    lines.append(f"## Slide: {title}")
                    for bullet in artifact.scoring_contract.required_bullets or artifact.scoring_contract.required_fragments:
                        lines.append(f"- {bullet}")
            elif artifact.scoring_contract.required_table_rows:
                lines.extend(["## Table", "| Metric | Value | Evidence |", "| --- | --- | --- |"])
                for row in artifact.scoring_contract.required_table_rows:
                    padded = row + [""] * max(0, 3 - len(row))
                    lines.append(f"| {padded[0]} | {padded[1]} | {padded[2]} |")
            elif artifact.scoring_contract.required_field_pairs:
                lines.append("## Form Fields")
                for field, value in artifact.scoring_contract.required_field_pairs.items():
                    lines.append(f"{field}: {value}")
            lines.extend(["## Output", " ".join(artifact.scoring_contract.required_fragments or ["golden"])])
            for index in range(max(artifact.scoring_contract.minimum_citations, 0)):
                lines.append(f"Source: seeded-{index + 1}")
            target.write_text("\n".join(lines), encoding="utf-8")


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
                        {"subject": "Board prep", "status": "needs brief"},
                        {"subject": "Vendor ambiguity", "status": "needs clarification"},
                    ]
                },
                indent=2,
            )
            + "\n",
            "calendar/events.json": json.dumps(
                {
                    "events": [
                        {"title": "Board meeting", "state": "scheduled"},
                        {"title": "Vendor sync", "state": "ambiguous"},
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
                },
                indent=2,
            )
            + "\n",
            "documents/resume.md": "# Resume\n\nCandidate experience and achievements.\n",
        }
    return {
        "data_room/filings.json": json.dumps(
            {
                "documents": [
                    {"name": "earnings_release.pdf", "status": "seeded"},
                    {"name": "management_commentary.md", "status": "seeded"},
                ]
            },
            indent=2,
        )
        + "\n",
        "models/template.csv": "metric,value,evidence\nrevenue,0,seeded\n",
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
                captured_fields=["title", "summary"],
            ),
            BrowserStep(
                action="capture_notes",
                target=f"live://{task_ref}#notes",
                surface="document",
                purpose="Extract facts into a dry-run working note",
                expected_signal="notes captured without side effects",
                verification_checks=["note captured", "no external mutation"],
                captured_fields=["fact", "source"],
            ),
            BrowserStep(
                action="prepare_sandbox_submission",
                target=f"sandbox://{workspace_id}/{task_ref}",
                surface="job_portal" if "jobs" in workspace_id else "document",
                purpose="Prepare a sandbox submission package without sending it to a production endpoint",
                expected_signal="sandbox payload ready",
                verification_checks=["sandbox target selected", "submission blocked from production"],
                captured_fields=["payload_id", "dry_run"],
                sandbox_endpoint=f"https://sandbox.local/{workspace_id}/{task_ref}",
            ),
        ]
    return [
        BrowserStep(
            action="open_seeded_workspace",
            target=f"workspace://{workspace_id}",
            surface="workspace",
            purpose=f"Open the seeded workspace for stage {index + 1}",
            expected_signal="seeded workspace loaded",
            verification_checks=["workspace opened"],
        ),
        BrowserStep(
            action="inspect_seeded_surface",
            target=f"workspace://{workspace_id}/{task_ref}",
            surface=_surface_for_task_ref(task_ref, workspace_id),
            purpose="Review the mirrored browser surface or local document state",
            expected_signal="required seeded context inspected",
            verification_checks=["surface inspected", "required fields visible"],
            captured_fields=_captured_fields_for_surface(task_ref, workspace_id),
        ),
    ]


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
