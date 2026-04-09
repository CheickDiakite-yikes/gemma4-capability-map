from __future__ import annotations

from gemma4_capability_map.evals.agent_eval import score_full_stack_trace
from gemma4_capability_map.evals.retrieval_eval import score_retrieval_trace
from gemma4_capability_map.schemas import (
    Document,
    Domain,
    ExpectedEvent,
    HardwareProfile,
    Message,
    ModelBundleSpec,
    RealWorldProfile,
    RetrievalHit,
    RunTrace,
    ScoringProfile,
    StateTransition,
    Task,
    ToolSpec,
    ToolResult,
    Track,
)


def _hardware() -> HardwareProfile:
    return HardwareProfile(
        platform="Darwin",
        platform_version="test",
        machine="arm64",
        cpu_count=12,
        memory_gb=24.0,
    )


def test_real_world_full_stack_metrics_include_readiness_score() -> None:
    task = Task(
        task_id="agent_real_world_guard",
        track=Track.FULL_STACK,
        domain=Domain.REPO,
        user_goal="Patch the rollout config and explain why.",
        messages=[Message(role="user", content="Patch the rollout config and explain why.")],
        tool_specs=[
            ToolSpec(name="read_repo_file", description="Read a repo file.", schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
            ToolSpec(name="propose_patch", description="Record a repo patch.", schema={"type": "object", "properties": {"path": {"type": "string"}, "patch": {"type": "string"}}, "required": ["path", "patch"]}),
        ],
        expected_events=[
            ExpectedEvent(event_type="tool_call", tool_name="read_repo_file", arguments={"path": "config/settings.yaml"}),
            ExpectedEvent(event_type="tool_call", tool_name="propose_patch", arguments={"path": "config/settings.yaml", "patch": "safe_mode: true"}),
        ],
        expected_final_state={"proposed_patches": [{"path": "config/settings.yaml", "patch": "safe_mode: true"}]},
        expected_answer_contains=["safe_mode: true", "production"],
        scoring_profile=ScoringProfile(tool_match=True, arg_match=True, final_state_match=True, answer_match=True),
        benchmark_tags=["real_world", "release_ops"],
        real_world_profile=RealWorldProfile(
            job_role="release engineer",
            scenario="Record the safer rollout patch without human intervention.",
            autonomy_level="bounded_autonomy",
            risk_tier="high",
            time_budget_minutes=12,
            human_equivalent_minutes=10,
            requires_multistep_state=True,
            requires_recovery=True,
            success_invariants=["Patch record matches the target file."],
            failure_costs=["Unsafe rollout recommendation"],
        ),
    )
    trace = RunTrace(
        run_id="run_real_world_agent",
        task_id=task.task_id,
        variant_id="clean",
        track=task.track,
        architecture="modular",
        model_bundle=ModelBundleSpec(reasoner="google/gemma-4-E2B-it", router="google/functiongemma-270m-it"),
        backend="hf",
        hardware_profile=_hardware(),
        tool_steps=[
            ToolResult(
                step=1,
                selected_tool="read_repo_file",
                arguments={"path": "config/settings.yaml"},
                validator_result="pass",
                output={"content": "safe_mode: false"},
                state_after={"repo_files": {"config/settings.yaml": "safe_mode: false"}, "proposed_patches": []},
            ),
            ToolResult(
                step=2,
                selected_tool="propose_patch",
                arguments={"path": "config/settings.yaml", "patch": "safe_mode: true"},
                validator_result="pass",
                output={"recorded": True},
                state_after={"repo_files": {"config/settings.yaml": "safe_mode: false"}, "proposed_patches": [{"path": "config/settings.yaml", "patch": "safe_mode: true"}]},
            ),
        ],
        state_transitions=[
            StateTransition(step=1, tool_name="read_repo_file", before={}, after={"repo_files": {"config/settings.yaml": "safe_mode: false"}, "proposed_patches": []}, diff={}),
            StateTransition(step=2, tool_name="propose_patch", before={"repo_files": {"config/settings.yaml": "safe_mode: false"}, "proposed_patches": []}, after={"repo_files": {"config/settings.yaml": "safe_mode: false"}, "proposed_patches": [{"path": "config/settings.yaml", "patch": "safe_mode: true"}]}, diff={}),
        ],
        prompt_artifacts={
            "planning_latency_ms": [200, 150],
            "planning_prompt_tokens": [50, 40],
            "planning_completion_tokens": [10, 12],
            "final_latency_ms": 300,
            "final_prompt_tokens": 20,
            "final_completion_tokens": 18,
        },
        final_answer="Set safe_mode: true because production runbooks require the safer default.",
        benchmark_tags=task.benchmark_tags,
        real_world_profile=task.real_world_profile,
    )

    metrics = score_full_stack_trace(task, trace)
    assert metrics["success"] == 1.0
    assert metrics["state_integrity_score"] == 1.0
    assert metrics["intervention_free_success"] == 1.0
    assert metrics["collateral_damage_free"] == 1.0
    assert metrics["real_world_readiness_score"] == 1.0
    assert float(metrics["human_time_ratio"]) > 0.0


def test_real_world_retrieval_metrics_do_not_blame_retriever_for_answer_surface() -> None:
    task = Task(
        task_id="retr_real_world_guard",
        track=Track.RETRIEVAL,
        domain=Domain.DOCS,
        user_goal="What approval requirement is current for production deploys?",
        messages=[Message(role="user", content="What approval requirement is current for production deploys?")],
        corpora={
            "default": [
                Document(doc_id="doc_new", content="Current production policy: every production deploy requires two-person approval."),
                Document(doc_id="doc_old", content="Older policy: approval was required only for major launches."),
            ]
        },
        expected_events=[ExpectedEvent(event_type="retrieval", expected_doc_ids=["doc_new"])],
        expected_answer_contains=["two-person approval"],
        scoring_profile=ScoringProfile(retrieval_match=True, answer_match=True),
        benchmark_tags=["real_world", "release_ops"],
        real_world_profile=RealWorldProfile(
            job_role="release manager",
            scenario="Answer a production approval question from policy evidence.",
            autonomy_level="bounded_autonomy",
            risk_tier="high",
            time_budget_minutes=6,
            human_equivalent_minutes=5,
            requires_multilingual=True,
            success_invariants=["Current approval policy is cited."],
            failure_costs=["Governance violation"],
        ),
    )
    trace = RunTrace(
        run_id="run_real_world_retrieval",
        task_id=task.task_id,
        variant_id="language_fr",
        track=task.track,
        architecture="hybrid",
        model_bundle=ModelBundleSpec(reasoner="google/gemma-4-E2B-it", retriever="google/embeddinggemma-300m"),
        backend="hf",
        hardware_profile=_hardware(),
        retrieval_hits=[],
        prompt_artifacts={
            "retrieved_doc_ids": ["doc_new"],
            "final_latency_ms": 1500,
            "final_prompt_tokens": 120,
            "final_completion_tokens": 12,
        },
        final_answer="Une approbation manuelle est requise.",
        benchmark_tags=task.benchmark_tags,
        real_world_profile=task.real_world_profile,
    )
    trace.retrieval_hits = [
        RetrievalHit(doc_id="doc_new", content="Current production policy: every production deploy requires two-person approval.", score=0.99),
        RetrievalHit(doc_id="doc_old", content="Older policy: approval was required only for major launches.", score=0.3),
    ]

    metrics = score_retrieval_trace(task, trace)
    assert metrics["recall_at_k"] == 1.0
    assert metrics["evidence_hit_rate"] == 1.0
    assert metrics["answer_match"] == 0.0
    assert metrics["success"] == 0.0
    assert metrics["real_world_readiness_score"] < 1.0
    assert float(metrics["human_time_ratio"]) > 0.0
