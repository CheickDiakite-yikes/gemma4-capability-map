from __future__ import annotations

import csv
import json
from pathlib import Path

from gemma4_capability_map.reporting.knowledge_work_board import (
    build_board_rows,
    build_intent_comparison_rows,
    latest_board_rows,
    load_model_registry,
    write_board_exports,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "model_registry.yaml"


def _write_snapshot(
    root: Path,
    name: str,
    manifest: dict,
    summary: dict,
    leaderboard_rows: list[dict[str, object]],
) -> None:
    target = root / name
    target.mkdir(parents=True, exist_ok=True)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (target / "episode_leaderboard.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "episode_id",
                "role_family",
                "lane",
                "workspace_id",
                "benchmark_tags",
                "artifact_quality_score",
                "browser_workflow_score",
                "strict_interface_score",
                "recovered_execution_score",
                "revision_responsiveness",
                "memory_retention_score",
                "escalation_correctness",
                "collateral_damage_free",
                "human_time_ratio",
                "role_readiness_score",
            ],
        )
        writer.writeheader()
        for row in leaderboard_rows:
            writer.writerow(row)


def test_board_rows_join_registry_and_compute_pass_refine_fail(tmp_path: Path) -> None:
    manifest = {
        "run_group_id": "20260410T150000Z_replayable_core",
        "created_at": "20260410T150000Z",
        "lane": "replayable_core",
        "run_intent": "exploratory",
        "backend": "hf_service",
        "reasoner": "google/gemma-4-E2B-it",
        "router": "google/functiongemma-270m-it",
        "retriever": "google/embeddinggemma-300m",
        "reasoner_backend": "hf_service",
        "router_backend": "hf",
        "retriever_backend": "hf",
        "episode_count": 3,
        "warmup": {
            "reasoner": {
                "service": {
                    "load_elapsed_ms": 34567,
                    "requests_completed": 9,
                }
            }
        },
        "runtime_bundle": {
            "reasoner": {
                "service_state": {
                    "last_request_elapsed_ms": 812,
                    "requests_completed": 11,
                }
            }
        },
    }
    summary = {
        "runs": 3,
        "artifact_quality_avg": 1.0,
        "browser_workflow_avg": 0.95,
        "strict_interface_avg": 0.8,
        "recovered_execution_avg": 0.8,
        "real_world_readiness_avg": 0.85,
        "escalation_correctness_avg": 1.0,
    }
    leaderboard_rows = [
        {
            "run_id": "run-pass",
            "episode_id": "ep-pass",
            "role_family": "finance",
            "lane": "replayable_core",
            "workspace_id": "ws-pass",
            "benchmark_tags": "knowledge_work_arena,replayable_core,finance",
            "artifact_quality_score": 1.0,
            "browser_workflow_score": 1.0,
            "strict_interface_score": 1.0,
            "recovered_execution_score": 1.0,
            "revision_responsiveness": 0.0,
            "memory_retention_score": 1.0,
            "escalation_correctness": 1.0,
            "collateral_damage_free": 1.0,
            "human_time_ratio": 0.1,
            "role_readiness_score": 0.9,
        },
        {
            "run_id": "run-refine",
            "episode_id": "ep-refine",
            "role_family": "finance",
            "lane": "replayable_core",
            "workspace_id": "ws-refine",
            "benchmark_tags": "knowledge_work_arena,replayable_core,finance",
            "artifact_quality_score": 1.0,
            "browser_workflow_score": 1.0,
            "strict_interface_score": 0.4,
            "recovered_execution_score": 1.0,
            "revision_responsiveness": 0.0,
            "memory_retention_score": 1.0,
            "escalation_correctness": 1.0,
            "collateral_damage_free": 1.0,
            "human_time_ratio": 0.1,
            "role_readiness_score": 0.8,
        },
        {
            "run_id": "run-fail",
            "episode_id": "ep-fail",
            "role_family": "finance",
            "lane": "replayable_core",
            "workspace_id": "ws-fail",
            "benchmark_tags": "knowledge_work_arena,replayable_core,finance",
            "artifact_quality_score": 1.0,
            "browser_workflow_score": 0.85,
            "strict_interface_score": 0.2,
            "recovered_execution_score": 0.0,
            "revision_responsiveness": 0.0,
            "memory_retention_score": 1.0,
            "escalation_correctness": 1.0,
            "collateral_damage_free": 0.0,
            "human_time_ratio": 0.1,
            "role_readiness_score": 0.5,
        },
    ]
    _write_snapshot(tmp_path, "model_backed_hf_specialists_test", manifest, summary, leaderboard_rows)

    rows = build_board_rows(tmp_path, REGISTRY_PATH)
    assert len(rows) == 1
    row = rows[0]
    assert row["system_id"] == "hf_service_gemma4_specialists_cpu"
    assert row["display_name"] == "Gemma 4 E2B + FunctionGemma + EmbeddingGemma (HF local)"
    assert row["pass_count"] == 1
    assert row["refine_count"] == 1
    assert row["fail_count"] == 1
    assert row["local"] is True
    assert row["total_params_b"] == 27.57
    assert row["capability_family"] == "specialist_stack"
    assert row["executor_mode"] == "local_specialists"
    assert row["modality"] == "multimodal"
    assert row["warmup_load_ms"] == 34567.0
    assert row["last_request_elapsed_ms"] == 812.0
    assert row["requests_completed"] == 11.0
    assert row["total_cost_per_mtok"] == 0.0
    role_breakdown = json.loads(row["role_breakdown_json"])
    assert role_breakdown["finance"]["pass_count"] == 1.0
    assert role_breakdown["finance"]["refine_count"] == 1.0
    assert role_breakdown["finance"]["fail_count"] == 1.0


def test_board_rows_fallback_to_registry_when_manifest_system_id_is_unknown(tmp_path: Path) -> None:
    manifest = {
        "run_group_id": "20260410T150500Z_replayable_core",
        "created_at": "20260410T150500Z",
        "lane": "replayable_core",
        "run_intent": "exploratory",
        "system_id": "model_backed_hf_reasoner_full",
        "backend": "hf_service",
        "reasoner": "google/gemma-4-E2B-it",
        "router": "google/functiongemma-270m-it",
        "retriever": "google/embeddinggemma-300m",
        "reasoner_backend": "hf_service",
        "router_backend": "heuristic",
        "retriever_backend": "heuristic",
        "episode_count": 1,
    }
    summary = {
        "runs": 1,
        "artifact_quality_avg": 1.0,
        "browser_workflow_avg": 1.0,
        "strict_interface_avg": 1.0,
        "recovered_execution_avg": 1.0,
        "real_world_readiness_avg": 0.95,
        "escalation_correctness_avg": 1.0,
    }
    leaderboard_rows = [
        {
            "run_id": "run-pass",
            "episode_id": "ep-pass",
            "role_family": "finance",
            "lane": "replayable_core",
            "workspace_id": "ws-pass",
            "benchmark_tags": "knowledge_work_arena,replayable_core,finance",
            "artifact_quality_score": 1.0,
            "browser_workflow_score": 1.0,
            "strict_interface_score": 1.0,
            "recovered_execution_score": 1.0,
            "revision_responsiveness": 1.0,
            "memory_retention_score": 1.0,
            "escalation_correctness": 1.0,
            "collateral_damage_free": 1.0,
            "human_time_ratio": 0.1,
            "role_readiness_score": 0.95,
        }
    ]
    _write_snapshot(tmp_path, "model_backed_hf_reasoner_full_replayable_v1", manifest, summary, leaderboard_rows)

    rows = build_board_rows(tmp_path, REGISTRY_PATH)
    assert len(rows) == 1
    row = rows[0]
    assert row["system_id"] == "hf_service_gemma4_reasoner_only"
    assert row["capability_family"] == "reasoner_only"
    assert row["executor_mode"] == "local_reasoner"


def test_board_rows_alias_legacy_slice_specific_system_id_to_generic_stack(tmp_path: Path) -> None:
    manifest = {
        "run_group_id": "20260410T151000Z_replayable_core",
        "created_at": "20260410T151000Z",
        "lane": "replayable_core",
        "run_intent": "exploratory",
        "system_id": "hf_specialists_cross_role_hardmix_visual",
        "backend": "hf_service",
        "reasoner": "google/gemma-4-E2B-it",
        "router": "google/functiongemma-270m-it",
        "retriever": "google/embeddinggemma-300m",
        "reasoner_backend": "hf_service",
        "router_backend": "hf",
        "retriever_backend": "hf",
        "episode_count": 24,
    }
    summary = {
        "runs": 24,
        "artifact_quality_avg": 0.98,
        "browser_workflow_avg": 0.99,
        "strict_interface_avg": 1.0,
        "recovered_execution_avg": 1.0,
        "real_world_readiness_avg": 0.95,
        "escalation_correctness_avg": 1.0,
    }
    leaderboard_rows = [
        {
            "run_id": "run-pass",
            "episode_id": "ep-pass",
            "role_family": "finance",
            "lane": "replayable_core",
            "workspace_id": "ws-pass",
            "benchmark_tags": "knowledge_work_arena,replayable_core,finance",
            "artifact_quality_score": 1.0,
            "browser_workflow_score": 1.0,
            "strict_interface_score": 1.0,
            "recovered_execution_score": 1.0,
            "revision_responsiveness": 1.0,
            "memory_retention_score": 1.0,
            "escalation_correctness": 1.0,
            "collateral_damage_free": 1.0,
            "human_time_ratio": 0.1,
            "role_readiness_score": 0.95,
        }
    ]
    _write_snapshot(tmp_path, "legacy-specialists", manifest, summary, leaderboard_rows)

    rows = build_board_rows(tmp_path, REGISTRY_PATH)
    assert len(rows) == 1
    row = rows[0]
    assert row["system_id"] == "hf_service_gemma4_specialists_cpu"
    assert row["capability_family"] == "specialist_stack"
    assert row["executor_mode"] == "local_specialists"


def test_board_latest_rows_keep_newest_snapshot_per_system_lane_and_intent(tmp_path: Path) -> None:
    registry = load_model_registry(REGISTRY_PATH)
    assert "hf_service_gemma4_specialists_cpu" in registry["systems"]
    rows = [
        {
            "system_id": "hf_service_gemma4_specialists_cpu",
            "display_name": "Gemma 4 + specialists local",
            "lane": "replayable_core",
            "run_intent": "exploratory",
            "run_scope": "full_lane",
            "created_at": "20260410T100000Z",
            "real_world_readiness_avg": 0.9,
            "strict_interface_avg": 1.0,
            "browser_workflow_avg": 1.0,
        },
        {
            "system_id": "hf_service_gemma4_specialists_cpu",
            "display_name": "Gemma 4 + specialists local",
            "lane": "replayable_core",
            "run_intent": "exploratory",
            "run_scope": "full_lane",
            "created_at": "20260410T110000Z",
            "real_world_readiness_avg": 0.91,
            "strict_interface_avg": 1.0,
            "browser_workflow_avg": 1.0,
        },
    ]
    latest = latest_board_rows(rows)
    assert len(latest) == 1
    assert latest[0]["created_at"] == "20260410T110000Z"

    payload = write_board_exports(rows, tmp_path)
    assert payload["latest_row_count"] == 1
    assert (tmp_path / "knowledge_work_board_runs.csv").exists()
    assert (tmp_path / "knowledge_work_board_latest.csv").exists()
    assert (tmp_path / "knowledge_work_scatter.csv").exists()
    assert (tmp_path / "knowledge_work_role_breakdown.csv").exists()
    assert (tmp_path / "knowledge_work_category_breakdown.csv").exists()
    assert (tmp_path / "knowledge_work_track_breakdown.csv").exists()
    with (tmp_path / "knowledge_work_board_latest.csv").open("r", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert "warmup_load_ms" in row
    assert "total_cost_per_mtok" in row


def test_board_latest_rows_prefer_full_lane_over_newer_subset() -> None:
    rows = [
        {
            "system_id": "hf_service_gemma4_specialists_cpu",
            "display_name": "Gemma 4 + specialists local",
            "lane": "replayable_core",
            "run_intent": "exploratory",
            "run_scope": "full_lane",
            "created_at": "20260410T100000Z",
            "real_world_readiness_avg": 0.95,
            "strict_interface_avg": 1.0,
            "browser_workflow_avg": 0.99,
        },
        {
            "system_id": "hf_service_gemma4_specialists_cpu",
            "display_name": "Gemma 4 + specialists local",
            "lane": "replayable_core",
            "run_intent": "exploratory",
            "run_scope": "subset",
            "created_at": "20260410T120000Z",
            "real_world_readiness_avg": 0.92,
            "strict_interface_avg": 1.0,
            "browser_workflow_avg": 0.99,
        },
    ]

    latest = latest_board_rows(rows)
    assert len(latest) == 1
    assert latest[0]["run_scope"] == "full_lane"
    assert latest[0]["created_at"] == "20260410T100000Z"


def test_board_builds_canonical_vs_exploratory_comparison_rows() -> None:
    rows = [
        {
            "system_id": "hf_service_gemma4_specialists_cpu",
            "display_name": "Gemma 4 + specialists local",
            "lane": "replayable_core",
            "run_intent": "canonical",
            "run_group_id": "canonical-1",
            "capability_family": "specialist_stack",
            "executor_mode": "local_specialists",
            "modality": "multimodal",
            "real_world_readiness_avg": 0.9,
            "strict_interface_avg": 1.0,
            "browser_workflow_avg": 0.95,
            "episode_count": 24,
            "pass_count": 20,
            "refine_count": 4,
            "fail_count": 0,
        },
        {
            "system_id": "hf_service_gemma4_specialists_cpu",
            "display_name": "Gemma 4 + specialists local",
            "lane": "replayable_core",
            "run_intent": "exploratory",
            "run_group_id": "exploratory-1",
            "capability_family": "specialist_stack",
            "executor_mode": "local_specialists",
            "modality": "multimodal",
            "real_world_readiness_avg": 0.94,
            "strict_interface_avg": 1.0,
            "browser_workflow_avg": 0.97,
            "episode_count": 24,
            "pass_count": 22,
            "refine_count": 2,
            "fail_count": 0,
        },
    ]

    comparison = build_intent_comparison_rows(rows)
    assert len(comparison) == 1
    row = comparison[0]
    assert row["system_id"] == "hf_service_gemma4_specialists_cpu"
    assert row["display_name"] == "Gemma 4 + specialists local"
    assert row["canonical_readiness"] == 0.9
    assert row["exploratory_readiness"] == 0.94
    assert row["readiness_delta"] == 0.04
    assert row["canonical_strict_interface"] == 1.0
    assert row["exploratory_strict_interface"] == 1.0
    assert row["browser_delta"] == 0.02


def test_board_rows_match_registry_for_direct_hf_reasoner_only(tmp_path: Path) -> None:
    manifest = {
        "run_group_id": "20260410T160000Z_replayable_core",
        "created_at": "20260410T160000Z",
        "lane": "replayable_core",
        "run_intent": "exploratory",
        "backend": "hf",
        "reasoner": "google/gemma-4-E2B-it",
        "router": "google/functiongemma-270m-it",
        "retriever": "google/embeddinggemma-300m",
        "reasoner_backend": "hf",
        "router_backend": "heuristic",
        "retriever_backend": "heuristic",
        "episode_count": 24,
    }
    summary = {
        "runs": 24,
        "artifact_quality_avg": 0.97,
        "browser_workflow_avg": 0.99,
        "strict_interface_avg": 1.0,
        "recovered_execution_avg": 1.0,
        "real_world_readiness_avg": 0.94,
        "escalation_correctness_avg": 1.0,
    }
    leaderboard_rows = [
        {
            "run_id": "run-pass",
            "episode_id": "ep-pass",
            "role_family": "finance",
            "lane": "replayable_core",
            "workspace_id": "ws-pass",
            "benchmark_tags": "knowledge_work_arena,replayable_core,finance",
            "artifact_quality_score": 1.0,
            "browser_workflow_score": 1.0,
            "strict_interface_score": 1.0,
            "recovered_execution_score": 1.0,
            "revision_responsiveness": 1.0,
            "memory_retention_score": 1.0,
            "escalation_correctness": 1.0,
            "collateral_damage_free": 1.0,
            "human_time_ratio": 0.1,
            "role_readiness_score": 0.94,
        }
    ]
    _write_snapshot(tmp_path, "model_backed_hf_inprocess_reasoner_full_replayable_v1", manifest, summary, leaderboard_rows)

    rows = build_board_rows(tmp_path, REGISTRY_PATH)
    assert len(rows) == 1
    row = rows[0]
    assert row["system_id"] == "hf_gemma4_e2b_reasoner_only"
    assert row["capability_family"] == "reasoner_only"
    assert row["executor_mode"] == "local_reasoner"
    assert row["provider"] == "huggingface"
