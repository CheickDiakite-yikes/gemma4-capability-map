from __future__ import annotations

import csv
import json
from pathlib import Path

from gemma4_capability_map.reporting.knowledge_work_board import (
    build_board_rows,
    build_comparison_batch_rows,
    build_external_benchmark_rows,
    build_external_benchmark_summary,
    build_intent_comparison_rows,
    build_lane_summary_rows,
    build_leaderboard_rows,
    build_public_summary,
    build_runtime_profile_rows,
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
    assert row["board_status"] == "completed"
    assert row["coverage_ratio"] == 1.0
    assert row["comparison_tier"] == "appendix"
    assert row["publishable_default"] is False
    assert row["full_lane_complete"] == 0.0
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


def test_board_rows_discover_nested_matrix_runs(tmp_path: Path) -> None:
    manifest = {
        "run_group_id": "20260411T151500Z_live_web_stress",
        "created_at": "20260411T151500Z",
        "lane": "live_web_stress",
        "run_intent": "exploratory",
        "backend": "hf_service",
        "reasoner": "google/gemma-4-E2B-it",
        "router": "",
        "retriever": "",
        "reasoner_backend": "hf_service",
        "router_backend": "heuristic",
        "retriever_backend": "heuristic",
        "episode_count": 20,
        "episodes_path": str((ROOT / "data" / "knowledge_work" / "live_web_stress" / "episodes.jsonl").resolve()),
    }
    summary = {
        "runs": 20,
        "artifact_quality_avg": 0.98,
        "browser_workflow_avg": 1.0,
        "strict_interface_avg": 1.0,
        "recovered_execution_avg": 1.0,
        "real_world_readiness_avg": 0.96,
        "escalation_correctness_avg": 1.0,
    }
    leaderboard_rows = [
        {
            "run_id": "run-pass",
            "episode_id": "ep-pass",
            "role_family": "finance",
            "lane": "live_web_stress",
            "workspace_id": "ws-pass",
            "benchmark_tags": "knowledge_work_arena,live_web_stress,finance",
            "artifact_quality_score": 1.0,
            "browser_workflow_score": 1.0,
            "strict_interface_score": 1.0,
            "recovered_execution_score": 1.0,
            "revision_responsiveness": 1.0,
            "memory_retention_score": 1.0,
            "escalation_correctness": 1.0,
            "collateral_damage_free": 1.0,
            "human_time_ratio": 0.1,
            "role_readiness_score": 0.96,
        }
    ]
    matrix_run = tmp_path / "results" / "knowledge_work_matrix" / "20260411T151500Z_knowledge_work_full_lane" / "mlx_gemma4_e2b_reasoner_only__live_web_stress"
    _write_snapshot(matrix_run.parent, matrix_run.name, manifest, summary, leaderboard_rows)

    rows = build_board_rows(
        [tmp_path / "results" / "knowledge_work", tmp_path / "results" / "knowledge_work_matrix"],
        REGISTRY_PATH,
    )

    assert len(rows) == 1
    assert rows[0]["result_family"] == "knowledge_work_matrix"
    assert rows[0]["comparison_batch"] == "20260411T151500Z_knowledge_work_full_lane"
    assert rows[0]["run_scope"] == "full_lane"
    assert rows[0]["board_status"] == "completed"


def test_board_rows_surface_matrix_failures_without_summary_snapshot(tmp_path: Path) -> None:
    batch_root = tmp_path / "results" / "knowledge_work_matrix" / "20260411T160000Z_knowledge_work_full_lane"
    batch_root.mkdir(parents=True, exist_ok=True)
    (batch_root / "manifest.json").write_text(
        json.dumps(
            {
                "run_group_id": "20260411T160000Z",
                "matrix_name": "knowledge_work_full_lane",
                "created_at": "2026-04-11T16:00:00+00:00",
                "run_intent": "exploratory",
                "runs": [
                    {
                        "run_id": "mlx_gemma4_e2b_reasoner_only__replayable_core",
                        "system_id": "mlx_gemma4_e2b_reasoner_only",
                        "lane": "replayable_core",
                        "limit": 26,
                        "backend": "mlx",
                        "reasoner_backend": "mlx",
                        "router_backend": "heuristic",
                        "retriever_backend": "heuristic",
                        "reasoner": "google/gemma-4-E2B-it",
                        "router": "",
                        "retriever": "",
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (batch_root / "results.json").write_text(
        json.dumps(
            [
                {
                    "run_id": "mlx_gemma4_e2b_reasoner_only__replayable_core",
                    "system_id": "mlx_gemma4_e2b_reasoner_only",
                    "lane": "replayable_core",
                    "output_dir": str((batch_root / "mlx_gemma4_e2b_reasoner_only__replayable_core").resolve()),
                    "returncode": 1,
                    "stderr": "RuntimeError: Install the mlx extra to use the MLX backend.",
                }
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = build_board_rows(
        [tmp_path / "results" / "knowledge_work", tmp_path / "results" / "knowledge_work_matrix"],
        REGISTRY_PATH,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["system_id"] == "mlx_gemma4_e2b_reasoner_only"
    assert row["comparison_batch"] == "20260411T160000Z_knowledge_work_full_lane"
    assert row["board_status"] == "failed"
    assert row["failure_excerpt"] == "RuntimeError: Install the mlx extra to use the MLX backend."
    assert row["planned_runs"] == 26
    assert row["completed_runs_observed"] == 0


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
    assert (tmp_path / "knowledge_work_leaderboard.csv").exists()
    assert (tmp_path / "knowledge_work_scatter.csv").exists()
    assert (tmp_path / "knowledge_work_lane_summary.csv").exists()
    assert (tmp_path / "knowledge_work_runtime_profiles.csv").exists()
    assert (tmp_path / "knowledge_work_comparison_batches.csv").exists()
    assert (tmp_path / "knowledge_work_role_breakdown.csv").exists()
    assert (tmp_path / "knowledge_work_category_breakdown.csv").exists()
    assert (tmp_path / "knowledge_work_track_breakdown.csv").exists()
    assert (tmp_path / "knowledge_work_external_benchmarks.csv").exists()
    assert (tmp_path / "knowledge_work_external_benchmark_summary.json").exists()
    assert (tmp_path / "knowledge_work_public_summary.json").exists()
    with (tmp_path / "knowledge_work_board_latest.csv").open("r", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert "warmup_load_ms" in row
    assert "total_cost_per_mtok" in row
    with (tmp_path / "knowledge_work_leaderboard.csv").open("r", encoding="utf-8") as handle:
        leaderboard_row = next(csv.DictReader(handle))
    assert "rank" in leaderboard_row
    assert "pass_rate_pct" in leaderboard_row
    with (tmp_path / "knowledge_work_lane_summary.csv").open("r", encoding="utf-8") as handle:
        lane_summary_row = next(csv.DictReader(handle))
    assert "best_display_name" in lane_summary_row
    assert "systems" in lane_summary_row
    with (tmp_path / "knowledge_work_runtime_profiles.csv").open("r", encoding="utf-8") as handle:
        runtime_profile_row = next(csv.DictReader(handle))
    assert "lane_readiness_json" in runtime_profile_row
    assert "avg_readiness" in runtime_profile_row
    with (tmp_path / "knowledge_work_comparison_batches.csv").open("r", encoding="utf-8") as handle:
        comparison_batch_row = next(csv.DictReader(handle), None)
    assert comparison_batch_row is None
    public_summary = json.loads((tmp_path / "knowledge_work_public_summary.json").read_text(encoding="utf-8"))
    assert "highest_readiness_local_profile" in public_summary
    assert public_summary["latest_runs"] == 1


def test_external_benchmark_rows_and_summary_load_from_registry(tmp_path: Path) -> None:
    registry_path = tmp_path / "external_benchmarks.yaml"
    registry_path.write_text(
        """
benchmarks:
  - model_id: openai_gpt_5_4
    display_name: GPT-5.4
    provider: openai
    access: closed
    benchmark: SWE-Bench Pro
    benchmark_group: coding_agent
    score: 57.7
    score_unit: percent
    source_scope: published_external
    source_kind: official_release
    source_org: OpenAI
    published_date: "2026-03-05"
    source_url: "https://openai.com/index/introducing-gpt-5-4/"
    notes: "Official score."
  - model_id: google_gemini_3_1_pro
    display_name: Gemini 3.1 Pro
    provider: google
    access: closed
    benchmark: BrowseComp
    benchmark_group: agentic_search
    score: 85.9
    score_unit: percent
    source_scope: published_external
    source_kind: official_model_card
    source_org: Google DeepMind
    published_date: "2026-02-19"
    source_url: "https://deepmind.google/models/model-cards/gemini-3-1-pro/"
    notes: "Official score."
""".strip()
        + "\n",
        encoding="utf-8",
    )

    rows = build_external_benchmark_rows(registry_path)
    summary = build_external_benchmark_summary(rows)

    assert len(rows) == 2
    assert rows[0]["benchmark_group"] == "agentic_search"
    assert rows[0]["score_display"] == "85.9%"
    assert summary["row_count"] == 2
    assert summary["model_count"] == 2
    assert summary["latest_published_date"] == "2026-03-05"
    assert summary["providers"] == ["google", "openai"]


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


def test_board_latest_rows_prefer_completed_over_newer_partial_attempt() -> None:
    rows = [
        {
            "system_id": "hf_service_gemma4_e4b_reasoner_only",
            "display_name": "Gemma 4 E4B local",
            "lane": "replayable_core",
            "run_intent": "exploratory",
            "run_scope": "full_lane",
            "board_status": "completed",
            "coverage_ratio": 1.0,
            "created_at": "20260411T100000Z",
            "real_world_readiness_avg": 0.95,
            "strict_interface_avg": 1.0,
            "browser_workflow_avg": 1.0,
        },
        {
            "system_id": "hf_service_gemma4_e4b_reasoner_only",
            "display_name": "Gemma 4 E4B local",
            "lane": "replayable_core",
            "run_intent": "exploratory",
            "run_scope": "full_lane",
            "board_status": "partial",
            "coverage_ratio": 0.3,
            "created_at": "20260411T120000Z",
            "real_world_readiness_avg": 0.98,
            "strict_interface_avg": 1.0,
            "browser_workflow_avg": 1.0,
        },
    ]

    latest = latest_board_rows(rows)
    assert len(latest) == 1
    assert latest[0]["board_status"] == "completed"
    assert latest[0]["created_at"] == "20260411T100000Z"


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


def test_board_builds_comparison_batch_rows() -> None:
    rows = [
        {
            "system_id": "hf_service_gemma4_specialists_cpu",
            "display_name": "Gemma 4 + specialists local",
            "lane": "replayable_core",
            "run_intent": "exploratory",
            "run_scope": "full_lane",
            "comparison_batch": "20260411T151500Z_knowledge_work_full_lane",
            "result_family": "knowledge_work_matrix",
            "real_world_readiness_avg": 0.92,
            "strict_interface_avg": 1.0,
            "browser_workflow_avg": 0.97,
            "artifact_quality_avg": 0.98,
            "recovered_execution_avg": 1.0,
            "episode_count": 24,
            "pass_count": 22,
            "refine_count": 2,
            "fail_count": 0,
            "board_status": "completed",
            "completed_runs_observed": 24,
            "planned_runs": 24,
        },
        {
            "system_id": "hf_service_gemma4_reasoner_only",
            "display_name": "Gemma 4 reasoner local",
            "lane": "replayable_core",
            "run_intent": "exploratory",
            "run_scope": "full_lane",
            "comparison_batch": "20260411T151500Z_knowledge_work_full_lane",
            "result_family": "knowledge_work_matrix",
            "real_world_readiness_avg": 0.87,
            "strict_interface_avg": 0.94,
            "browser_workflow_avg": 0.91,
            "artifact_quality_avg": 0.92,
            "recovered_execution_avg": 0.88,
            "episode_count": 24,
            "pass_count": 18,
            "refine_count": 4,
            "fail_count": 2,
            "board_status": "timed_out",
            "completed_runs_observed": 7,
            "planned_runs": 24,
        },
    ]

    batch_rows = build_comparison_batch_rows(rows)
    assert len(batch_rows) == 1
    row = batch_rows[0]
    assert row["comparison_batch"] == "20260411T151500Z_knowledge_work_full_lane"
    assert row["systems"] == 2
    assert row["completed_systems"] == 1
    assert row["timed_out_systems"] == 1
    assert row["best_display_name"] == "Gemma 4 + specialists local"
    assert row["best_readiness"] == 0.92
    assert row["total_episodes"] == 48
    assert row["observed_episodes"] == 31


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


def test_leaderboard_lane_summary_runtime_profiles_and_public_summary() -> None:
    rows = [
        {
            "system_id": "hf_service_gemma4_specialists_cpu",
            "display_name": "Gemma 4 + specialists local",
            "short_label": "Gemma 4 specialists",
            "lane": "replayable_core",
            "run_intent": "exploratory",
            "run_scope": "full_lane",
            "coverage_ratio": 1.0,
            "local": True,
                "provider": "huggingface",
                "comparison_tier": "appendix",
                "publishable_default": False,
                "capability_family": "specialist_stack",
            "executor_mode": "local_specialists",
            "modality": "multimodal",
            "episode_count": 24,
            "pass_count": 22,
            "refine_count": 2,
            "fail_count": 0,
            "artifact_quality_avg": 0.99,
            "browser_workflow_avg": 0.98,
            "strict_interface_avg": 1.0,
            "recovered_execution_avg": 1.0,
            "real_world_readiness_avg": 0.95,
            "escalation_correctness_avg": 1.0,
            "total_params_b": 27.57,
            "warmup_load_ms": 1200.0,
            "last_request_elapsed_ms": 420.0,
            "requests_completed": 24.0,
            "total_cost_per_mtok": 0.0,
            "created_at": "20260411T100000Z",
        },
        {
            "system_id": "hf_service_gemma4_specialists_cpu",
            "display_name": "Gemma 4 + specialists local",
            "short_label": "Gemma 4 specialists",
            "lane": "live_web_stress",
            "run_intent": "exploratory",
            "run_scope": "full_lane",
            "coverage_ratio": 1.0,
            "local": True,
                "provider": "huggingface",
                "comparison_tier": "appendix",
                "publishable_default": False,
                "capability_family": "specialist_stack",
            "executor_mode": "local_specialists",
            "modality": "multimodal",
            "episode_count": 18,
            "pass_count": 18,
            "refine_count": 0,
            "fail_count": 0,
            "artifact_quality_avg": 1.0,
            "browser_workflow_avg": 1.0,
            "strict_interface_avg": 1.0,
            "recovered_execution_avg": 1.0,
            "real_world_readiness_avg": 0.97,
            "escalation_correctness_avg": 1.0,
            "total_params_b": 27.57,
            "warmup_load_ms": 1300.0,
            "last_request_elapsed_ms": 390.0,
            "requests_completed": 18.0,
            "total_cost_per_mtok": 0.0,
            "created_at": "20260411T101000Z",
        },
        {
            "system_id": "hf_gemma4_e2b_reasoner_only",
            "display_name": "Gemma 4 reasoner only",
            "short_label": "Gemma 4 reasoner",
            "lane": "replayable_core",
            "run_intent": "exploratory",
            "run_scope": "full_lane",
            "coverage_ratio": 1.0,
            "local": True,
                "provider": "huggingface",
                "comparison_tier": "control",
                "publishable_default": True,
                "capability_family": "reasoner_only",
            "executor_mode": "local_reasoner",
            "modality": "multimodal",
            "episode_count": 24,
            "pass_count": 20,
            "refine_count": 2,
            "fail_count": 2,
            "artifact_quality_avg": 0.97,
            "browser_workflow_avg": 0.96,
            "strict_interface_avg": 0.94,
            "recovered_execution_avg": 0.92,
            "real_world_readiness_avg": 0.93,
            "escalation_correctness_avg": 0.98,
            "total_params_b": 27.0,
            "warmup_load_ms": 1000.0,
            "last_request_elapsed_ms": 310.0,
            "requests_completed": 24.0,
            "total_cost_per_mtok": 0.0,
            "created_at": "20260411T102000Z",
        },
    ]

    leaderboard = build_leaderboard_rows(rows)
    replayable_leader = next(
        row for row in leaderboard
        if row["display_name"] == "Gemma 4 + specialists local" and row["lane"] == "replayable_core"
    )
    assert replayable_leader["rank"] == 1
    assert replayable_leader["pass_rate_pct"] == round((22 / 24) * 100, 2)

    lane_summary = build_lane_summary_rows(rows)
    replayable_summary = next(row for row in lane_summary if row["lane"] == "replayable_core")
    assert replayable_summary["systems"] == 2
    assert replayable_summary["avg_coverage"] == 1.0
    assert replayable_summary["best_display_name"] == "Gemma 4 + specialists local"

    runtime_profiles = build_runtime_profile_rows(rows)
    specialist_profile = next(row for row in runtime_profiles if row["system_id"] == "hf_service_gemma4_specialists_cpu")
    assert specialist_profile["lane_count"] == 2
    assert specialist_profile["best_lane"] == "live_web_stress"
    assert specialist_profile["comparison_tier"] == "appendix"
    assert json.loads(specialist_profile["lane_readiness_json"])["live_web_stress"] == 0.97

    public_summary = build_public_summary(rows)
    assert public_summary["latest_runs"] == 3
    assert public_summary["comparison_health"]["completed"] == 3
    assert public_summary["publishable_profiles"] == 1
    assert public_summary["highest_readiness_local_profile"]["system_id"] == "hf_service_gemma4_specialists_cpu"
    assert public_summary["fastest_local_profile"]["metric"] == "last_request_elapsed_ms"
