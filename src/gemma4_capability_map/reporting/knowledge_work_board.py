from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_capability_map.io import load_yaml


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_ROOT = ROOT / "results" / "knowledge_work"
DEFAULT_MATRIX_RESULTS_ROOT = ROOT / "results" / "knowledge_work_matrix"
DEFAULT_RESULTS_ROOTS = (DEFAULT_RESULTS_ROOT, DEFAULT_MATRIX_RESULTS_ROOT)
DEFAULT_HISTORY_DIR = ROOT / "results" / "history"
DEFAULT_REGISTRY_PATH = ROOT / "configs" / "model_registry.yaml"
DEFAULT_EXTERNAL_BENCHMARKS_PATH = ROOT / "configs" / "external_benchmarks.yaml"
DEFAULT_COMMUNITY_SIGNALS_PATH = ROOT / "configs" / "community_signals.yaml"
SYSTEM_ID_ALIASES = {
    "hf_specialists_cross_role_hardmix_visual": "hf_service_gemma4_specialists_cpu",
    "model_backed_hf_reasoner_full": "hf_service_gemma4_reasoner_only",
}


def load_model_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, dict[str, Any]]:
    payload = load_yaml(path) or {}
    return {
        "models": payload.get("models", {}),
        "systems": payload.get("systems", {}),
    }


def load_external_benchmark_registry(
    path: str | Path = DEFAULT_EXTERNAL_BENCHMARKS_PATH,
) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    payload = load_yaml(target) or {}
    records = payload.get("benchmarks", [])
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def build_external_benchmark_rows(
    path: str | Path = DEFAULT_EXTERNAL_BENCHMARKS_PATH,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in load_external_benchmark_registry(path):
        score = _first_number(record.get("score"))
        score_unit = str(record.get("score_unit", "percent") or "percent")
        score_display = f"{score:.1f}%" if score_unit == "percent" else f"{score:.1f}"
        rows.append(
            {
                "source_scope": str(record.get("source_scope", "published_external") or "published_external"),
                "source_kind": str(record.get("source_kind", "") or ""),
                "source_org": str(record.get("source_org", "") or ""),
                "model_id": str(record.get("model_id", "") or ""),
                "display_name": str(record.get("display_name", "") or ""),
                "provider": str(record.get("provider", "") or ""),
                "access": str(record.get("access", "") or ""),
                "benchmark": str(record.get("benchmark", "") or ""),
                "benchmark_group": str(record.get("benchmark_group", "") or ""),
                "score": score,
                "score_unit": score_unit,
                "score_display": score_display,
                "published_date": str(record.get("published_date", "") or ""),
                "source_url": str(record.get("source_url", "") or ""),
                "notes": str(record.get("notes", "") or ""),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("benchmark_group", "")),
            str(row.get("benchmark", "")),
            -float(row.get("score", 0.0) or 0.0),
            str(row.get("display_name", "")),
        ),
    )


def build_external_benchmark_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    providers = sorted({str(row.get("provider", "")) for row in rows if str(row.get("provider", ""))})
    benchmarks = sorted({str(row.get("benchmark", "")) for row in rows if str(row.get("benchmark", ""))})
    benchmark_groups = sorted({str(row.get("benchmark_group", "")) for row in rows if str(row.get("benchmark_group", ""))})
    models = sorted({str(row.get("model_id", "")) for row in rows if str(row.get("model_id", ""))})
    return {
        "row_count": len(rows),
        "model_count": len(models),
        "providers": providers,
        "benchmarks": benchmarks,
        "benchmark_groups": benchmark_groups,
        "latest_published_date": max((str(row.get("published_date", "")) for row in rows), default=""),
        "provider_counts": [
            {
                "provider": provider,
                "rows": sum(1 for row in rows if str(row.get("provider", "")) == provider),
                "models": len({str(row.get("model_id", "")) for row in rows if str(row.get("provider", "")) == provider}),
            }
            for provider in providers
        ],
        "benchmark_counts": [
            {
                "benchmark": benchmark,
                "rows": sum(1 for row in rows if str(row.get("benchmark", "")) == benchmark),
                "models": len({str(row.get("model_id", "")) for row in rows if str(row.get("benchmark", "")) == benchmark}),
                "best_score": max(
                    (float(row.get("score", 0.0) or 0.0) for row in rows if str(row.get("benchmark", "")) == benchmark),
                    default=0.0,
                ),
                "leader": max(
                    (
                        row
                        for row in rows
                        if str(row.get("benchmark", "")) == benchmark
                    ),
                    key=lambda row: float(row.get("score", 0.0) or 0.0),
                    default={},
                ).get("display_name", ""),
            }
            for benchmark in benchmarks
        ],
    }


def load_community_signal_registry(
    path: str | Path = DEFAULT_COMMUNITY_SIGNALS_PATH,
) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    payload = load_yaml(target) or {}
    records = payload.get("community_signals", payload.get("signals", []))
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def build_community_signal_rows(
    path: str | Path = DEFAULT_COMMUNITY_SIGNALS_PATH,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in load_community_signal_registry(path):
        status = str(record.get("status", "untriaged") or "untriaged").strip().lower()
        source = str(record.get("source", "") or record.get("source_org", "") or "").strip()
        source_date = str(record.get("source_date", "") or record.get("published_date", "") or "").strip()
        slice_name = str(record.get("benchmark_slice", "") or record.get("experiment", "") or "").strip()
        rows.append(
            {
                "claim": str(record.get("claim", "") or "").strip(),
                "source": source,
                "source_date": source_date,
                "source_url": str(record.get("source_url", "") or "").strip(),
                "why_it_matters": str(record.get("why_it_matters", "") or "").strip(),
                "moonie_hypothesis": str(record.get("moonie_hypothesis", "") or record.get("hypothesis", "") or "").strip(),
                "benchmark_slice": slice_name,
                "status": status if status in {"untriaged", "planned", "running", "answered"} else "untriaged",
                "notes": str(record.get("notes", "") or "").strip(),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            _community_signal_status_priority(row.get("status")),
            str(row.get("source_date", "")),
            str(row.get("source", "")),
            str(row.get("claim", "")),
        ),
    )


def build_community_signal_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = ["untriaged", "planned", "running", "answered"]
    status_counts = {status: sum(1 for row in rows if str(row.get("status", "")) == status) for status in statuses}
    slices = sorted({str(row.get("benchmark_slice", "")) for row in rows if str(row.get("benchmark_slice", ""))})
    sources = sorted({str(row.get("source", "")) for row in rows if str(row.get("source", ""))})
    return {
        "row_count": len(rows),
        "status_counts": status_counts,
        "benchmark_slices": slices,
        "sources": sources,
        "latest_source_date": max((str(row.get("source_date", "")) for row in rows), default=""),
        "planned_or_running": status_counts["planned"] + status_counts["running"],
        "answered": status_counts["answered"],
    }


def _is_completed_row(row: dict[str, Any]) -> bool:
    return str(row.get("board_status", "completed") or "completed") == "completed"


def build_board_rows(
    results_root: str | Path | list[str | Path] | tuple[str | Path, ...] = DEFAULT_RESULTS_ROOTS,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> list[dict[str, Any]]:
    registry = load_model_registry(registry_path)
    roots = _normalize_results_roots(results_root)
    matrix_records = _load_matrix_result_records(roots)
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for root, path in discover_run_dirs(roots):
        manifest_path = path / "manifest.json"
        summary_path = path / "summary.json"
        progress_path = path / "progress.json"
        leaderboard_path = path / "episode_leaderboard.csv"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        progress = _load_json(progress_path) or {}
        leaderboard_rows = _load_csv_rows(leaderboard_path)
        matrix_record = matrix_records.get(str(path.resolve()))
        rows.append(
            _snapshot_row(
                path,
                manifest,
                summary,
                leaderboard_rows,
                registry,
                source_root=root,
                progress=progress,
                matrix_record=matrix_record,
            )
        )
        seen_paths.add(str(path.resolve()))
    for output_dir, matrix_record in sorted(matrix_records.items()):
        if output_dir in seen_paths:
            continue
        path = Path(output_dir)
        manifest = _matrix_manifest(matrix_record)
        summary = _matrix_summary(matrix_record)
        progress = _matrix_progress(matrix_record)
        leaderboard_rows = _load_csv_rows(path / "episode_leaderboard.csv")
        rows.append(
            _snapshot_row(
                path,
                manifest,
                summary,
                leaderboard_rows,
                registry,
                source_root=matrix_record["source_root"],
                progress=progress,
                matrix_record=matrix_record,
            )
        )
    return rows


def discover_run_dirs(
    results_root: str | Path | list[str | Path] | tuple[str | Path, ...] = DEFAULT_RESULTS_ROOTS,
) -> list[tuple[Path, Path]]:
    roots = _normalize_results_roots(results_root)
    discovered: dict[Path, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for summary_path in root.rglob("summary.json"):
            run_dir = summary_path.parent
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            discovered.setdefault(run_dir.resolve(), root)
    return sorted(((root, path) for path, root in discovered.items()), key=lambda item: str(item[1]))


def _load_matrix_result_records(roots: list[Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for root in roots:
        if not root.exists():
            continue
        for results_path in root.rglob("results.json"):
            batch_dir = results_path.parent
            batch_manifest = _load_json(batch_dir / "manifest.json") or {}
            if not batch_manifest.get("matrix_name"):
                continue
            payload = _load_json(results_path)
            if not isinstance(payload, list):
                continue
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                output_dir_raw = str(entry.get("output_dir", "") or "").strip()
                output_dir = Path(output_dir_raw).resolve() if output_dir_raw else (batch_dir / str(entry.get("run_id", ""))).resolve()
                records[str(output_dir)] = {
                    "entry": entry,
                    "batch_manifest": batch_manifest,
                    "batch_dir": batch_dir,
                    "source_root": root,
                }
    return records


def write_board_exports(
    rows: list[dict[str, Any]],
    history_dir: str | Path = DEFAULT_HISTORY_DIR,
    external_benchmarks_path: str | Path = DEFAULT_EXTERNAL_BENCHMARKS_PATH,
    community_signals_path: str | Path = DEFAULT_COMMUNITY_SIGNALS_PATH,
) -> dict[str, Any]:
    target = Path(history_dir)
    target.mkdir(parents=True, exist_ok=True)
    latest = latest_board_rows(rows)
    leaderboard = build_leaderboard_rows(rows)
    lane_summary = build_lane_summary_rows(rows)
    runtime_profiles = build_runtime_profile_rows(rows)
    intent_comparison = build_intent_comparison_rows(rows)
    comparison_batches = build_comparison_batch_rows(rows)
    public_summary = build_public_summary(rows)
    external_benchmarks = build_external_benchmark_rows(external_benchmarks_path)
    external_benchmark_summary = build_external_benchmark_summary(external_benchmarks)
    community_signals = build_community_signal_rows(community_signals_path)
    community_signal_summary = build_community_signal_summary(community_signals)
    harnessability_breakdown_rows = _flatten_breakdown_rows(latest, "harnessability_breakdown_json", "harnessability_tag")
    direction_following_breakdown_rows = _flatten_breakdown_rows(latest, "direction_following_breakdown_json", "direction_following_tag")
    tool_family_breakdown_rows = _flatten_breakdown_rows(latest, "tool_family_breakdown_json", "tool_family_tag")
    scatter_rows = [
        {
            "system_id": row.get("system_id", ""),
            "display_name": row.get("display_name", ""),
            "short_label": row.get("short_label", row.get("display_name", "")),
            "lane": row.get("lane", ""),
            "run_intent": row.get("run_intent", ""),
            "provider": row.get("provider", ""),
            "local": row.get("local", ""),
            "total_params_b": row.get("total_params_b", ""),
            "reasoner_params_b": row.get("reasoner_params_b", ""),
            "router_params_b": row.get("router_params_b", ""),
            "retriever_params_b": row.get("retriever_params_b", ""),
            "episode_count": row.get("episode_count", ""),
            "pass_count": row.get("pass_count", ""),
            "refine_count": row.get("refine_count", ""),
            "fail_count": row.get("fail_count", ""),
            "artifact_quality_avg": row.get("artifact_quality_avg", ""),
            "browser_workflow_avg": row.get("browser_workflow_avg", ""),
            "strict_interface_avg": row.get("strict_interface_avg", ""),
            "recovered_execution_avg": row.get("recovered_execution_avg", ""),
            "real_world_readiness_avg": row.get("real_world_readiness_avg", ""),
            "escalation_correctness_avg": row.get("escalation_correctness_avg", ""),
            "controller_repair_avg": row.get("controller_repair_avg", ""),
            "argument_repair_avg": row.get("argument_repair_avg", ""),
            "controller_fallback_avg": row.get("controller_fallback_avg", ""),
            "intent_override_avg": row.get("intent_override_avg", ""),
            "raw_planning_clean_rate_avg": row.get("raw_planning_clean_rate_avg", ""),
            "input_cost_per_mtok": row.get("input_cost_per_mtok", ""),
            "output_cost_per_mtok": row.get("output_cost_per_mtok", ""),
            "total_cost_per_mtok": row.get("total_cost_per_mtok", ""),
            "warmup_load_ms": row.get("warmup_load_ms", ""),
            "last_request_elapsed_ms": row.get("last_request_elapsed_ms", ""),
            "requests_completed": row.get("requests_completed", ""),
        }
        for row in latest
    ]
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "row_count": len(rows),
        "latest_row_count": len(latest),
        "leaderboard_row_count": len(leaderboard),
        "lane_summary_row_count": len(lane_summary),
        "runtime_profile_row_count": len(runtime_profiles),
        "intent_comparison_row_count": len(intent_comparison),
        "comparison_batch_row_count": len(comparison_batches),
        "public_summary": public_summary,
        "external_benchmark_row_count": len(external_benchmarks),
        "external_benchmark_summary": external_benchmark_summary,
        "community_signal_row_count": len(community_signals),
        "community_signal_summary": community_signal_summary,
        "harnessability_breakdown_row_count": len(harnessability_breakdown_rows),
        "direction_following_breakdown_row_count": len(direction_following_breakdown_rows),
        "tool_family_breakdown_row_count": len(tool_family_breakdown_rows),
        "rows": rows,
        "latest": latest,
    }
    (target / "knowledge_work_board.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(target / "knowledge_work_board_runs.csv", rows)
    _write_csv(target / "knowledge_work_board_latest.csv", latest)
    _write_csv(target / "knowledge_work_leaderboard.csv", leaderboard)
    _write_csv(target / "knowledge_work_scatter.csv", scatter_rows)
    _write_csv(target / "knowledge_work_lane_summary.csv", lane_summary)
    _write_csv(target / "knowledge_work_runtime_profiles.csv", runtime_profiles)
    _write_csv(target / "knowledge_work_intent_comparison.csv", intent_comparison)
    _write_csv(target / "knowledge_work_comparison_batches.csv", comparison_batches)
    _write_csv(target / "knowledge_work_role_breakdown.csv", _flatten_breakdown_rows(latest, "role_breakdown_json", "role_family"))
    _write_csv(target / "knowledge_work_category_breakdown.csv", _flatten_breakdown_rows(latest, "category_breakdown_json", "category"))
    _write_csv(target / "knowledge_work_track_breakdown.csv", _flatten_breakdown_rows(latest, "track_breakdown_json", "track_tag"))
    _write_csv(target / "knowledge_work_harnessability_breakdown.csv", harnessability_breakdown_rows)
    _write_csv(target / "knowledge_work_direction_following_breakdown.csv", direction_following_breakdown_rows)
    _write_csv(target / "knowledge_work_tool_family_breakdown.csv", tool_family_breakdown_rows)
    _write_csv(
        target / "knowledge_work_external_benchmarks.csv",
        external_benchmarks,
        preferred_fields=[
            "display_name",
            "model_id",
            "provider",
            "access",
            "benchmark",
            "benchmark_group",
            "score",
            "score_display",
            "score_unit",
            "published_date",
            "source_scope",
            "source_kind",
            "source_org",
            "notes",
            "source_url",
        ],
    )
    _write_csv(target / "knowledge_work_community_signals.csv", community_signals)
    (target / "knowledge_work_public_summary.json").write_text(json.dumps(public_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (target / "knowledge_work_external_benchmark_summary.json").write_text(
        json.dumps(external_benchmark_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (target / "knowledge_work_community_signal_summary.json").write_text(
        json.dumps(community_signal_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def latest_board_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("system_id", "")), str(row.get("lane", "")), str(row.get("run_intent", "")))
        current = latest.get(key)
        if current is None or (
            _board_status_priority(row.get("board_status")) > _board_status_priority(current.get("board_status"))
            or (
                _board_status_priority(row.get("board_status")) == _board_status_priority(current.get("board_status"))
                and _run_scope_priority(row.get("run_scope")) > _run_scope_priority(current.get("run_scope"))
            )
            or (
                _board_status_priority(row.get("board_status")) == _board_status_priority(current.get("board_status"))
                and _run_scope_priority(row.get("run_scope")) == _run_scope_priority(current.get("run_scope"))
                and float(row.get("coverage_ratio", 0.0) or 0.0) > float(current.get("coverage_ratio", 0.0) or 0.0)
            )
            or (
                _board_status_priority(row.get("board_status")) == _board_status_priority(current.get("board_status"))
                and _run_scope_priority(row.get("run_scope")) == _run_scope_priority(current.get("run_scope"))
                and str(row.get("created_at", "")) >= str(current.get("created_at", ""))
            )
        ):
            latest[key] = row
    return sorted(
        latest.values(),
        key=lambda row: (
            str(row.get("lane", "")),
            str(row.get("run_intent", "")),
            -float(row.get("real_world_readiness_avg", 0.0)),
            -float(row.get("strict_interface_avg", 0.0)),
            str(row.get("display_name", "")),
        ),
    )


def build_leaderboard_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest = [row for row in latest_board_rows(rows) if _is_completed_row(row)]
    ranked = sorted(
        latest,
        key=lambda row: (
            str(row.get("lane", "")),
            str(row.get("run_intent", "")),
            -float(row.get("real_world_readiness_avg", 0.0)),
            -float(row.get("strict_interface_avg", 0.0)),
            -float(row.get("browser_workflow_avg", 0.0)),
            str(row.get("display_name", "")),
        ),
    )
    leaderboard: list[dict[str, Any]] = []
    per_group_rank: dict[tuple[str, str], int] = {}
    for row in ranked:
        key = (str(row.get("lane", "")), str(row.get("run_intent", "")))
        per_group_rank[key] = per_group_rank.get(key, 0) + 1
        leaderboard.append(
            {
                **row,
                "rank": per_group_rank[key],
                "rank_group": f"{row.get('lane', '')}:{row.get('run_intent', '')}",
                "readiness_pct": round(float(row.get("real_world_readiness_avg", 0.0) or 0.0) * 100, 2),
                "strict_pct": round(float(row.get("strict_interface_avg", 0.0) or 0.0) * 100, 2),
                "browser_pct": round(float(row.get("browser_workflow_avg", 0.0) or 0.0) * 100, 2),
                "artifact_pct": round(float(row.get("artifact_quality_avg", 0.0) or 0.0) * 100, 2),
                "recovered_pct": round(float(row.get("recovered_execution_avg", 0.0) or 0.0) * 100, 2),
                "pass_rate_pct": round(
                    (
                        float(row.get("pass_count", 0.0) or 0.0)
                        / max(1.0, float(row.get("episode_count", 0.0) or 0.0))
                    )
                    * 100,
                    2,
                ),
            }
        )
    return leaderboard


def build_lane_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest = [row for row in latest_board_rows(rows) if _is_completed_row(row)]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in latest:
        key = (str(row.get("lane", "")), str(row.get("run_intent", "")))
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for (lane, run_intent), items in grouped.items():
        best = max(items, key=lambda row: float(row.get("real_world_readiness_avg", 0.0) or 0.0))
        summaries.append(
            {
                "lane": lane,
                "run_intent": run_intent,
                "systems": len(items),
                "local_systems": sum(1 for row in items if bool(row.get("local", False))),
                "publishable_systems": sum(1 for row in items if bool(row.get("publishable_default", False))),
                "avg_readiness": _average_metric(items, "real_world_readiness_avg"),
                "avg_strict_interface": _average_metric(items, "strict_interface_avg"),
                "avg_browser_workflow": _average_metric(items, "browser_workflow_avg"),
                "avg_artifact_quality": _average_metric(items, "artifact_quality_avg"),
                "avg_recovered_execution": _average_metric(items, "recovered_execution_avg"),
                "avg_controller_repairs": _average_metric(items, "controller_repair_avg"),
                "avg_argument_repairs": _average_metric(items, "argument_repair_avg"),
                "avg_controller_fallbacks": _average_metric(items, "controller_fallback_avg"),
                "avg_intent_overrides": _average_metric(items, "intent_override_avg"),
                "avg_raw_planning_clean_rate": _average_metric(items, "raw_planning_clean_rate_avg"),
                "avg_coverage": _average_metric(items, "coverage_ratio"),
                "total_episodes": int(sum(float(row.get("episode_count", 0.0) or 0.0) for row in items)),
                "total_pass_count": int(sum(float(row.get("pass_count", 0.0) or 0.0) for row in items)),
                "total_refine_count": int(sum(float(row.get("refine_count", 0.0) or 0.0) for row in items)),
                "total_fail_count": int(sum(float(row.get("fail_count", 0.0) or 0.0) for row in items)),
                "best_system_id": best.get("system_id", ""),
                "best_display_name": best.get("display_name", ""),
                "best_readiness": float(best.get("real_world_readiness_avg", 0.0) or 0.0),
                "best_strict_interface": float(best.get("strict_interface_avg", 0.0) or 0.0),
                "best_browser_workflow": float(best.get("browser_workflow_avg", 0.0) or 0.0),
            }
        )
    return sorted(summaries, key=lambda row: (str(row.get("lane", "")), str(row.get("run_intent", ""))))


def build_runtime_profile_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest = [row for row in latest_board_rows(rows) if _is_completed_row(row)]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in latest:
        key = (str(row.get("system_id", "")), str(row.get("run_intent", "")))
        grouped.setdefault(key, []).append(row)

    profiles: list[dict[str, Any]] = []
    for (system_id, run_intent), items in grouped.items():
        reference = max(items, key=lambda row: float(row.get("real_world_readiness_avg", 0.0) or 0.0))
        lane_map = {
            str(item.get("lane", "")): round(float(item.get("real_world_readiness_avg", 0.0) or 0.0), 4)
            for item in items
        }
        profiles.append(
            {
                "system_id": system_id,
                "display_name": reference.get("display_name", ""),
                "short_label": reference.get("short_label", ""),
                "run_intent": run_intent,
                "provider": reference.get("provider", ""),
                "deployment": reference.get("deployment", ""),
                "comparison_tier": reference.get("comparison_tier", ""),
                "publishable_default": bool(reference.get("publishable_default", False)),
                "capability_family": reference.get("capability_family", ""),
                "executor_mode": reference.get("executor_mode", ""),
                "modality": reference.get("modality", ""),
                "local": bool(reference.get("local", False)),
                "lane_count": len(items),
                "lanes": ",".join(sorted(lane_map)),
                "lane_readiness_json": json.dumps(lane_map, ensure_ascii=False, sort_keys=True),
                "avg_readiness": _average_metric(items, "real_world_readiness_avg"),
                "avg_strict_interface": _average_metric(items, "strict_interface_avg"),
                "avg_browser_workflow": _average_metric(items, "browser_workflow_avg"),
                "avg_artifact_quality": _average_metric(items, "artifact_quality_avg"),
                "avg_recovered_execution": _average_metric(items, "recovered_execution_avg"),
                "avg_escalation_correctness": _average_metric(items, "escalation_correctness_avg"),
                "avg_controller_repairs": _average_metric(items, "controller_repair_avg"),
                "avg_argument_repairs": _average_metric(items, "argument_repair_avg"),
                "avg_controller_fallbacks": _average_metric(items, "controller_fallback_avg"),
                "avg_intent_overrides": _average_metric(items, "intent_override_avg"),
                "avg_raw_planning_clean_rate": _average_metric(items, "raw_planning_clean_rate_avg"),
                "episode_count": int(sum(float(item.get("episode_count", 0.0) or 0.0) for item in items)),
                "pass_count": int(sum(float(item.get("pass_count", 0.0) or 0.0) for item in items)),
                "refine_count": int(sum(float(item.get("refine_count", 0.0) or 0.0) for item in items)),
                "fail_count": int(sum(float(item.get("fail_count", 0.0) or 0.0) for item in items)),
                "total_params_b": _first_number(reference.get("total_params_b")),
                "warmup_load_ms": _average_metric(items, "warmup_load_ms"),
                "last_request_elapsed_ms": _average_metric(items, "last_request_elapsed_ms"),
                "requests_completed": int(sum(float(item.get("requests_completed", 0.0) or 0.0) for item in items)),
                "total_cost_per_mtok": _first_number(reference.get("total_cost_per_mtok")),
                "input_cost_per_mtok": _first_number(reference.get("input_cost_per_mtok")),
                "output_cost_per_mtok": _first_number(reference.get("output_cost_per_mtok")),
                "best_lane": max(lane_map, key=lane_map.get) if lane_map else "",
                "best_lane_readiness": max(lane_map.values()) if lane_map else 0.0,
            }
        )
    return sorted(
        profiles,
        key=lambda row: (
            str(row.get("run_intent", "")),
            -float(row.get("avg_readiness", 0.0) or 0.0),
            str(row.get("display_name", "")),
        ),
    )


def build_public_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latest = latest_board_rows(rows)
    leaderboard = build_leaderboard_rows(rows)
    lane_summary = build_lane_summary_rows(rows)
    runtime_profiles = build_runtime_profile_rows(rows)
    comparison_batches = build_comparison_batch_rows(rows)
    completed_latest = [row for row in latest if _is_completed_row(row)]
    headline_rows = [
        row for row in completed_latest
        if bool(row.get("publishable_default", False)) and bool(row.get("full_lane_complete", False))
    ]
    headline_rows = sorted(
        headline_rows,
        key=lambda row: (
            str(row.get("lane", "")),
            -float(row.get("real_world_readiness_avg", 0.0) or 0.0),
            str(row.get("display_name", "")),
        ),
    )
    leaders: dict[str, dict[str, Any]] = {}
    for row in leaderboard:
        key = f"{row.get('lane', '')}:{row.get('run_intent', '')}"
        if key not in leaders:
            leaders[key] = {
                "display_name": row.get("display_name", ""),
                "system_id": row.get("system_id", ""),
                "readiness": row.get("real_world_readiness_avg", 0.0),
                "strict_interface": row.get("strict_interface_avg", 0.0),
                "browser_workflow": row.get("browser_workflow_avg", 0.0),
                "artifact_quality": row.get("artifact_quality_avg", 0.0),
            }
    return {
        "total_runs": len(rows),
        "latest_runs": len(latest),
        "local_profiles": sum(1 for row in runtime_profiles if bool(row.get("local", False))),
        "publishable_profiles": sum(1 for row in runtime_profiles if bool(row.get("publishable_default", False))),
        "lanes": lane_summary,
        "comparison_batches": comparison_batches,
        "comparison_health": {
            "completed": sum(1 for row in latest if str(row.get("board_status", "completed") or "completed") == "completed"),
            "partial": sum(1 for row in latest if str(row.get("board_status", "")) == "partial"),
            "timed_out": sum(1 for row in latest if str(row.get("board_status", "")) == "timed_out"),
            "failed": sum(1 for row in latest if str(row.get("board_status", "")) == "failed"),
            "avg_coverage": _average_metric(latest, "coverage_ratio"),
        },
        "harnessability_slice_count": _count_breakdown_tags(completed_latest, "harnessability_breakdown_json"),
        "direction_following_slice_count": _count_breakdown_tags(completed_latest, "direction_following_breakdown_json"),
        "tool_family_slice_count": _count_breakdown_tags(completed_latest, "tool_family_breakdown_json"),
        "avg_controller_repairs": _average_metric(completed_latest, "controller_repair_avg"),
        "avg_argument_repairs": _average_metric(completed_latest, "argument_repair_avg"),
        "avg_controller_fallbacks": _average_metric(completed_latest, "controller_fallback_avg"),
        "avg_intent_overrides": _average_metric(completed_latest, "intent_override_avg"),
        "avg_raw_planning_clean_rate": _average_metric(completed_latest, "raw_planning_clean_rate_avg"),
        "leaders": leaders,
        "headline_systems": [
            {
                "display_name": row.get("display_name", ""),
                "system_id": row.get("system_id", ""),
                "lane": row.get("lane", ""),
                "run_intent": row.get("run_intent", ""),
                "comparison_tier": row.get("comparison_tier", ""),
                "real_world_readiness_avg": row.get("real_world_readiness_avg", 0.0),
                "strict_interface_avg": row.get("strict_interface_avg", 0.0),
                "browser_workflow_avg": row.get("browser_workflow_avg", 0.0),
                "artifact_quality_avg": row.get("artifact_quality_avg", 0.0),
                "coverage_ratio": row.get("coverage_ratio", 0.0),
                "episode_count": row.get("episode_count", 0),
                "pass_count": row.get("pass_count", 0),
                "refine_count": row.get("refine_count", 0),
                "fail_count": row.get("fail_count", 0),
            }
            for row in headline_rows
        ],
        "fastest_local_profile": _best_runtime_profile(
            runtime_profiles,
            key="last_request_elapsed_ms",
            prefer_lowest=True,
        ),
        "highest_readiness_local_profile": _best_runtime_profile(runtime_profiles, key="avg_readiness"),
        "most_efficient_local_profile": _best_runtime_profile(
            runtime_profiles,
            key="total_cost_per_mtok",
            prefer_lowest=True,
        ),
    }


def build_intent_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest = [row for row in latest_board_rows(rows) if _is_completed_row(row)]
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in latest:
        system_id = str(row.get("system_id", ""))
        lane = str(row.get("lane", ""))
        intent = str(row.get("run_intent", ""))
        grouped.setdefault((system_id, lane), {})[intent] = row

    comparison_rows: list[dict[str, Any]] = []
    for (system_id, lane), entries in grouped.items():
        canonical = entries.get("canonical")
        exploratory = entries.get("exploratory")
        if not canonical and not exploratory:
            continue
        reference = canonical or exploratory or {}
        canonical_readiness = _comparison_metric(canonical, "real_world_readiness_avg")
        exploratory_readiness = _comparison_metric(exploratory, "real_world_readiness_avg")
        canonical_strict = _comparison_metric(canonical, "strict_interface_avg")
        exploratory_strict = _comparison_metric(exploratory, "strict_interface_avg")
        canonical_browser = _comparison_metric(canonical, "browser_workflow_avg")
        exploratory_browser = _comparison_metric(exploratory, "browser_workflow_avg")
        comparison_rows.append(
            {
                "system_id": system_id,
                "display_name": reference.get("display_name", ""),
                "lane": lane,
                "capability_family": reference.get("capability_family", ""),
                "executor_mode": reference.get("executor_mode", ""),
                "modality": reference.get("modality", ""),
                "canonical_run_group_id": canonical.get("run_group_id", "") if canonical else "",
                "exploratory_run_group_id": exploratory.get("run_group_id", "") if exploratory else "",
                "canonical_readiness": canonical_readiness,
                "exploratory_readiness": exploratory_readiness,
                "readiness_delta": _delta(exploratory_readiness, canonical_readiness),
                "canonical_strict_interface": canonical_strict,
                "exploratory_strict_interface": exploratory_strict,
                "strict_delta": _delta(exploratory_strict, canonical_strict),
                "canonical_browser_workflow": canonical_browser,
                "exploratory_browser_workflow": exploratory_browser,
                "browser_delta": _delta(exploratory_browser, canonical_browser),
                "canonical_episode_count": canonical.get("episode_count", "") if canonical else "",
                "exploratory_episode_count": exploratory.get("episode_count", "") if exploratory else "",
                "canonical_pass_count": canonical.get("pass_count", "") if canonical else "",
                "exploratory_pass_count": exploratory.get("pass_count", "") if exploratory else "",
                "canonical_refine_count": canonical.get("refine_count", "") if canonical else "",
                "exploratory_refine_count": exploratory.get("refine_count", "") if exploratory else "",
                "canonical_fail_count": canonical.get("fail_count", "") if canonical else "",
                "exploratory_fail_count": exploratory.get("fail_count", "") if exploratory else "",
            }
        )
    return sorted(
        comparison_rows,
        key=lambda row: (
            str(row.get("lane", "")),
            -float(row.get("exploratory_readiness", row.get("canonical_readiness", 0.0)) or 0.0),
            str(row.get("display_name", "")),
        ),
    )


def build_comparison_batch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest = latest_board_rows(rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in latest:
        batch = str(row.get("comparison_batch", "") or "").strip()
        if not batch:
            continue
        grouped.setdefault(batch, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for batch, items in grouped.items():
        completed_items = [row for row in items if _is_completed_row(row)]
        ranked_items = completed_items or items
        best = max(ranked_items, key=lambda row: float(row.get("real_world_readiness_avg", 0.0) or 0.0))
        summary_rows.append(
            {
                "comparison_batch": batch,
                "result_families": ",".join(
                    sorted(
                        {
                            str(item.get("result_family", ""))
                            for item in items
                            if str(item.get("result_family", ""))
                        }
                    )
                ),
                "systems": len(items),
                "completed_systems": sum(1 for row in items if _is_completed_row(row)),
                "partial_systems": sum(1 for row in items if str(row.get("board_status", "")) == "partial"),
                "timed_out_systems": sum(1 for row in items if str(row.get("board_status", "")) == "timed_out"),
                "failed_systems": sum(1 for row in items if str(row.get("board_status", "")) == "failed"),
                "matrix_complete": float(all(_is_completed_row(row) for row in items)),
                "lanes": ",".join(sorted({str(item.get("lane", "")) for item in items if str(item.get("lane", ""))})),
                "run_intents": ",".join(
                    sorted({str(item.get("run_intent", "")) for item in items if str(item.get("run_intent", ""))})
                ),
                "avg_readiness": _average_metric(ranked_items, "real_world_readiness_avg"),
                "avg_strict_interface": _average_metric(ranked_items, "strict_interface_avg"),
                "avg_browser_workflow": _average_metric(ranked_items, "browser_workflow_avg"),
                "avg_artifact_quality": _average_metric(ranked_items, "artifact_quality_avg"),
                "avg_recovered_execution": _average_metric(ranked_items, "recovered_execution_avg"),
                "best_system_id": best.get("system_id", ""),
                "best_display_name": best.get("display_name", ""),
                "best_readiness": float(best.get("real_world_readiness_avg", 0.0) or 0.0),
                "total_episodes": int(sum(float(row.get("planned_runs", row.get("episode_count", 0.0)) or 0.0) for row in items)),
                "observed_episodes": int(sum(float(row.get("completed_runs_observed", row.get("episode_count", 0.0)) or 0.0) for row in items)),
                "coverage_ratio": round(
                    min(
                        (
                            sum(float(row.get("completed_runs_observed", row.get("episode_count", 0.0)) or 0.0) for row in items)
                            / max(1.0, sum(float(row.get("planned_runs", row.get("episode_count", 0.0)) or 0.0) for row in items))
                        ),
                        1.0,
                    ),
                    4,
                ),
                "total_pass_count": int(sum(float(row.get("pass_count", 0.0) or 0.0) for row in ranked_items)),
                "total_refine_count": int(sum(float(row.get("refine_count", 0.0) or 0.0) for row in ranked_items)),
                "total_fail_count": int(sum(float(row.get("fail_count", 0.0) or 0.0) for row in ranked_items)),
            }
        )
    return sorted(
        summary_rows,
        key=lambda row: (
            -float(row.get("avg_readiness", 0.0) or 0.0),
            str(row.get("comparison_batch", "")),
        ),
    )


def _snapshot_row(
    path: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    leaderboard_rows: list[dict[str, str]],
    registry: dict[str, dict[str, Any]],
    source_root: Path,
    progress: dict[str, Any] | None = None,
    matrix_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    system_id, system_meta = _resolve_system(manifest, registry)
    pass_count, refine_count, fail_count = _status_counts(leaderboard_rows)
    role_breakdown = _group_metric_breakdown(leaderboard_rows, "role_family")
    category_breakdown = _tag_metric_breakdown(leaderboard_rows)
    track_breakdown = _tag_metric_breakdown(leaderboard_rows, prefix_filters=["visual_", "knowledge_work_arena"])
    harnessability_breakdown = _tag_metric_breakdown(leaderboard_rows, prefix_filters=["harnessability_"])
    direction_following_breakdown = _tag_metric_breakdown(leaderboard_rows, prefix_filters=["direction_following_"])
    tool_family_breakdown = _tag_metric_breakdown(
        leaderboard_rows,
        prefix_filters=["tool_", "function_call", "cli", "api", "mcp_", "skill_"],
    )
    reasoner = str(manifest.get("reasoner", "") or "")
    router = str(manifest.get("router", "") or "")
    retriever = str(manifest.get("retriever", "") or "")
    reasoner_meta = registry["models"].get(reasoner, {})
    router_meta = registry["models"].get(router, {})
    retriever_meta = registry["models"].get(retriever, {})
    reasoner_params_b = _first_number(system_meta.get("reasoner_params_b"), reasoner_meta.get("params_b"))
    router_params_b = _first_number(system_meta.get("router_params_b"), router_meta.get("params_b"))
    retriever_params_b = _first_number(system_meta.get("retriever_params_b"), retriever_meta.get("params_b"))
    input_cost_per_mtok = _first_number(system_meta.get("input_cost_per_mtok"))
    output_cost_per_mtok = _first_number(system_meta.get("output_cost_per_mtok"))
    total_params_b = _first_number(
        system_meta.get("total_params_b"),
        _sum_present(reasoner_params_b, router_params_b, retriever_params_b),
    )
    runtime_metrics = _extract_runtime_metrics(manifest)
    result_family = "knowledge_work_matrix" if source_root.name == "knowledge_work_matrix" else "knowledge_work"
    comparison_batch = path.parent.name if result_family == "knowledge_work_matrix" and path.parent != source_root else ""
    run_health = _run_health_payload(
        manifest,
        summary,
        progress or {},
        matrix_record,
        leaderboard_rows,
        comparison_batch,
    )

    return {
        "run_group_id": manifest.get("run_group_id", path.name),
        "created_at": manifest.get("created_at", ""),
        "lane": manifest.get("lane", ""),
        "run_intent": _infer_run_intent(path, manifest),
        "run_scope": _infer_run_scope(manifest, comparison_batch),
        "result_family": result_family,
        "comparison_batch": comparison_batch,
        "snapshot_name": path.name,
        "system_id": system_id,
        "display_name": system_meta.get("display_name") or _fallback_display_name(manifest),
        "short_label": system_meta.get("short_label") or system_meta.get("display_name") or _fallback_display_name(manifest),
        "provider": system_meta.get("provider") or reasoner_meta.get("provider") or manifest.get("backend", ""),
        "deployment": system_meta.get("deployment") or manifest.get("backend", ""),
        "comparison_tier": system_meta.get("comparison_tier", ""),
        "publishable_default": bool(system_meta.get("publishable_default", False)),
        "local": bool(system_meta.get("local", manifest.get("backend") == "hf_service")),
        "capability_family": system_meta.get("capability_family", ""),
        "executor_mode": system_meta.get("executor_mode", ""),
        "modality": system_meta.get("modality") or reasoner_meta.get("modality", ""),
        "color": system_meta.get("color", ""),
        "backend": manifest.get("backend", ""),
        "reasoner_backend": manifest.get("reasoner_backend", ""),
        "router_backend": manifest.get("router_backend", ""),
        "retriever_backend": manifest.get("retriever_backend", ""),
        "reasoner": reasoner,
        "router": router,
        "retriever": retriever,
        "reasoner_display_name": reasoner_meta.get("display_name", reasoner),
        "router_display_name": router_meta.get("display_name", router),
        "retriever_display_name": retriever_meta.get("display_name", retriever),
        "reasoner_params_b": reasoner_params_b,
        "router_params_b": router_params_b,
        "retriever_params_b": retriever_params_b,
        "total_params_b": total_params_b,
        "input_cost_per_mtok": input_cost_per_mtok,
        "output_cost_per_mtok": output_cost_per_mtok,
        "total_cost_per_mtok": _sum_present(input_cost_per_mtok, output_cost_per_mtok),
        "warmup_load_ms": runtime_metrics["warmup_load_ms"],
        "last_request_elapsed_ms": runtime_metrics["last_request_elapsed_ms"],
        "requests_completed": runtime_metrics["requests_completed"],
        "episode_count": int(manifest.get("episode_count", len(leaderboard_rows) or summary.get("runs", 0))),
        "pass_count": pass_count,
        "refine_count": refine_count,
        "fail_count": fail_count,
        "artifact_quality_avg": float(summary.get("artifact_quality_avg", 0.0)),
        "browser_workflow_avg": float(summary.get("browser_workflow_avg", 0.0)),
        "strict_interface_avg": float(summary.get("strict_interface_avg", 0.0)),
        "recovered_execution_avg": float(summary.get("recovered_execution_avg", 0.0)),
        "real_world_readiness_avg": float(summary.get("real_world_readiness_avg", 0.0)),
        "escalation_correctness_avg": float(summary.get("escalation_correctness_avg", 0.0)),
        "controller_repair_avg": float(summary.get("controller_repair_avg", 0.0)),
        "argument_repair_avg": float(summary.get("argument_repair_avg", 0.0)),
        "controller_fallback_avg": float(summary.get("controller_fallback_avg", 0.0)),
        "intent_override_avg": float(summary.get("intent_override_avg", 0.0)),
        "raw_planning_clean_rate_avg": float(summary.get("raw_planning_clean_rate_avg", 1.0)),
        "role_breakdown_json": json.dumps(role_breakdown, ensure_ascii=False, sort_keys=True),
        "category_breakdown_json": json.dumps(category_breakdown, ensure_ascii=False, sort_keys=True),
        "track_breakdown_json": json.dumps(track_breakdown, ensure_ascii=False, sort_keys=True),
        "harnessability_breakdown_json": json.dumps(harnessability_breakdown, ensure_ascii=False, sort_keys=True),
        "direction_following_breakdown_json": json.dumps(direction_following_breakdown, ensure_ascii=False, sort_keys=True),
        "tool_family_breakdown_json": json.dumps(tool_family_breakdown, ensure_ascii=False, sort_keys=True),
        **run_health,
        "output_dir": str(path.resolve()),
    }


def _resolve_system(
    manifest: dict[str, Any],
    registry: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    explicit = str(manifest.get("system_id", "") or "").strip()
    explicit = SYSTEM_ID_ALIASES.get(explicit, explicit)
    systems = registry.get("systems", {})
    if explicit and explicit in systems:
        return explicit, systems.get(explicit, {})

    matched = _match_registry_system(manifest, systems)
    if matched is not None:
        return matched

    derived = _slugify(
        "__".join(
            part
            for part in (
                backend,
                reasoner.replace("/", "_"),
                router.replace("/", "_"),
                retriever.replace("/", "_"),
            )
            if part
        )
    )
    if explicit:
        return explicit, {}
    return derived or "unknown_system", {}


def _match_registry_system(
    manifest: dict[str, Any],
    systems: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]] | None:
    backend = str(manifest.get("backend", "") or "")
    reasoner = str(manifest.get("reasoner", "") or "")
    router = str(manifest.get("router", "") or "")
    retriever = str(manifest.get("retriever", "") or "")
    router_backend = str(manifest.get("router_backend", "") or "").strip().lower()
    retriever_backend = str(manifest.get("retriever_backend", "") or "").strip().lower()

    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for system_id, meta in systems.items():
        if str(meta.get("backend", "") or "") != backend:
            continue
        if str(meta.get("reasoner", "") or "") and str(meta.get("reasoner", "") or "") != reasoner:
            continue
        if str(meta.get("router", "") or "") and str(meta.get("router", "") or "") != router:
            continue
        if str(meta.get("retriever", "") or "") and str(meta.get("retriever", "") or "") != retriever:
            continue

        executor_mode = str(meta.get("executor_mode", "") or "")
        if executor_mode == "local_reasoner":
            if router_backend not in {"", "heuristic"} or retriever_backend not in {"", "heuristic"}:
                continue
        elif executor_mode == "local_specialists":
            if router_backend not in {"hf", "hf_service"} or retriever_backend not in {"hf", "hf_service"}:
                continue
        elif executor_mode == "seeded" and backend != "oracle":
            continue

        score = 0
        if str(meta.get("reasoner", "") or "") == reasoner:
            score += 4
        if str(meta.get("router", "") or "") == router:
            score += 2
        if str(meta.get("retriever", "") or "") == retriever:
            score += 2
        if executor_mode == "local_reasoner" and router_backend in {"", "heuristic"} and retriever_backend in {"", "heuristic"}:
            score += 3
        if executor_mode == "local_specialists" and router_backend in {"hf", "hf_service"} and retriever_backend in {"hf", "hf_service"}:
            score += 3
        if system_id.startswith("hf_service_"):
            score += 1
        candidates.append((score, system_id, meta))

    if not candidates:
        return None
    _, system_id, meta = max(candidates, key=lambda item: (item[0], item[1]))
    return system_id, meta


def _status_counts(rows: list[dict[str, str]]) -> tuple[int, int, int]:
    pass_count = 0
    refine_count = 0
    fail_count = 0
    for row in rows:
        strict = float(row.get("strict_interface_score", 0.0) or 0.0)
        recovered = float(row.get("recovered_execution_score", 0.0) or 0.0)
        if strict >= 0.999 and recovered >= 0.999:
            pass_count += 1
        elif recovered >= 0.999:
            refine_count += 1
        else:
            fail_count += 1
    return pass_count, refine_count, fail_count


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred_fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = preferred_fields or [
        "rank",
        "display_name",
        "short_label",
        "system_id",
        "lane",
        "run_intent",
        "run_scope",
        "result_family",
        "comparison_batch",
        "snapshot_name",
        "provider",
        "comparison_tier",
        "publishable_default",
        "capability_family",
        "executor_mode",
        "modality",
        "deployment",
        "local",
        "backend",
        "reasoner",
        "router",
        "retriever",
        "reasoner_display_name",
        "router_display_name",
        "retriever_display_name",
        "reasoner_params_b",
        "router_params_b",
        "retriever_params_b",
        "total_params_b",
        "input_cost_per_mtok",
        "output_cost_per_mtok",
        "total_cost_per_mtok",
        "warmup_load_ms",
        "last_request_elapsed_ms",
        "requests_completed",
        "board_status",
        "progress_status",
        "validation_error",
        "returncode",
        "timed_out",
        "completed_runs_observed",
        "planned_runs",
        "coverage_ratio",
        "coverage_pct",
        "completion_pct",
        "matrix_complete",
        "full_lane_complete",
        "failure_excerpt",
        "episode_count",
        "pass_count",
        "refine_count",
        "fail_count",
        "artifact_quality_avg",
        "browser_workflow_avg",
        "strict_interface_avg",
        "recovered_execution_avg",
        "real_world_readiness_avg",
        "escalation_correctness_avg",
        "readiness_pct",
        "strict_pct",
        "browser_pct",
        "artifact_pct",
        "recovered_pct",
        "pass_rate_pct",
        "rank_group",
        "avg_readiness",
        "avg_strict_interface",
        "avg_browser_workflow",
        "avg_artifact_quality",
        "avg_recovered_execution",
        "avg_escalation_correctness",
        "best_lane",
        "best_lane_readiness",
        "lane_count",
        "lanes",
        "lane_readiness_json",
        "systems",
        "completed_systems",
        "partial_systems",
        "timed_out_systems",
        "failed_systems",
        "local_systems",
        "total_episodes",
        "observed_episodes",
        "total_pass_count",
        "total_refine_count",
        "total_fail_count",
        "best_system_id",
        "best_display_name",
        "best_readiness",
        "best_strict_interface",
        "best_browser_workflow",
        "created_at",
        "run_group_id",
        "output_dir",
        "role_breakdown_json",
        "category_breakdown_json",
        "track_breakdown_json",
        "harnessability_breakdown_json",
        "direction_following_breakdown_json",
        "tool_family_breakdown_json",
        "color",
        "reasoner_backend",
        "router_backend",
        "retriever_backend",
    ]
    discovered = {key for row in rows for key in row.keys()}
    fieldnames = preferred + sorted(key for key in discovered if key not in preferred)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _matrix_manifest(record: dict[str, Any]) -> dict[str, Any]:
    entry = record.get("entry", {}) or {}
    manifest = entry.get("manifest")
    if isinstance(manifest, dict):
        return manifest
    output_dir = Path(str(entry.get("output_dir", "") or "")).resolve()
    on_disk = _load_json(output_dir / "manifest.json")
    if on_disk is not None:
        return on_disk
    batch_manifest = record.get("batch_manifest", {}) or {}
    spec = _matrix_run_spec(record)
    lane = str(entry.get("lane", spec.get("lane", "")) or "")
    episodes_path = _lane_episodes_path(lane)
    return {
        "run_group_id": batch_manifest.get("run_group_id", output_dir.name),
        "created_at": batch_manifest.get("created_at", ""),
        "lane": lane,
        "run_intent": batch_manifest.get("run_intent", "exploratory"),
        "backend": spec.get("backend", ""),
        "reasoner_backend": spec.get("reasoner_backend", ""),
        "router_backend": spec.get("router_backend", ""),
        "retriever_backend": spec.get("retriever_backend", ""),
        "system_id": entry.get("system_id", spec.get("system_id", "")),
        "reasoner": spec.get("reasoner", ""),
        "router": spec.get("router", ""),
        "retriever": spec.get("retriever", ""),
        "episode_count": spec.get("limit") or _lane_episode_count(lane),
        "episodes_path": str(episodes_path.resolve()) if episodes_path is not None else "",
    }


def _matrix_summary(record: dict[str, Any]) -> dict[str, Any]:
    entry = record.get("entry", {}) or {}
    summary = entry.get("summary")
    if isinstance(summary, dict):
        return summary
    output_dir = Path(str(entry.get("output_dir", "") or "")).resolve()
    return _load_json(output_dir / "summary.json") or {}


def _matrix_progress(record: dict[str, Any]) -> dict[str, Any]:
    entry = record.get("entry", {}) or {}
    progress = entry.get("progress")
    if isinstance(progress, dict):
        return progress
    output_dir = Path(str(entry.get("output_dir", "") or "")).resolve()
    return _load_json(output_dir / "progress.json") or {}


def _matrix_run_spec(record: dict[str, Any]) -> dict[str, Any]:
    entry = record.get("entry", {}) or {}
    batch_manifest = record.get("batch_manifest", {}) or {}
    run_id = str(entry.get("run_id", "") or "")
    return next((spec for spec in batch_manifest.get("runs", []) if str(spec.get("run_id", "")) == run_id), {})


def _run_health_payload(
    manifest: dict[str, Any],
    summary: dict[str, Any],
    progress: dict[str, Any],
    matrix_record: dict[str, Any] | None,
    leaderboard_rows: list[dict[str, str]],
    comparison_batch: str = "",
) -> dict[str, Any]:
    entry = (matrix_record or {}).get("entry", {}) or {}
    planned_runs = int(progress.get("planned_runs", manifest.get("episode_count", 0)) or 0)
    completed_runs = int(progress.get("completed_runs", summary.get("runs", len(leaderboard_rows))) or 0)
    progress_status = str(progress.get("status", "") or "").strip().lower()
    validation_error = str(entry.get("validation_error", "") or "").strip()
    returncode = int(entry.get("returncode", 0) or 0) if entry else 0
    timed_out = bool(entry.get("timed_out", False)) if entry else False
    stderr = str(entry.get("stderr", "") or "").strip()
    if timed_out or validation_error == "run_timeout":
        board_status = "timed_out"
    elif validation_error or (progress_status not in {"", "completed"} and completed_runs < max(planned_runs, completed_runs)):
        board_status = "partial" if completed_runs > 0 else "failed"
    elif returncode not in {0}:
        board_status = "partial" if completed_runs > 0 else "failed"
    elif planned_runs and completed_runs and completed_runs < planned_runs:
        board_status = "partial"
    else:
        board_status = "completed"
    coverage_ratio = 0.0
    if planned_runs > 0:
        coverage_ratio = round(min(completed_runs / planned_runs, 1.0), 4)
    elif completed_runs > 0:
        coverage_ratio = 1.0
    return {
        "board_status": board_status,
        "progress_status": progress_status or ("completed" if board_status == "completed" else ""),
        "validation_error": validation_error,
        "returncode": returncode,
        "timed_out": timed_out,
        "completed_runs_observed": completed_runs,
        "planned_runs": planned_runs or int(manifest.get("episode_count", 0) or 0),
        "coverage_ratio": coverage_ratio,
        "coverage_pct": round(coverage_ratio * 100, 2),
        "completion_pct": round(coverage_ratio * 100, 2),
        "matrix_complete": float(board_status == "completed" and coverage_ratio >= 0.999),
        "full_lane_complete": float(
            _infer_run_scope(manifest, comparison_batch) == "full_lane"
            and board_status == "completed"
            and coverage_ratio >= 0.999
        ),
        "failure_excerpt": _first_failure_excerpt(validation_error, stderr),
    }


def _first_failure_excerpt(validation_error: str, stderr: str) -> str:
    if validation_error:
        return validation_error
    for line in stderr.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _board_status_priority(value: object) -> int:
    status = str(value or "")
    if status in {"", "completed"}:
        return 3
    if status == "partial":
        return 2
    if status == "timed_out":
        return 1
    return 0


def _community_signal_status_priority(value: object) -> int:
    status = str(value or "")
    if status == "answered":
        return 3
    if status == "running":
        return 2
    if status == "planned":
        return 1
    return 0


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _lane_episodes_path(lane: str) -> Path | None:
    if not lane:
        return None
    path = ROOT / "data" / "knowledge_work" / lane / "episodes.jsonl"
    return path if path.exists() else None


def _lane_episode_count(lane: str) -> int:
    path = _lane_episodes_path(lane)
    if path is None:
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def _infer_run_intent(path: Path, manifest: dict[str, Any]) -> str:
    manifest_intent = str(manifest.get("run_intent", "")).strip().lower()
    if manifest_intent in {"canonical", "exploratory"}:
        return manifest_intent
    lane = str(manifest.get("lane", "")).strip()
    if lane and path.name == lane:
        return "canonical"
    return "exploratory"


def _infer_run_scope(manifest: dict[str, Any], comparison_batch: str = "") -> str:
    batch_name = comparison_batch.strip().lower()
    matrix_name = str(manifest.get("matrix_name", "") or "").strip().lower()
    if "full_lane" in batch_name or matrix_name == "knowledge_work_full_lane":
        return "full_lane"
    episodes_path_value = str(manifest.get("episodes_path", "") or "").strip()
    if not episodes_path_value:
        return "unknown"
    episodes_path = Path(episodes_path_value)
    if not episodes_path.exists():
        return "unknown"
    try:
        planned_episode_count = sum(1 for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return "unknown"
    observed_episode_count = int(manifest.get("episode_count", 0) or 0)
    if planned_episode_count and observed_episode_count >= planned_episode_count:
        return "full_lane"
    return "subset"


def _normalize_results_roots(
    results_root: str | Path | list[str | Path] | tuple[str | Path, ...],
) -> list[Path]:
    if isinstance(results_root, (str, Path)):
        return [Path(results_root)]
    return [Path(root) for root in results_root]


def _run_scope_priority(value: object) -> int:
    scope = str(value or "")
    if scope == "full_lane":
        return 2
    if scope == "subset":
        return 1
    return 0


def _fallback_display_name(manifest: dict[str, Any]) -> str:
    backend = str(manifest.get("backend", "") or "")
    reasoner = str(manifest.get("reasoner", "") or "")
    router = str(manifest.get("router", "") or "")
    retriever = str(manifest.get("retriever", "") or "")
    parts = [backend, reasoner]
    if router:
        parts.append(f"router={router}")
    if retriever:
        parts.append(f"retriever={retriever}")
    return " | ".join(part for part in parts if part)


def _flatten_breakdown_rows(
    rows: list[dict[str, Any]],
    column: str,
    label: str,
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        raw = str(row.get(column, "") or "").strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for key, values in payload.items():
            if not isinstance(values, dict):
                continue
            flattened.append(
                {
                    "system_id": row.get("system_id", ""),
                    "display_name": row.get("display_name", ""),
                    "lane": row.get("lane", ""),
                    "run_intent": row.get("run_intent", ""),
                    label: key,
                    **values,
                }
            )
    return flattened


def _group_metric_breakdown(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        label = str(row.get(key, "")).strip()
        if not label:
            continue
        grouped.setdefault(label, []).append(row)
    return {
        label: {
            "episodes": float(len(items)),
            "pass_count": float(sum(1.0 for item in items if float(item.get("strict_interface_score", 0.0) or 0.0) >= 0.999 and float(item.get("recovered_execution_score", 0.0) or 0.0) >= 0.999)),
            "refine_count": float(sum(1.0 for item in items if float(item.get("strict_interface_score", 0.0) or 0.0) < 0.999 and float(item.get("recovered_execution_score", 0.0) or 0.0) >= 0.999)),
            "fail_count": float(sum(1.0 for item in items if float(item.get("recovered_execution_score", 0.0) or 0.0) < 0.999)),
            "readiness_avg": round(sum(float(item.get("role_readiness_score", 0.0) or 0.0) for item in items) / len(items), 4),
            "strict_avg": round(sum(float(item.get("strict_interface_score", 0.0) or 0.0) for item in items) / len(items), 4),
            "browser_avg": round(sum(float(item.get("browser_workflow_score", 0.0) or 0.0) for item in items) / len(items), 4),
            "artifact_avg": round(sum(float(item.get("artifact_quality_score", 0.0) or 0.0) for item in items) / len(items), 4),
            "recovered_avg": round(sum(float(item.get("recovered_execution_score", 0.0) or 0.0) for item in items) / len(items), 4),
        }
        for label, items in grouped.items()
    }


def _tag_metric_breakdown(
    rows: list[dict[str, str]],
    *,
    prefix_filters: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        tags = [tag.strip() for tag in str(row.get("benchmark_tags", "")).split(",") if tag.strip()]
        for tag in tags:
            if prefix_filters and not any(tag.startswith(prefix) or tag == prefix for prefix in prefix_filters):
                continue
            grouped.setdefault(tag, []).append(row)
    return {
        tag: {
            "episodes": float(len(items)),
            "pass_count": float(sum(1.0 for item in items if float(item.get("strict_interface_score", 0.0) or 0.0) >= 0.999 and float(item.get("recovered_execution_score", 0.0) or 0.0) >= 0.999)),
            "refine_count": float(sum(1.0 for item in items if float(item.get("strict_interface_score", 0.0) or 0.0) < 0.999 and float(item.get("recovered_execution_score", 0.0) or 0.0) >= 0.999)),
            "fail_count": float(sum(1.0 for item in items if float(item.get("recovered_execution_score", 0.0) or 0.0) < 0.999)),
            "readiness_avg": round(sum(float(item.get("role_readiness_score", 0.0) or 0.0) for item in items) / len(items), 4),
            "strict_avg": round(sum(float(item.get("strict_interface_score", 0.0) or 0.0) for item in items) / len(items), 4),
            "artifact_avg": round(sum(float(item.get("artifact_quality_score", 0.0) or 0.0) for item in items) / len(items), 4),
        }
        for tag, items in grouped.items()
    }


def _count_breakdown_tags(rows: list[dict[str, Any]], column: str) -> int:
    tags: set[str] = set()
    for row in rows:
        raw = str(row.get(column, "") or "").strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            tags.update(str(tag) for tag in payload if str(tag))
    return len(tags)


def _extract_runtime_metrics(manifest: dict[str, Any]) -> dict[str, float | None]:
    warmup_reasoner = ((manifest.get("warmup") or {}).get("reasoner") or {})
    warmup_service = (warmup_reasoner.get("service") or {})
    runtime_reasoner = ((manifest.get("runtime_bundle") or {}).get("reasoner") or {})
    service_state = (runtime_reasoner.get("service_state") or {})
    service_info = (runtime_reasoner.get("service") or {})
    return {
        "warmup_load_ms": _first_number(
            warmup_service.get("load_elapsed_ms"),
            service_state.get("load_elapsed_ms"),
        ),
        "last_request_elapsed_ms": _first_number(service_state.get("last_request_elapsed_ms")),
        "requests_completed": _first_number(
            service_state.get("requests_completed"),
            service_info.get("requests_completed"),
            warmup_service.get("requests_completed"),
        ),
    }


def _average_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0) or 0.0) for row in rows if row.get(key) not in {"", None}]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _best_runtime_profile(
    rows: list[dict[str, Any]],
    *,
    key: str,
    prefer_lowest: bool = False,
) -> dict[str, Any]:
    local_rows = [row for row in rows if bool(row.get("local", False)) and row.get(key) not in {"", None}]
    if not local_rows:
        return {}
    ranked = sorted(
        local_rows,
        key=lambda row: (
            float(row.get(key, 0.0) or 0.0),
            -float(row.get("avg_readiness", 0.0) or 0.0),
            str(row.get("display_name", "")),
        ),
        reverse=not prefer_lowest,
    )
    leader = ranked[0]
    return {
        "display_name": leader.get("display_name", ""),
        "system_id": leader.get("system_id", ""),
        "metric": key,
        "value": leader.get(key, 0.0),
        "avg_readiness": leader.get("avg_readiness", 0.0),
    }


def _first_number(*values: object) -> float | None:
    for value in values:
        if value in (None, "", "n/a"):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _sum_present(*values: float | None) -> float | None:
    present = [float(value) for value in values if value is not None]
    return round(sum(present), 4) if present else None

def _comparison_metric(row: dict[str, Any] | None, key: str) -> float:
    if not row:
        return 0.0
    return float(row.get(key, 0.0) or 0.0)


def _delta(left: float, right: float) -> float:
    return round(left - right, 4)


def _slugify(text: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in text]
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")
