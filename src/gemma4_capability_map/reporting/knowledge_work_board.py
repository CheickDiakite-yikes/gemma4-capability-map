from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_capability_map.io import load_yaml


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_ROOT = ROOT / "results" / "knowledge_work"
DEFAULT_HISTORY_DIR = ROOT / "results" / "history"
DEFAULT_REGISTRY_PATH = ROOT / "configs" / "model_registry.yaml"


def load_model_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, dict[str, Any]]:
    payload = load_yaml(path) or {}
    return {
        "models": payload.get("models", {}),
        "systems": payload.get("systems", {}),
    }


def build_board_rows(
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> list[dict[str, Any]]:
    registry = load_model_registry(registry_path)
    rows: list[dict[str, Any]] = []
    root = Path(results_root)
    if not root.exists():
        return rows

    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        manifest_path = path / "manifest.json"
        summary_path = path / "summary.json"
        leaderboard_path = path / "episode_leaderboard.csv"
        if not manifest_path.exists() or not summary_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        leaderboard_rows = _load_csv_rows(leaderboard_path)
        rows.append(_snapshot_row(path, manifest, summary, leaderboard_rows, registry))
    return rows


def write_board_exports(
    rows: list[dict[str, Any]],
    history_dir: str | Path = DEFAULT_HISTORY_DIR,
) -> dict[str, Any]:
    target = Path(history_dir)
    target.mkdir(parents=True, exist_ok=True)
    latest = latest_board_rows(rows)
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
            "input_cost_per_mtok": row.get("input_cost_per_mtok", ""),
            "output_cost_per_mtok": row.get("output_cost_per_mtok", ""),
        }
        for row in latest
    ]
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "row_count": len(rows),
        "latest_row_count": len(latest),
        "rows": rows,
        "latest": latest,
    }
    (target / "knowledge_work_board.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(target / "knowledge_work_board_runs.csv", rows)
    _write_csv(target / "knowledge_work_board_latest.csv", latest)
    _write_csv(target / "knowledge_work_scatter.csv", scatter_rows)
    return payload


def latest_board_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("system_id", "")), str(row.get("lane", "")), str(row.get("run_intent", "")))
        current = latest.get(key)
        if current is None or str(row.get("created_at", "")) >= str(current.get("created_at", "")):
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


def _snapshot_row(
    path: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    leaderboard_rows: list[dict[str, str]],
    registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    system_id, system_meta = _resolve_system(manifest, registry)
    pass_count, refine_count, fail_count = _status_counts(leaderboard_rows)
    reasoner = str(manifest.get("reasoner", "") or "")
    router = str(manifest.get("router", "") or "")
    retriever = str(manifest.get("retriever", "") or "")
    reasoner_meta = registry["models"].get(reasoner, {})
    router_meta = registry["models"].get(router, {})
    retriever_meta = registry["models"].get(retriever, {})
    reasoner_params_b = _first_number(system_meta.get("reasoner_params_b"), reasoner_meta.get("params_b"))
    router_params_b = _first_number(system_meta.get("router_params_b"), router_meta.get("params_b"))
    retriever_params_b = _first_number(system_meta.get("retriever_params_b"), retriever_meta.get("params_b"))
    total_params_b = _first_number(
        system_meta.get("total_params_b"),
        _sum_present(reasoner_params_b, router_params_b, retriever_params_b),
    )

    return {
        "run_group_id": manifest.get("run_group_id", path.name),
        "created_at": manifest.get("created_at", ""),
        "lane": manifest.get("lane", ""),
        "run_intent": _infer_run_intent(path, manifest),
        "system_id": system_id,
        "display_name": system_meta.get("display_name") or _fallback_display_name(manifest),
        "short_label": system_meta.get("short_label") or system_meta.get("display_name") or _fallback_display_name(manifest),
        "provider": system_meta.get("provider") or reasoner_meta.get("provider") or manifest.get("backend", ""),
        "deployment": system_meta.get("deployment") or manifest.get("backend", ""),
        "local": bool(system_meta.get("local", manifest.get("backend") == "hf_service")),
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
        "input_cost_per_mtok": _first_number(system_meta.get("input_cost_per_mtok")),
        "output_cost_per_mtok": _first_number(system_meta.get("output_cost_per_mtok")),
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
        "output_dir": str(path.resolve()),
    }


def _resolve_system(
    manifest: dict[str, Any],
    registry: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    explicit = str(manifest.get("system_id", "") or "").strip()
    systems = registry.get("systems", {})
    if explicit:
        return explicit, systems.get(explicit, {})

    backend = str(manifest.get("backend", "") or "")
    reasoner = str(manifest.get("reasoner", "") or "")
    router = str(manifest.get("router", "") or "")
    retriever = str(manifest.get("retriever", "") or "")

    for system_id, meta in systems.items():
        if (
            str(meta.get("backend", "") or "") == backend
            and str(meta.get("reasoner", "") or "") == reasoner
            and str(meta.get("router", "") or "") == router
            and str(meta.get("retriever", "") or "") == retriever
        ):
            return system_id, meta

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
    return derived or "unknown_system", {}


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [
        "display_name",
        "short_label",
        "system_id",
        "lane",
        "run_intent",
        "provider",
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
        "created_at",
        "run_group_id",
        "output_dir",
        "color",
        "reasoner_backend",
        "router_backend",
        "retriever_backend",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _infer_run_intent(path: Path, manifest: dict[str, Any]) -> str:
    manifest_intent = str(manifest.get("run_intent", "")).strip().lower()
    if manifest_intent in {"canonical", "exploratory"}:
        return manifest_intent
    lane = str(manifest.get("lane", "")).strip()
    if lane and path.name == lane:
        return "canonical"
    return "exploratory"


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


def _slugify(text: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in text]
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")
