from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_capability_map.io import load_yaml
from gemma4_capability_map.models.hf_service import read_service_state, service_paths_for
from gemma4_capability_map.reporting.knowledge_work_board import DEFAULT_REGISTRY_PATH, load_model_registry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "knowledge_work_matrix.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "knowledge_work_matrix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a registry-driven KnowledgeWorkArena comparison matrix.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-group-id", default=None)
    parser.add_argument("--system-id", action="append", dest="system_ids", default=[])
    parser.add_argument("--lane", action="append", dest="lanes", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config) or {}
    matrix = config.get("matrix", {})
    registry = load_model_registry(args.registry)
    run_group_id = args.run_group_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_root) / f"{run_group_id}_{matrix.get('name', 'knowledge_work_matrix')}"
    output_root.mkdir(parents=True, exist_ok=True)

    run_specs = _build_run_specs(matrix, registry, allowed_system_ids=args.system_ids, allowed_lanes=args.lanes)
    manifest = {
        "run_group_id": run_group_id,
        "matrix_name": matrix.get("name", "knowledge_work_matrix"),
        "description": matrix.get("description", ""),
        "created_at": datetime.now(UTC).isoformat(),
        "config_path": str(Path(args.config).resolve()),
        "registry_path": str(Path(args.registry).resolve()),
        "run_intent": matrix.get("run_intent", "exploratory"),
        "update_latest": bool(matrix.get("update_latest", False)),
        "runs": run_specs,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return

    results: list[dict[str, Any]] = []
    for spec in run_specs:
        output_dir = output_root / spec["run_id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_knowledge_work_arena.py"),
            "--lane",
            spec["lane"],
            "--output-dir",
            str(output_dir),
            "--run-intent",
            spec["run_intent"],
            "--system-id",
            spec["system_id"],
            "--backend",
            spec["backend"],
            "--reasoner-backend",
            spec["reasoner_backend"],
            "--reasoner",
            spec["reasoner"],
        ]
        if not spec["update_latest"]:
            command.append("--no-update-latest")
        if spec["router"]:
            command.extend(["--router", spec["router"]])
        if spec["retriever"]:
            command.extend(["--retriever", spec["retriever"]])
        if spec["router_backend"]:
            command.extend(["--router-backend", spec["router_backend"]])
        if spec["retriever_backend"]:
            command.extend(["--retriever-backend", spec["retriever_backend"]])
        if spec["limit"] is not None:
            command.extend(["--limit", str(spec["limit"])])
        if spec.get("thinking"):
            command.append("--thinking")
        if spec.get("reasoner_max_new_tokens"):
            command.extend(["--reasoner-max-new-tokens", str(spec["reasoner_max_new_tokens"])])
        if spec.get("request_timeout_seconds"):
            command.extend(["--request-timeout-seconds", str(spec["request_timeout_seconds"])])

        timeout_seconds = spec.get("run_timeout_seconds") or None
        try:
            process = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
            result = _collect_run_result(spec, output_dir, process)
        except subprocess.TimeoutExpired as exc:
            result = _collect_timeout_result(spec, output_dir, exc, timeout_seconds=float(timeout_seconds or 0.0))

        if int(result.get("returncode", 0) or 0) != 0 and spec.get("backend") == "hf_service":
            cleanup = _stop_hf_service_for_spec(spec)
            if cleanup:
                result["service_cleanup"] = cleanup
        results.append(result)
        (output_root / f"{spec['run_id']}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    (output_root / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    failed_runs = sum(1 for result in results if int(result.get("returncode", 0) or 0) != 0)
    print(
        json.dumps(
            {
                "run_group_id": run_group_id,
                "runs": len(run_specs),
                "failed_runs": failed_runs,
                "output_dir": str(output_root.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _build_run_specs(
    matrix: dict[str, Any],
    registry: dict[str, dict[str, Any]],
    allowed_system_ids: list[str] | None = None,
    allowed_lanes: list[str] | None = None,
) -> list[dict[str, Any]]:
    systems = list(matrix.get("systems", []))
    lanes = list(matrix.get("lanes", []))
    if allowed_system_ids:
        allowed = set(allowed_system_ids)
        systems = [system_id for system_id in systems if system_id in allowed]
    if allowed_lanes:
        allowed = set(allowed_lanes)
        lanes = [lane for lane in lanes if lane in allowed]

    lane_limits = matrix.get("lane_limits", {})
    run_intent = str(matrix.get("run_intent", "exploratory"))
    update_latest = bool(matrix.get("update_latest", False))
    run_specs: list[dict[str, Any]] = []
    for system_id in systems:
        system_args = _system_run_args(system_id, registry)
        for lane in lanes:
            run_specs.append(
                {
                    "run_id": f"{system_id}__{lane}",
                    "system_id": system_id,
                    "lane": lane,
                    "limit": lane_limits.get(lane),
                    "run_intent": run_intent,
                    "update_latest": update_latest,
                    **system_args,
                }
            )
    return run_specs


def _system_run_args(system_id: str, registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    systems = registry.get("systems", {})
    meta = systems.get(system_id)
    if meta is None:
        raise ValueError(f"Unknown system `{system_id}`.")
    backend = str(meta.get("backend", ""))
    executor_mode = str(meta.get("executor_mode", ""))
    router = str(meta.get("router", "") or "")
    retriever = str(meta.get("retriever", "") or "")
    router_backend = ""
    retriever_backend = ""
    if executor_mode == "local_specialists":
        router_backend = "hf_service" if backend == "hf_service" else "hf"
        retriever_backend = "hf_service" if backend == "hf_service" else "hf"
    elif executor_mode == "local_reasoner":
        router_backend = "heuristic"
        retriever_backend = "heuristic"
        router = ""
        retriever = ""
    return {
        "backend": backend,
        "reasoner_backend": backend,
        "router_backend": router_backend,
        "retriever_backend": retriever_backend,
        "reasoner": str(meta.get("reasoner", "")),
        "router": router,
        "retriever": retriever,
        "thinking": bool(meta.get("thinking", False)),
        "reasoner_max_new_tokens": int(meta.get("reasoner_max_new_tokens", 96) or 96),
        "request_timeout_seconds": float(meta.get("request_timeout_seconds", 600.0) or 600.0),
        "run_timeout_seconds": float(meta.get("run_timeout_seconds", 0.0) or 0.0),
    }


def _collect_run_result(spec: dict[str, Any], output_dir: Path, process: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    result = {
        "run_id": spec["run_id"],
        "system_id": spec["system_id"],
        "lane": spec["lane"],
        "output_dir": str(output_dir.resolve()),
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"
    progress_path = output_dir / "progress.json"
    summary = _read_json(summary_path)
    manifest = _read_json(manifest_path)
    progress = _read_json(progress_path)
    if summary is not None:
        result["summary"] = summary
    if manifest is not None:
        result["manifest"] = manifest
    if progress is not None:
        result["progress"] = progress

    validation_error = _validation_error(manifest, progress, summary)
    if validation_error:
        result["validation_error"] = validation_error
        if process.returncode == 0:
            result["returncode"] = 1
    return result


def _collect_timeout_result(
    spec: dict[str, Any],
    output_dir: Path,
    exc: subprocess.TimeoutExpired,
    timeout_seconds: float,
) -> dict[str, Any]:
    stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout.decode("utf-8", errors="replace") if exc.stdout else "")
    stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode("utf-8", errors="replace") if exc.stderr else "")
    result = {
        "run_id": spec["run_id"],
        "system_id": spec["system_id"],
        "lane": spec["lane"],
        "output_dir": str(output_dir.resolve()),
        "returncode": 124,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": True,
        "timeout_seconds": timeout_seconds,
        "validation_error": "run_timeout",
    }
    summary = _read_json(output_dir / "summary.json")
    manifest = _read_json(output_dir / "manifest.json")
    progress = _read_json(output_dir / "progress.json")
    if summary is not None:
        result["summary"] = summary
    if manifest is not None:
        result["manifest"] = manifest
    if progress is not None:
        result["progress"] = progress
    return result


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _validation_error(
    manifest: dict[str, Any] | None,
    progress: dict[str, Any] | None,
    summary: dict[str, Any] | None,
) -> str | None:
    if progress is None:
        return "missing_progress"
    progress_status = str(progress.get("status", "") or "").strip().lower()
    if progress_status != "completed":
        return f"incomplete_progress:{progress_status or 'unknown'}"
    if manifest is None or summary is None:
        return "missing_summary_or_manifest"
    expected_runs = int(manifest.get("episode_count", 0) or 0)
    observed_runs = int(float(summary.get("runs", 0.0) or 0.0))
    if expected_runs and observed_runs < expected_runs:
        return f"incomplete_summary:{observed_runs}/{expected_runs}"
    return None


def _stop_hf_service_for_spec(spec: dict[str, Any]) -> str | None:
    reasoner = str(spec.get("reasoner", "") or "").strip()
    if not reasoner:
        return None
    paths = service_paths_for(reasoner, "auto")
    state = read_service_state(paths["state_path"]) or {}
    pid = state.get("pid")
    try:
        numeric_pid = int(pid)
    except (TypeError, ValueError):
        return None
    try:
        os.kill(numeric_pid, 15)
    except OSError:
        return None
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            os.kill(numeric_pid, 0)
        except OSError:
            return f"terminated:{numeric_pid}"
        time.sleep(0.1)
    try:
        os.kill(numeric_pid, 9)
    except OSError:
        return f"terminated:{numeric_pid}"
    return f"killed:{numeric_pid}"


if __name__ == "__main__":
    main()
