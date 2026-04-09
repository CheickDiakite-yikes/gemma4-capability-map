from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import platform
import signal
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_capability_map.benchmark import ROOT
from gemma4_capability_map.hardware import detect_hardware_profile
from gemma4_capability_map.io import load_yaml
from gemma4_capability_map.metrics.failure_taxonomy import summarize_failure_tags
from gemma4_capability_map.schemas import RunTrace
from gemma4_capability_map.traces.exporters import export_leaderboard_csv
from gemma4_capability_map.traces.replay import load_traces
from gemma4_capability_map.traces.recorder import TraceRecorder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "alpha_matrix.yaml"))
    parser.add_argument("--output-root", default=str(ROOT / "results" / "alpha_matrix"))
    parser.add_argument("--experiment-id", action="append", dest="experiment_ids", default=[])
    parser.add_argument("--run-group-id")
    args = parser.parse_args()

    config = load_yaml(args.config)
    matrix = config["matrix"]
    experiments = matrix["experiments"]
    if args.experiment_ids:
        allowed = set(args.experiment_ids)
        experiments = [experiment for experiment in experiments if experiment["id"] in allowed]

    run_group_id = args.run_group_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    matrix_name = matrix["name"]
    output_dir = Path(args.output_root) / f"{run_group_id}_{matrix_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    backend_preflight = _load_backend_preflight()

    manifest_path = output_dir / "manifest.json"
    manifest = _load_existing_manifest(manifest_path) or {
        "run_group_id": run_group_id,
        "matrix_name": matrix_name,
        "description": matrix.get("description"),
        "created_at": datetime.now(UTC).isoformat(),
        "config_path": str(Path(args.config).resolve()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "hardware_profile": detect_hardware_profile().model_dump(mode="json"),
        "experiments": experiments,
        "probes": matrix.get("probes", []),
        "backend_preflight": backend_preflight,
    }
    manifest["experiments"] = experiments
    manifest["probes"] = matrix.get("probes", [])
    manifest["backend_preflight"] = backend_preflight
    _write_json(manifest_path, manifest)

    combined_traces: list[RunTrace] = []
    experiment_rows: list[dict[str, Any]] = []
    improvement_rows: list[dict[str, Any]] = []
    run_log_path = ROOT / "results" / "history" / "experiment_runs.jsonl"
    improvement_log_path = ROOT / "results" / "history" / "improvements.jsonl"
    previous_runs = _load_jsonl_records(run_log_path)

    for experiment in experiments:
        experiment_id = experiment["id"]
        experiment_dir = output_dir / experiment_id
        existing_summary = _read_existing_summary(experiment_dir / "summary.json")
        if existing_summary and existing_summary.get("status") in {"completed", "blocked"}:
            print(f"Skipping {existing_summary.get('status')} {experiment_id}...", flush=True)
            result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        else:
            blockers = _access_blockers_for_experiment(experiment, backend_preflight)
            if blockers:
                print(f"Blocking {experiment_id} before launch due to model access: {blockers}", flush=True)
                summary = _blocked_summary(
                    run_group_id=run_group_id,
                    matrix_name=matrix_name,
                    experiment=experiment,
                    blockers=blockers,
                )
                _write_json(experiment_dir / "summary.json", summary)
                _write_json(experiment_dir / "progress.json", summary)
                result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            else:
                action = "Resuming" if existing_summary else "Starting"
                print(f"{action} {experiment_id}...", flush=True)
                execution_mode = str(experiment.get("execution_mode", "subprocess"))
                if execution_mode == "in_process":
                    result = _run_experiment_in_process(
                        config_path=args.config,
                        experiment_id=experiment_id,
                        run_group_id=run_group_id,
                        experiment_dir=experiment_dir,
                        matrix_name=matrix_name,
                        experiment=experiment,
                        env_overrides=experiment.get("env", {}),
                    )
                else:
                    result = _run_experiment_subprocess(
                        config_path=args.config,
                        experiment_id=experiment_id,
                        run_group_id=run_group_id,
                        experiment_dir=experiment_dir,
                        env_overrides=experiment.get("env", {}),
                        timeout_minutes=experiment.get("timeout_minutes"),
                    )
                if result.stdout:
                    print(result.stdout, end="")
                if result.stderr:
                    print(result.stderr, end="", file=sys.stderr)
        summary, traces = _read_experiment_outputs(
            experiment_dir=experiment_dir,
            experiment=experiment,
            run_group_id=run_group_id,
            matrix_name=matrix_name,
            process=result,
        )
        experiment_rows.append(summary)
        if traces:
            combined_traces.extend(traces)

        run_record = {
            "run_group_id": run_group_id,
            "matrix_name": matrix_name,
            "created_at": datetime.now(UTC).isoformat(),
            "experiment": experiment,
            "summary": summary,
            "output_dir": str((output_dir / experiment_id).resolve()),
        }
        already_logged = _has_logged_run(previous_runs, run_group_id=run_group_id, experiment_id=experiment_id)
        if not already_logged:
            _append_jsonl_record(run_log_path, run_record)

        previous = _latest_matching_run(previous_runs, experiment_id=experiment_id, exclude_run_group_id=run_group_id)
        if previous is not None and not already_logged and summary.get("status") == "completed":
            improvement = _compute_improvement(
                current_summary=summary,
                previous_summary=previous["summary"],
                current_run_group_id=run_group_id,
                previous_run_group_id=previous["run_group_id"],
                experiment_id=experiment_id,
            )
            improvement_rows.append(improvement)
            _append_jsonl_record(improvement_log_path, improvement)
        if not already_logged:
            previous_runs.append(run_record)

    probes = _load_probe_rows(matrix.get("probes", []))
    _write_json(output_dir / "experiment_summaries.json", experiment_rows)
    _write_json(output_dir / "probe_summaries.json", probes)
    _write_json(output_dir / "improvements.json", improvement_rows)

    if combined_traces:
        recorder = TraceRecorder()
        for trace in combined_traces:
            recorder.add(trace)
        combined_trace_path = output_dir / "combined_traces.jsonl"
        recorder.write(combined_trace_path)
        export_leaderboard_csv(combined_traces, output_dir / "combined_leaderboard.csv")

    report = {
        "run_group_id": run_group_id,
        "matrix_name": matrix_name,
        "expected_experiments": len(experiments),
        "completed_experiments": sum(1 for row in experiment_rows if row.get("status") == "completed"),
        "blocked_experiments": sum(1 for row in experiment_rows if row.get("status") == "blocked"),
        "failed_experiments": sum(1 for row in experiment_rows if row.get("status") == "failed"),
        "matrix_complete": all(row.get("status") in {"completed", "blocked"} for row in experiment_rows) if experiment_rows else False,
        "experiments": experiment_rows,
        "probes": probes,
        "improvements": improvement_rows,
        "failure_breakdown": summarize_failure_tags(combined_traces),
        "backend_preflight": backend_preflight,
    }
    _write_json(output_dir / "summary.json", report)
    (output_dir / "summary.md").write_text(_markdown_summary(report), encoding="utf-8")
    print(f"Wrote alpha matrix report to {output_dir}")


def _read_experiment_outputs(
    experiment_dir: Path,
    experiment: dict[str, Any],
    run_group_id: str,
    matrix_name: str,
    process: subprocess.CompletedProcess[str],
) -> tuple[dict[str, Any], list[RunTrace]]:
    summary_path = experiment_dir / "summary.json"
    trace_path = experiment_dir / "traces.jsonl"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "runs": 0.0,
            "avg_latency_ms": 0.0,
            "avg_prompt_tokens": 0.0,
            "avg_completion_tokens": 0.0,
            "run_group_id": run_group_id,
            "matrix_name": matrix_name,
            "experiment_id": experiment["id"],
            "execution_mode": experiment.get("execution_mode", "subprocess"),
            "pipeline": experiment["pipeline"],
            "track": experiment.get("track"),
            "backend": experiment["backend"],
            "reasoner_backend": experiment.get("reasoner_backend") or experiment["backend"],
            "router_backend": experiment.get("router_backend"),
            "retriever_backend": experiment.get("retriever_backend"),
            "reasoner": experiment["reasoner"],
            "router": experiment.get("router"),
            "retriever": experiment.get("retriever"),
            "max_new_tokens": experiment.get("max_new_tokens"),
            "planning_max_new_tokens": experiment.get("planning_max_new_tokens"),
            "final_max_new_tokens": experiment.get("final_max_new_tokens"),
            "thinking_enabled": bool(experiment.get("thinking", False)),
            "variants": bool(experiment.get("variants", False)),
            "status": "failed",
            "completed_runs": 0,
            "total_runs": 0,
            "remaining_runs": 0,
            "notes": experiment.get("notes", []),
            "error": f"Missing summary output. exit_code={process.returncode}",
        }
    if process.returncode == 124 and summary.get("status") not in {"completed", "blocked"}:
        summary["status"] = "failed"
        summary["error"] = "Timed out while running experiment subprocess."
    traces = load_traces(trace_path) if trace_path.exists() else []
    return summary, traces


def _run_experiment_subprocess(
    config_path: str,
    experiment_id: str,
    run_group_id: str,
    experiment_dir: Path,
    env_overrides: dict[str, Any] | None,
    timeout_minutes: int | float | None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_alpha_experiment.py"),
        "--config",
        config_path,
        "--experiment-id",
        experiment_id,
        "--run-group-id",
        run_group_id,
        "--output-dir",
        str(experiment_dir),
    ]
    child_env = os.environ.copy()
    if env_overrides:
        child_env.update({str(key): str(value) for key, value in env_overrides.items()})
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(
            timeout=None if timeout_minutes in {None, 0} else float(timeout_minutes) * 60.0
        )
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        timeout_note = (
            f"\n[run_alpha_matrix] Timed out after {timeout_minutes} minute(s) for experiment {experiment_id}.\n"
        )
        stderr = (stderr or "") + timeout_note
        return subprocess.CompletedProcess(command, 124, stdout, stderr)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _run_experiment_in_process(
    config_path: str,
    experiment_id: str,
    run_group_id: str,
    experiment_dir: Path,
    matrix_name: str,
    experiment: dict[str, Any],
    env_overrides: dict[str, Any] | None,
) -> subprocess.CompletedProcess[str]:
    module = _load_alpha_experiment_module()
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    returncode = 0
    with _temporary_env(env_overrides), contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        try:
            module.run_experiment(
                run_group_id=run_group_id,
                matrix_name=matrix_name,
                output_dir=experiment_dir,
                experiment=experiment,
            )
        except Exception:
            returncode = 1
            traceback.print_exc()
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_alpha_experiment.py"),
        "--config",
        config_path,
        "--experiment-id",
        experiment_id,
        "--run-group-id",
        run_group_id,
        "--output-dir",
        str(experiment_dir),
    ]
    return subprocess.CompletedProcess(command, returncode, stdout_buffer.getvalue(), stderr_buffer.getvalue())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_existing_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_existing_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_backend_preflight() -> dict[str, Any] | None:
    preflight_path = ROOT / "results" / "tables" / "backend_preflight.json"
    if not preflight_path.exists():
        return None
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    payload["source"] = str(preflight_path.resolve())
    return payload


def _access_blockers_for_experiment(
    experiment: dict[str, Any],
    backend_preflight: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not backend_preflight:
        return []
    specialist_probe = backend_preflight.get("specialist_access_probe")
    if not isinstance(specialist_probe, dict):
        return []
    rows = specialist_probe.get("models", [])
    if not isinstance(rows, list):
        return []
    access_by_model = {
        str(row.get("model")): row
        for row in rows
        if isinstance(row, dict) and row.get("model")
    }
    candidates = [
        ("reasoner", experiment.get("reasoner"), experiment.get("reasoner_backend") or experiment.get("backend")),
        ("router", experiment.get("router"), experiment.get("router_backend")),
        ("retriever", experiment.get("retriever"), experiment.get("retriever_backend")),
    ]
    blockers: list[dict[str, Any]] = []
    for role, model_id, backend in candidates:
        if not model_id or backend not in {"hf", "hf_service"}:
            continue
        row = access_by_model.get(str(model_id))
        if not row:
            continue
        access = str(row.get("access"))
        if access != "available":
            blockers.append(
                {
                    "role": role,
                    "model": model_id,
                    "backend": backend,
                    "access": access,
                    "api_status": row.get("api_status"),
                    "config_status": row.get("config_status"),
                    "gated": row.get("gated"),
                }
            )
    return blockers


def _blocked_summary(
    run_group_id: str,
    matrix_name: str,
    experiment: dict[str, Any],
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "runs": 0.0,
        "avg_latency_ms": 0.0,
        "avg_prompt_tokens": 0.0,
        "avg_completion_tokens": 0.0,
        "run_group_id": run_group_id,
        "matrix_name": matrix_name,
        "experiment_id": experiment["id"],
        "execution_mode": experiment.get("execution_mode", "subprocess"),
        "pipeline": experiment["pipeline"],
        "track": experiment.get("track"),
        "backend": experiment["backend"],
        "reasoner_backend": experiment.get("reasoner_backend") or experiment["backend"],
        "router_backend": experiment.get("router_backend"),
        "retriever_backend": experiment.get("retriever_backend"),
        "reasoner": experiment["reasoner"],
        "router": experiment.get("router"),
        "retriever": experiment.get("retriever"),
        "max_new_tokens": experiment.get("max_new_tokens"),
        "planning_max_new_tokens": experiment.get("planning_max_new_tokens"),
        "final_max_new_tokens": experiment.get("final_max_new_tokens"),
        "thinking_enabled": bool(experiment.get("thinking", False)),
        "variants": bool(experiment.get("variants", False)),
        "status": "blocked",
        "completed_runs": 0,
        "total_runs": 0,
        "remaining_runs": 0,
        "notes": experiment.get("notes", []),
        "access_blockers": blockers,
        "error": "Blocked before launch due to model access restrictions.",
    }


_ALPHA_EXPERIMENT_MODULE: Any | None = None


def _load_alpha_experiment_module() -> Any:
    global _ALPHA_EXPERIMENT_MODULE
    if _ALPHA_EXPERIMENT_MODULE is not None:
        return _ALPHA_EXPERIMENT_MODULE
    module_path = ROOT / "scripts" / "run_alpha_experiment.py"
    spec = importlib.util.spec_from_file_location("moonie_run_alpha_experiment", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load alpha experiment module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _ALPHA_EXPERIMENT_MODULE = module
    return module


@contextlib.contextmanager
def _temporary_env(overrides: dict[str, Any] | None):
    previous: dict[str, str | None] = {}
    try:
        if overrides:
            for key, value in overrides.items():
                env_key = str(key)
                previous[env_key] = os.environ.get(env_key)
                os.environ[env_key] = str(value)
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _has_logged_run(rows: list[dict[str, Any]], run_group_id: str, experiment_id: str) -> bool:
    return any(
        row.get("run_group_id") == run_group_id and row.get("summary", {}).get("experiment_id") == experiment_id
        for row in rows
    )


def _latest_matching_run(
    rows: list[dict[str, Any]],
    experiment_id: str,
    exclude_run_group_id: str | None = None,
) -> dict[str, Any] | None:
    for row in reversed(rows):
        summary = row.get("summary", {})
        if exclude_run_group_id is not None and row.get("run_group_id") == exclude_run_group_id:
            continue
        if summary.get("experiment_id") == experiment_id and summary.get("status") == "completed":
            return row
    return None


def _compute_improvement(
    current_summary: dict[str, Any],
    previous_summary: dict[str, Any],
    current_run_group_id: str,
    previous_run_group_id: str,
    experiment_id: str,
) -> dict[str, Any]:
    current_success = float(current_summary.get("success_rate", 0.0))
    previous_success = float(previous_summary.get("success_rate", 0.0))
    current_latency = float(current_summary.get("avg_latency_ms", 0.0))
    previous_latency = float(previous_summary.get("avg_latency_ms", 0.0))
    return {
        "experiment_id": experiment_id,
        "current_run_group_id": current_run_group_id,
        "previous_run_group_id": previous_run_group_id,
        "success_rate_delta": round(current_success - previous_success, 4),
        "avg_latency_ms_delta": round(current_latency - previous_latency, 2),
        "latency_improvement_ms": round(previous_latency - current_latency, 2),
        "current_success_rate": current_success,
        "previous_success_rate": previous_success,
        "current_avg_latency_ms": current_latency,
        "previous_avg_latency_ms": previous_latency,
    }


def _load_probe_rows(probes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for probe in probes:
        source = ROOT / probe["source"]
        payload: dict[str, Any] = {"id": probe["id"], "kind": probe.get("kind"), "backend": probe.get("backend"), "model": probe.get("model"), "source": str(source)}
        if source.exists():
            data = json.loads(source.read_text(encoding="utf-8"))
            payload["status"] = data.get("status")
            payload["load_elapsed_ms"] = _stage_elapsed_ms(data, "runner_loaded")
            payload["runtime_device"] = _stage_value(data, "runner_loaded", "runtime_device")
            payload["load_mode"] = _stage_value(data, "runner_loaded", "load_mode")
        else:
            payload["status"] = "missing"
        payload["notes"] = probe.get("notes", [])
        rows.append(payload)
    return rows


def _stage_elapsed_ms(payload: dict[str, Any], stage_name: str) -> int | None:
    for stage in payload.get("stages", []):
        if isinstance(stage, dict) and stage.get("name") == stage_name:
            value = stage.get("elapsed_ms")
            if isinstance(value, int | float):
                return int(value)
    return None


def _stage_value(payload: dict[str, Any], stage_name: str, key: str) -> Any:
    for stage in payload.get("stages", []):
        if isinstance(stage, dict) and stage.get("name") == stage_name:
            return stage.get(key)
    return None


def _markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Alpha Matrix Summary",
        "",
        f"- Run group: `{report['run_group_id']}`",
        f"- Matrix: `{report['matrix_name']}`",
        f"- Completed experiments: `{report['completed_experiments']}/{report['expected_experiments']}`",
        f"- Matrix complete: `{report['matrix_complete']}`",
        "",
        "## Backend Posture",
        "",
    ]
    backend_preflight = report.get("backend_preflight")
    if isinstance(backend_preflight, dict):
        lines.extend(
            [
                f"- Recommended local reasoner backend: `{backend_preflight.get('recommended_local_reasoner_backend')}`",
                f"- HF token present: `{backend_preflight.get('auth', {}).get('token_present')}`",
                f"- Offline mode enabled: `{backend_preflight.get('offline_mode_enabled')}`",
                f"- Preflight source: `{backend_preflight.get('source')}`",
                "",
            ]
        )
    else:
        lines.extend(["- No backend preflight snapshot found.", ""])
    lines.extend(
        [
        "## Failure Breakdown",
        "",
        ]
    )
    if report.get("failure_breakdown"):
        for tag, count in report["failure_breakdown"].items():
            lines.append(f"- `{tag}`: {count}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Experiments",
            "",
            "| Experiment | Status | Exec mode | Pipeline | Track | Backend | Success | Strict interface | Recovered execution | Readiness | Avg latency ms | Runs |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["experiments"]:
        lines.append(
            f"| {row['experiment_id']} | {row['status']} | {row.get('execution_mode', 'subprocess')} | {row['pipeline']} | {row['track']} | {row['backend']} | "
            f"{row.get('success_rate', 0.0)} | {row.get('strict_interface_rate', 0.0)} | "
            f"{row.get('recovered_execution_rate', 0.0)} | {row.get('real_world_readiness_avg', 0.0)} | "
            f"{row.get('avg_latency_ms', 0.0)} | {row.get('runs', 0.0)} |"
        )
    lines.extend(["", "## Probes", ""])
    if report["probes"]:
        lines.extend(
            [
                "| Probe | Status | Backend | Model | Load ms | Device |",
                "| --- | --- | --- | --- | ---: | --- |",
            ]
        )
        for probe in report["probes"]:
            lines.append(
                f"| {probe['id']} | {probe.get('status')} | {probe.get('backend')} | {probe.get('model')} | "
                f"{probe.get('load_elapsed_ms')} | {probe.get('runtime_device')} |"
            )
    else:
        lines.append("No probes configured.")
    if report["improvements"]:
        lines.extend(["", "## Improvements", "", "| Experiment | Success delta | Latency improvement ms |", "| --- | ---: | ---: |"])
        for item in report["improvements"]:
            lines.append(
                f"| {item['experiment_id']} | {item['success_rate_delta']} | {item['latency_improvement_ms']} |"
            )
    failing_rows: list[dict[str, Any]] = []
    for experiment in report["experiments"]:
        for failure in experiment.get("failing_variants", []):
            failing_rows.append(
                {
                    "experiment_id": experiment["experiment_id"],
                    "variant_id": failure.get("variant_id"),
                    "failure_tags": failure.get("failure_tags", []),
                    "interface_reliability_score": failure.get("interface_reliability_score", 0.0),
                }
            )
    if failing_rows:
        lines.extend(
            [
                "",
                "## Failing Variants",
                "",
                "| Experiment | Variant | Failure tags | Interface reliability |",
                "| --- | --- | --- | ---: |",
            ]
        )
        for row in failing_rows[:12]:
            lines.append(
                f"| {row['experiment_id']} | {row['variant_id']} | {', '.join(row['failure_tags'])} | "
                f"{row['interface_reliability_score']} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
