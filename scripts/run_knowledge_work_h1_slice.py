from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_capability_map.knowledge_work.h1 import DEFAULT_H1_SLICE_PATH, build_h1_run_specs, load_h1_slice, validate_h1_slice
from gemma4_capability_map.reporting.knowledge_work_board import DEFAULT_REGISTRY_PATH, load_model_registry


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "knowledge_work_h1_slice"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the config-backed H1 packaged-workflow KWA slice.")
    parser.add_argument("--config", default=str(DEFAULT_H1_SLICE_PATH))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-group-id", default=None)
    parser.add_argument("--run-set", choices=["primary", "comparison", "ablation", "all"], default="primary")
    parser.add_argument("--system-id", action="append", dest="system_ids", default=[])
    parser.add_argument("--lane", action="append", dest="lanes", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_h1_slice(args.config)
    validation_errors = validate_h1_slice(config)
    if validation_errors:
        raise SystemExit("Invalid H1 slice config:\n" + "\n".join(f"- {error}" for error in validation_errors))

    registry = load_model_registry(args.registry)
    run_specs = build_h1_run_specs(
        config,
        registry,
        lanes=args.lanes or None,
        run_set=args.run_set,
        system_ids=args.system_ids or None,
    )
    run_group_id = args.run_group_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_root) / f"{run_group_id}_{config.name}_{config.version}"
    output_root.mkdir(parents=True, exist_ok=True)
    commands = [_arena_command(spec, output_root / spec["run_id"]) for spec in run_specs]
    manifest = {
        "run_group_id": run_group_id,
        "created_at": datetime.now(UTC).isoformat(),
        "h1_slice": config.model_dump(mode="json"),
        "config_path": str(Path(args.config).resolve()),
        "registry_path": str(Path(args.registry).resolve()),
        "run_set": args.run_set,
        "dry_run": bool(args.dry_run),
        "runs": run_specs,
        "commands": commands,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"run_group_id": run_group_id, "runs": len(run_specs), "output_dir": str(output_root.resolve())}, indent=2, ensure_ascii=False))
        return

    results: list[dict[str, Any]] = []
    for spec, command in zip(run_specs, commands, strict=True):
        output_dir = output_root / spec["run_id"]
        output_dir.mkdir(parents=True, exist_ok=True)
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
            result = _collect_result(spec, output_dir, process)
        except subprocess.TimeoutExpired as exc:
            result = _collect_timeout(spec, output_dir, exc, timeout_seconds=float(timeout_seconds or 0.0))
        results.append(result)
        (output_root / f"{spec['run_id']}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    failed_runs = sum(1 for result in results if int(result.get("returncode", 0) or 0) != 0)
    (output_root / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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
    if failed_runs:
        raise SystemExit(1)


def _arena_command(spec: dict[str, Any], output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_knowledge_work_arena.py"),
        "--lane",
        spec["lane"],
        "--output-dir",
        str(output_dir),
        "--no-update-latest",
        "--run-intent",
        spec["run_intent"],
        "--system-id",
        spec["system_id"],
        "--pipeline-name",
        spec["pipeline_name"],
        "--backend",
        spec["backend"],
        "--reasoner-backend",
        spec["reasoner_backend"],
        "--reasoner",
        spec["reasoner"],
        "--reasoner-max-new-tokens",
        str(spec["reasoner_max_new_tokens"]),
        "--request-timeout-seconds",
        str(spec["request_timeout_seconds"]),
    ]
    if spec.get("router"):
        command.extend(["--router", spec["router"]])
    if spec.get("retriever"):
        command.extend(["--retriever", spec["retriever"]])
    if spec.get("router_backend"):
        command.extend(["--router-backend", spec["router_backend"]])
    if spec.get("retriever_backend"):
        command.extend(["--retriever-backend", spec["retriever_backend"]])
    if spec.get("thinking"):
        command.append("--thinking")
    if spec.get("disable_controller_repair"):
        command.append("--disable-controller-repair")
    if spec.get("disable_controller_fallback"):
        command.append("--disable-controller-fallback")
    if spec.get("disable_visual_rescue"):
        command.append("--disable-visual-rescue")
    if spec.get("disable_intent_priority"):
        command.append("--disable-intent-priority")
    if spec.get("disable_argument_repair"):
        command.append("--disable-argument-repair")
    if spec.get("disable_deterministic_visual_follow_on"):
        command.append("--disable-deterministic-visual-follow-on")
    if spec.get("disable_tool_turn_directive"):
        command.append("--disable-tool-turn-directive")
    for episode_id in spec["episode_ids"]:
        command.extend(["--episode-id", episode_id])
    return command


def _collect_result(spec: dict[str, Any], output_dir: Path, process: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    result = {
        "run_id": spec["run_id"],
        "system_id": spec["system_id"],
        "lane": spec["lane"],
        "output_dir": str(output_dir.resolve()),
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }
    for name in ("summary", "manifest", "progress"):
        payload = _read_json(output_dir / f"{name}.json")
        if payload is not None:
            result[name] = payload
    return result


def _collect_timeout(
    spec: dict[str, Any],
    output_dir: Path,
    exc: subprocess.TimeoutExpired,
    timeout_seconds: float,
) -> dict[str, Any]:
    stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout.decode("utf-8", errors="replace") if exc.stdout else "")
    stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode("utf-8", errors="replace") if exc.stderr else "")
    return {
        "run_id": spec["run_id"],
        "system_id": spec["system_id"],
        "lane": spec["lane"],
        "output_dir": str(output_dir.resolve()),
        "returncode": 124,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": True,
        "timeout_seconds": timeout_seconds,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
