from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from gemma4_capability_map.io import load_jsonl


def analyze_ablation_packet(packet_dir: str | Path) -> dict[str, Any]:
    root = Path(packet_dir)
    if not root.exists():
        raise FileNotFoundError(f"Packet directory not found: {root}")

    system_summaries: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    note_counter: Counter[tuple[str, str]] = Counter()
    note_episodes: dict[tuple[str, str], set[str]] = defaultdict(set)
    note_tasks: dict[tuple[str, str], set[str]] = defaultdict(set)

    for run_dir in sorted(root.iterdir()):
        traces_path = run_dir / "episode_traces.jsonl"
        summary_path = run_dir / "summary.json"
        manifest_path = run_dir / "manifest.json"
        if not traces_path.exists() or not summary_path.exists() or not manifest_path.exists():
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        system_id = str(manifest.get("system_id") or run_dir.name.rsplit("__", 1)[0])
        lane = str(manifest.get("lane") or summary.get("lane") or "")

        system_summaries.append(_system_summary(system_id, lane, summary, run_dir))

        for episode in load_jsonl(traces_path):
            episode_id = str(episode.get("episode_id", ""))
            task_notes: Counter[str] = Counter()
            planning_turns = 0
            raw_planning_samples: list[str] = []
            for stage_id, task in _iter_task_traces(episode):
                task_id = str(task.get("task_id", ""))
                raw_outputs = [str(raw) for raw in _task_list_value(task, "planning_raw_outputs") if raw]
                planning_turns += len(raw_outputs)
                raw_planning_samples.extend(raw_outputs)
                for note in _flatten_notes(_task_list_value(task, "planning_repair_notes")):
                    task_notes[note] += 1
                    key = (system_id, note)
                    note_counter[key] += 1
                    note_episodes[key].add(episode_id)
                    note_tasks[key].add(f"{stage_id}:{task_id}" if stage_id else task_id)

            scorecard = episode.get("scorecard", {}) or {}
            failed_tools = _failed_tool_labels(episode)
            notes = sorted(task_notes)
            raw_planning_text = _joined_samples(raw_planning_samples)
            raw_planning_sample = _compact_sample(raw_planning_samples)
            episode_rows.append(
                {
                    "system_id": system_id,
                    "lane": lane,
                    "episode_id": episode_id,
                    "role_family": str(episode.get("role_family", "")),
                    "real_world_readiness_score": _float(scorecard.get("role_readiness_score")),
                    "strict_interface_score": _float(scorecard.get("strict_interface_score")),
                    "recovered_execution_score": _float(scorecard.get("recovered_execution_score")),
                    "controller_repair_count": _float(scorecard.get("controller_repair_count")),
                    "argument_repair_count": _float(scorecard.get("argument_repair_count")),
                    "controller_fallback_count": _float(scorecard.get("controller_fallback_count")),
                    "intent_override_count": _float(scorecard.get("intent_override_count")),
                    "raw_planning_clean_rate": _float(scorecard.get("raw_planning_clean_rate")),
                    "planning_turns": planning_turns,
                    "repair_notes": ";".join(notes),
                    "repair_note_counts": ";".join(f"{note}={task_notes[note]}" for note in notes),
                    "failed_tools": ";".join(failed_tools),
                    "failure_modes": ";".join(
                        _failure_modes(
                            notes,
                            failed_tools,
                            raw_planning_text,
                            system_id=system_id,
                            tool_calls=episode.get("tool_calls", []) or [],
                        )
                    ),
                    "raw_planning_sample": raw_planning_sample,
                    "failure_candidate": _is_failure_candidate(scorecard),
                }
            )

    note_rows = [
        {
            "system_id": system_id,
            "note": note,
            "count": count,
            "episode_count": len(note_episodes[(system_id, note)]),
            "task_count": len(note_tasks[(system_id, note)]),
            "episodes": ";".join(sorted(note_episodes[(system_id, note)])),
        }
        for (system_id, note), count in sorted(note_counter.items())
    ]
    failure_rows = [row for row in episode_rows if row["failure_candidate"]]
    failure_mode_rows = _failure_mode_counts(failure_rows)

    return {
        "packet_dir": str(root.resolve()),
        "system_count": len(system_summaries),
        "episode_count": len(episode_rows),
        "note_count": sum(note_counter.values()),
        "failure_candidate_count": len(failure_rows),
        "system_summaries": system_summaries,
        "note_counts": note_rows,
        "episode_rows": episode_rows,
        "failure_rows": failure_rows,
        "failure_mode_counts": failure_mode_rows,
    }


def write_trace_analysis(packet_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, str]:
    analysis = analyze_ablation_packet(packet_dir)
    target = Path(output_dir) if output_dir else Path(packet_dir)
    target.mkdir(parents=True, exist_ok=True)

    summary_path = target / "trace_note_summary.json"
    note_path = target / "trace_note_counts.csv"
    failures_path = target / "trace_episode_failures.csv"
    failure_modes_path = target / "trace_failure_mode_counts.csv"

    summary_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(note_path, analysis["note_counts"])
    _write_csv(failures_path, analysis["failure_rows"])
    _write_csv(failure_modes_path, analysis["failure_mode_counts"])
    return {
        "summary": str(summary_path.resolve()),
        "note_counts": str(note_path.resolve()),
        "failures": str(failures_path.resolve()),
        "failure_modes": str(failure_modes_path.resolve()),
    }


def _system_summary(system_id: str, lane: str, summary: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    fields = [
        "runs",
        "real_world_readiness_avg",
        "strict_interface_avg",
        "recovered_execution_avg",
        "controller_repair_avg",
        "argument_repair_avg",
        "controller_fallback_avg",
        "intent_override_avg",
        "raw_planning_clean_rate_avg",
    ]
    row = {
        "system_id": system_id,
        "lane": lane,
        "output_dir": str(run_dir.resolve()),
    }
    for field in fields:
        row[field] = _float(summary.get(field))
    return row


def _iter_task_traces(episode: dict[str, Any]):
    for stage in episode.get("stage_traces", []) or []:
        stage_id = str(stage.get("stage_id", ""))
        for task in stage.get("task_traces", []) or []:
            if isinstance(task, dict):
                yield stage_id, task


def _task_list_value(task: dict[str, Any], key: str) -> list[Any]:
    values: list[Any] = []
    direct = task.get(key)
    if isinstance(direct, list):
        values.extend(direct)
    prompt_artifacts = task.get("prompt_artifacts", {})
    if isinstance(prompt_artifacts, dict):
        nested = prompt_artifacts.get(key)
        if isinstance(nested, list):
            values.extend(nested)
    return values


def _flatten_notes(notes: Any) -> list[str]:
    flattened: list[str] = []
    for item in notes or []:
        if isinstance(item, list):
            flattened.extend(str(note) for note in item if note)
        elif item:
            flattened.append(str(item))
    return flattened


def _failed_tool_labels(episode: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for call in episode.get("tool_calls", []) or []:
        if call.get("validator_result") == "pass":
            continue
        tool_name = str(call.get("tool_name", ""))
        task_id = str(call.get("task_id", ""))
        labels.append(f"{task_id}:{tool_name}" if task_id else tool_name)
    return sorted(label for label in labels if label)


def _failure_modes(
    notes: list[str],
    failed_tools: list[str],
    raw_planning_text: str,
    *,
    system_id: str,
    tool_calls: list[Any],
) -> list[str]:
    modes: list[str] = []

    def add(mode: str) -> None:
        if mode not in modes:
            modes.append(mode)

    raw_lower = raw_planning_text.lower()
    if (
        "i cannot assist" in raw_lower
        or "i cannot do" in raw_lower
        or "i cannot proceed" in raw_lower
        or "current capabilities" in raw_lower
        or "tools are not applicable" in raw_lower
    ):
        add("raw_refusal")
    if "call:tool_name" in raw_planning_text or any(label.endswith(":tool_name") or label == "tool_name" for label in failed_tools):
        add("generic_tool_name")
    if "controller_fallback_disabled" in notes:
        add("fallback_disabled")
    if "controller_repair_disabled" in notes:
        add("repair_disabled")
    if "controller_fallback_planner" in notes:
        add("fallback_planner")
    if any(note.startswith("repaired_arguments:") or note == "argument_repair_disabled" for note in notes):
        add("argument_repair")
    if any(note.startswith("intent_prior:") or note == "intent_priority_disabled" for note in notes):
        add("intent_prior")
    if (
        any(note.startswith("feedback_prior:") or note == "deterministic_visual_follow_on_disabled" for note in notes)
        or "no_deterministic_visual_follow_on" in system_id
    ):
        add("visual_follow_on")
    if _has_visual_stepwise_sequence(tool_calls):
        add("visual_stepwise_control")
    if _has_repeated_visual_refinement(tool_calls):
        add("visual_repeated_refinement")
    if _has_visual_refinement_without_readback(tool_calls):
        add("visual_readback_missing")
    if any(note.startswith("canonicalized_tool:") for note in notes):
        add("tool_canonicalization")
    return modes


def _visual_tool_calls(tool_calls: list[Any]) -> list[dict[str, Any]]:
    visual_names = {"extract_layout", "segment_entities", "refine_selection", "read_region_text"}
    calls: list[dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        if str(call.get("tool_name", "")) in visual_names:
            calls.append(call)
    return calls


def _has_visual_stepwise_sequence(tool_calls: list[Any]) -> bool:
    names = [str(call.get("tool_name", "")) for call in _visual_tool_calls(tool_calls)]
    return any(name in names for name in {"extract_layout", "segment_entities"}) and "refine_selection" in names


def _has_repeated_visual_refinement(tool_calls: list[Any]) -> bool:
    seen: set[tuple[str, str]] = set()
    for call in _visual_tool_calls(tool_calls):
        if str(call.get("tool_name", "")) != "refine_selection":
            continue
        arguments = call.get("arguments", {}) if isinstance(call.get("arguments"), dict) else {}
        key = (str(arguments.get("selection_id", "")), str(arguments.get("filter_query", "")))
        if key in seen:
            return True
        seen.add(key)
    return False


def _has_visual_refinement_without_readback(tool_calls: list[Any]) -> bool:
    names = [str(call.get("tool_name", "")) for call in _visual_tool_calls(tool_calls)]
    return "refine_selection" in names and "read_region_text" not in names


def _failure_mode_counts(failure_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    systems: dict[str, set[str]] = defaultdict(set)
    episodes: dict[str, set[str]] = defaultdict(set)
    for row in failure_rows:
        system_id = str(row.get("system_id", ""))
        episode_id = str(row.get("episode_id", ""))
        for mode in str(row.get("failure_modes", "")).split(";"):
            if not mode:
                continue
            counter[mode] += 1
            systems[mode].add(system_id)
            episodes[mode].add(episode_id)
    return [
        {
            "failure_mode": mode,
            "count": count,
            "system_count": len(systems[mode]),
            "episode_count": len(episodes[mode]),
            "systems": ";".join(sorted(systems[mode])),
            "episodes": ";".join(sorted(episodes[mode])),
        }
        for mode, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _is_failure_candidate(scorecard: dict[str, Any]) -> bool:
    return _float(scorecard.get("strict_interface_score"), 1.0) < 1.0 or _float(scorecard.get("recovered_execution_score"), 1.0) < 1.0


def _compact_sample(samples: list[str], limit: int = 240) -> str:
    text = _joined_samples(samples)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _joined_samples(samples: list[str]) -> str:
    return " | ".join(sample.replace("\n", " ").strip() for sample in samples if sample.strip())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
