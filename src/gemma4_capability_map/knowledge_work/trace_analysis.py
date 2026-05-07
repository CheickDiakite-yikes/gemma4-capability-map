from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from gemma4_capability_map.io import load_jsonl
from gemma4_capability_map.knowledge_work.h1 import H1WorkflowFamily, load_h1_slice


SYSTEM_DELTA_FIELDS = [
    "real_world_readiness_avg",
    "strict_interface_avg",
    "recovered_execution_avg",
    "controller_repair_avg",
    "argument_repair_avg",
    "controller_fallback_avg",
    "intent_override_avg",
    "raw_planning_clean_rate_avg",
]

DEFAULT_TOOL_CONTRACT_SYSTEM_ID = "mlx_gemma4_e2b_reasoner_only"
DEFAULT_NO_DIRECTIVE_SYSTEM_ID = "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive"


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


def compare_ablation_packets(baseline_packet_dir: str | Path, candidate_packet_dir: str | Path) -> dict[str, Any]:
    baseline = analyze_ablation_packet(baseline_packet_dir)
    candidate = analyze_ablation_packet(candidate_packet_dir)
    baseline_systems = {str(row["system_id"]): row for row in baseline["system_summaries"]}
    candidate_systems = {str(row["system_id"]): row for row in candidate["system_summaries"]}
    shared_system_ids = sorted(set(baseline_systems) & set(candidate_systems))

    system_delta_rows = []
    for system_id in shared_system_ids:
        baseline_row = baseline_systems[system_id]
        candidate_row = candidate_systems[system_id]
        row: dict[str, Any] = {
            "system_id": system_id,
            "baseline_lane": baseline_row.get("lane", ""),
            "candidate_lane": candidate_row.get("lane", ""),
        }
        for field in SYSTEM_DELTA_FIELDS:
            baseline_value = _float(baseline_row.get(field))
            candidate_value = _float(candidate_row.get(field))
            row[f"baseline_{field}"] = baseline_value
            row[f"candidate_{field}"] = candidate_value
            row[f"delta_{field}"] = _delta(candidate_value, baseline_value)
        system_delta_rows.append(row)

    note_delta_rows = _counter_delta_rows(
        baseline["note_counts"],
        candidate["note_counts"],
        key_fields=["system_id", "note"],
        count_field="count",
    )
    failure_mode_delta_rows = _counter_delta_rows(
        baseline["failure_mode_counts"],
        candidate["failure_mode_counts"],
        key_fields=["failure_mode"],
        count_field="count",
    )
    return {
        "baseline_packet_dir": str(Path(baseline_packet_dir).resolve()),
        "candidate_packet_dir": str(Path(candidate_packet_dir).resolve()),
        "baseline": {
            "system_count": baseline["system_count"],
            "episode_count": baseline["episode_count"],
            "note_count": baseline["note_count"],
            "failure_candidate_count": baseline["failure_candidate_count"],
        },
        "candidate": {
            "system_count": candidate["system_count"],
            "episode_count": candidate["episode_count"],
            "note_count": candidate["note_count"],
            "failure_candidate_count": candidate["failure_candidate_count"],
        },
        "deltas": {
            "shared_system_count": len(shared_system_ids),
            "note_count_delta": candidate["note_count"] - baseline["note_count"],
            "failure_candidate_count_delta": candidate["failure_candidate_count"] - baseline["failure_candidate_count"],
        },
        "system_deltas": system_delta_rows,
        "note_deltas": note_delta_rows,
        "failure_mode_deltas": failure_mode_delta_rows,
    }


def write_packet_comparison(
    baseline_packet_dir: str | Path,
    candidate_packet_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    comparison = compare_ablation_packets(baseline_packet_dir, candidate_packet_dir)
    target = Path(output_dir) if output_dir else Path(candidate_packet_dir)
    target.mkdir(parents=True, exist_ok=True)

    summary_path = target / "trace_packet_comparison.json"
    system_deltas_path = target / "trace_packet_system_deltas.csv"
    note_deltas_path = target / "trace_packet_note_deltas.csv"
    failure_mode_deltas_path = target / "trace_packet_failure_mode_deltas.csv"
    summary_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(system_deltas_path, comparison["system_deltas"])
    _write_csv(note_deltas_path, comparison["note_deltas"])
    _write_csv(failure_mode_deltas_path, comparison["failure_mode_deltas"])
    return {
        "summary": str(summary_path.resolve()),
        "system_deltas": str(system_deltas_path.resolve()),
        "note_deltas": str(note_deltas_path.resolve()),
        "failure_mode_deltas": str(failure_mode_deltas_path.resolve()),
    }


def summarize_tool_contract_packet(
    packet_dir: str | Path,
    *,
    contracted_system_id: str = DEFAULT_TOOL_CONTRACT_SYSTEM_ID,
    no_directive_system_id: str = DEFAULT_NO_DIRECTIVE_SYSTEM_ID,
) -> dict[str, Any]:
    root = Path(packet_dir)
    rows = _load_packet_system_rows(root)
    by_system = {str(row["system_id"]): row for row in rows}
    contracted = by_system.get(contracted_system_id)
    no_directive = by_system.get(no_directive_system_id)
    if contracted is None:
        raise ValueError(f"Contracted system `{contracted_system_id}` not found in {root}.")
    if no_directive is None:
        raise ValueError(f"No-directive system `{no_directive_system_id}` not found in {root}.")

    delta_rows = []
    for row in rows:
        delta_row = {
            "system_id": row["system_id"],
            "lane": row["lane"],
            "disabled_controls": row["disabled_controls"],
            "tool_turn_directive_enabled": row["tool_turn_directive_enabled"],
        }
        for field in SYSTEM_DELTA_FIELDS:
            value = _float(row.get(field))
            delta_row[field] = value
            delta_row[f"delta_vs_contracted_{field}"] = _delta(value, _float(contracted.get(field)))
            delta_row[f"delta_vs_no_directive_{field}"] = _delta(value, _float(no_directive.get(field)))
        delta_rows.append(delta_row)

    findings = {
        "contracted_system_id": contracted_system_id,
        "no_directive_system_id": no_directive_system_id,
        "contracted_readiness": _float(contracted.get("real_world_readiness_avg")),
        "no_directive_readiness": _float(no_directive.get("real_world_readiness_avg")),
        "no_directive_controller_repair": _float(no_directive.get("controller_repair_avg")),
        "no_directive_controller_fallback": _float(no_directive.get("controller_fallback_avg")),
        "no_directive_argument_repair": _float(no_directive.get("argument_repair_avg")),
        "no_directive_raw_planning_clean_rate": _float(no_directive.get("raw_planning_clean_rate_avg")),
    }
    findings["readiness_delta_no_directive_vs_contracted"] = _delta(
        findings["no_directive_readiness"],
        findings["contracted_readiness"],
    )
    return {
        "packet_dir": str(root.resolve()),
        "system_count": len(rows),
        "findings": findings,
        "system_rows": rows,
        "delta_rows": delta_rows,
    }


def write_tool_contract_summary(
    packet_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    contracted_system_id: str = DEFAULT_TOOL_CONTRACT_SYSTEM_ID,
    no_directive_system_id: str = DEFAULT_NO_DIRECTIVE_SYSTEM_ID,
) -> dict[str, str]:
    summary = summarize_tool_contract_packet(
        packet_dir,
        contracted_system_id=contracted_system_id,
        no_directive_system_id=no_directive_system_id,
    )
    target = Path(output_dir) if output_dir else Path(packet_dir)
    target.mkdir(parents=True, exist_ok=True)

    summary_path = target / "tool_contract_summary.json"
    csv_path = target / "tool_contract_system_deltas.csv"
    markdown_path = target / "tool_contract_summary.md"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(csv_path, summary["delta_rows"])
    markdown_path.write_text(_tool_contract_markdown(summary), encoding="utf-8")
    return {
        "summary": str(summary_path.resolve()),
        "system_deltas": str(csv_path.resolve()),
        "markdown": str(markdown_path.resolve()),
    }


def summarize_h1_workflow_families(packet_dir: str | Path, config_path: str | Path) -> dict[str, Any]:
    analysis = analyze_ablation_packet(packet_dir)
    config = load_h1_slice(config_path)
    family_by_episode = _family_by_episode(config.workflow_families)
    rows = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in analysis["episode_rows"]:
        family = family_by_episode.get(str(row["episode_id"]))
        if family is None:
            continue
        grouped[(str(row["system_id"]), family["workflow_id"])].append({**row, **family})

    for (system_id, workflow_id), episode_rows in sorted(grouped.items()):
        first = episode_rows[0]
        failures = [row for row in episode_rows if row["failure_candidate"]]
        failure_modes = sorted(
            {
                mode
                for row in failures
                for mode in str(row.get("failure_modes", "")).split(";")
                if mode
            }
        )
        rows.append(
            {
                "system_id": system_id,
                "workflow_id": workflow_id,
                "role_family": first["workflow_role_family"],
                "h1_stressors": ";".join(first["h1_stressors"]),
                "episode_count": len(episode_rows),
                "failure_candidate_count": len(failures),
                "failure_modes": ";".join(failure_modes),
                "real_world_readiness_avg": _avg(row["real_world_readiness_score"] for row in episode_rows),
                "strict_interface_avg": _avg(row["strict_interface_score"] for row in episode_rows),
                "recovered_execution_avg": _avg(row["recovered_execution_score"] for row in episode_rows),
                "controller_repair_avg": _avg(row["controller_repair_count"] for row in episode_rows),
                "controller_fallback_avg": _avg(row["controller_fallback_count"] for row in episode_rows),
                "argument_repair_avg": _avg(row["argument_repair_count"] for row in episode_rows),
                "raw_planning_clean_rate_avg": _avg(row["raw_planning_clean_rate"] for row in episode_rows),
                "episodes": ";".join(sorted(str(row["episode_id"]) for row in episode_rows)),
            }
        )

    failure_rows = [row for row in rows if int(row["failure_candidate_count"]) > 0]
    return {
        "packet_dir": str(Path(packet_dir).resolve()),
        "config_path": str(Path(config_path).resolve()),
        "workflow_row_count": len(rows),
        "workflow_failure_row_count": len(failure_rows),
        "workflow_rows": rows,
        "workflow_failure_rows": failure_rows,
    }


def write_h1_workflow_family_summary(
    packet_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    summary = summarize_h1_workflow_families(packet_dir, config_path)
    target = Path(output_dir) if output_dir else Path(packet_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "workflow_family_summary.json"
    rows_path = target / "workflow_family_system_rows.csv"
    failures_path = target / "workflow_family_failures.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(rows_path, summary["workflow_rows"])
    _write_csv(failures_path, summary["workflow_failure_rows"])
    return {
        "summary": str(summary_path.resolve()),
        "system_rows": str(rows_path.resolve()),
        "failures": str(failures_path.resolve()),
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


def _load_packet_system_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for run_dir in sorted(root.iterdir()):
        summary_path = run_dir / "summary.json"
        manifest_path = run_dir / "manifest.json"
        if not summary_path.exists() or not manifest_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        system_id = str(manifest.get("system_id") or run_dir.name.rsplit("__", 1)[0])
        controls = manifest.get("research_controls", {}) or {}
        runtime_reasoner = (manifest.get("runtime_bundle", {}) or {}).get("reasoner", {}) or {}
        row = _system_summary(system_id, str(manifest.get("lane") or summary.get("lane") or ""), summary, run_dir)
        row["disabled_controls"] = ";".join(sorted(key for key, value in controls.items() if value))
        row["tool_turn_directive_enabled"] = bool(runtime_reasoner.get("tool_turn_directive_enabled", True))
        rows.append(row)
    return rows


def _family_by_episode(workflow_families: list[H1WorkflowFamily]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for family in workflow_families:
        payload = {
            "workflow_id": family.workflow_id,
            "workflow_role_family": family.role_family,
            "workflow_purpose": family.purpose,
            "h1_stressors": list(family.h1_stressors),
        }
        mapping[family.replayable_episode_id] = payload
        mapping[family.live_episode_id] = payload
    return mapping


def _avg(values: Any) -> float:
    items = [_float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _tool_contract_markdown(summary: dict[str, Any]) -> str:
    findings = summary["findings"]
    lines = [
        "# Tool Contract Summary",
        "",
        f"- Contracted readiness: {findings['contracted_readiness']:.5f}",
        f"- No-directive readiness: {findings['no_directive_readiness']:.5f}",
        f"- No-directive controller repair/fallback/argument repair: {findings['no_directive_controller_repair']:.2f} / {findings['no_directive_controller_fallback']:.2f} / {findings['no_directive_argument_repair']:.2f}",
        f"- No-directive raw planning clean rate: {findings['no_directive_raw_planning_clean_rate']:.2f}",
        "",
        "| system_id | controls | readiness | strict | recovered | repair | fallback | arg repair | raw clean |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["delta_rows"]:
        lines.append(
            "| {system_id} | {controls} | {readiness:.5f} | {strict:.3f} | {recovered:.3f} | {repair:.2f} | {fallback:.2f} | {arg_repair:.2f} | {raw_clean:.2f} |".format(
                system_id=row["system_id"],
                controls=row["disabled_controls"] or "none",
                readiness=_float(row.get("real_world_readiness_avg")),
                strict=_float(row.get("strict_interface_avg")),
                recovered=_float(row.get("recovered_execution_avg")),
                repair=_float(row.get("controller_repair_avg")),
                fallback=_float(row.get("controller_fallback_avg")),
                arg_repair=_float(row.get("argument_repair_avg")),
                raw_clean=_float(row.get("raw_planning_clean_rate_avg")),
            )
        )
    return "\n".join(lines) + "\n"


def _counter_delta_rows(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    key_fields: list[str],
    count_field: str,
) -> list[dict[str, Any]]:
    baseline_counts = {
        tuple(str(row.get(field, "")) for field in key_fields): _float(row.get(count_field))
        for row in baseline_rows
    }
    candidate_counts = {
        tuple(str(row.get(field, "")) for field in key_fields): _float(row.get(count_field))
        for row in candidate_rows
    }
    rows = []
    for key in sorted(set(baseline_counts) | set(candidate_counts)):
        baseline_count = baseline_counts.get(key, 0.0)
        candidate_count = candidate_counts.get(key, 0.0)
        row = {field: value for field, value in zip(key_fields, key, strict=False)}
        row["baseline_count"] = baseline_count
        row["candidate_count"] = candidate_count
        row["delta_count"] = _delta(candidate_count, baseline_count)
        rows.append(row)
    return rows


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


def _delta(candidate_value: float, baseline_value: float) -> float:
    return round(candidate_value - baseline_value, 10)
