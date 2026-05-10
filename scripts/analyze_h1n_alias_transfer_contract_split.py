from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_capability_map.schemas import ToolCall, ToolSpec
from gemma4_capability_map.tools.executor import DeterministicExecutor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKET_DIR = (
    ROOT
    / "results"
    / "tool_probe_replay_packets"
    / "20260509T_visual_hard_slice_live_stress_alias_transfer_dry_run_v1"
)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "h1n_alias_transfer_contract_split"


@dataclass(frozen=True)
class ReplayRun:
    label: str
    path: Path


DEFAULT_RUNS: tuple[ReplayRun, ...] = (
    ReplayRun(
        "no_directive",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260509T_visual_hard_slice_live_stress_alias_transfer_no_directive_execute_v1",
    ),
    ReplayRun(
        "contracted",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260509T_visual_hard_slice_live_stress_alias_transfer_contracted_execute_v1",
    ),
    ReplayRun(
        "role_catalog_v1",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260509T_visual_hard_slice_live_stress_alias_transfer_role_catalog_execute_v1",
    ),
    ReplayRun(
        "argument_hints_v2",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260509T_visual_hard_slice_live_stress_alias_transfer_argument_hints_execute_v1",
    ),
    ReplayRun(
        "schema_field_hints_v4",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260509T_visual_hard_slice_live_stress_alias_transfer_schema_field_hints_execute_v1",
    ),
    ReplayRun(
        "schema_literal_targets_v5",
        ROOT
        / "results"
        / "tool_probe_replay_live"
        / "20260509T_visual_hard_slice_live_stress_alias_transfer_schema_literal_targets_execute_v1",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose the H1n alias-transfer strict-call vs executor-target contract split."
    )
    parser.add_argument("--packet-dir", default=str(DEFAULT_PACKET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = analyze_h1n_alias_transfer_contract_split(
        packet_dir=Path(args.packet_dir),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(payload["manifest"], indent=2, ensure_ascii=False))


def analyze_h1n_alias_transfer_contract_split(
    *,
    packet_dir: Path = DEFAULT_PACKET_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    runs: tuple[ReplayRun, ...] = DEFAULT_RUNS,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    packet_cases = _read_packet_cases(packet_dir)
    expected_call_rows = [_expected_call_row(case) for case in packet_cases]
    expected_by_case = {row["case_id"]: row for row in expected_call_rows}
    replay_rows = [
        _replay_row(run.label, row, expected_by_case[row["case_id"]])
        for run in runs
        for row in _read_probe_rows(run.path)
    ]
    summary_rows = _summary_rows(expected_call_rows, replay_rows)
    finding_rows = _finding_rows(expected_call_rows, replay_rows)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output_dir.resolve()),
        "packet_dir": str(packet_dir.resolve()),
        "case_count": len(packet_cases),
        "run_count": len(runs),
        "expected_call_contract_mismatch_count": sum(
            1 for row in expected_call_rows if row["expected_call_satisfies_execution"] is False
        ),
        "contracted_exact_non_executor_count": sum(
            1
            for row in replay_rows
            if row["label"] == "contracted"
            and row["exact_match"] is True
            and row["executor_target_match"] is False
        ),
        "argument_hints_executor_success_count": sum(
            1
            for row in replay_rows
            if row["label"] == "argument_hints_v2" and row["executor_target_match"] is True
        ),
        "purpose": "Separate H1n planner strict-call fidelity from executor-target oracle success.",
    }
    payload = {
        "manifest": manifest,
        "expected_call_rows": expected_call_rows,
        "replay_rows": replay_rows,
        "summary_rows": summary_rows,
        "finding_rows": finding_rows,
    }

    _write_csv(tables_dir / "h1n_expected_call_contract_audit.csv", expected_call_rows)
    _write_csv(tables_dir / "h1n_replay_contract_split.csv", replay_rows)
    _write_csv(tables_dir / "h1n_contract_split_summary.csv", summary_rows)
    _write_csv(tables_dir / "h1n_contract_split_findings.csv", finding_rows)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "diagnostic.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "diagnostic.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _read_packet_cases(packet_dir: Path) -> list[dict[str, Any]]:
    return json.loads((packet_dir / "replay_cases.json").read_text(encoding="utf-8"))


def _read_probe_rows(run_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_path / "runs").glob("*/probe_results.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _expected_call_row(case: dict[str, Any]) -> dict[str, Any]:
    expected_calls = [_tool_call(payload) for payload in case.get("expected_calls", [])]
    tool_specs = [ToolSpec.model_validate(payload) for payload in case.get("tool_specs", [])]
    execution = _execute_calls(case.get("initial_state", {}), tool_specs, expected_calls)
    execution_satisfies = _execution_satisfies_contract(execution, case.get("expected_execution", {}))
    validator_pass = bool(execution) and all(result.get("validator_result") == "pass" for result in execution)
    expected_call_satisfies = bool(expected_calls) and validator_pass and execution_satisfies
    return {
        "case_id": case["case_id"],
        "family": case.get("family", ""),
        "expected_call_count": len(expected_calls),
        "expected_call_satisfies_execution": expected_call_satisfies,
        "expected_call_validator_pass": validator_pass,
        "expected_call_classification": _classify_expected_call(execution, case.get("expected_execution", {})),
        "expected_calls": _json_compact(case.get("expected_calls", [])),
        "expected_execution": _json_compact(case.get("expected_execution", {})),
        "expected_call_execution": _json_compact(_compact_execution(execution)),
    }


def _replay_row(label: str, row: dict[str, Any], expected_row: dict[str, Any]) -> dict[str, Any]:
    exact_match = bool(row.get("exact_match"))
    executor_target_match = _optional_bool(row.get("executor_target_match"))
    executable_match = _optional_bool(row.get("executable_match"))
    return {
        "label": label,
        "case_id": row["case_id"],
        "family": row.get("family", ""),
        "exact_match": exact_match,
        "executable_match": executable_match,
        "executor_target_match": executor_target_match,
        "expected_call_satisfies_execution": expected_row["expected_call_satisfies_execution"],
        "expected_call_classification": expected_row["expected_call_classification"],
        "replay_classification": _classify_replay_row(
            exact_match=exact_match,
            executable_match=executable_match,
            executor_target_match=executor_target_match,
            actual_calls=row.get("actual_calls", []),
            expected_row=expected_row,
        ),
        "expected_calls": _json_compact(row.get("expected_calls", [])),
        "actual_calls": _json_compact(row.get("actual_calls", [])),
        "expected_execution": _json_compact(row.get("expected_execution", {})),
        "actual_execution": _json_compact(_compact_execution(row.get("actual_execution", []))),
    }


def _summary_rows(expected_call_rows: list[dict[str, Any]], replay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "label": "expected_call_contract",
            "case_count": len(expected_call_rows),
            "strict_exact_count": "",
            "executor_target_count": sum(
                1 for row in expected_call_rows if row["expected_call_satisfies_execution"] is True
            ),
            "exact_but_executor_miss_count": "",
            "nonexact_executor_success_count": "",
            "contract_mismatch_count": sum(
                1 for row in expected_call_rows if row["expected_call_satisfies_execution"] is False
            ),
            "interpretation": "Generated expected calls audited against the packet oracle.",
        }
    ]
    labels = sorted({row["label"] for row in replay_rows})
    for label in labels:
        label_rows = [row for row in replay_rows if row["label"] == label]
        rows.append(
            {
                "label": label,
                "case_count": len(label_rows),
                "strict_exact_count": sum(1 for row in label_rows if row["exact_match"] is True),
                "executor_target_count": sum(1 for row in label_rows if row["executor_target_match"] is True),
                "exact_but_executor_miss_count": sum(
                    1
                    for row in label_rows
                    if row["exact_match"] is True and row["executor_target_match"] is False
                ),
                "nonexact_executor_success_count": sum(
                    1
                    for row in label_rows
                    if row["exact_match"] is False and row["executor_target_match"] is True
                ),
                "contract_mismatch_count": sum(
                    1 for row in label_rows if row["expected_call_satisfies_execution"] is False
                ),
                "interpretation": _summary_interpretation(label, label_rows),
            }
        )
    return rows


def _finding_rows(expected_call_rows: list[dict[str, Any]], replay_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    contract_mismatch_count = sum(
        1 for row in expected_call_rows if row["expected_call_satisfies_execution"] is False
    )
    contracted_exact_non_executor = [
        row
        for row in replay_rows
        if row["label"] == "contracted" and row["exact_match"] is True and row["executor_target_match"] is False
    ]
    argument_hints_executor = [
        row for row in replay_rows if row["label"] == "argument_hints_v2" and row["executor_target_match"] is True
    ]
    return [
        {
            "finding_id": "expected_calls_are_not_oracle_calls",
            "finding": (
                f"{contract_mismatch_count} / {len(expected_call_rows)} generated expected-call contracts do not "
                "satisfy the packet's own expected_execution oracle."
            ),
            "implication": "H1n strict exactness partly measures matching the heuristic planner, not reaching the visual target.",
        },
        {
            "finding_id": "contracted_exactness_is_overstated_for_h1n",
            "finding": (
                f"Contracted MLX has {len(contracted_exact_non_executor)} exact rows that are not executor-equivalent."
            ),
            "implication": "The contracted 5/6 strict score should not be treated as a clean model-only upper bound on H1n target success.",
        },
        {
            "finding_id": "argument_hints_are_executor_oracle_winner",
            "finding": f"Argument hints v2 reaches {len(argument_hints_executor)} / 6 executor-target successes.",
            "implication": "For this transfer slice, executor-equivalence is the more faithful outcome metric than strict planner-call exactness.",
        },
    ]


def _summary_interpretation(label: str, rows: list[dict[str, Any]]) -> str:
    exact = sum(1 for row in rows if row["exact_match"] is True)
    executor = sum(1 for row in rows if row["executor_target_match"] is True)
    exact_non_executor = sum(1 for row in rows if row["exact_match"] is True and row["executor_target_match"] is False)
    if exact_non_executor:
        return f"{exact}/{len(rows)} strict but {exact_non_executor} exact rows miss the executor target."
    return f"{exact}/{len(rows)} strict and {executor}/{len(rows)} executor-target successes."


def _classify_expected_call(execution: list[dict[str, Any]], expected_execution: dict[str, Any]) -> str:
    if not execution:
        return "no_expected_call"
    if any(result.get("validator_result") != "pass" for result in execution):
        if any(_has_empty_region_id(result) for result in execution):
            return "expected_call_invalid_empty_region_id"
        return "expected_call_validator_failure"
    if _execution_satisfies_contract(execution, expected_execution):
        return "expected_call_reaches_executor_target"
    actual_region_ids = _last_output_list(execution, "region_ids")
    if actual_region_ids == []:
        return "expected_call_returns_empty_region_selection"
    return "expected_call_reaches_wrong_executor_target"


def _classify_replay_row(
    *,
    exact_match: bool,
    executable_match: bool | None,
    executor_target_match: bool | None,
    actual_calls: list[dict[str, Any]],
    expected_row: dict[str, Any],
) -> str:
    if not actual_calls:
        return "no_tool_call"
    if exact_match and executor_target_match is True:
        return "exact_and_executor_target_success"
    if exact_match and executor_target_match is False and expected_row["expected_call_satisfies_execution"] is False:
        return "exact_against_nonoracle_expected_call"
    if exact_match and executable_match is False:
        return "exact_but_execution_failure"
    if not exact_match and executor_target_match is True:
        return "nonexact_executor_target_success"
    if executor_target_match is False:
        return "executor_target_miss"
    return "not_evaluable"


def _execute_calls(initial_state: dict[str, Any], tool_specs: list[ToolSpec], calls: list[ToolCall]) -> list[dict[str, Any]]:
    executor = DeterministicExecutor(tool_specs=tool_specs)
    state = deepcopy(initial_state)
    rows = []
    for step, call in enumerate(calls, start=1):
        result = executor.step(state, call, step=step)
        state = result.state_after
        rows.append(
            {
                "step": result.step,
                "selected_tool": result.selected_tool,
                "arguments": result.arguments,
                "validator_result": result.validator_result,
                "output": result.output,
                "error": result.error,
            }
        )
    return rows


def _execution_satisfies_contract(execution: list[dict[str, Any]], expected_execution: dict[str, Any]) -> bool:
    if "region_ids" in expected_execution:
        expected_region_ids = [str(region_id) for region_id in expected_execution["region_ids"]]
        return _last_output_list(execution, "region_ids") == expected_region_ids
    if "region_id" in expected_execution:
        expected_region_id = str(expected_execution["region_id"])
        actual_region_id = _last_output_value(execution, "region_id")
        return actual_region_id == expected_region_id or _last_output_list(execution, "region_ids") == [
            expected_region_id
        ]
    return True


def _tool_call(payload: dict[str, Any]) -> ToolCall:
    return ToolCall(
        name=str(payload.get("name", "")),
        arguments=dict(payload.get("arguments", {})),
        source_format="oracle",
        raw=json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
    )


def _compact_execution(execution: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in execution:
        output = result.get("output", {}) or {}
        rows.append(
            {
                "selected_tool": result.get("selected_tool", ""),
                "arguments": result.get("arguments", {}),
                "validator_result": result.get("validator_result", ""),
                "region_ids": output.get("region_ids", []),
                "region_id": output.get("region_id", ""),
                "error": result.get("error"),
            }
        )
    return rows


def _last_output_list(execution: list[dict[str, Any]], key: str) -> list[str]:
    for result in reversed(execution):
        value = result.get("output", {}).get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    return []


def _last_output_value(execution: list[dict[str, Any]], key: str) -> str:
    for result in reversed(execution):
        value = result.get("output", {}).get(key)
        if value is not None:
            return str(value)
    return ""


def _has_empty_region_id(result: dict[str, Any]) -> bool:
    return result.get("selected_tool") == "read_region_text" and str(result.get("arguments", {}).get("region_id", "")) == ""


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(payload: dict[str, Any]) -> str:
    manifest = payload["manifest"]
    lines = [
        "# H1n Alias-Transfer Contract Split Diagnostic",
        "",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Cases: `{manifest['case_count']}`",
        f"- Replay runs: `{manifest['run_count']}`",
        f"- Expected-call contract mismatches: `{manifest['expected_call_contract_mismatch_count']}`",
        f"- Contracted exact-but-not-executor rows: `{manifest['contracted_exact_non_executor_count']}`",
        f"- Argument-hints executor successes: `{manifest['argument_hints_executor_success_count']}`",
        "",
        "## Findings",
        "",
        _markdown_table(payload["finding_rows"]),
        "",
        "## Summary",
        "",
        _markdown_table(payload["summary_rows"]),
        "",
        "## Expected-Call Contract Audit",
        "",
        _markdown_table(payload["expected_call_rows"]),
        "",
        "## Interpretation",
        "",
        "H1n exposed a benchmark-contract flaw: the packet's strict expected calls were generated by the heuristic planner, and most of those calls do not reach the visual oracle target when executed. Executor-equivalence is therefore the faithful outcome metric for this slice, while strict exactness should be reported as planner-call fidelity until H1n is rebuilt with oracle expected calls.",
        "",
    ]
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
