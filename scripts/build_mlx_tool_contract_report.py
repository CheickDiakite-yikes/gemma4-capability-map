from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_capability_map.reporting.knowledge_work_board import DEFAULT_REGISTRY_PATH, load_model_registry
from gemma4_capability_map.research_controls import ResearchControls
from gemma4_capability_map.tools.prompt_contracts import TOOL_PROMPT_CONTRACTS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reports" / "mlx_tool_contract_harnessing"
DEFAULT_H1F_PACKET = (
    ROOT
    / "results"
    / "knowledge_work_h1_slice"
    / "20260506T_h1f_mlx_no_directive_v1_knowledge_work_h1f_mlx_tool_contract_ablation_v1"
)
DEFAULT_H1H_PACKET = (
    ROOT
    / "results"
    / "knowledge_work_h1_slice"
    / "20260507T_h1h_mlx_full_no_directive_v1_knowledge_work_h1h_mlx_full_tool_contract_ablation_v1"
)
DEFAULT_H1I_PACKET = (
    ROOT
    / "results"
    / "knowledge_work_h1_slice"
    / "20260507T_h1i_mlx_worst_no_directive_v1_knowledge_work_h1i_mlx_worst_family_tool_contract_v1"
)
DEFAULT_PROBE_COMPARISON = (
    ROOT / "results" / "tool_directive_probe" / "20260507T_mlx_no_directive_probe_v1" / "probe_comparison.json"
)
DEFAULT_GEMINI_PACKET = ROOT / "results" / "gemini_cli" / "20260507T_h1h_gemini_cli_dry_run_baseline_v1"
DEFAULT_PROMPT_CONTRACT_PACKET = (
    ROOT / "results" / "tool_prompt_contract_probe_packets" / "20260507T_prompt_contract_candidates_execute_v1"
)

SYSTEM_LABELS = {
    "mlx_gemma4_e2b_reasoner_only": "contracted",
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive": "no directive",
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_repair": "no directive + no repair",
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_fallback": "no directive + no fallback",
    "mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_argument_repair": "no directive + no arg repair",
}


def build_report(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    h1f_packet: str | Path = DEFAULT_H1F_PACKET,
    h1h_packet: str | Path = DEFAULT_H1H_PACKET,
    h1i_packet: str | Path = DEFAULT_H1I_PACKET,
    probe_comparison_path: str | Path = DEFAULT_PROBE_COMPARISON,
    gemini_packet: str | Path = DEFAULT_GEMINI_PACKET,
    prompt_contract_packet: str | Path = DEFAULT_PROMPT_CONTRACT_PACKET,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    target = Path(output_dir)
    tables_dir = target / "tables"
    figures_dir = target / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    packets = [
        _packet_payload("H1f compact", Path(h1f_packet)),
        _packet_payload("H1h full", Path(h1h_packet)),
        _packet_payload("H1i worst-family", Path(h1i_packet)),
    ]
    probe = json.loads(Path(probe_comparison_path).read_text(encoding="utf-8"))
    gemini_manifest = json.loads((Path(gemini_packet) / "manifest.json").read_text(encoding="utf-8"))
    registry = load_model_registry(registry_path)

    packet_rows = [_packet_summary_row(packet) for packet in packets]
    h1i_system_rows = _system_metric_rows(packets[-1]["tool_contract"]["system_rows"])
    probe_case_rows = probe["case_deltas"]
    probe_failure_rows = _probe_failure_rows(probe_case_rows)
    h1i_failure_rows = _csv_rows(Path(h1i_packet) / "trace_failure_mode_counts.csv")
    h1i_workflow_failures = _csv_rows(Path(h1i_packet) / "workflow_family_failures.csv")
    candidate_rows = _prompt_contract_candidate_rows(registry)
    prompt_contract_gate_rows = _csv_rows(Path(prompt_contract_packet) / "candidate_gate_summary.csv")
    prompt_contract_failure_rows = _csv_rows(Path(prompt_contract_packet) / "candidate_failure_mode_counts.csv")

    _write_csv(tables_dir / "packet_summary.csv", packet_rows)
    _write_csv(tables_dir / "h1i_system_metrics.csv", h1i_system_rows)
    _write_csv(tables_dir / "probe_case_deltas.csv", probe_case_rows)
    _write_csv(tables_dir / "probe_failure_modes.csv", probe_failure_rows)
    _write_csv(tables_dir / "h1i_failure_modes.csv", h1i_failure_rows)
    _write_csv(tables_dir / "h1i_workflow_failures.csv", h1i_workflow_failures)
    _write_csv(tables_dir / "prompt_contract_candidates.csv", candidate_rows)
    _write_csv(tables_dir / "prompt_contract_probe_gates.csv", prompt_contract_gate_rows)
    _write_csv(tables_dir / "prompt_contract_probe_failure_modes.csv", prompt_contract_failure_rows)

    _write_grouped_metric_svg(
        figures_dir / "h1i_readiness_strict_recovered.svg",
        title="H1i readiness vs interface recovery",
        rows=h1i_system_rows,
        label_field="label",
        metrics=[
            ("real_world_readiness_avg", "readiness", "#2563EB"),
            ("strict_interface_avg", "strict", "#059669"),
            ("recovered_execution_avg", "recovered", "#D97706"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "h1h_h1i_controller_burden.svg",
        title="No-directive controller burden: H1h vs H1i",
        rows=[
            _burden_row("H1h full", packets[1]),
            _burden_row("H1i worst-family", packets[2]),
        ],
        label_field="packet",
        metrics=[
            ("controller_repair_avg", "repair", "#7C3AED"),
            ("controller_fallback_avg", "fallback", "#DC2626"),
            ("argument_repair_avg", "arg repair", "#0891B2"),
            ("raw_planning_clean_rate_avg", "raw clean", "#16A34A"),
        ],
    )
    _write_grouped_metric_svg(
        figures_dir / "tool_probe_contract_gap.svg",
        title="Tool probe contract gap",
        rows=[
            {
                "label": "contracted",
                "exact_match_rate": probe["baseline_exact_match_rate"],
                "executable_match_rate": probe["baseline_executable_match_rate"],
            },
            {
                "label": "no directive",
                "exact_match_rate": probe["candidate_exact_match_rate"],
                "executable_match_rate": probe["candidate_executable_match_rate"],
            },
        ],
        label_field="label",
        metrics=[
            ("exact_match_rate", "exact", "#2563EB"),
            ("executable_match_rate", "executable", "#059669"),
        ],
    )
    _write_bar_svg(
        figures_dir / "h1i_failure_modes.svg",
        title="H1i failure candidate modes",
        rows=[{"label": row["failure_mode"], "value": int(row["count"])} for row in h1i_failure_rows],
        color="#B91C1C",
    )
    _write_bar_svg(
        figures_dir / "prompt_contract_candidate_targets.svg",
        title="Prompt contract candidate target tags",
        rows=_candidate_tag_rows(candidate_rows),
        color="#0F766E",
    )
    _write_grouped_metric_svg(
        figures_dir / "prompt_contract_probe_gate.svg",
        title="Executed prompt contract probe gate",
        rows=prompt_contract_gate_rows,
        label_field="tool_prompt_contract_id",
        metrics=[
            ("exact_match_rate", "exact", "#2563EB"),
            ("executable_match_rate", "executable", "#059669"),
            ("delta_exact_vs_no_directive", "delta exact", "#D97706"),
        ],
    )

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(target.resolve()),
        "source_packets": {
            packet["name"]: str(packet["packet_dir"]) for packet in packets
        },
        "probe_comparison": str(Path(probe_comparison_path).resolve()),
        "gemini_packet": str(Path(gemini_packet).resolve()),
        "prompt_contract_packet": str(Path(prompt_contract_packet).resolve()),
        "registry_path": str(Path(registry_path).resolve()),
        "table_count": 9,
        "figure_count": 6,
    }
    report_payload = {
        "manifest": manifest,
        "packet_summary": packet_rows,
        "h1i_system_metrics": h1i_system_rows,
        "probe_failure_modes": probe_failure_rows,
        "prompt_contract_candidates": candidate_rows,
        "prompt_contract_probe_gates": prompt_contract_gate_rows,
        "prompt_contract_probe_failure_modes": prompt_contract_failure_rows,
        "gemini": gemini_manifest,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (target / "report.json").write_text(json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (target / "report.md").write_text(_markdown_report(report_payload), encoding="utf-8")
    return report_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the current MLX tool-contract research report.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--h1f-packet", default=str(DEFAULT_H1F_PACKET))
    parser.add_argument("--h1h-packet", default=str(DEFAULT_H1H_PACKET))
    parser.add_argument("--h1i-packet", default=str(DEFAULT_H1I_PACKET))
    parser.add_argument("--probe-comparison", default=str(DEFAULT_PROBE_COMPARISON))
    parser.add_argument("--gemini-packet", default=str(DEFAULT_GEMINI_PACKET))
    parser.add_argument("--prompt-contract-packet", default=str(DEFAULT_PROMPT_CONTRACT_PACKET))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_report(
        output_dir=args.output_dir,
        h1f_packet=args.h1f_packet,
        h1h_packet=args.h1h_packet,
        h1i_packet=args.h1i_packet,
        probe_comparison_path=args.probe_comparison,
        gemini_packet=args.gemini_packet,
        prompt_contract_packet=args.prompt_contract_packet,
        registry_path=args.registry,
    )
    print(
        json.dumps(
            {
                "output_dir": payload["manifest"]["output_dir"],
                "table_count": payload["manifest"]["table_count"],
                "figure_count": payload["manifest"]["figure_count"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _packet_payload(name: str, packet_dir: Path) -> dict[str, Any]:
    return {
        "name": name,
        "packet_dir": packet_dir.resolve(),
        "tool_contract": json.loads((packet_dir / "tool_contract_summary.json").read_text(encoding="utf-8")),
        "trace_summary": json.loads((packet_dir / "trace_note_summary.json").read_text(encoding="utf-8")),
    }


def _packet_summary_row(packet: dict[str, Any]) -> dict[str, Any]:
    findings = packet["tool_contract"]["findings"]
    rows = {row["system_id"]: row for row in packet["tool_contract"]["system_rows"]}
    no_repair = rows["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_repair"]
    no_fallback = rows["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_controller_fallback"]
    no_args = rows["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive_no_argument_repair"]
    return {
        "packet": packet["name"],
        "episode_count": int(rows["mlx_gemma4_e2b_reasoner_only"]["runs"]),
        "contracted_readiness": _round(findings["contracted_readiness"]),
        "no_directive_readiness": _round(findings["no_directive_readiness"]),
        "readiness_delta_no_directive_vs_contracted": _round(findings["readiness_delta_no_directive_vs_contracted"]),
        "no_directive_controller_repair": _round(findings["no_directive_controller_repair"]),
        "no_directive_controller_fallback": _round(findings["no_directive_controller_fallback"]),
        "no_directive_argument_repair": _round(findings["no_directive_argument_repair"]),
        "no_directive_raw_clean": _round(findings["no_directive_raw_planning_clean_rate"]),
        "no_repair_readiness": _round(no_repair["real_world_readiness_avg"]),
        "no_fallback_readiness": _round(no_fallback["real_world_readiness_avg"]),
        "no_argument_repair_readiness": _round(no_args["real_world_readiness_avg"]),
        "failure_candidates": int(packet["trace_summary"]["failure_candidate_count"]),
    }


def _system_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: _system_order(str(row["system_id"])))
    return [
        {
            "label": SYSTEM_LABELS.get(str(row["system_id"]), str(row["system_id"])),
            "system_id": row["system_id"],
            "runs": int(float(row["runs"])),
            "real_world_readiness_avg": _round(row["real_world_readiness_avg"]),
            "strict_interface_avg": _round(row["strict_interface_avg"]),
            "recovered_execution_avg": _round(row["recovered_execution_avg"]),
            "controller_repair_avg": _round(row["controller_repair_avg"]),
            "controller_fallback_avg": _round(row["controller_fallback_avg"]),
            "argument_repair_avg": _round(row["argument_repair_avg"]),
            "raw_planning_clean_rate_avg": _round(row["raw_planning_clean_rate_avg"]),
            "disabled_controls": row.get("disabled_controls", ""),
        }
        for row in ordered
    ]


def _burden_row(packet_name: str, packet: dict[str, Any]) -> dict[str, Any]:
    rows = {row["system_id"]: row for row in packet["tool_contract"]["system_rows"]}
    no_directive = rows["mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive"]
    return {
        "packet": packet_name,
        "controller_repair_avg": _round(no_directive["controller_repair_avg"]),
        "controller_fallback_avg": _round(no_directive["controller_fallback_avg"]),
        "argument_repair_avg": _round(no_directive["argument_repair_avg"]),
        "raw_planning_clean_rate_avg": _round(no_directive["raw_planning_clean_rate_avg"]),
    }


def _probe_failure_rows(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for side, field in [
        ("candidate", "candidate_failure_mode"),
        ("baseline_non_exact", "baseline_failure_mode"),
    ]:
        counter: Counter[str] = Counter(
            str(row.get(field, "")) for row in case_rows if str(row.get(field, "")) not in {"", "exact"}
        )
        rows.extend(
            {"side": side, "failure_mode": mode, "case_count": count}
            for mode, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        )
    return rows


def _prompt_contract_candidate_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for system_id, meta in sorted((registry.get("systems") or {}).items()):
        controls = ResearchControls.from_mapping(meta.get("research_controls"))
        if not controls.tool_prompt_contract_id:
            continue
        contract = TOOL_PROMPT_CONTRACTS.get(controls.tool_prompt_contract_id)
        rows.append(
            {
                "system_id": system_id,
                "short_label": str(meta.get("short_label", system_id)),
                "tool_prompt_contract_id": controls.tool_prompt_contract_id,
                "disable_tool_turn_directive": controls.disable_tool_turn_directive,
                "label": contract.label if contract else "",
                "hypothesis": contract.hypothesis if contract else "",
                "tags": ";".join(contract.tags) if contract else "",
            }
        )
    return rows


def _candidate_tag_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in candidate_rows:
        for tag in str(row.get("tags", "")).split(";"):
            if tag:
                counter[tag] += 1
    return [
        {"label": tag, "value": count}
        for tag, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _markdown_report(payload: dict[str, Any]) -> str:
    packet_rows = payload["packet_summary"]
    h1i_rows = payload["h1i_system_metrics"]
    probe_rows = payload["probe_failure_modes"]
    candidate_rows = payload["prompt_contract_candidates"]
    gate_rows = payload["prompt_contract_probe_gates"]
    gemini = payload["gemini"]
    lines = [
        "# MLX Tool-Contract Harnessing Report",
        "",
        f"Generated: `{payload['manifest']['generated_at']}`",
        "",
        "## Executive Read",
        "",
        "The current local-Gemma research frontier is no longer top-line readiness on the aligned `32 / 26` surface. "
        "The strongest remaining signal is whether MLX Gemma can stay inside Moonie's tool interface without controller repair, fallback, and argument normalization.",
        "",
        "H1h confirmed that the compact H1f no-directive finding survives the full ten-workflow live surface. "
        "H1i then compressed the worst H1h workflow families into a faster packet and amplified the same causal ordering.",
        "",
        "The main finding is blunt: the tool-turn directive is a real model-side harness intervention, not presentation polish. "
        "When it is removed, no-directive MLX can still match readiness only because the controller repairs or substitutes calls. "
        "Raw no-directive tool compliance collapses on the probe suite.",
        "",
        "## Figures",
        "",
        "![H1i readiness, strict interface, and recovered execution](figures/h1i_readiness_strict_recovered.svg)",
        "",
        "![H1h vs H1i no-directive controller burden](figures/h1h_h1i_controller_burden.svg)",
        "",
        "![Tool probe contract gap](figures/tool_probe_contract_gap.svg)",
        "",
        "![H1i failure modes](figures/h1i_failure_modes.svg)",
        "",
        "![Prompt contract candidate targets](figures/prompt_contract_candidate_targets.svg)",
        "",
        "![Executed prompt contract probe gate](figures/prompt_contract_probe_gate.svg)",
        "",
        "## Packet Summary",
        "",
        _markdown_table(packet_rows),
        "",
        "## H1i System Metrics",
        "",
        _markdown_table(h1i_rows),
        "",
        "## Probe Failure Modes",
        "",
        _markdown_table(probe_rows),
        "",
        "## Prompt-Contract Candidate Queue",
        "",
        _markdown_table(candidate_rows),
        "",
        "These candidates are generic prompt contracts for the no-directive row. They deliberately avoid embedding the expected planned call, so they can be tested on the probe before spending H1i or H1h runs.",
        "",
        "## Executed Prompt-Contract Probe Gate",
        "",
        _markdown_table(gate_rows),
        "",
        "The first executed probe gate shows only partial gains. `schema_anchor_v1` recovers one exact visual readback case over no-directive, while `literal_argument_guard_v1` and `tool_required_parallel_v1` recover the executable visual target without improving exact JSON copy rate. All three remain far below the contracted MLX probe row.",
        "",
        "## Gemini CLI Baseline Status",
        "",
        f"- Packet: `{gemini['packet_run_id']}`",
        f"- H1 slice: `{gemini['h1_slice']}`",
        f"- Workflow count: `{gemini['workflow_count']}`",
        f"- Dry run: `{gemini['dry_run']}`",
        f"- Binary: `{gemini['binary']}`",
        "",
        "This packet is deliberately a dry-run prompt and command manifest. It is an external-reference baseline, not a replacement for Moonie's local MLX harness.",
        "",
        "## Interpretation",
        "",
        "- H1f established the compact causal ordering: no directive plus no controller repair was the largest drop.",
        "- H1h verified that the ordering survives all ten H1e live workflow families.",
        "- H1i is now the best fast loop because it targets the worst H1h no-repair families and makes the repair/fallback gaps larger.",
        "- The no-directive probe explains why: CLI/API calls often keep the right tool but drift on canonical arguments, while visual referent and parallel-tool cases collapse to no tool call.",
        "- The next prompt-contract experiment should be evaluated first on the probe suite and then on H1i before spending another full H1h run.",
        "",
        "## Source Artifacts",
        "",
    ]
    for name, path in payload["manifest"]["source_packets"].items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(
        [
            f"- Probe comparison: `{payload['manifest']['probe_comparison']}`",
            f"- Prompt-contract probe packet: `{payload['manifest']['prompt_contract_packet']}`",
            f"- Gemini dry-run baseline: `{payload['manifest']['gemini_packet']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_grouped_metric_svg(
    path: Path,
    *,
    title: str,
    rows: list[dict[str, Any]],
    label_field: str,
    metrics: list[tuple[str, str, str]],
) -> None:
    width = 1120
    left = 250
    top = 80
    group_height = 72
    bar_height = 12
    gap = 6
    chart_width = 760
    axis_max = _axis_max(rows, metrics)
    height = top + len(rows) * group_height + 90
    parts = _svg_header(width, height, title)
    for tick in range(0, 6):
        x = left + chart_width * tick / 5
        parts.append(f'<line x1="{x:.1f}" y1="55" x2="{x:.1f}" y2="{height - 45}" stroke="#E5E7EB" stroke-width="1"/>')
        tick_value = axis_max * tick / 5
        parts.append(f'<text x="{x:.1f}" y="{height - 22}" text-anchor="middle" font-size="12" fill="#475569">{tick_value:.2g}</text>')
    for index, row in enumerate(rows):
        y0 = top + index * group_height
        parts.append(f'<text x="20" y="{y0 + 23}" font-size="13" fill="#111827">{_escape(str(row[label_field]))}</text>')
        for metric_index, (field, label, color) in enumerate(metrics):
            value = float(row.get(field) or 0.0)
            y = y0 + metric_index * (bar_height + gap) + 10
            bar_width = max(0.0, min(value, axis_max)) / axis_max * chart_width
            parts.append(f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" rx="2" fill="{color}"/>')
            parts.append(f'<text x="{left + chart_width + 12}" y="{y + 10}" font-size="12" fill="#334155">{label}: {value:.3f}</text>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def _axis_max(rows: list[dict[str, Any]], metrics: list[tuple[str, str, str]]) -> float:
    highest = max((float(row.get(field) or 0.0) for row in rows for field, _, _ in metrics), default=1.0)
    if highest <= 1.0:
        return 1.0
    if highest <= 1.25:
        return 1.25
    if highest <= 1.5:
        return 1.5
    return highest


def _write_bar_svg(path: Path, *, title: str, rows: list[dict[str, Any]], color: str) -> None:
    width = 980
    left = 260
    top = 80
    bar_height = 24
    row_gap = 18
    chart_width = 560
    max_value = max((float(row["value"]) for row in rows), default=1.0)
    height = top + len(rows) * (bar_height + row_gap) + 70
    parts = _svg_header(width, height, title)
    for index, row in enumerate(rows):
        y = top + index * (bar_height + row_gap)
        value = float(row["value"])
        bar_width = value / max_value * chart_width if max_value else 0
        parts.append(f'<text x="20" y="{y + 17}" font-size="13" fill="#111827">{_escape(str(row["label"]))}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{left + bar_width + 10}" y="{y + 17}" font-size="12" fill="#334155">{int(value)}</text>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        f'<text x="20" y="35" font-size="20" font-family="Inter, Arial, sans-serif" font-weight="700" fill="#0F172A">{_escape(title)}</text>',
        '<style>text{font-family:Inter,Arial,sans-serif}</style>',
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    fields = list(rows[0].keys())
    output = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(output)


def _system_order(system_id: str) -> int:
    ordered = list(SYSTEM_LABELS)
    return ordered.index(system_id) if system_id in ordered else len(ordered)


def _round(value: Any) -> float:
    return round(float(value), 5)


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()
