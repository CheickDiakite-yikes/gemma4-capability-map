from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_h1o_control_factorial_synthesis.py"
SPEC = importlib.util.spec_from_file_location("build_h1o_control_factorial_synthesis_script", MODULE_PATH)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def test_h1o_control_factorial_synthesis_marks_mechanism_split(tmp_path: Path) -> None:
    payload = SCRIPT.build_h1o_control_factorial_synthesis(output_dir=tmp_path)

    assert payload["manifest"]["profile_count"] == 6
    assert payload["manifest"]["family_row_count"] == 18
    assert payload["manifest"]["case_count"] == 12
    assert payload["manifest"]["strict_upper_bound"] == ["argument_hints_v2", "component_value_guard_v9"]
    assert payload["manifest"]["executor_upper_bound"] == [
        "argument_hints_v2",
        "hybrid_label_guard_v8",
        "component_value_guard_v9",
    ]

    profiles = {row["label"]: row for row in payload["profile_rows"]}
    assert profiles["no_directive"]["exact_success_count"] == 5
    assert profiles["no_directive"]["executor_success_count"] == 6
    assert profiles["argument_hints_v2"]["exact_success_count"] == 9
    assert profiles["component_value_guard_v9"]["executor_success_count"] == 10
    assert profiles["no_call_control_rescue_v10"]["exact_success_count"] == 7

    family_rows = {
        (row["label"], row["family"]): row
        for row in payload["family_rows"]
    }
    assert family_rows[("no_directive", "h1o_activation_no_call")]["exact_success_count"] == 4
    assert family_rows[("no_call_control_rescue_v10", "h1o_activation_no_call")]["exact_success_count"] == 3
    assert family_rows[("argument_hints_v2", "h1o_code_negation_preservation")]["executor_success_count"] == 4
    assert family_rows[("component_value_guard_v9", "h1o_component_value_boundary")]["exact_success_count"] == 2

    delta_rows = {
        (row["label"], row["family"]): row
        for row in payload["family_delta_rows"]
    }
    assert delta_rows[("argument_hints_v2", "h1o_code_negation_preservation")][
        "delta_exact_success_count"
    ] == 2
    assert delta_rows[("component_value_guard_v9", "h1o_component_value_boundary")][
        "delta_exact_success_count"
    ] == 2
    assert delta_rows[("no_call_control_rescue_v10", "h1o_activation_no_call")][
        "delta_exact_success_count"
    ] == -1

    findings = {row["finding_id"]: row["finding"] for row in payload["finding_rows"]}
    assert "argument_hints_v2, component_value_guard_v9" in findings["strict_upper_bound"]
    assert "no H1o profile reaches full executor success" in findings["executor_upper_bound"]
    assert "Activation/no-call is not the remaining bottleneck" in findings[
        "activation_saturated_without_rescue"
    ]
    assert "Code/negation failures are controller-sensitive" in findings["code_negation_is_repairable"]
    assert "Component/value boundaries remain the hard residue" in findings[
        "component_boundary_remains_residual"
    ]

    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "synthesis.json").exists()
    assert (tmp_path / "tables" / "h1o_control_factorial_profile_summary.csv").exists()
