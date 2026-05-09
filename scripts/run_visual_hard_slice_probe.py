from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma4_capability_map.runtime.tool_directive_probe import run_tool_directive_probe
from gemma4_capability_map.runtime.visual_hard_slice import build_visual_hard_slice_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the executable visual hard-slice probe.")
    parser.add_argument("--system-id", default="mlx_gemma4_e2b_reasoner_only")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--registry-path", "--registry", dest="registry_path", default=None)
    parser.add_argument("--case-id", action="append", dest="case_ids", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = build_visual_hard_slice_cases()
    if args.case_ids:
        cases_by_id = {case.case_id: case for case in cases}
        missing = [case_id for case_id in args.case_ids if case_id not in cases_by_id]
        if missing:
            raise SystemExit(f"Unknown visual hard-slice case id(s): {', '.join(missing)}")
        cases = [cases_by_id[case_id] for case_id in args.case_ids]
    result = run_tool_directive_probe(
        system_id=args.system_id,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        registry_path=Path(args.registry_path) if args.registry_path else Path("configs/model_registry.yaml"),
        cases=cases,
    )
    print(json.dumps({"output_dir": result["output_dir"], "summary": result["summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
