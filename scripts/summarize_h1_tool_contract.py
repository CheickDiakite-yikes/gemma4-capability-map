from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma4_capability_map.knowledge_work.trace_analysis import write_tool_contract_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize tool-turn directive ablation deltas inside an H1 packet.")
    parser.add_argument("packet_dir")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--contracted-system-id", default="mlx_gemma4_e2b_reasoner_only")
    parser.add_argument("--no-directive-system-id", default="mlx_gemma4_e2b_reasoner_only_no_tool_turn_directive")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = write_tool_contract_summary(
        args.packet_dir,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        contracted_system_id=args.contracted_system_id,
        no_directive_system_id=args.no_directive_system_id,
    )
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
