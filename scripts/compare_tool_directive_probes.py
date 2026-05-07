from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma4_capability_map.runtime.tool_directive_probe import write_tool_directive_probe_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two tool directive probe packets.")
    parser.add_argument("baseline_dir")
    parser.add_argument("candidate_dir")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = write_tool_directive_probe_comparison(
        args.baseline_dir,
        args.candidate_dir,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
