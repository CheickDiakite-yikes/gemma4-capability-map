from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma4_capability_map.knowledge_work.h1 import DEFAULT_H1_SLICE_PATH
from gemma4_capability_map.knowledge_work.trace_analysis import write_h1_workflow_family_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize H1 packet metrics by packaged workflow family.")
    parser.add_argument("packet_dir")
    parser.add_argument("--config", default=str(DEFAULT_H1_SLICE_PATH))
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = write_h1_workflow_family_summary(
        args.packet_dir,
        args.config,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
