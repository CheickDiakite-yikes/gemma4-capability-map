from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma4_capability_map.runtime.packet_analysis import (
    analyze_runtime_live_smoke_packet,
    write_runtime_packet_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize runtime live-smoke packet repair and policy families.")
    parser.add_argument("packet_dir", help="Runtime live-smoke packet directory.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Defaults to packet_dir.")
    parser.add_argument("--json", action="store_true", help="Print the full analysis JSON after writing outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = write_runtime_packet_analysis(args.packet_dir, args.output_dir)
    analysis = analyze_runtime_live_smoke_packet(args.packet_dir)
    payload = {
        "packet_dir": str(Path(args.packet_dir).resolve()),
        "run_group_id": analysis["run_group_id"],
        "workflow_count": analysis["workflow_count"],
        "repeat_count": analysis["repeat_count"],
        "session_count": analysis["session_count"],
        "failed_sessions": analysis["failed_sessions"],
        "controller_finding_count": analysis["controller_finding_count"],
        "policy_block_count": analysis["policy_block_count"],
        "stable_repair_family_count": analysis["stable_repair_family_count"],
        "stable_policy_block_family_count": analysis["stable_policy_block_family_count"],
        "outputs": output_paths,
    }
    print(json.dumps(analysis if args.json else payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
