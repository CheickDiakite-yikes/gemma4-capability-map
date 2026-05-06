from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma4_capability_map.knowledge_work.trace_analysis import analyze_ablation_packet, write_trace_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine H1 KnowledgeWorkArena ablation traces for controller note families.")
    parser.add_argument("packet_dir", help="H1 ablation packet directory containing per-system trace outputs.")
    parser.add_argument("--output-dir", default=None, help="Optional directory for analysis outputs. Defaults to packet_dir.")
    parser.add_argument("--json", action="store_true", help="Print the full analysis JSON after writing outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = write_trace_analysis(args.packet_dir, args.output_dir)
    analysis = analyze_ablation_packet(args.packet_dir)
    payload = {
        "packet_dir": str(Path(args.packet_dir).resolve()),
        "system_count": analysis["system_count"],
        "episode_count": analysis["episode_count"],
        "note_count": analysis["note_count"],
        "failure_candidate_count": analysis["failure_candidate_count"],
        "outputs": output_paths,
    }
    print(json.dumps(analysis if args.json else payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
