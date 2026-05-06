from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma4_capability_map.knowledge_work.trace_analysis import compare_ablation_packets, write_packet_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two H1 KnowledgeWorkArena ablation packets.")
    parser.add_argument("baseline_packet_dir", help="Baseline H1 packet directory.")
    parser.add_argument("candidate_packet_dir", help="Candidate H1 packet directory.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Defaults to candidate_packet_dir.")
    parser.add_argument("--json", action="store_true", help="Print the full comparison JSON after writing outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = write_packet_comparison(args.baseline_packet_dir, args.candidate_packet_dir, args.output_dir)
    comparison = compare_ablation_packets(args.baseline_packet_dir, args.candidate_packet_dir)
    payload = {
        "baseline_packet_dir": str(Path(args.baseline_packet_dir).resolve()),
        "candidate_packet_dir": str(Path(args.candidate_packet_dir).resolve()),
        "shared_system_count": comparison["deltas"]["shared_system_count"],
        "note_count_delta": comparison["deltas"]["note_count_delta"],
        "failure_candidate_count_delta": comparison["deltas"]["failure_candidate_count_delta"],
        "outputs": output_paths,
    }
    print(json.dumps(comparison if args.json else payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
