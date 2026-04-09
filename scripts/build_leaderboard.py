from __future__ import annotations

import argparse
from pathlib import Path

from gemma4_capability_map.traces.exporters import export_leaderboard_csv
from gemma4_capability_map.traces.replay import load_traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-path", default="results/raw/traces.jsonl")
    parser.add_argument("--leaderboard-path", default="results/leaderboard.csv")
    args = parser.parse_args()
    traces = load_traces(Path(args.trace_path))
    export_leaderboard_csv(traces, Path(args.leaderboard_path))
    print(f"Wrote leaderboard to {args.leaderboard_path}")


if __name__ == "__main__":
    main()
