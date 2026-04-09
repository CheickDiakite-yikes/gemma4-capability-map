from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from gemma4_capability_map.traces.replay import load_traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-path", default="results/raw/traces.jsonl")
    args = parser.parse_args()
    traces = load_traces(Path(args.trace_path))
    summary: dict[tuple[str, str], list[float]] = defaultdict(list)
    for trace in traces:
        summary[(trace.architecture, trace.track.value)].append(float(trace.metrics.get("success", 0.0)))
    for (architecture, track), values in sorted(summary.items()):
        print(f"{architecture:9s} {track:12s} success={sum(values)/len(values):.3f} runs={len(values)}")


if __name__ == "__main__":
    main()

