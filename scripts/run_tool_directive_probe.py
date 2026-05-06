from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma4_capability_map.runtime.tool_directive_probe import run_tool_directive_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local model probe for exact tool-turn directive copying.")
    parser.add_argument("--system-id", default="mlx_gemma4_e2b_reasoner_only")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--registry-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_tool_directive_probe(
        system_id=args.system_id,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        registry_path=Path(args.registry_path) if args.registry_path else Path("configs/model_registry.yaml"),
    )
    print(json.dumps({"output_dir": result["output_dir"], "summary": result["summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
