from __future__ import annotations

from gemma4_capability_map.reporting.knowledge_work_board import build_board_rows, write_board_exports


def main() -> None:
    payload = write_board_exports(build_board_rows())
    print(
        f"Wrote KnowledgeWorkArena board exports with {payload['row_count']} rows "
        f"and {payload['latest_row_count']} latest rows."
    )


if __name__ == "__main__":
    main()
