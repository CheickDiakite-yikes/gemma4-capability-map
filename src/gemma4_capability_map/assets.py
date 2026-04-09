from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_IMAGE = ROOT / "data" / "assets" / "security_2fa_disabled.png"


def default_benchmark_image_path() -> str:
    return str(DEFAULT_BENCHMARK_IMAGE.resolve())
