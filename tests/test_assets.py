from __future__ import annotations

from pathlib import Path

from gemma4_capability_map.assets import default_benchmark_image_path


def test_default_benchmark_image_path_points_to_local_asset() -> None:
    image_path = Path(default_benchmark_image_path())
    assert image_path.exists()
    assert image_path.name == "security_2fa_disabled.png"
