from __future__ import annotations

from pathlib import Path

from gemma4_capability_map.models.gemma4_runner import _render_media_content


def test_render_media_content_uses_plain_local_paths_for_images(tmp_path: Path) -> None:
    image_path = tmp_path / "fixture.png"
    image_path.write_bytes(b"png")

    content = _render_media_content(str(image_path))

    assert content == [{"type": "image", "url": str(image_path.resolve())}]
