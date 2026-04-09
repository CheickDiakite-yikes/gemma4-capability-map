from __future__ import annotations

from gemma4_capability_map.env import load_project_env


def test_load_project_env_reads_env_local_before_env(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / ".env").write_text("HF_TOKEN=from_env\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("HF_TOKEN=from_env_local\n", encoding="utf-8")

    load_project_env(tmp_path, force=True)

    assert load_project_env(tmp_path, force=False) == {}
    assert __import__("os").environ["HF_TOKEN"] == "from_env_local"


def test_load_project_env_does_not_override_shell_value(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "from_shell")
    (tmp_path / ".env.local").write_text('HF_TOKEN="from_file"\nHUGGINGFACE_HUB_TOKEN=secondary\n', encoding="utf-8")

    loaded = load_project_env(tmp_path, force=True)

    assert __import__("os").environ["HF_TOKEN"] == "from_shell"
    assert __import__("os").environ["HUGGINGFACE_HUB_TOKEN"] == "secondary"
    assert loaded == {"HUGGINGFACE_HUB_TOKEN": "secondary"}
