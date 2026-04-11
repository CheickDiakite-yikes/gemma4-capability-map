from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_knowledge_work_arena.py"
SPEC = importlib.util.spec_from_file_location("run_knowledge_work_arena_script", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

_infer_system_id = MODULE._infer_system_id


def test_infer_system_id_matches_registry_for_reasoner_only_hf() -> None:
    args = argparse.Namespace(
        system_id=None,
        backend="hf",
        reasoner_backend="hf",
        router_backend="heuristic",
        retriever_backend="heuristic",
        reasoner="google/gemma-4-E2B-it",
        router="google/functiongemma-270m-it",
        retriever="google/embeddinggemma-300m",
    )

    assert _infer_system_id(args) == "hf_gemma4_e2b_reasoner_only"


def test_infer_system_id_matches_registry_for_specialist_stack() -> None:
    args = argparse.Namespace(
        system_id=None,
        backend="hf_service",
        reasoner_backend="hf_service",
        router_backend="hf_service",
        retriever_backend="hf_service",
        reasoner="google/gemma-4-E2B-it",
        router="google/functiongemma-270m-it",
        retriever="google/embeddinggemma-300m",
    )

    assert _infer_system_id(args) == "hf_service_gemma4_specialists_cpu"


def test_infer_system_id_matches_registry_for_qwen_reasoner_only() -> None:
    args = argparse.Namespace(
        system_id=None,
        backend="hf",
        reasoner_backend="hf",
        router_backend="heuristic",
        retriever_backend="heuristic",
        reasoner="Qwen/Qwen3-8B",
        router="",
        retriever="",
    )

    assert _infer_system_id(args) == "hf_qwen3_8b_reasoner_only"
