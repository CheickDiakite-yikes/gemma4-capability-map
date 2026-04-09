from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from gemma4_capability_map.io import load_jsonl
from gemma4_capability_map.models.embeddinggemma_runner import EmbeddingGemmaRetriever
from gemma4_capability_map.models.functiongemma_runner import FunctionGemmaRunner
from gemma4_capability_map.models.gemma4_runner import Gemma4Runner
from gemma4_capability_map.models.hf_service import read_service_state
from gemma4_capability_map.pipelines.base import RuntimeBundle
from gemma4_capability_map.pipelines.hybrid import HybridPipeline
from gemma4_capability_map.pipelines.modular import ModularPipeline
from gemma4_capability_map.pipelines.monolith import MonolithPipeline
from gemma4_capability_map.schemas import RunTrace, Task, Variant
from gemma4_capability_map.tools.executor import DeterministicExecutor
from gemma4_capability_map.tools.registry import build_default_registry


ROOT = Path(__file__).resolve().parents[2]


def load_tasks(track: str | None, task_ids: list[str] | None = None) -> list[Task]:
    tasks: list[Task] = []
    gold_dir = ROOT / "data" / "gold"
    allowed = set(task_ids or [])
    for path in sorted(gold_dir.glob("*.jsonl")):
        batch = load_jsonl(path, Task)
        if track:
            batch = [task for task in batch if task.track.value == track]
        if allowed:
            batch = [task for task in batch if task.task_id in allowed]
        tasks.extend(batch)
    return tasks


def load_variants(tasks: list[Task], include_generated: bool) -> list[Variant]:
    if not include_generated:
        return [Variant(variant_id=f"{task.task_id}_clean", base_task_id=task.task_id) for task in tasks]
    variant_path = ROOT / "data" / "variants" / "generated_variants.jsonl"
    variants = load_jsonl(variant_path, Variant)
    task_ids = {task.task_id for task in tasks}
    return [variant for variant in variants if variant.base_task_id in task_ids]


def build_pipeline(
    name: str,
    thinking_enabled: bool,
    planning_max_new_tokens: int | None = None,
    final_max_new_tokens: int | None = None,
):
    mapping = {
        "monolith": MonolithPipeline,
        "hybrid": HybridPipeline,
        "modular": ModularPipeline,
    }
    return mapping[name](
        thinking_enabled=thinking_enabled,
        planning_max_new_tokens=planning_max_new_tokens,
        final_max_new_tokens=final_max_new_tokens,
    )


def build_runtime_bundle(
    tasks: list[Task],
    pipeline_name: str,
    backend: str,
    reasoner_backend: str | None,
    router_backend: str | None,
    retriever_backend: str | None,
    reasoner_id: str,
    router_id: str,
    retriever_id: str,
    reasoner_device: str,
    reasoner_max_new_tokens: int,
) -> RuntimeBundle:
    registry = build_default_registry()
    resolved_reasoner_backend = reasoner_backend or backend
    resolved_router_backend = router_backend or ("heuristic" if backend in {"hf", "hf_service", "mlx"} else backend)
    resolved_retriever_backend = retriever_backend or ("hf" if resolved_reasoner_backend in {"hf", "hf_service"} else "heuristic")
    return RuntimeBundle(
        reasoner=Gemma4Runner(
            reasoner_id,
            backend=resolved_reasoner_backend,
            device=reasoner_device,
            max_new_tokens=reasoner_max_new_tokens,
        ),
        router=FunctionGemmaRunner(router_id, backend=resolved_router_backend) if pipeline_name == "modular" else None,
        retriever=EmbeddingGemmaRetriever(retriever_id, backend=resolved_retriever_backend) if pipeline_name in {"hybrid", "modular"} or any(task.track.value == "retrieval" for task in tasks) else None,
        executor=DeterministicExecutor(registry=registry),
    )


def warm_runtime_bundle(bundle: RuntimeBundle, tasks: list[Task]) -> dict[str, dict]:
    warmup: dict[str, dict] = {}
    media: list[str] = []
    for task in tasks:
        if task.image_refs:
            media = list(task.image_refs)
            break
    if hasattr(bundle.reasoner, "ensure_loaded"):
        warmup["reasoner"] = bundle.reasoner.ensure_loaded(media=media)
    if bundle.router and hasattr(bundle.router, "ensure_loaded"):
        warmup["router"] = bundle.router.ensure_loaded()
    return warmup


def runtime_bundle_snapshot(bundle: RuntimeBundle) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    components = {
        "reasoner": bundle.reasoner,
        "router": bundle.router,
        "retriever": bundle.retriever,
    }
    for name, component in components.items():
        if component is None or not hasattr(component, "runtime_info"):
            continue
        info = component.runtime_info()
        if not isinstance(info, dict):
            continue
        component_snapshot = dict(info)
        service = component_snapshot.get("service")
        if isinstance(service, dict):
            state_path = service.get("state_path")
            if state_path:
                state = read_service_state(state_path)
                if state is not None:
                    component_snapshot["service_state"] = state
        snapshot[name] = component_snapshot
    return snapshot


def plan_runs(tasks: list[Task], variants: list[Variant]) -> list[tuple[Task, Variant]]:
    variant_map: dict[str, list[Variant]] = {}
    for variant in variants:
        variant_map.setdefault(variant.base_task_id, []).append(variant)

    planned: list[tuple[Task, Variant]] = []
    for task in tasks:
        task_variants = variant_map.get(task.task_id)
        if not task_variants:
            task_variants = [Variant(variant_id=f"{task.task_id}_clean", base_task_id=task.task_id)]
        for variant in task_variants:
            planned.append((task, variant))
    return planned


def run_benchmark(
    tasks: list[Task],
    variants: list[Variant],
    pipeline_name: str,
    backend: str,
    reasoner_backend: str | None,
    router_backend: str | None,
    retriever_backend: str | None,
    reasoner_id: str,
    router_id: str,
    retriever_id: str,
    reasoner_device: str,
    reasoner_max_new_tokens: int,
    planning_max_new_tokens: int | None,
    final_max_new_tokens: int | None,
    limit: int,
    thinking_enabled: bool,
    on_trace: Callable[[RunTrace, int, int], None] | None = None,
    bundle: RuntimeBundle | None = None,
) -> list[RunTrace]:
    bundle = bundle or build_runtime_bundle(
        tasks=tasks,
        pipeline_name=pipeline_name,
        backend=backend,
        reasoner_backend=reasoner_backend,
        router_backend=router_backend,
        retriever_backend=retriever_backend,
        reasoner_id=reasoner_id,
        router_id=router_id,
        retriever_id=retriever_id,
        reasoner_device=reasoner_device,
        reasoner_max_new_tokens=reasoner_max_new_tokens,
    )
    pipeline = build_pipeline(
        pipeline_name,
        thinking_enabled=thinking_enabled,
        planning_max_new_tokens=planning_max_new_tokens,
        final_max_new_tokens=final_max_new_tokens,
    )

    traces: list[RunTrace] = []
    planned = plan_runs(tasks, variants)[:limit]
    planned_runs = len(planned)
    for task, variant in planned:
        trace = pipeline.run(task=task, variant=variant, bundle=bundle)
        traces.append(trace)
        if on_trace is not None:
            on_trace(trace, len(traces), planned_runs)
    return traces
