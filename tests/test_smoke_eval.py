from __future__ import annotations

from pathlib import Path

from gemma4_capability_map.io import load_jsonl
from gemma4_capability_map.models.embeddinggemma_runner import EmbeddingGemmaRetriever
from gemma4_capability_map.models.functiongemma_runner import FunctionGemmaRunner
from gemma4_capability_map.models.gemma4_runner import Gemma4Runner
from gemma4_capability_map.pipelines.base import RuntimeBundle
from gemma4_capability_map.pipelines.modular import ModularPipeline
from gemma4_capability_map.pipelines.monolith import MonolithPipeline
from gemma4_capability_map.pipelines.hybrid import HybridPipeline
from gemma4_capability_map.schemas import ModelTurn
from gemma4_capability_map.schemas import Task, Variant
from gemma4_capability_map.tools.executor import DeterministicExecutor
from gemma4_capability_map.tools.registry import build_default_registry
from gemma4_capability_map.traces.exporters import export_leaderboard_csv
from gemma4_capability_map.traces.replay import load_traces, summarize_traces


ROOT = Path(__file__).resolve().parents[1]


class _ThoughtOnlyRunner:
    model_id = "test/thought-only"
    backend = "test"

    def generate(self, messages, media, tool_specs, thinking, max_new_tokens=None) -> ModelTurn:  # noqa: ANN001
        return ModelTurn(
            raw_model_output="<|channel>thought\nNeed more room to finish.",
            final_answer="",
            thinking_text="Need more room to finish.",
        )


class _FrenchRescueRunner:
    model_id = "test/french-rescue"
    backend = "test"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, media, tool_specs, thinking, max_new_tokens=None) -> ModelTurn:  # noqa: ANN001
        self.calls += 1
        if self.calls == 1:
            return ModelTurn(raw_model_output="Veuillez patienter.", final_answer="Veuillez patienter.")
        return ModelTurn(
            raw_model_output="Action: defer. L'accès fournisseur doit attendre l'approbation de sécurité.",
            final_answer="Action: defer. L'accès fournisseur doit attendre l'approbation de sécurité.",
        )


def load_all_tasks() -> list[Task]:
    tasks: list[Task] = []
    for path in sorted((ROOT / "data" / "gold").glob("*.jsonl")):
        tasks.extend(load_jsonl(path, Task))
    return tasks


def test_monolith_oracle_smoke_bundle(tmp_path: Path) -> None:
    tasks = load_all_tasks()[:12]
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        retriever=EmbeddingGemmaRetriever("google/embeddinggemma-300m", backend="heuristic"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    pipeline = MonolithPipeline()
    traces = [pipeline.run(task, Variant(variant_id=f"{task.task_id}_clean", base_task_id=task.task_id), bundle) for task in tasks]
    assert len(traces) == 12
    assert all(float(trace.metrics.get("success", 0.0)) == 1.0 for trace in traces)

    trace_path = tmp_path / "traces.jsonl"
    leaderboard_path = tmp_path / "leaderboard.csv"
    from gemma4_capability_map.io import dump_jsonl

    dump_jsonl(trace_path, traces)
    export_leaderboard_csv(traces, leaderboard_path)
    reloaded = load_traces(trace_path)
    summary = summarize_traces(reloaded)
    assert summary["runs"] == 12.0
    assert summary["success_rate"] == 1.0
    assert "metric_averages" in summary
    assert summary["failing_variants"] == []
    assert leaderboard_path.exists()


def test_modular_pipeline_uses_router_path() -> None:
    task = load_jsonl(ROOT / "data" / "gold" / "agents.jsonl", Task)[0]
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        router=FunctionGemmaRunner("google/functiongemma-270m-it", backend="oracle"),
        retriever=EmbeddingGemmaRetriever("google/embeddinggemma-300m", backend="heuristic"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    pipeline = ModularPipeline()
    trace = pipeline.run(task, Variant(variant_id=f"{task.task_id}_clean", base_task_id=task.task_id), bundle)
    assert trace.architecture == "modular"
    assert float(trace.metrics["success"]) == 1.0


def test_parallel_tool_group_runs_in_single_oracle_turn() -> None:
    task = [task for task in load_jsonl(ROOT / "data" / "gold" / "tools.jsonl", Task) if task.task_id == "tool_009_parallel_context_check"][0]
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    pipeline = MonolithPipeline()
    trace = pipeline.run(task, Variant(variant_id=f"{task.task_id}_clean", base_task_id=task.task_id), bundle)
    assert [step.selected_tool for step in trace.tool_steps] == ["inspect_image", "read_repo_file"]
    assert len(trace.prompt_artifacts["planning_raw_outputs"]) == 1


def test_monolith_thinking_stuffs_corpus_context_when_no_retriever_is_present() -> None:
    task = [task for task in load_jsonl(ROOT / "data" / "gold" / "thinking.jsonl", Task) if task.task_id == "think_003_policy_reasoning"][0]
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    pipeline = MonolithPipeline()
    trace = pipeline.run(task, Variant(variant_id=f"{task.task_id}_clean", base_task_id=task.task_id), bundle)

    assert trace.prompt_artifacts["stuffed_doc_ids"] == ["doc_policy_old", "doc_policy_new"]
    assert float(trace.metrics["success"]) == 1.0


def test_pipeline_does_not_fallback_to_raw_thought_text_as_final_answer() -> None:
    task = [task for task in load_jsonl(ROOT / "data" / "gold" / "thinking.jsonl", Task) if task.task_id == "think_001_math"][0]
    bundle = RuntimeBundle(
        reasoner=_ThoughtOnlyRunner(),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    pipeline = MonolithPipeline(thinking_enabled=True)
    trace = pipeline.run(task, Variant(variant_id=f"{task.task_id}_clean", base_task_id=task.task_id), bundle)

    assert trace.final_answer == ""
    assert float(trace.metrics["success"]) == 0.0


def test_thinking_tasks_receive_direct_answer_guidance() -> None:
    task = [task for task in load_jsonl(ROOT / "data" / "gold" / "thinking.jsonl", Task) if task.task_id == "think_001_math"][0]
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    pipeline = MonolithPipeline()
    trace = pipeline.run(task, Variant(variant_id=f"{task.task_id}_clean", base_task_id=task.task_id), bundle)

    system_messages = [message["content"] for message in trace.prompt_artifacts["final_messages"] if message["role"] == "system"]
    assert any("Answer directly and minimally." in content for content in system_messages)


def test_image_thinking_tasks_receive_toggle_name_guidance() -> None:
    task = [task for task in load_jsonl(ROOT / "data" / "gold" / "thinking.jsonl", Task) if task.task_id == "think_007_doc_image_summary"][0]
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    pipeline = MonolithPipeline()
    trace = pipeline.run(task, Variant(variant_id=f"{task.task_id}_clean", base_task_id=task.task_id), bundle)

    system_messages = [message["content"] for message in trace.prompt_artifacts["final_messages"] if message["role"] == "system"]
    assert any("Answer from the image and name the setting or recommended action explicitly." in content for content in system_messages)
    assert any("Do not answer with only status words like On, Off, Enabled, or Disabled." in content for content in system_messages)


def test_effective_max_new_tokens_are_recorded_when_pipeline_overrides_are_absent() -> None:
    task = [task for task in load_jsonl(ROOT / "data" / "gold" / "thinking.jsonl", Task) if task.task_id == "think_001_math"][0]
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle", max_new_tokens=77),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    pipeline = MonolithPipeline()
    trace = pipeline.run(task, Variant(variant_id=f"{task.task_id}_clean", base_task_id=task.task_id), bundle)

    assert trace.prompt_artifacts["final_max_new_tokens"] == 77


def test_judgment_mode_skips_tools_for_refusal_tasks() -> None:
    task = [task for task in load_jsonl(ROOT / "data" / "gold" / "tools.jsonl", Task) if task.task_id == "tool_013_refuse_unsafe_billing_disable"][0]
    bundle = RuntimeBundle(
        reasoner=Gemma4Runner("google/gemma-4-E4B-it", backend="oracle"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    pipeline = MonolithPipeline()
    trace = pipeline.run(task, Variant(variant_id=f"{task.task_id}_clean", base_task_id=task.task_id), bundle)

    assert trace.tool_steps == []
    assert float(trace.metrics["success"]) == 1.0


def test_french_second_pass_rescue_rewrites_answer_without_mutating_trace() -> None:
    task = [task for task in load_jsonl(ROOT / "data" / "gold" / "retrieval.jsonl", Task) if task.task_id == "retr_013_vendor_access_defer"][0]
    bundle = RuntimeBundle(
        reasoner=_FrenchRescueRunner(),
        retriever=EmbeddingGemmaRetriever("google/embeddinggemma-300m", backend="heuristic"),
        executor=DeterministicExecutor(registry=build_default_registry()),
    )
    pipeline = HybridPipeline()
    trace = pipeline.run(
        task,
        Variant(
            variant_id=f"{task.task_id}_fr",
            base_task_id=task.task_id,
            stressors={"language": "fr"},
        ),
        bundle,
    )

    assert trace.prompt_artifacts["second_pass_used"] is True
    assert trace.final_answer.startswith("Action: defer")
    assert trace.tool_steps == []
    assert float(trace.metrics["success"]) == 1.0
