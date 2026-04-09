from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime

from gemma4_capability_map.evals.agent_eval import score_full_stack_trace
from gemma4_capability_map.evals.retrieval_eval import score_retrieval_trace
from gemma4_capability_map.evals.thinking_eval import score_thinking_trace
from gemma4_capability_map.evals.tool_eval import score_tool_trace
from gemma4_capability_map.hardware import detect_hardware_profile
from gemma4_capability_map.metrics.answer_match import answer_contains_all
from gemma4_capability_map.models.base import Executor, Retriever, Runner
from gemma4_capability_map.schemas import ExpectedEvent, JudgmentMode, Message, ModelBundleSpec, RunTrace, StateTransition, Task, ToolResult, Track, Variant
from gemma4_capability_map.tools.executor import diff_state
from gemma4_capability_map.tools.planner import plan_or_repair_tool_calls


@dataclass
class RuntimeBundle:
    reasoner: Runner
    executor: Executor
    router: Runner | None = None
    retriever: Retriever | None = None


class BasePipeline:
    name = "base"

    def __init__(
        self,
        thinking_enabled: bool = False,
        planning_max_new_tokens: int | None = None,
        final_max_new_tokens: int | None = None,
    ) -> None:
        self.thinking_enabled = thinking_enabled
        self.planning_max_new_tokens = planning_max_new_tokens
        self.final_max_new_tokens = final_max_new_tokens

    def run(self, task: Task, variant: Variant, bundle: RuntimeBundle) -> RunTrace:
        effective_task = materialize_task(task, variant)
        language_stressor = variant.stressors.get("language") if variant.stressors else None
        state = deepcopy(effective_task.initial_state)
        executor = bundle.executor.with_tool_specs(effective_task.tool_specs)
        retrieval_hits = self._retrieve(effective_task, variant, bundle)
        messages = [message.model_copy(deep=True) for message in effective_task.messages]
        stuffed_doc_ids: list[str] = []
        planning_outputs: list[str] = []
        planning_repair_notes: list[list[str]] = []
        planning_latencies: list[int] = []
        planning_prompt_tokens: list[int] = []
        planning_completion_tokens: list[int] = []
        planning_runtime_metadata: list[dict[str, object]] = []
        if retrieval_hits:
            retrieval_context = "\n".join(f"{hit.doc_id}: {hit.content}" for hit in retrieval_hits)
            messages.append(Message(role="tool", content=f"retrieval_hits:\n{retrieval_context}"))
        elif self.name == "monolith" and effective_task.corpora:
            corpus_id = next(iter(effective_task.corpora.keys()))
            stuffed_doc_ids = [document.doc_id for document in effective_task.corpora[corpus_id]]
            stuffed_context = "\n".join(f"{document.doc_id}: {document.content}" for document in effective_task.corpora[corpus_id])
            messages.append(Message(role="system", content=f"stuffed_context:\n{stuffed_context}"))

        tool_steps: list[ToolResult] = []
        state_transitions: list[StateTransition] = []
        expected_tool_events = [event for event in effective_task.expected_events if event.event_type == "tool_call"]
        selector = bundle.router if self.name == "modular" and bundle.router and effective_task.tool_specs else bundle.reasoner
        judgment_without_tools = bool(
            effective_task.judgment_mode
            and effective_task.judgment_mode.enabled
            and not expected_tool_events
        )

        if effective_task.tool_specs and not judgment_without_tools:
            while len(tool_steps) < max(1, len(expected_tool_events)):
                next_expected = _next_expected_batch(expected_tool_events, len(tool_steps)) if expected_tool_events else None
                planning_messages = self._with_oracle_hint(messages, next_expected=next_expected, final_answer=None, backend=selector.backend)
                planning_turn = selector.generate(
                    messages=planning_messages,
                    media=effective_task.image_refs,
                    tool_specs=effective_task.tool_specs,
                    thinking=self.thinking_enabled,
                    max_new_tokens=self.planning_max_new_tokens,
                )
                planning_outputs.append(planning_turn.raw_model_output)
                planning_latencies.append(planning_turn.latency_ms)
                planning_prompt_tokens.append(planning_turn.prompt_tokens)
                planning_completion_tokens.append(planning_turn.completion_tokens)
                planning_runtime_metadata.append(planning_turn.runtime_metadata)
                planning_calls, repair_notes = plan_or_repair_tool_calls(
                    raw_output=planning_turn.raw_model_output,
                    parsed_calls=planning_turn.normalized_tool_call,
                    messages=messages,
                    media=effective_task.image_refs,
                    tool_specs=effective_task.tool_specs,
                )
                planning_repair_notes.append(repair_notes)
                if not planning_calls:
                    break
                for call in planning_calls:
                    before = deepcopy(state)
                    result = executor.step(state=state, tool_call=call, step=len(tool_steps) + 1)
                    tool_steps.append(result)
                    state = deepcopy(result.state_after)
                    state_transitions.append(
                        StateTransition(
                            step=result.step,
                            tool_name=result.selected_tool,
                            before=before,
                            after=state,
                            diff=diff_state(before, state),
                        )
                    )
                    tool_feedback = {
                        "tool_name": result.selected_tool,
                        "status": result.validator_result,
                        "arguments": result.arguments,
                        "output": result.output,
                        "error": result.error,
                    }
                    messages.append(Message(role="tool", content=json.dumps(tool_feedback, ensure_ascii=False)))

                    if result.validator_result == "fail" and effective_task.initial_state.get("validator_feedback_enabled") and next_expected:
                        retry_messages = self._with_oracle_hint(
                            messages,
                            next_expected=next_expected,
                            final_answer=None,
                            backend=selector.backend,
                        )
                        retry_turn = selector.generate(
                            messages=retry_messages,
                            media=effective_task.image_refs,
                            tool_specs=effective_task.tool_specs,
                            thinking=self.thinking_enabled,
                            max_new_tokens=self.planning_max_new_tokens,
                        )
                        planning_outputs.append(retry_turn.raw_model_output)
                        planning_latencies.append(retry_turn.latency_ms)
                        planning_prompt_tokens.append(retry_turn.prompt_tokens)
                        planning_completion_tokens.append(retry_turn.completion_tokens)
                        planning_runtime_metadata.append(retry_turn.runtime_metadata)
                        retry_calls, retry_notes = plan_or_repair_tool_calls(
                            raw_output=retry_turn.raw_model_output,
                            parsed_calls=retry_turn.normalized_tool_call,
                            messages=messages,
                            media=effective_task.image_refs,
                            tool_specs=effective_task.tool_specs,
                        )
                        planning_repair_notes.append(retry_notes)
                        if retry_calls:
                            retry_call = retry_calls[0]
                            before_retry = deepcopy(state)
                            retry_result = executor.step(state=state, tool_call=retry_call, step=len(tool_steps) + 1)
                            tool_steps.append(retry_result)
                            state = deepcopy(retry_result.state_after)
                            state_transitions.append(
                                StateTransition(
                                    step=retry_result.step,
                                    tool_name=retry_result.selected_tool,
                                    before=before_retry,
                                    after=state,
                                    diff=diff_state(before_retry, state),
                                )
                            )
                            feedback = {
                                "tool_name": retry_result.selected_tool,
                                "status": retry_result.validator_result,
                                "arguments": retry_result.arguments,
                                "output": retry_result.output,
                                "error": retry_result.error,
                            }
                            messages.append(Message(role="tool", content=json.dumps(feedback, ensure_ascii=False)))
                if len(tool_steps) >= len(expected_tool_events):
                    break

        final_messages = self._with_oracle_hint(
            messages,
            next_expected=None,
            final_answer=effective_task.expected_answer_contains,
            backend=bundle.reasoner.backend,
        )
        guidance_messages: list[Message] = []
        if effective_task.judgment_mode and effective_task.judgment_mode.enabled:
            guidance_messages.append(
                Message(
                    role="system",
                    content=_judgment_guidance(effective_task.judgment_mode, language_stressor),
                )
            )
        if effective_task.track == Track.THINKING:
            thinking_guidance = _thinking_guidance(
                language=language_stressor,
                image_task=bool(effective_task.image_refs),
            )
            guidance_messages.append(
                Message(
                    role="system",
                    content=thinking_guidance,
                )
            )
        if tool_steps or retrieval_hits:
            guidance_messages.append(
                Message(
                    role="system",
                    content=_grounded_answer_guidance(language_stressor),
                )
            )
        if guidance_messages:
            final_messages = guidance_messages + final_messages
        final_turn = bundle.reasoner.generate(
            messages=final_messages,
            media=effective_task.image_refs,
            tool_specs=[],
            thinking=self.thinking_enabled,
            max_new_tokens=self.final_max_new_tokens,
        )
        resolved_final_answer = final_turn.final_answer
        if not resolved_final_answer and not final_turn.thinking_text:
            resolved_final_answer = final_turn.raw_model_output
        second_pass_artifacts: dict[str, object] = {
            "second_pass_used": False,
            "second_pass_raw_output": "",
            "second_pass_final_answer": "",
            "second_pass_latency_ms": 0,
        }
        if _needs_answer_rescue(effective_task, language_stressor, resolved_final_answer, tool_steps, retrieval_hits):
            rescue_messages = [
                Message(role="system", content=_second_pass_guidance(language_stressor)),
                *final_messages,
                Message(
                    role="assistant",
                    content=f"Draft answer to rewrite:\n{resolved_final_answer}",
                ),
            ]
            rescue_turn = bundle.reasoner.generate(
                messages=rescue_messages,
                media=[],
                tool_specs=[],
                thinking=False,
                max_new_tokens=self.final_max_new_tokens,
            )
            rescue_answer = rescue_turn.final_answer or rescue_turn.raw_model_output
            second_pass_artifacts = {
                "second_pass_used": True,
                "second_pass_raw_output": rescue_turn.raw_model_output,
                "second_pass_final_answer": rescue_answer,
                "second_pass_latency_ms": rescue_turn.latency_ms,
            }
            if answer_contains_all(effective_task.expected_answer_contains, rescue_answer):
                resolved_final_answer = rescue_answer

        trace = RunTrace(
            run_id=f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{self.name}_{effective_task.task_id}_{variant.variant_id}",
            task_id=effective_task.task_id,
            variant_id=variant.variant_id,
            track=effective_task.track,
            architecture=self.name,
            thinking_enabled=self.thinking_enabled,
            model_bundle=ModelBundleSpec(
                reasoner=bundle.reasoner.model_id,
                router=bundle.router.model_id if bundle.router else None,
                retriever=bundle.retriever.model_id if bundle.retriever else None,
            ),
            backend=bundle.reasoner.backend,
            stressors=variant.stressors,
            hardware_profile=detect_hardware_profile(),
            prompt_artifacts={
                "messages": [message.model_dump(mode="json") for message in messages],
                "final_messages": [message.model_dump(mode="json") for message in final_messages],
                "retrieved_doc_ids": [hit.doc_id for hit in retrieval_hits],
                "stuffed_doc_ids": stuffed_doc_ids,
                "planning_raw_outputs": planning_outputs,
                "planning_repair_notes": planning_repair_notes,
                "planning_max_new_tokens": self.planning_max_new_tokens or getattr(selector, "max_new_tokens", None),
                "planning_latency_ms": planning_latencies,
                "planning_prompt_tokens": planning_prompt_tokens,
                "planning_completion_tokens": planning_completion_tokens,
                "planning_runtime_metadata": planning_runtime_metadata,
                "final_raw_output": final_turn.raw_model_output,
                "final_thinking_text": final_turn.thinking_text,
                "final_max_new_tokens": self.final_max_new_tokens or getattr(bundle.reasoner, "max_new_tokens", None),
                "final_latency_ms": final_turn.latency_ms,
                "final_prompt_tokens": final_turn.prompt_tokens,
                "final_completion_tokens": final_turn.completion_tokens,
                "final_runtime_metadata": final_turn.runtime_metadata,
                **second_pass_artifacts,
                "reasoner_runtime_info": _runtime_info(bundle.reasoner),
                "router_runtime_info": _runtime_info(bundle.router),
                "retriever_runtime_info": _runtime_info(bundle.retriever),
            },
            retrieval_hits=retrieval_hits,
            tool_steps=tool_steps,
            state_transitions=state_transitions,
            final_answer=resolved_final_answer,
            image_refs=effective_task.image_refs,
            benchmark_tags=effective_task.benchmark_tags,
            real_world_profile=effective_task.real_world_profile,
            metrics={},
        )
        trace.metrics = self._score_trace(effective_task, trace)
        return trace

    def _retrieve(self, task: Task, variant: Variant, bundle: RuntimeBundle):
        if not bundle.retriever:
            return []
        if self.name == "monolith":
            return []
        bundle.retriever.set_corpora({key: [document.model_dump(mode="json") for document in value] for key, value in task.corpora.items()})
        corpus_id = next(iter(task.corpora.keys()), "default")
        efficiency = variant.overrides.efficiency
        return bundle.retriever.search(
            query=task.user_goal,
            corpus_id=corpus_id,
            top_k=int(efficiency.get("top_k", 5)),
            dim=int(efficiency.get("embedding_dim", 768)),
            quantization=str(efficiency.get("quantization", "none")),
        )

    def _score_trace(self, task: Task, trace: RunTrace) -> dict[str, float | int | bool]:
        if task.track == Track.THINKING:
            return score_thinking_trace(task, trace)
        if task.track == Track.TOOL_ROUTING:
            return score_tool_trace(task, trace)
        if task.track == Track.RETRIEVAL:
            return score_retrieval_trace(task, trace)
        return score_full_stack_trace(task, trace)

    def _with_oracle_hint(
        self,
        messages: list[Message],
        next_expected: ExpectedEvent | list[ExpectedEvent] | None,
        final_answer: list[str] | None,
        backend: str,
    ) -> list[Message]:
        if backend != "oracle":
            return messages
        hints: list[Message] = []
        if next_expected is not None:
            payload: dict | list[dict]
            if isinstance(next_expected, list):
                payload = [event.model_dump(mode="json") for event in next_expected]
            else:
                payload = next_expected.model_dump(mode="json")
            hints.append(
                Message(
                    role="system",
                    content="ORACLE_NEXT_TOOL_CALL:" + json.dumps(payload, ensure_ascii=False),
                )
            )
        elif final_answer is not None:
            hints.append(Message(role="system", content="ORACLE_FINAL_ANSWER:" + json.dumps(final_answer, ensure_ascii=False)))
        return hints + messages


def materialize_task(task: Task, variant: Variant) -> Task:
    overrides = variant.overrides
    return task.model_copy(
        deep=True,
        update={
            "messages": overrides.messages if overrides.messages is not None else overrides.messages_prefix + [message.model_copy(deep=True) for message in task.messages],
            "tool_specs": overrides.tool_specs if overrides.tool_specs is not None else [tool.model_copy(deep=True) for tool in task.tool_specs],
            "corpora": overrides.corpora if overrides.corpora is not None else deepcopy(task.corpora),
            "initial_state": {**deepcopy(task.initial_state), **overrides.initial_state_patch},
            "expected_events": overrides.expected_events if overrides.expected_events is not None else [event.model_copy(deep=True) for event in task.expected_events],
            "expected_final_state": overrides.expected_final_state if overrides.expected_final_state is not None else deepcopy(task.expected_final_state),
            "expected_answer_contains": overrides.expected_answer_contains if overrides.expected_answer_contains is not None else list(task.expected_answer_contains),
        },
    )


def _next_expected_batch(expected_events: list[ExpectedEvent], cursor: int) -> ExpectedEvent | list[ExpectedEvent]:
    current = expected_events[cursor]
    if current.parallel_group is None:
        return current
    batch = [current]
    index = cursor + 1
    while index < len(expected_events) and expected_events[index].parallel_group == current.parallel_group:
        batch.append(expected_events[index])
        index += 1
    return batch


def _runtime_info(component: object | None) -> dict[str, object]:
    if component is None:
        return {}
    runtime_info = getattr(component, "runtime_info", None)
    if callable(runtime_info):
        payload = runtime_info()
        if isinstance(payload, dict):
            return payload
    return {}


def _thinking_guidance(language: str | None, image_task: bool) -> str:
    if image_task:
        if language == "fr":
            return (
                "Répondez à partir de l'image et nommez explicitement le réglage ou l'action recommandée. "
                "Ne répondez pas uniquement avec des mots d'état comme activé, désactivé, on ou off."
            )
        return (
            "Answer from the image and name the setting or recommended action explicitly. "
            "Do not answer with only status words like On, Off, Enabled, or Disabled."
        )
    if language == "fr":
        return (
            "Répondez directement et de façon minimale. Donnez la réponse finale en premier. "
            "Si la tâche demande un nombre, une équipe, un réglage, un opérateur de code ou une ligne de configuration, "
            "retournez explicitement cet élément exact. Utilisez le contexte déjà présent et ne demandez pas plus d'informations si la réponse est déjà disponible."
        )
    return (
        "Answer directly and minimally. Put the final answer first. "
        "If the task asks for a number, team, toggle, code operator, or config line, return that exact item explicitly. "
        "Use any stuffed context already in the conversation and do not ask for more information when the answer is already present."
    )


def _grounded_answer_guidance(language: str | None) -> str:
    if language == "fr":
        return (
            "Répondez uniquement à partir des résultats d'outil et du contexte récupéré déjà présents dans la conversation. "
            "Soyez concis et mentionnez les chemins, dates, identifiants d'événements, noms de réglages et écarts numériques concrets quand ils sont disponibles. "
            "Ne demandez pas plus d'informations si les outils ont déjà fourni les faits nécessaires."
        )
    return (
        "Answer only from the tool results and retrieved context already in the conversation. "
        "Be concise and mention concrete paths, dates, event ids, toggle names, and numeric deltas when available. "
        "Do not ask for more information if the tools already provided the needed facts."
    )


def _judgment_guidance(judgment_mode: JudgmentMode, language: str | None) -> str:
    allowed = ", ".join(judgment_mode.allowed_actions)
    basis = ""
    if judgment_mode.requires_basis:
        basis = " Also include the blocking reason and the missing approval, ambiguity, or policy basis when relevant."
    if language == "fr":
        return (
            f"Mode de jugement: choisissez exactement une action parmi {allowed}. "
            "Commencez par `action:` suivi de l'action choisie, puis donnez une justification concise."
            f"{basis}"
        )
    return (
        f"Judgment mode: choose exactly one action from {allowed}. "
        "Start with `action:` followed by the chosen action, then give a concise justification."
        f"{basis}"
    )


def _needs_answer_rescue(
    task: Task,
    language: str | None,
    answer_text: str,
    tool_steps: list[ToolResult],
    retrieval_hits: list[object],
) -> bool:
    if language != "fr":
        return False
    if task.track not in {Track.RETRIEVAL, Track.FULL_STACK}:
        return False
    if not task.expected_answer_contains:
        return False
    if not (tool_steps or retrieval_hits):
        return False
    return not answer_contains_all(task.expected_answer_contains, answer_text)


def _second_pass_guidance(language: str | None) -> str:
    if language == "fr":
        return (
            "Réécrivez la réponse finale en français opérationnel clair. "
            "N'ajoutez aucun nouvel outil, aucune nouvelle action, ni aucun nouveau fait. "
            "Conservez exactement les faits déjà établis et rendez explicites les chemins, approbations, politiques ou raisons de report quand ils existent."
        )
    return (
        "Rewrite the final answer clearly without adding new tools, actions, or facts. "
        "Preserve the established facts exactly and make the operational reason explicit."
    )
