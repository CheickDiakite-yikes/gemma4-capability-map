from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Protocol


_TOKEN_ALIASES = {
    "vehicles": "vehicle",
    "cars": "car",
    "metrics": "metric",
    "forms": "form",
    "errors": "error",
    "totals": "total",
    "slides": "slide",
    "white": "white",
    "blanches": "white",
    "blanche": "white",
    "blanc": "white",
    "slots": "slot",
    "parking": "parking",
    "empty": "empty",
    "vacants": "empty",
    "vacant": "empty",
    "blocked": "blocked",
    "bloque": "blocked",
    "bloquee": "blocked",
    "sorties": "exit",
    "sortie": "exit",
    "exits": "exit",
    "invoice": "invoice",
    "facture": "invoice",
    "factures": "invoice",
    "totals": "total",
    "below": "below",
    "under": "below",
    "target": "target",
    "disabled": "disabled",
    "destructive": "destructive",
    "risk": "risk",
    "risks": "risk",
    "anomaly": "anomaly",
    "anomalies": "anomaly",
    "dashboards": "dashboard",
    "dashboard": "dashboard",
}


class VisualExecutor(Protocol):
    def segment_entities(self, state: dict[str, Any], image_id: str, entity_query: str) -> tuple[dict[str, Any], dict[str, Any]]:
        ...

    def refine_selection(self, state: dict[str, Any], selection_id: str, filter_query: str) -> tuple[dict[str, Any], dict[str, Any]]:
        ...

    def extract_layout(self, state: dict[str, Any], image_id: str, target_query: str) -> tuple[dict[str, Any], dict[str, Any]]:
        ...

    def read_region_text(self, state: dict[str, Any], image_id: str, region_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        ...


@dataclass
class SeededVisualExecutor:
    mode: str = "seeded"

    def segment_entities(self, state: dict[str, Any], image_id: str, entity_query: str) -> tuple[dict[str, Any], dict[str, Any]]:
        image = _load_image_record(state, image_id)
        matches = [entity for entity in _entities_for_mode(image, self.mode) if _candidate_matches(entity, entity_query)]
        selection_id = _store_selection(
            state,
            image_id=image_id,
            selection_kind="entities",
            items=matches,
            query=entity_query,
            parent_selection_id=None,
        )
        return state, _selection_output(selection_id, image_id, matches, selection_kind="entities", executor_mode=self.mode)

    def refine_selection(self, state: dict[str, Any], selection_id: str, filter_query: str) -> tuple[dict[str, Any], dict[str, Any]]:
        selection = _selection_from_state(state, selection_id)
        items = list(selection.get("items", []))
        filtered = [item for item in items if _candidate_matches(item, filter_query)]
        refined_id = _store_selection(
            state,
            image_id=str(selection.get("image_id", "")),
            selection_kind=str(selection.get("selection_kind", "entities")),
            items=filtered,
            query=filter_query,
            parent_selection_id=selection_id,
        )
        return state, _selection_output(
            refined_id,
            str(selection.get("image_id", "")),
            filtered,
            selection_kind=str(selection.get("selection_kind", "entities")),
            executor_mode=self.mode,
            parent_selection_id=selection_id,
        )

    def extract_layout(self, state: dict[str, Any], image_id: str, target_query: str) -> tuple[dict[str, Any], dict[str, Any]]:
        image = _load_image_record(state, image_id)
        matches = [region for region in _layouts_for_mode(image, self.mode) if _candidate_matches(region, target_query)]
        selection_id = _store_selection(
            state,
            image_id=image_id,
            selection_kind="regions",
            items=matches,
            query=target_query,
            parent_selection_id=None,
        )
        output = _selection_output(selection_id, image_id, matches, selection_kind="regions", executor_mode=self.mode)
        if matches:
            output["region_id"] = str(matches[0].get("region_id", ""))
            output["text_preview"] = str(matches[0].get("text", ""))
        return state, output

    def read_region_text(self, state: dict[str, Any], image_id: str, region_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        image = _load_image_record(state, image_id)
        for region in _layouts_for_mode(image, self.mode):
            if str(region.get("region_id", "")) == region_id:
                return state, {
                    "image_id": image_id,
                    "region_id": region_id,
                    "text": str(region.get("text", "")),
                    "label": str(region.get("label", "")),
                    "executor_mode": self.mode,
                }
        raise KeyError(f"Region not found: {region_id}")


@dataclass
class LocalVisualExecutor(SeededVisualExecutor):
    mode: str = "local"


@dataclass
class RoutedVisualExecutor:
    seeded: VisualExecutor
    local: VisualExecutor

    def segment_entities(self, state: dict[str, Any], image_id: str, entity_query: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._executor_for_state(state).segment_entities(state, image_id, entity_query)

    def refine_selection(self, state: dict[str, Any], selection_id: str, filter_query: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._executor_for_state(state).refine_selection(state, selection_id, filter_query)

    def extract_layout(self, state: dict[str, Any], image_id: str, target_query: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._executor_for_state(state).extract_layout(state, image_id, target_query)

    def read_region_text(self, state: dict[str, Any], image_id: str, region_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._executor_for_state(state).read_region_text(state, image_id, region_id)

    def _executor_for_state(self, state: dict[str, Any]) -> VisualExecutor:
        if str(state.get("visual_executor_mode", "seeded")).lower() == "local":
            return self.local
        return self.seeded


def build_visual_executor() -> RoutedVisualExecutor:
    return RoutedVisualExecutor(seeded=SeededVisualExecutor(), local=LocalVisualExecutor())


def _selection_output(
    selection_id: str,
    image_id: str,
    items: list[dict[str, Any]],
    *,
    selection_kind: str,
    executor_mode: str,
    parent_selection_id: str | None = None,
) -> dict[str, Any]:
    output = {
        "selection_id": selection_id,
        "image_id": image_id,
        "selection_kind": selection_kind,
        "count": len(items),
        "executor_mode": executor_mode,
    }
    if parent_selection_id:
        output["parent_selection_id"] = parent_selection_id
    if selection_kind == "entities":
        output["entity_ids"] = [str(item.get("entity_id", "")) for item in items]
    else:
        output["region_ids"] = [str(item.get("region_id", "")) for item in items]
    return output


def _store_selection(
    state: dict[str, Any],
    *,
    image_id: str,
    selection_kind: str,
    items: list[dict[str, Any]],
    query: str,
    parent_selection_id: str | None,
) -> str:
    counter = int(state.get("visual_selection_counter", 0)) + 1
    state["visual_selection_counter"] = counter
    selection_id = f"sel-{counter:03d}"
    state.setdefault("visual_selections", {})[selection_id] = {
        "image_id": image_id,
        "selection_kind": selection_kind,
        "items": items,
        "query": query,
        "parent_selection_id": parent_selection_id,
    }
    state["visual_last_selection_id"] = selection_id
    return selection_id


def _selection_from_state(state: dict[str, Any], selection_id: str) -> dict[str, Any]:
    selections = state.get("visual_selections", {})
    if selection_id not in selections:
        raise KeyError(f"Selection not found: {selection_id}")
    return selections[selection_id]


def _load_image_record(state: dict[str, Any], image_id: str) -> dict[str, Any]:
    images = state.get("images", {})
    image = images.get(image_id)
    if image is None:
        if len(images) == 1:
            return next(iter(images.values()))
        resolved_id = _resolve_image_id_from_path(image_id, images)
        if resolved_id:
            return images[resolved_id]
        raise KeyError(f"Image not found: {image_id}")
    return image


def _entities_for_mode(image: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    if mode == "local" and isinstance(image.get("local_entities"), list):
        return list(image["local_entities"])
    return list(image.get("entities", []))


def _layouts_for_mode(image: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    if mode == "local" and isinstance(image.get("local_layouts"), list):
        return list(image["local_layouts"])
    return list(image.get("layouts", []))


def _candidate_matches(candidate: dict[str, Any], query: str) -> bool:
    query_tokens = _informative_tokens(query)
    if not query_tokens:
        return True
    candidate_tokens = _candidate_tokens(candidate)
    if not candidate_tokens:
        return False
    return all(token in candidate_tokens for token in query_tokens)


def _candidate_tokens(candidate: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key in ("label", "text", "entity_id", "region_id", "type", "category"):
        value = candidate.get(key)
        if isinstance(value, str):
            tokens.update(_informative_tokens(value))
    attributes = candidate.get("attributes", {})
    if isinstance(attributes, dict):
        for key, value in attributes.items():
            tokens.update(_informative_tokens(str(key)))
            tokens.update(_informative_tokens(str(value)))
    return tokens


def _informative_tokens(text: str) -> set[str]:
    normalized = _normalize(text)
    return {
        _TOKEN_ALIASES.get(token, token)
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) > 2
    }


def _normalize(text: str) -> str:
    lowered = unicodedata.normalize("NFKD", text)
    lowered = "".join(character for character in lowered if not unicodedata.combining(character))
    return lowered.lower()


def _resolve_image_id_from_path(image_ref: str, images: dict[str, Any]) -> str:
    normalized = image_ref.replace("\\", "/").lower()
    stem = normalized.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    stem = stem.removeprefix("visual_").replace("_", "-")
    candidates = [str(key) for key in images.keys()]
    if not stem:
        return ""
    exact = [candidate for candidate in candidates if candidate.lower() == f"img-{stem}"]
    if exact:
        return exact[0]
    partial = [candidate for candidate in candidates if stem in candidate.lower()]
    if len(partial) == 1:
        return partial[0]
    live_partial = [candidate for candidate in partial if candidate.lower().endswith("-live")]
    if len(live_partial) == 1:
        return live_partial[0]
    return ""
