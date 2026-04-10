from __future__ import annotations

import re
import unicodedata
from datetime import datetime

from gemma4_capability_map.schemas import Task


PHRASE_ALIASES = {
    "authentification a deux facteurs": "two factor authentication",
    "double authentification": "two factor authentication",
    "mode sans echec": "safe mode",
    "apres midi": "afternoon",
    "high risk": "unsafe",
    "safety control": "unsafe",
    "safety controls": "unsafe",
    "verrouillage des factures": "invoice lock",
    "verrouillage de facture": "invoice lock",
    "controle des factures": "invoice lock",
    "acces fournisseur": "vendor access",
    "approbation de securite": "security approval",
    "approbation requise": "approval required",
    "mise a jour de facturation": "billing update",
    "demander des precisions": "clarify",
    "besoin de precisions": "clarify",
    "demander une clarification": "clarify",
    "demander une approbation": "escalate",
    "demander l approbation": "escalate",
    "escalader": "escalate",
    "refuser la demande": "refuse",
    "reporter la demande": "defer",
    "je ne peux pas": "cannot",
}

TOKEN_ALIASES = {
    "activer": "enable",
    "activez": "enable",
    "active": "enable",
    "activation": "enable",
    "desactiver": "disable",
    "desactivez": "disable",
    "desactive": "disable",
    "specific": "which",
    "authentification": "authentication",
    "deux": "two",
    "facteurs": "factor",
    "facteur": "factor",
    "mardi": "tuesday",
    "lundi": "monday",
    "mercredi": "wednesday",
    "jeudi": "thursday",
    "vendredi": "friday",
    "samedi": "saturday",
    "dimanche": "sunday",
    "verrouillage": "lock",
    "factures": "invoice",
    "facture": "invoice",
    "escalader": "escalate",
    "escalade": "escalate",
    "escalation": "escalate",
    "deferer": "defer",
    "reporter": "defer",
    "clarifier": "clarify",
    "clarification": "clarify",
    "preciser": "clarify",
    "precisions": "clarify",
    "approbation": "approval",
    "approuveur": "approver",
    "politique": "policy",
    "risk": "unsafe",
    "safety": "unsafe",
    "dangereux": "unsafe",
    "dangereuse": "unsafe",
    "risque": "unsafe",
    "refuser": "refuse",
    "refus": "refuse",
    "annuler": "rollback",
    "retablir": "rollback",
    "fournisseur": "vendor",
    "acces": "access",
    "securite": "security",
    "facturation": "billing",
    "verrou": "lock",
}


def answer_contains_all(expected_fragments: list[str], answer_text: str) -> bool:
    answer_lower = _normalize_text(answer_text)
    answer_tokens = set(_normalize_tokens(answer_text))
    return all(_fragment_matches(fragment, answer_lower, answer_tokens) for fragment in expected_fragments)


def answer_matches_task(task: Task, answer_text: str) -> bool:
    judgment_mode = task.judgment_mode
    if judgment_mode is None or not judgment_mode.enabled:
        return answer_contains_all(task.expected_answer_contains, answer_text)
    return judgment_answer_matches(task, answer_text)


def judgment_answer_matches(task: Task, answer_text: str) -> bool:
    judgment_mode = task.judgment_mode
    if judgment_mode is None or not judgment_mode.enabled:
        return answer_contains_all(task.expected_answer_contains, answer_text)

    extracted_action = extract_judgment_action(answer_text)
    expected_action = judgment_mode.expected_action
    legacy_expected_match = bool(task.expected_answer_contains) and answer_contains_all(task.expected_answer_contains, answer_text)
    if expected_action:
        if extracted_action is None:
            if not legacy_expected_match:
                return False
        elif extracted_action != expected_action:
            return False

    supporting_fragments = list(judgment_mode.basis_fragments)
    if not supporting_fragments:
        supporting_fragments = list(task.expected_answer_contains)
    if not supporting_fragments:
        return True

    if answer_contains_all(supporting_fragments, answer_text):
        return True
    if legacy_expected_match:
        return True
    return False


def extract_judgment_action(answer_text: str) -> str | None:
    normalized = _normalize_text(answer_text)
    match = re.search(r"\baction\s*:\s*([a-z]+)", normalized)
    if not match:
        return None
    action = TOKEN_ALIASES.get(match.group(1), match.group(1))
    if action in {"proceed", "escalate", "defer", "clarify", "refuse"}:
        return action
    return None


def _fragment_matches(fragment: str, answer_lower: str, answer_tokens: set[str]) -> bool:
    fragment_lower = _normalize_text(fragment)
    if fragment_lower in answer_lower:
        return True
    fragment_tokens = _raw_tokens(fragment)
    if not fragment_tokens and fragment_lower not in {"morning", "afternoon", "evening", "night"}:
        return False
    if fragment_lower in {"morning", "afternoon", "evening", "night"} and fragment_lower in answer_tokens:
        return True
    return all(bool(_token_variants(token) & answer_tokens) for token in fragment_tokens)


def _normalize_tokens(text: str) -> list[str]:
    tokens = _raw_tokens(text)
    expanded: list[str] = []
    for token in tokens:
        expanded.extend(sorted(_token_variants(token)))
    expanded.extend(_semantic_time_tokens(text))
    return expanded


def _raw_tokens(text: str) -> list[str]:
    normalized = _normalize_text(text).replace("_", " ").replace("-", " ")
    return re.findall(r"[a-z0-9]+", normalized)


def _token_variants(token: str) -> set[str]:
    token = TOKEN_ALIASES.get(token, token)
    variants = {token}
    if len(token) < 4:
        return variants
    if token.endswith("ies") and len(token) > 4:
        variants.add(token[:-3] + "y")
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        variants.add(stem)
        variants.add(_restore_terminal_e(stem))
    if token.endswith("ed") and len(token) > 4:
        stem = token[:-2]
        variants.add(stem)
        variants.add(_restore_terminal_e(stem))
    if token.endswith("es") and len(token) > 4:
        stem = token[:-2]
        variants.add(stem)
        variants.add(_restore_terminal_e(stem))
    if token.endswith("s") and not token.endswith("ss") and len(token) > 4:
        variants.add(token[:-1])
    return {variant for variant in variants if variant}


def _restore_terminal_e(stem: str) -> str:
    if not stem:
        return stem
    if stem.endswith("e"):
        return stem
    return stem + "e"


def _semantic_time_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for match in re.findall(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?", _deaccent_lower(text)):
        try:
            instant = datetime.fromisoformat(match)
        except ValueError:
            continue
        tokens.append(instant.strftime("%A").lower())
        hour = instant.hour
        if 5 <= hour < 12:
            tokens.append("morning")
        elif 12 <= hour < 18:
            tokens.append("afternoon")
        elif 18 <= hour < 22:
            tokens.append("evening")
        else:
            tokens.append("night")
    return tokens


def _normalize_text(text: str) -> str:
    normalized = _deaccent_lower(text)
    normalized = normalized.replace("-", " ").replace("_", " ")
    for source, target in sorted(PHRASE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        normalized = normalized.replace(source, target)
    return normalized


def _deaccent_lower(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(character for character in normalized if not unicodedata.combining(character))
    return normalized.lower()
