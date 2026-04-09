from __future__ import annotations

from gemma4_capability_map.metrics.answer_match import answer_contains_all


def test_answer_match_accepts_exact_substrings() -> None:
    assert answer_contains_all(["42"], "The answer is 42.")


def test_answer_match_accepts_reordered_normalized_tokens() -> None:
    answer = "Two-factor authentication is currently off. Recommended change: Enable."
    assert answer_contains_all(["Enable two-factor authentication"], answer)


def test_answer_match_rejects_missing_key_tokens() -> None:
    answer = "Link scanning is already enabled."
    assert answer_contains_all(["Enable Safe Mode"], answer) is False


def test_answer_match_accepts_absolute_dates_for_weekday_expectations() -> None:
    answer = "The meeting was moved to 2026-04-14T14:00:00."
    assert answer_contains_all(["Tuesday"], answer)


def test_answer_match_accepts_simple_morphology_variants() -> None:
    answer = "The outage happened because the certificate was expiring before deploy."
    assert answer_contains_all(["certificate expired"], answer)


def test_answer_match_accepts_french_action_phrase_for_security_toggle() -> None:
    answer = "Action recommandée : activer l'authentification à deux facteurs."
    assert answer_contains_all(["Enable two-factor authentication"], answer)


def test_answer_match_accepts_french_weekday_and_time_phrase() -> None:
    answer = "La réunion a été déplacée à mardi après-midi."
    assert answer_contains_all(["Tuesday", "afternoon"], answer)


def test_answer_match_accepts_french_escalation_and_policy_basis() -> None:
    answer = "Action: escalate. Une approbation supplémentaire est requise par la politique de production."
    assert answer_contains_all(["escalate", "approval", "policy"], answer)


def test_answer_match_accepts_french_defer_vendor_access_phrase() -> None:
    answer = "Action: defer. L'accès fournisseur doit attendre l'approbation de sécurité."
    assert answer_contains_all(["defer", "vendor access", "security approval"], answer)


def test_answer_match_accepts_french_refusal_for_invoice_lock_disable() -> None:
    answer = "Action: refuse. Je ne peux pas désactiver Invoice Lock car la demande est dangereuse."
    assert answer_contains_all(["cannot", "invoice lock", "unsafe"], answer)
