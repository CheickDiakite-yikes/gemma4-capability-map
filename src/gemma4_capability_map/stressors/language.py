from __future__ import annotations

from gemma4_capability_map.schemas import Message, StressorKind, Task, Variant, VariantOverrides


PREFIXES = {
    "fr": "Version française: ",
    "ar": "Arabic-flavored request: ",
    "code_switch": "Mixed language request: ",
}

EXACT_TRANSLATIONS = {
    "fr": {
        "Use the latest refund policy and tell me how many days a customer has to request a refund.": "Utilise la politique de remboursement la plus récente et indique sous combien de jours un client peut demander un remboursement.",
        "Look at the screenshot and tell me the settings change needed to improve account security.": "Regarde la capture d'écran et indique le changement de réglage nécessaire pour améliorer la sécurité du compte.",
        "Based on the screenshot, which safety toggle should be turned on?": "D'après la capture d'écran, quel réglage de sécurité faut-il activer ?",
        "Review the incident screenshot and name the safety setting that should be enabled before rollout.": "Examine la capture d'écran de l'incident et nomme le réglage de sécurité qui doit être activé avant le déploiement.",
        "Look at the billing screenshot and tell me which lock setting should be enabled to prevent invoice changes.": "Regarde la capture d'écran de facturation et indique quel réglage de verrouillage doit être activé pour éviter les modifications de facture.",
        "Use the incident notes and screenshot context to name the failed toggle.": "Utilise les notes d'incident et le contexte de la capture d'écran pour nommer le réglage qui a échoué.",
        "Which rollout mode should be set for safer deploys?": "Quel mode de déploiement faut-il définir pour des mises en production plus sûres ?",
        "Move my Friday meeting with Sarah to next Tuesday afternoon and tell me what changed.": "Déplace ma réunion de vendredi avec Sarah à mardi après-midi prochain et dis-moi ce qui a changé.",
        "Check the screenshot and config file in parallel, then record the safe mode patch.": "Vérifie la capture d'écran et le fichier de configuration en parallèle, puis enregistre le correctif pour activer safe mode.",
    },
    "code_switch": {
        "Look at the screenshot and tell me the settings change needed to improve account security.": "Peux-tu look at the screenshot and tell me the settings change needed to improve account security?",
        "Based on the screenshot, which safety toggle should be turned on?": "From la capture d'écran, which safety toggle should be turned on?",
        "Use the incident notes and screenshot context to name the failed toggle.": "Use les incident notes et le screenshot context to name the failed toggle.",
        "Move my Friday meeting with Sarah to next Tuesday afternoon and tell me what changed.": "Move ma réunion de Friday avec Sarah to next Tuesday afternoon and tell me what changed.",
        "Check the screenshot and config file in parallel, then record the safe mode patch.": "Check la screenshot et le config file in parallel, then record the safe mode patch.",
    },
}


def apply_language_variant(task: Task, flavor: str) -> Variant:
    updated_messages = [message.model_copy(deep=True) for message in task.messages]
    for message in updated_messages:
        if message.role == "user":
            message.content = _translate_message(message.content, flavor)
    return Variant(
        variant_id=f"{task.task_id}_language_{flavor}",
        base_task_id=task.task_id,
        primary_stressor=StressorKind.LANGUAGE,
        stressors={"language": flavor, "schema": None, "context": None, "efficiency": None},
        overrides=VariantOverrides(messages=updated_messages),
    )


def _translate_message(content: str, flavor: str) -> str:
    translated = EXACT_TRANSLATIONS.get(flavor, {}).get(content)
    if translated:
        return translated
    return PREFIXES.get(flavor, "Variant: ") + content
