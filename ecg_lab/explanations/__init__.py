"""CardioSense Explain Results Package — Safe, decoupled Groq LLM explanation layer."""

from ecg_lab.explanations.groq_client import (
    DEFAULT_GROQ_MODEL,
    execute_chat_completion,
    get_groq_client,
    resolve_groq_api_key,
    resolve_groq_model,
)
from ecg_lab.explanations.schemas import (
    ExplanationPayload,
    ExplanationResult,
    WorkflowType,
)
from ecg_lab.explanations.service import (
    clear_explanation_cache,
    explain_results,
    sanitize_payload,
)
from ecg_lab.explanations.ui import (
    render_explain_results_button,
    render_explanation_panel,
)

__all__ = [
    "DEFAULT_GROQ_MODEL",
    "ExplanationPayload",
    "ExplanationResult",
    "WorkflowType",
    "clear_explanation_cache",
    "execute_chat_completion",
    "explain_results",
    "get_groq_client",
    "render_explain_results_button",
    "render_explanation_panel",
    "resolve_groq_api_key",
    "resolve_groq_model",
    "sanitize_payload",
]
