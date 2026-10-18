"""Shared explanation service managing payload sanitization, caching, and Groq execution."""

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Union

from ecg_lab.explanations.groq_client import (
    execute_chat_completion,
    resolve_groq_api_key,
    resolve_groq_model,
)
from ecg_lab.explanations.prompts import SYSTEM_PROMPT, build_explanation_prompt
from ecg_lab.explanations.schemas import ExplanationPayload, ExplanationResult

# Global in-memory cache for explanation results
_EXPLANATION_CACHE: Dict[str, ExplanationResult] = {}

PROMPT_VERSION: str = "1.0.0"

# Strict whitelist of allowed top-level payload keys
ALLOWED_PAYLOAD_KEYS = {
    "schema_version",
    "workflow",
    "source_category",
    "is_simulation",
    "sampling_rate_hz",
    "units_and_calibration",
    "summary_metrics",
    "class_predictions",
    "data_quality_flags",
    "evaluation_context",
    "known_limitations",
}

# Blacklist patterns for disallowed sensitive content
DISALLOWED_KEY_PATTERNS = [
    r"patient",
    r"subject",
    r"name",
    r"email",
    r"dob",
    r"birth",
    r"hospital",
    r"path",
    r"filename",
    r"file_",
    r"raw_",
    r"signal",
    r"waveform",
    r"trace",
    r"bytes",
    r"image",
    r"secret",
    r"key",
    r"token",
]


def is_sensitive_key(key: str) -> bool:
    """Check if a dictionary key matches any sensitive or identifying patterns."""
    k_lower = key.lower()
    return any(re.search(pat, k_lower) for pat in DISALLOWED_KEY_PATTERNS)


def sanitize_value(val: Any) -> Any:
    """Recursively sanitize payload values, removing arrays, paths, or large objects."""
    if isinstance(val, (int, float, bool)):
        return val
    if isinstance(val, str):
        # Scrub potential file paths
        if "/" in val or "\\" in val:
            if ".." in val or val.startswith(("/", "C:", "D:")):
                return "[FILE_PATH_REDACTED]"
        return val[:500]  # Cap string length
    if isinstance(val, dict):
        sanitized_dict = {}
        for k, v in val.items():
            if is_sensitive_key(str(k)):
                continue
            sanitized_dict[str(k)] = sanitize_value(v)
        return sanitized_dict
    if isinstance(val, (list, tuple)):
        # Never serialize raw numerical arrays or long lists of beats
        if len(val) > 25:
            return f"[Truncated list of {len(val)} items]"
        return [sanitize_value(item) for item in val]
    return str(val)[:200]


def sanitize_payload(payload: Union[ExplanationPayload, Dict[str, Any]]) -> Dict[str, Any]:
    """Strictly filter and sanitize payload data before passing to LLM prompts.

    Guarantees:
        - Only ALLOWED_PAYLOAD_KEYS are retained.
        - All sub-dictionaries are checked for sensitive keys (patient IDs, paths, raw arrays).
        - No raw waveform arrays or scanned image data can leak through.
    """
    if hasattr(payload, "to_dict"):
        raw_dict = payload.to_dict()
    else:
        raw_dict = dict(payload)

    clean_payload: Dict[str, Any] = {}
    for key, value in raw_dict.items():
        if key not in ALLOWED_PAYLOAD_KEYS:
            continue
        clean_payload[key] = sanitize_value(value)

    return clean_payload


def compute_payload_cache_key(sanitized_payload: Dict[str, Any], model_id: str) -> str:
    """Compute deterministic SHA-256 cache key based on sanitized payload and model version."""
    payload_str = json.dumps(sanitized_payload, sort_keys=True)
    composite = f"{model_id}:{PROMPT_VERSION}:{payload_str}"
    return hashlib.sha256(composite.encode("utf-8")).hexdigest()


def parse_explanation_json(raw_text: str) -> Dict[str, str]:
    """Parse JSON string from Groq output with markdown fence stripping."""
    text = raw_text.strip()
    if text.startswith("```"):
        # Strip opening fence (``` or ```json)
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        # Strip closing fence
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Fallback regex extraction of keys if strict json parsing failed
    extracted = {}
    for k in (
        "simple_terms",
        "what_numbers_mean",
        "what_stands_out",
        "limitations",
        "what_this_does_not_tell_you",
        "suggested_next_step",
    ):
        pattern = rf'"{k}"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
        m = re.search(pattern, raw_text)
        if m:
            extracted[k] = m.group(1).replace('\\"', '"').replace("\\n", "\n")
        else:
            extracted[k] = "Information not provided in output."
    return extracted


def explain_results(
    payload: Union[ExplanationPayload, Dict[str, Any]],
    client: Optional[Any] = None,
    model: Optional[str] = None,
    use_cache: bool = True,
) -> ExplanationResult:
    """Generate structured, plain-language explanation for an analysis run.

    Args:
        payload: ExplanationPayload or equivalent dictionary.
        client: Optional pre-configured Groq client (useful for unit tests with mocks).
        model: Optional model identifier override.
        use_cache: Whether to check and write to in-memory/session cache.

    Returns:
        ExplanationResult containing all 6 structured sections.
    """
    clean_payload = sanitize_payload(payload)
    workflow = clean_payload.get("workflow", "general")
    active_model = model or resolve_groq_model()

    # Cache check
    cache_key = compute_payload_cache_key(clean_payload, active_model)
    if use_cache and cache_key in _EXPLANATION_CACHE:
        cached_res = _EXPLANATION_CACHE[cache_key]
        return ExplanationResult(
            success=cached_res.success,
            workflow=cached_res.workflow,
            model_id=cached_res.model_id,
            simple_terms=cached_res.simple_terms,
            what_numbers_mean=cached_res.what_numbers_mean,
            what_stands_out=cached_res.what_stands_out,
            limitations=cached_res.limitations,
            what_this_does_not_tell_you=cached_res.what_this_does_not_tell_you,
            suggested_next_step=cached_res.suggested_next_step,
            cached=True,
            error_message=cached_res.error_message,
            raw_response=cached_res.raw_response,
        )

    # Check if API key is present
    api_key = resolve_groq_api_key()
    if client is None and not api_key:
        return ExplanationResult(
            success=False,
            workflow=workflow,
            model_id=active_model,
            simple_terms=(
                "Groq API key is not configured. Set GROQ_API_KEY in your environment, "
                ".env file, or .streamlit/secrets.toml to generate live AI explanations."
            ),
            what_numbers_mean="Numbers represent deterministic algorithm outputs computed locally by CardioSense.",
            what_stands_out="Live AI explanation is offline because no API key was provided.",
            limitations="AI explanation unavailable until GROQ_API_KEY is supplied.",
            what_this_does_not_tell_you="This prototype is not a certified medical device and predictions are not diagnoses.",
            suggested_next_step="Configure an API key in your environment or secrets to enable automated explanations.",
            error_message="API key not configured.",
        )

    # Build prompt
    user_prompt = build_explanation_prompt(clean_payload)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Execute chat completion
    success, raw_content, error_msg = execute_chat_completion(
        messages=messages,
        model=active_model,
        client=client,
        json_mode=True,
    )

    if not success:
        return ExplanationResult(
            success=False,
            workflow=workflow,
            model_id=active_model,
            simple_terms="An error occurred while communicating with the explanation service.",
            what_numbers_mean="Computed algorithm metrics remain visible in the dashboard above.",
            what_stands_out="API explanation request failed.",
            limitations="AI explanation temporarily unavailable.",
            what_this_does_not_tell_you="Software outputs are experimental research prototypes and not medical advice.",
            suggested_next_step="Check your internet connection or verify your API configuration and rate limits.",
            error_message=error_msg,
        )

    # Parse and validate output JSON
    parsed = parse_explanation_json(raw_content)

    result = ExplanationResult(
        success=True,
        workflow=workflow,
        model_id=active_model,
        simple_terms=parsed.get("simple_terms", "Explanation generated."),
        what_numbers_mean=parsed.get("what_numbers_mean", "Refer to metrics table."),
        what_stands_out=parsed.get("what_stands_out", "No anomalies identified."),
        limitations=parsed.get("limitations", "Research prototype limitations apply."),
        what_this_does_not_tell_you=parsed.get("what_this_does_not_tell_you", "Not a medical diagnosis."),
        suggested_next_step=parsed.get("suggested_next_step", "Inspect raw waveforms."),
        cached=False,
        raw_response=parsed,
    )

    if use_cache:
        _EXPLANATION_CACHE[cache_key] = result

    return result


def clear_explanation_cache() -> None:
    """Clear all in-memory cached explanations."""
    global _EXPLANATION_CACHE
    _EXPLANATION_CACHE.clear()
