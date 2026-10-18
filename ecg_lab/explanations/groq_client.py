"""Groq API client wrapper with robust secret resolution, timeout, and bounded retry policy."""

import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from groq import (
        APIConnectionError,
        APIError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        Groq,
        RateLimitError,
    )
    GROQ_SDK_AVAILABLE = True
except ImportError:
    GROQ_SDK_AVAILABLE = False
    Groq = None
    AuthenticationError = Exception
    RateLimitError = Exception
    APITimeoutError = Exception
    APIConnectionError = Exception
    APIStatusError = Exception
    APIError = Exception

DEFAULT_GROQ_MODEL: str = "qwen/qwen3.8-27b"
DEFAULT_TIMEOUT_SECONDS: float = 25.0
DEFAULT_MAX_TOKENS: int = 1000


def resolve_groq_api_key() -> Optional[str]:
    """Resolve Groq API key in deterministic priority order without leaking secrets.

    Priority:
        1. os.environ["GROQ_API_KEY"] or os.environ["GROQ_API"]
        2. Streamlit secrets st.secrets["GROQ_API_KEY"] or st.secrets["GROQ_API"]
        3. Local .env file in workspace root
        4. Parent workspace .env file
    """
    # 1. Environment variables
    for var_name in ("GROQ_API_KEY", "GROQ_API"):
        val = os.environ.get(var_name)
        if val and val.strip():
            return val.strip()

    # 2. Streamlit secrets (if running inside Streamlit runtime)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for sec_key in ("GROQ_API_KEY", "GROQ_API"):
                if sec_key in st.secrets:
                    val = str(st.secrets[sec_key]).strip()
                    if val:
                        return val
    except Exception:
        pass

    # 3. .env file resolution
    env_paths = [
        Path(".env"),
        Path(__file__).resolve().parent.parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent.parent / ".env",
    ]

    for ep in env_paths:
        if ep.is_file():
            try:
                from dotenv import dotenv_values
                vals = dotenv_values(ep)
                for k, v in vals.items():
                    if k and k.strip() in ("GROQ_API_KEY", "GROQ_API"):
                        if v and v.strip():
                            return v.strip()
            except Exception:
                continue

    return None


def resolve_groq_model() -> str:
    """Resolve active Groq model identifier."""
    # Environment variable override
    env_model = os.environ.get("GROQ_MODEL")
    if env_model and env_model.strip():
        return env_model.strip()

    # Streamlit secrets override
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GROQ_MODEL" in st.secrets:
            return str(st.secrets["GROQ_MODEL"]).strip()
    except Exception:
        pass

    return DEFAULT_GROQ_MODEL


def sanitize_error_message(error: Exception) -> str:
    """Scrub potential API keys, auth tokens, or internal URLs from exception strings."""
    msg = str(error)
    # Mask gsk_... keys
    msg = re.sub(r"gsk_[a-zA-Z0-9_\-]+", "gsk_***", msg)
    # Mask Bearer tokens
    msg = re.sub(r"Bearer\s+[a-zA-Z0-9_\-\.]+", "Bearer ***", msg)
    return msg


def get_groq_client(api_key: Optional[str] = None) -> Any:
    """Construct a Groq client instance if the SDK and API key are available."""
    if not GROQ_SDK_AVAILABLE:
        raise RuntimeError("The 'groq' Python package is not installed.")

    resolved_key = api_key or resolve_groq_api_key()
    if not resolved_key:
        raise ValueError(
            "Groq API key not configured. Set GROQ_API_KEY in environment or .env file."
        )

    return Groq(api_key=resolved_key, timeout=DEFAULT_TIMEOUT_SECONDS)


def execute_chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    client: Optional[Any] = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.2,
    json_mode: bool = True,
) -> Tuple[bool, str, Optional[str]]:
    """Execute a chat completion request against Groq with bounded retry and error sanitization.

    Returns:
        Tuple of (success: bool, content: str, error_message: Optional[str])
    """
    selected_model = model or resolve_groq_model()

    active_client = client
    if active_client is None:
        try:
            active_client = get_groq_client()
        except Exception as e:
            return False, "", sanitize_error_message(e)

    kwargs: Dict[str, Any] = {
        "model": selected_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout": timeout_seconds,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    # Attempt execution with at most 1 controlled retry for transient errors
    max_attempts = 2
    last_err: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = active_client.chat.completions.create(**kwargs)
            if not response.choices or not response.choices[0].message:
                return False, "", "Groq API returned an empty response choices payload."

            content = response.choices[0].message.content or ""
            if not content.strip():
                return False, "", "Groq API returned empty text content."

            return True, content.strip(), None

        except AuthenticationError as auth_err:
            return False, "", f"Authentication failed: {sanitize_error_message(auth_err)}"

        except RateLimitError as rl_err:
            last_err = rl_err
            if attempt < max_attempts:
                time.sleep(1.0)  # Brief backoff before single retry
                continue
            return False, "", f"Rate limit exceeded (429): {sanitize_error_message(rl_err)}"

        except (APITimeoutError, APIConnectionError) as net_err:
            last_err = net_err
            if attempt < max_attempts:
                time.sleep(1.0)
                continue
            return False, "", f"Connection/Timeout failure: {sanitize_error_message(net_err)}"

        except APIStatusError as status_err:
            last_err = status_err
            if status_err.status_code and status_err.status_code >= 500 and attempt < max_attempts:
                time.sleep(1.0)
                continue
            return False, "", f"API Error (HTTP {status_err.status_code}): {sanitize_error_message(status_err)}"

        except Exception as general_err:
            return False, "", f"Unexpected error: {sanitize_error_message(general_err)}"

    return False, "", sanitize_error_message(last_err) if last_err else "Failed after retry."
