"""Comprehensive test suite for Groq-Powered Clinical Explanation Service.

Verifies:
1. Whitelist-only payload sanitization and privacy guarantees.
2. Prompt construction and clinical safety constraints.
3. Groq client resolution, error sanitization, and bounded retries.
4. In-memory caching and payload deduplication.
5. Multi-workflow payload adapters (Workflows A, B, C, and Hardware Prototype).
6. Streamlit UI components and AppTest integration.
7. Frozen single-lead model SHA-256 weight integrity.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from ecg_lab.explanations.schemas import (
    ExplanationPayload,
    ExplanationResult,
    WorkflowType,
)
from ecg_lab.explanations.groq_client import (
    DEFAULT_GROQ_MODEL,
    execute_chat_completion,
    get_groq_client,
    resolve_groq_api_key,
    resolve_groq_model,
    sanitize_error_message,
)
from ecg_lab.explanations.prompts import (
    SYSTEM_PROMPT,
    WORKFLOW_INSTRUCTIONS,
    build_explanation_prompt,
)
from ecg_lab.explanations.service import (
    clear_explanation_cache,
    compute_payload_cache_key,
    explain_results,
    sanitize_payload,
)
from ecg_lab.explanations.ui import (
    DISCLAIMER_EXPLANATION_BANNER,
    render_explain_results_button,
    render_explanation_panel,
)
from ecg_lab.single_lead_ml.config import (
    DEFAULT_ARTIFACT_DIR,
    MODEL_NAME,
)

FROZEN_MODEL_SHA256 = "3889a5f5ab97803da339710d4ae9fde8ea50f8e5a5ff76ed2015a0fb0b6ec8e9"
APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


# ==============================================================================
# 1. Payload Sanitization & Privacy Tests
# ==============================================================================

class TestPayloadSanitization:
    """Ensure strict privacy by testing payload scrubbing and whitelist filtering."""

    def test_sanitize_payload_strips_disallowed_keys(self):
        dirty_input = {
            "workflow": "workflow_b_12lead",
            "source_category": "PTB-XL Record #00123",
            "is_simulation": False,
            "units_and_calibration": "Calibrated 12-lead (mV)",
            "summary_metrics": {"NORM": "92.0%"},
            "data_quality_flags": ["Clean trace"],
            "known_limitations": ["Research prototype only"],
            # Disallowed private keys
            "patient_id": "Pt-18829",
            "raw_waveform_samples": [0.12, 0.15, -0.04] * 1000,
            "filepath": "/home/user/medical_records/patient_001.dat",
            "secret_token": "gsk_super_secret_token",
            "api_key": "private_key",
        }

        sanitized = sanitize_payload(dirty_input)
        assert isinstance(sanitized, dict)

        # Disallowed keys must be completely absent
        assert "patient_id" not in sanitized
        assert "raw_waveform_samples" not in sanitized
        assert "filepath" not in sanitized
        assert "secret_token" not in sanitized
        assert "api_key" not in sanitized

        # Whitelisted keys must be preserved
        assert sanitized["workflow"] == "workflow_b_12lead"
        assert sanitized["source_category"] == "PTB-XL Record #00123"
        assert sanitized["is_simulation"] is False
        assert sanitized["summary_metrics"] == {"NORM": "92.0%"}

    def test_sanitize_payload_preserves_allowed_keys(self):
        sparse_input = {"workflow": "workflow_c_single_lead"}
        sanitized = sanitize_payload(sparse_input)
        assert sanitized["workflow"] == "workflow_c_single_lead"
        assert "patient_id" not in sanitized

    def test_cache_key_generation_is_deterministic_and_canonical(self):
        d1 = {
            "workflow": "workflow_b_12lead",
            "source_category": "Record A",
            "is_simulation": False,
            "units_and_calibration": "mV",
            "summary_metrics": {"NORM": "0.85"},
            "data_quality_flags": ["Clean"],
            "known_limitations": ["Limit A"],
        }
        d2 = copy.deepcopy(d1)
        d3 = copy.deepcopy(d1)
        d3["source_category"] = "Record B"

        k1 = compute_payload_cache_key(d1, model_id="qwen/qwen3.8-27b")
        k2 = compute_payload_cache_key(d2, model_id="qwen/qwen3.8-27b")
        k3 = compute_payload_cache_key(d3, model_id="qwen/qwen3.8-27b")
        k4 = compute_payload_cache_key(d1, model_id="other-model")

        assert len(k1) == 64  # SHA-256 hex string
        assert k1 == k2
        assert k1 != k3
        assert k1 != k4


# ==============================================================================
# 2. Prompts & Medical Safety Rules Tests
# ==============================================================================

class TestPromptsAndMedicalSafety:
    """Verify system safety instructions enforce strict non-diagnostic claims."""

    def test_system_prompt_contains_mandatory_safety_clauses(self):
        prompt = SYSTEM_PROMPT
        # Non-clinical notice
        assert "NON-CLINICAL PROTOTYPE" in prompt
        assert "NO DIAGNOSTIC CLAIMS" in prompt
        assert "Never state or imply that a user or patient \"has\" a heart attack" in prompt

        # 4-class taxonomy disclosure
        assert "project-defined 4-class beat grouping" in prompt
        assert "ventricular fusion beats F are merged into Q" in prompt
        assert "NOT an unqualified 5-class AAMI EC57 model" in prompt

        # Decoupling & Hardware simulation notice
        assert "HARDWARE PROTOTYPE SIMULATION" in prompt
        assert "NO physical sensors" in prompt
        assert "GROUND TRUTH DECOUPLING" in prompt

    def test_build_explanation_prompt_embeds_sanitized_json(self):
        payload = {
            "workflow": "workflow_b_12lead",
            "source_category": "PTB-XL Record #00042",
            "is_simulation": False,
            "units_and_calibration": "mV",
            "summary_metrics": {"NORM": "85.2%", "MI": "4.1%"},
            "data_quality_flags": ["Zero-phase Butterworth"],
            "known_limitations": ["Research prototype"],
        }

        user_msg = build_explanation_prompt(payload)
        assert "PTB-XL Record #00042" in user_msg
        assert "NORM" in user_msg
        assert "85.2%" in user_msg


# ==============================================================================
# 3. Groq Client & Environment Key Resolution Tests
# ==============================================================================

class TestGroqClientResolution:
    """Verify multi-source API key resolution and error sanitization."""

    def test_resolve_key_from_groq_api_key_env(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test_key_12345", "GROQ_API": ""}):
            assert resolve_groq_api_key() == "gsk_test_key_12345"

    def test_resolve_key_from_groq_api_env_fallback(self):
        with patch.dict(os.environ, {"GROQ_API": "gsk_test_key_alternate"}, clear=True):
            assert resolve_groq_api_key() == "gsk_test_key_alternate"

    def test_resolve_key_missing_returns_none(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("dotenv.dotenv_values", return_value={}):
                with patch("pathlib.Path.is_file", return_value=False):
                    assert resolve_groq_api_key() is None

    def test_model_selection_default_and_override(self):
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_groq_model() == DEFAULT_GROQ_MODEL
        with patch.dict(os.environ, {"GROQ_MODEL": "llama-3.3-70b-versatile"}):
            assert resolve_groq_model() == "llama-3.3-70b-versatile"

    def test_sanitize_error_message_scrubs_keys(self):
        raw_err = Exception("Auth failed for key gsk_abc1234567890xyz and Bearer tok_99887766")
        scrubbed = sanitize_error_message(raw_err)
        assert "gsk_abc1234567890xyz" not in scrubbed
        assert "gsk_***" in scrubbed
        assert "Bearer ***" in scrubbed

    def test_execute_chat_completion_mock_success(self):
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "simple_terms": "The model processed the signal and classified patterns.",
            "what_numbers_mean": "Probabilities indicate relative pattern resemblance.",
            "what_stands_out": "Lead V1 shows prominent amplitude.",
            "limitations": "Trained on retrospective data only.",
            "what_this_does_not_tell_you": "Does not provide a clinical diagnosis.",
            "suggested_next_step": "Review raw signals with an expert cardiologist.",
        })
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        success, content, err = execute_chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            client=mock_client,
        )
        assert success is True
        assert err is None
        parsed = json.loads(content)
        assert parsed["simple_terms"].startswith("The model processed")


# ==============================================================================
# 4. Explanation Service & Cache Tests
# ==============================================================================

class TestExplanationService:
    """Verify end-to-end service orchestration, caching, and fallback handling."""

    def setup_method(self):
        clear_explanation_cache()

    def test_explain_results_with_mock_client_and_caching(self):
        mock_json_str = json.dumps({
            "simple_terms": "Normal sinus rhythm pattern observed.",
            "what_numbers_mean": "High probability for NORM class.",
            "what_stands_out": "No ST-segment elevation detected.",
            "limitations": "Prototype algorithm.",
            "what_this_does_not_tell_you": "Not a substitute for 12-lead clinical ECG review.",
            "suggested_next_step": "Explore other patient records in the lab.",
        })

        with patch("ecg_lab.explanations.service.execute_chat_completion", return_value=(True, mock_json_str, None)) as mock_call:
            payload_data = {
                "workflow": "workflow_b_12lead",
                "source_category": "PTB-XL Record",
                "is_simulation": False,
                "summary_metrics": {"NORM": "95.0%"},
            }

            # 1. First invocation: calls LLM
            res1 = explain_results(payload_data)
            assert res1.success is True
            assert res1.cached is False
            assert res1.simple_terms == "Normal sinus rhythm pattern observed."
            assert mock_call.call_count == 1

            # 2. Second invocation: served from cache
            res2 = explain_results(payload_data)
            assert res2.success is True
            assert res2.cached is True
            assert res2.simple_terms == "Normal sinus rhythm pattern observed."
            assert mock_call.call_count == 1  # No additional network call

    def test_explain_results_graceful_fallback_on_failure(self):
        with patch("ecg_lab.explanations.service.execute_chat_completion", return_value=(False, "", "Rate limit reached (429)")):
            payload_data = {
                "workflow": "workflow_c_single_lead",
                "source_category": "MIT-BIH Record 100",
                "is_simulation": False,
            }
            res = explain_results(payload_data)
            assert res.success is False
            assert "error occurred" in res.simple_terms.lower()
            assert "429" in (res.error_message or "")
            assert "temporarily unavailable" in res.limitations.lower()
            assert res.what_this_does_not_tell_you != ""


# ==============================================================================
# 5. Multi-Workflow Adapters Tests
# ==============================================================================

class TestWorkflowAdapters:
    """Verify valid payload payloads for all 4 distinct project workflows."""

    def test_workflow_a_digitizer_payload(self):
        payload_data = {
            "workflow": WorkflowType.WORKFLOW_A_DIGITIZATION.value,
            "source_category": "Scanned ECG Paper Document",
            "is_simulation": False,
            "sampling_rate_hz": 100.0,
            "units_and_calibration": "Amplitude: relative_units (UNRECOVERED); 25 mm/s reference",
            "summary_metrics": {
                "extracted_leads_count": 12,
                "calibration_status": "UNRECOVERED",
                "runtime_ms": 142.5,
            },
            "class_predictions": [],
            "data_quality_flags": ["AIR-GAPPED WORKFLOW: Completely decoupled from ML models"],
            "known_limitations": [
                "Amplitudes are uncalibrated relative units",
                "Decoupled from diagnostic classifiers",
            ],
        }
        sanitized = sanitize_payload(payload_data)
        assert sanitized["workflow"] == "workflow_a_digitization"
        assert len(sanitized["class_predictions"]) == 0
        assert "UNRECOVERED" in sanitized["units_and_calibration"]

    def test_workflow_b_12lead_payload(self):
        payload_data = {
            "workflow": WorkflowType.WORKFLOW_B_12LEAD.value,
            "source_category": "PTB-XL Record (Fold 10)",
            "is_simulation": False,
            "sampling_rate_hz": 100.0,
            "units_and_calibration": "Calibrated 12-lead digital ECG (mV)",
            "summary_metrics": {"NORM": "89.4%", "MI": "5.2%", "STTC": "2.1%"},
            "class_predictions": [
                {"superclass": "NORM", "probability": 0.894, "cutoff": 0.448, "status": "POS"},
                {"superclass": "MI", "probability": 0.052, "cutoff": 0.334, "status": "NEG"},
            ],
            "data_quality_flags": ["Zero-phase forward-backward Butterworth filter applied"],
            "known_limitations": ["Predicted labels are statistical patterns, NOT clinical diagnoses"],
        }
        sanitized = sanitize_payload(payload_data)
        assert sanitized["workflow"] == "workflow_b_12lead"
        assert len(sanitized["class_predictions"]) == 2

    def test_workflow_c_single_lead_payload(self):
        payload_data = {
            "workflow": WorkflowType.WORKFLOW_C_SINGLE_LEAD.value,
            "source_category": "MIT-BIH Holter Record 100",
            "is_simulation": False,
            "sampling_rate_hz": 360.0,
            "units_and_calibration": "Calibrated single-lead Holter ECG (mV)",
            "summary_metrics": {
                "total_detected_beats": 14,
                "class_counts": {"N": 13, "V": 1},
            },
            "data_quality_flags": [
                "RECORDED DATA REPLAY — NOT LIVE HARDWARE",
                "Pan-Tompkins autonomous QRS peak detector cascade",
            ],
            "evaluation_context": {
                "taxonomy": "Project-defined 4-class beat grouping: N, S, V, Q (F merged into Q)",
            },
            "known_limitations": [
                "Ventricular fusion (F) merged into Q",
                "Scores are uncalibrated Random Forest voting fractions",
            ],
        }
        sanitized = sanitize_payload(payload_data)
        assert sanitized["workflow"] == "workflow_c_single_lead"
        assert "RECORDED DATA REPLAY" in sanitized["data_quality_flags"][0]

    def test_hardware_simulation_payload(self):
        payload_data = {
            "workflow": WorkflowType.HARDWARE_SIMULATION.value,
            "source_category": "Offline Simulation (MIT-BIH Record 100, Lead MLII)",
            "is_simulation": True,
            "sampling_rate_hz": 360.0,
            "units_and_calibration": "Simulated 10-bit integer ADC counts mapped to mV-equivalent; not hardware-calibrated",
            "summary_metrics": {
                "total_detected_peaks": 12,
                "clipping_samples": 0,
                "reconstruction_mae_mv": 0.0012,
            },
            "data_quality_flags": [
                "OFFLINE SIMULATION — NO PHYSICAL SENSOR CONNECTED",
                "Inference blocked on physical serial ADC streams lacking calibration certificates",
            ],
            "known_limitations": [
                "Offline software simulation only; no physical sensor connected",
            ],
        }
        sanitized = sanitize_payload(payload_data)
        assert sanitized["workflow"] == "hardware_simulation"
        assert sanitized["is_simulation"] is True


# ==============================================================================
# 6. Streamlit Integration Tests
# ==============================================================================

class TestStreamlitIntegration:
    """Verify Streamlit app renders Explain Results components cleanly."""

    def test_app_tabs_load_without_exceptions(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        assert len(at.exception) == 0

        # Switch to Live Experiment
        at.radio(key="top_nav_mode_radio").set_value("Live Experiment").run()
        assert len(at.exception) == 0

    def test_explain_results_button_present_in_single_lead_tab(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.radio(key="top_nav_mode_radio").set_value("Live Experiment").run()

        # In Tab 2, single-lead beats are rendered automatically for Record 100
        # Button should exist with key starting with btn_explain_workflow_c_single_lead
        matching_buttons = [b for b in at.button if "explain_workflow_c" in str(b.key)]
        assert len(matching_buttons) >= 1


# ==============================================================================
# 7. Frozen Model Weight Verification
# ==============================================================================

class TestFrozenModelIntegrity:
    """Verify that single-lead Random Forest model weights remain unchanged."""

    def test_single_lead_rf_model_sha256(self):
        model_path = DEFAULT_ARTIFACT_DIR / MODEL_NAME
        assert model_path.is_file(), f"Model file missing: {model_path}"

        hasher = hashlib.sha256()
        with open(model_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        actual_sha = hasher.hexdigest()

        assert actual_sha == FROZEN_MODEL_SHA256, (
            f"FROZEN MODEL INTEGRITY VIOLATION!\n"
            f"Expected: {FROZEN_MODEL_SHA256}\n"
            f"Actual:   {actual_sha}"
        )
