"""Tests for Offline AD8232 Simulation and Streamlit Single-Lead Inference."""

import copy
import hashlib
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from ecg_lab.single_lead_ml.config import (
    AAMI_TARGET_CLASSES,
    DEFAULT_ARTIFACT_DIR,
    MODEL_NAME,
    NATIVE_SAMPLING_RATE_HZ,
)
from ecg_lab.single_lead_ml.dataset import (
    BeatAnnotation,
    load_record_with_annotations,
)
from ecg_lab.single_lead_ml.inference import (
    BeatPrediction,
    SingleLeadInferenceResult,
    run_single_lead_inference,
)
from hardware.host.serial_protocol import (
    ADC_MAX_COUNT,
    ADC_MIN_COUNT,
    SerialSample,
    SerialStreamParser,
    format_serial_line,
)
from hardware.host.simulate_adc_stream import (
    DEFAULT_ADC_BASELINE,
    DEFAULT_ADC_GAIN_COUNTS_PER_MV,
    SIMULATION_PRESETS,
    generate_synthetic_ecg,
    simulate_adc_stream,
    simulated_adc_to_voltage,
    voltage_to_simulated_adc,
)
from hardware.signal_processing.calibration import (
    IncompatibleSignalError,
    UncalibratedHardwareError,
    validate_input_for_classifier,
)

FROZEN_MODEL_SHA256 = "3889a5f5ab97803da339710d4ae9fde8ea50f8e5a5ff76ed2015a0fb0b6ec8e9"


class TestFrozenModelIntegrity:
    """Verify that the model weights and SHA-256 digest are strictly frozen."""

    def test_model_sha256_unmodified(self):
        model_path = DEFAULT_ARTIFACT_DIR / MODEL_NAME
        assert model_path.is_file(), f"Model file not found: {model_path}"
        h = hashlib.sha256()
        with open(model_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        actual_sha = h.hexdigest()
        assert actual_sha == FROZEN_MODEL_SHA256, (
            f"FROZEN MODEL INTEGRITY VIOLATION: expected {FROZEN_MODEL_SHA256}, got {actual_sha}"
        )


class TestSimulationPipeline:
    """Test the full end-to-end forward/reverse simulation and inference flow."""

    def test_full_offline_simulation_mitbih_record_100(self):
        # 1. Forward simulation
        samples, prov = simulate_adc_stream(
            source_type="mitbih",
            record_id="100",
            duration_seconds=5.0,
            sampling_rate_hz=360,
            baseline_count=512.0,
            gain_counts_per_mv=150.0,
        )
        assert len(samples) == 1800  # 5.0s @ 360 Hz
        assert prov["source_type"] == "MITBIH"
        assert prov["mitbih_record_id"] == "100"

        # 2. Serialize and parse back
        lines = [format_serial_line(s) for s in samples]
        parser = SerialStreamParser()
        for line in lines:
            parser.feed_line(line)

        assert parser.sample_count == 1800
        assert parser.corrupted_lines_count == 0

        # 3. Inverse conversion
        parsed_adc = parser.to_dataframe()["raw_adc"].to_numpy(dtype=np.int32)
        recon = simulated_adc_to_voltage(parsed_adc, baseline_count=512.0, gain_counts_per_mv=150.0)
        reconstructed_mv = recon["reconstructed_mv"]
        assert len(reconstructed_mv) == 1800

        # 4. Single-lead inference
        result = run_single_lead_inference(
            signal=reconstructed_mv,
            sampling_rate_hz=360,
            units="mV",
            calibration_provenance=prov["calibration_provenance"],
            source_kind="offline AD8232-like simulation",
        )

        assert result.success is True
        assert result.detected_peaks_count > 0
        assert result.classifiable_windows_count > 0
        assert len(result.predictions) == result.classifiable_windows_count

        # Verify predictions structure
        for p in result.predictions:
            assert p.predicted_class in AAMI_TARGET_CLASSES
            assert 0.0 <= p.confidence_score <= 1.0
            assert set(p.class_probabilities.keys()) == set(AAMI_TARGET_CLASSES)

        # Verify DataFrame conversion
        df = result.to_dataframe()
        assert not df.empty
        assert "Predicted Class" in df.columns
        assert "Time (s)" in df.columns
        assert "Confidence" in df.columns

    def test_synthetic_ecg_fixture_simulation(self):
        synth_mv = generate_synthetic_ecg(duration_seconds=6.0, sampling_rate_hz=360, noise_amplitude_mv=0.0, seed=123)
        samples, prov = simulate_adc_stream(
            source_type="synthetic",
            duration_seconds=6.0,
            sampling_rate_hz=360,
            noise_amplitude_mv=0.0,
            seed=123,
        )
        parsed_adc = np.array([s.raw_adc for s in samples])
        recon = simulated_adc_to_voltage(parsed_adc, baseline_count=512.0, gain_counts_per_mv=200.0, reference_signal_mv=synth_mv)
        assert recon["mae_mv"] is not None
        assert recon["mae_mv"] < 0.005  # Numerical quantization error only

        result = run_single_lead_inference(
            signal=recon["reconstructed_mv"],
            sampling_rate_hz=360,
            units="mV",
            calibration_provenance="SIMULATION PRESET — NOT HARDWARE CALIBRATION",
            source_kind="offline AD8232-like simulation",
        )
        assert result.success is True
        assert result.detected_peaks_count >= 5


class TestAnnotationDecoupling:
    """Verify ground truth annotations do NOT influence detection or predictions."""

    def test_shuffling_annotations_leaves_inference_predictions_invariant(self):
        rec, anns = load_record_with_annotations("100")
        fs = rec.sampling_rate_hz
        sig_slice = rec.signal_mv[: int(10.0 * fs)]
        sub_anns = [a for a in anns if a.sample_index < len(sig_slice)]

        # Run 1: with true reference annotations
        res1 = run_single_lead_inference(
            signal=sig_slice,
            sampling_rate_hz=fs,
            units="mV",
            calibration_provenance="MIT-BIH calibrated mV",
            source_kind="MIT-BIH source waveform",
            reference_annotations=sub_anns,
        )

        # Run 2: with scrambled/permuted class labels in annotations
        scrambled_anns = []
        for i, a in enumerate(sub_anns):
            fake_class = "V" if a.aami_class == "N" else "N"
            scrambled_anns.append(
                BeatAnnotation(
                    sample_index=a.sample_index,
                    original_symbol="X",
                    aami_class=fake_class,
                    time_seconds=a.time_seconds,
                )
            )

        res2 = run_single_lead_inference(
            signal=sig_slice,
            sampling_rate_hz=fs,
            units="mV",
            calibration_provenance="MIT-BIH calibrated mV",
            source_kind="MIT-BIH source waveform",
            reference_annotations=scrambled_anns,
        )

        # Run 3: with zero annotations
        res3 = run_single_lead_inference(
            signal=sig_slice,
            sampling_rate_hz=fs,
            units="mV",
            calibration_provenance="MIT-BIH calibrated mV",
            source_kind="MIT-BIH source waveform",
            reference_annotations=None,
        )

        # Assert: Detected peaks are bit-for-bit identical across all 3 runs
        np.testing.assert_array_equal(res1.detected_peak_samples, res2.detected_peak_samples)
        np.testing.assert_array_equal(res1.detected_peak_samples, res3.detected_peak_samples)

        # Assert: Predictions (classes and probabilities) are 100% identical across all 3 runs
        assert len(res1.predictions) == len(res2.predictions) == len(res3.predictions)
        for p1, p2, p3 in zip(res1.predictions, res2.predictions, res3.predictions):
            assert p1.peak_sample == p2.peak_sample == p3.peak_sample
            assert p1.predicted_class == p2.predicted_class == p3.predicted_class
            assert np.isclose(p1.confidence_score, p2.confidence_score)
            assert np.isclose(p1.confidence_score, p3.confidence_score)

        # Post-hoc evaluation differs because annotations were scrambled, proving decouple
        assert res1.evaluation is not None
        assert res2.evaluation is not None
        assert res3.evaluation is None
        f1_1 = res1.evaluation["classifier_evaluation"]["macro_f1"]
        f1_2 = res2.evaluation["classifier_evaluation"]["macro_f1"]
        assert f1_1 != f1_2  # Scrambled annotations change evaluation metrics, but NOT predictions!


class TestCalibrationSafetyAndInputContracts:
    """Verify hardware calibration guards and signal prerequisites."""

    def test_uncalibrated_physical_hardware_stream_blocked(self):
        sig = np.random.normal(0, 1, 3600)
        res = run_single_lead_inference(
            signal=sig,
            sampling_rate_hz=360,
            units="mV",
            calibration_provenance="UNKNOWN_HARDWARE",
            source_kind="physical Arduino/AD8232 stream",
        )
        assert res.success is False
        assert "Inference blocked: hardware calibration unknown" in res.error_message

    def test_raw_adc_counts_rejected(self):
        raw_adc = np.full(3600, 512.0)
        res = run_single_lead_inference(
            signal=raw_adc,
            sampling_rate_hz=360,
            units="raw_adc_counts",
            calibration_provenance="SIMULATION PRESET",
            source_kind="offline AD8232-like simulation",
        )
        assert res.success is False
        assert "Classifier requires input in calibrated millivolts" in res.error_message

    def test_incompatible_sampling_rate_rejected(self):
        sig = np.zeros(1000)
        res = run_single_lead_inference(
            signal=sig,
            sampling_rate_hz=100,  # 100 Hz instead of 360 Hz
            units="mV",
            calibration_provenance="SIMULATION PRESET",
            source_kind="offline AD8232-like simulation",
        )
        assert res.success is False
        assert "Sampling rate 100 Hz is incompatible" in res.error_message

    def test_non_finite_signal_rejected(self):
        sig = np.zeros(3600)
        sig[100] = np.nan
        res = run_single_lead_inference(
            signal=sig,
            sampling_rate_hz=360,
            units="mV",
            calibration_provenance="SIMULATION PRESET",
            source_kind="offline AD8232-like simulation",
        )
        assert res.success is False
        assert "non-finite" in res.error_message

    def test_too_short_signal_rejected(self):
        sig = np.zeros(100)  # ~0.27s @ 360 Hz (< 0.5s)
        res = run_single_lead_inference(
            signal=sig,
            sampling_rate_hz=360,
            units="mV",
            calibration_provenance="SIMULATION PRESET",
            source_kind="offline AD8232-like simulation",
        )
        assert res.success is False
        assert "too short" in res.error_message


class TestSimulationPresetsAndInversion:
    """Verify numerical fidelity and error bounds of simulator inversion."""

    def test_near_lossless_roundtrip_error(self):
        ecg = generate_synthetic_ecg(duration_seconds=5.0, sampling_rate_hz=360, seed=42)
        adc, clipped = voltage_to_simulated_adc(ecg, baseline_count=512.0, gain_counts_per_mv=150.0)
        assert clipped == 0

        recon = simulated_adc_to_voltage(adc, baseline_count=512.0, gain_counts_per_mv=150.0, reference_signal_mv=ecg)
        assert recon["max_abs_error_mv"] < 0.005  # < 5 µV
        assert recon["mae_mv"] < 0.002
        assert recon["clipping_count"] == 0

    def test_clipping_detection_on_extreme_signal(self):
        ecg_extreme = np.array([-10.0, 0.0, 10.0])
        adc, clipped = voltage_to_simulated_adc(ecg_extreme, baseline_count=512.0, gain_counts_per_mv=150.0)
        assert clipped == 2
        assert adc[0] == ADC_MIN_COUNT
        assert adc[2] == ADC_MAX_COUNT


APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


class TestStreamlitAppTestIntegration:
    """Verify Streamlit UI components and visible prediction tables using AppTest."""

    def test_app_tabs_include_hardware_prototype(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.radio(key="top_nav_mode_radio").set_value("Live Experiment").run()

        tab_labels = [t.label for t in at.tabs]
        assert "Hardware Prototype AD8232" in tab_labels
        assert "Single-Lead Lab" in tab_labels

    def test_ad8232_simulation_button_runs_and_displays_predictions(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.radio(key="top_nav_mode_radio").set_value("Live Experiment").run()

        # Click the simulation execution button
        btn = at.button(key="btn_run_ad8232_sim")
        btn.click().run()

        assert len(at.exception) == 0
        assert "ad8232_sim_result" in at.session_state
        sim_res = at.session_state["ad8232_sim_result"]
        assert sim_res.success is True
        assert len(sim_res.predictions) > 0

        # Verify DataFrame is rendered
        assert len(at.dataframe) >= 1
