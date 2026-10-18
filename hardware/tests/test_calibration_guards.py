"""Tests for calibration safeguards and physical unit isolation."""

import pytest
import numpy as np

from hardware.signal_processing.calibration import (
    IncompatibleSignalError,
    SignalUnits,
    UncalibratedHardwareError,
    adc_counts_to_input_volts,
    convert_hardware_to_ecg_mv,
    validate_input_for_classifier,
)


class TestVoltageConversions:
    """Test ADC counts to pin voltage conversion."""

    def test_adc_counts_to_input_volts_scalar(self):
        # At 3.3V Vref and 10-bit ADC (1023 counts), 0 count = 0.0V, 1023 = 3.3V
        assert adc_counts_to_input_volts(0, vref_volts=3.3, resolution_bits=10) == 0.0
        assert adc_counts_to_input_volts(1023, vref_volts=3.3, resolution_bits=10) == 3.3
        # Midpoint ~ 1.65V
        v_mid = adc_counts_to_input_volts(511.5, vref_volts=3.3, resolution_bits=10)
        assert pytest.approx(v_mid, abs=1e-3) == 1.65

    def test_adc_counts_to_input_volts_array(self):
        counts = np.array([0, 512, 1023])
        volts = adc_counts_to_input_volts(counts, vref_volts=3.3, resolution_bits=10)
        assert len(volts) == 3
        assert volts[0] == 0.0
        assert volts[2] == 3.3


class TestCalibrationGuardrails:
    """Test safety blocking when uncalibrated hardware streams are processed."""

    def test_uncalibrated_hardware_raises_error_without_params(self):
        counts = np.array([512, 513, 510, 508])
        with pytest.raises(UncalibratedHardwareError, match="Uncalibrated hardware stream"):
            convert_hardware_to_ecg_mv(counts, calibration_params=None)

    def test_uncalibrated_hardware_raises_error_with_unverified_params(self):
        counts = np.array([512, 513])
        unverified = {"verified": False, "analog_gain_vv": 100.0}
        with pytest.raises(UncalibratedHardwareError, match="Uncalibrated hardware stream"):
            convert_hardware_to_ecg_mv(counts, calibration_params=unverified)

    def test_verified_params_convert_correctly(self):
        counts = np.array([512])  # Exactly midscale ~ 1.65V
        verified_params = {
            "verified": True,
            "vref_volts": 3.3,
            "virtual_ground_volts": 1.65,
            "analog_gain_vv": 1000.0,
            "calibration_source": "Authorized Benchtop Generator Test",
        }
        # 1.65V - 1.65V = 0.0V -> 0.0 mV
        mv = convert_hardware_to_ecg_mv(counts, calibration_params=verified_params)
        assert pytest.approx(mv[0], abs=1e-2) == 0.0


class TestClassifierInputValidation:
    """Test strict input validation protecting machine learning classifiers."""

    def test_raw_adc_counts_rejected(self):
        signal = np.array([512.0, 513.0, 514.0])
        with pytest.raises(IncompatibleSignalError, match="calibrated millivolts"):
            validate_input_for_classifier(signal, sampling_rate_hz=360, unit_name=SignalUnits.RAW_ADC_COUNTS)

    def test_volts_rejected(self):
        signal = np.array([1.65, 1.66, 1.64])
        with pytest.raises(IncompatibleSignalError, match="calibrated millivolts"):
            validate_input_for_classifier(signal, sampling_rate_hz=360, unit_name=SignalUnits.INPUT_VOLTS)

    def test_calibrated_millivolts_accepted(self):
        signal = np.array([0.05, 0.12, 1.25, -0.20])
        assert validate_input_for_classifier(signal, sampling_rate_hz=360, unit_name="mV") is True
        assert validate_input_for_classifier(signal, sampling_rate_hz=360, unit_name=SignalUnits.ECG_MILLIVOLTS) is True

    def test_sampling_rate_mismatch_rejected(self):
        signal = np.array([0.05, 0.12, 1.25, -0.20])
        with pytest.raises(IncompatibleSignalError, match="Sampling rate 100 Hz is incompatible"):
            validate_input_for_classifier(signal, sampling_rate_hz=100, unit_name="mV", expected_sampling_rate_hz=360)

    def test_empty_or_non_finite_rejected(self):
        with pytest.raises(IncompatibleSignalError, match="empty or None"):
            validate_input_for_classifier(np.array([]), sampling_rate_hz=360, unit_name="mV")

        with pytest.raises(IncompatibleSignalError, match="non-finite"):
            validate_input_for_classifier(np.array([1.0, np.nan]), sampling_rate_hz=360, unit_name="mV")
