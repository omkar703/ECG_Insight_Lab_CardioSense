"""Integration tests bridging root quality gates with the hardware prototype module."""

import hashlib
from pathlib import Path
import pytest
import numpy as np

from hardware.host.serial_protocol import (
    PROTOCOL_VERSION,
    NOMINAL_SAMPLING_RATE_HZ,
    parse_serial_line,
    SerialStreamParser,
)
from hardware.host.simulate_adc_stream import simulate_adc_stream
from hardware.signal_processing.calibration import (
    UncalibratedHardwareError,
    IncompatibleSignalError,
    SignalUnits,
    convert_hardware_to_ecg_mv,
    validate_input_for_classifier,
)
from hardware.signal_processing.ecg_filters import preprocess_offline_ecg


class TestHardwareIntegrationGates:
    """Validate root test suite integration with Phase 7 hardware prototype."""

    SINGLE_LEAD_MODEL_SHA256 = "3889a5f5ab97803da339710d4ae9fde8ea50f8e5a5ff76ed2015a0fb0b6ec8e9"
    PTBXL_MODEL_SHA256 = "55d364ed57cc971167f7ad813443c249f792567e406cce4c3dd84188829328d7"

    def test_model_artifact_hashes_remain_frozen(self):
        """Zero mutation gate: Model weights must never be altered during hardware development."""
        sl_path = Path("artifacts/single_lead_ml/single_lead_rf_model.joblib")
        assert sl_path.is_file()
        sl_hash = hashlib.sha256(sl_path.read_bytes()).hexdigest()
        assert sl_hash == self.SINGLE_LEAD_MODEL_SHA256

        ptb_path = Path("artifacts/waveform_ml/waveform_lr_model.joblib")
        assert ptb_path.is_file()
        ptb_hash = hashlib.sha256(ptb_path.read_bytes()).hexdigest()
        assert ptb_hash == self.PTBXL_MODEL_SHA256

    def test_hardware_simulation_and_filter_pipeline(self):
        """End-to-end simulation, protocol parse, and digital filtering."""
        samples, prov = simulate_adc_stream(
            source_type="synthetic",
            duration_seconds=2.0,
            sampling_rate_hz=NOMINAL_SAMPLING_RATE_HZ,
            seed=42,
        )
        assert len(samples) == 720
        parser = SerialStreamParser()
        for s in samples:
            parser.feed_line(f"{s.seq},{s.timestamp_micros},{s.raw_adc},{s.lo_plus},{s.lo_minus}")

        df = parser.to_dataframe()
        assert len(df) == 720
        assert "raw_adc" in df.columns
        assert "mV" not in df.columns

        # Digital filter execution on simulated raw ADC
        filtered_adc = preprocess_offline_ecg(df["raw_adc"].values, fs=360.0)
        assert len(filtered_adc) == 720
        assert np.all(np.isfinite(filtered_adc))

    def test_model_inference_strictly_blocked_on_uncalibrated_hardware(self):
        """Safety gate: Raw hardware ADC counts cannot enter existing beat classifiers."""
        raw_adc = np.array([512.0, 520.0, 510.0])
        # Direct conversion blocked without calibration params
        with pytest.raises(UncalibratedHardwareError):
            convert_hardware_to_ecg_mv(raw_adc)

        # Ingestion blocked by classifier input validator
        with pytest.raises(IncompatibleSignalError, match="calibrated millivolts"):
            validate_input_for_classifier(raw_adc, sampling_rate_hz=360, unit_name=SignalUnits.RAW_ADC_COUNTS)

    def test_wiring_diagram_and_safety_notices_present(self):
        """Documentation gate: Diagram, pinout, and safety manuals must exist."""
        svg_file = Path("hardware/diagrams/ad8232_arduino_uno_wiring.svg")
        assert svg_file.is_file()
        svg_text = svg_file.read_text(encoding="utf-8")
        assert "CAUTION: PROTOTYPE REFERENCE ONLY" in svg_text
        assert "NO HUMAN ELECTRODE TEST" in svg_text

        pin_file = Path("hardware/diagrams/pin_map.md")
        assert pin_file.is_file()

        safety_file = Path("hardware/PINOUT_AND_SAFETY.md")
        assert safety_file.is_file()
        assert "IEC 60601-1" in safety_file.read_text(encoding="utf-8")
