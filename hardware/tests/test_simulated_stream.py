"""Tests for offline ADC stream simulator and roundtrip parser verification."""

import json
from pathlib import Path
import pytest
import numpy as np

from hardware.host.serial_protocol import SerialStreamParser
from hardware.host.simulate_adc_stream import (
    DEFAULT_ADC_BASELINE,
    DEFAULT_ADC_GAIN_COUNTS_PER_MV,
    generate_synthetic_ecg,
    simulate_adc_stream,
    voltage_to_simulated_adc,
    write_simulated_stream_files,
)


class TestSyntheticECGGenerator:
    """Test mathematical synthetic waveform generation fixture."""

    def test_synthetic_ecg_determinism_with_seed(self):
        ecg1 = generate_synthetic_ecg(duration_seconds=2.0, sampling_rate_hz=360, seed=123)
        ecg2 = generate_synthetic_ecg(duration_seconds=2.0, sampling_rate_hz=360, seed=123)
        np.testing.assert_array_equal(ecg1, ecg2)

        # Different seed produces different noise
        ecg3 = generate_synthetic_ecg(duration_seconds=2.0, sampling_rate_hz=360, seed=999)
        assert not np.array_equal(ecg1, ecg3)

    def test_synthetic_ecg_sample_count(self):
        ecg = generate_synthetic_ecg(duration_seconds=3.5, sampling_rate_hz=360)
        assert len(ecg) == int(3.5 * 360)


class TestVoltageToADCQuantization:
    """Test mapping from millivolts to 10-bit integer ADC counts."""

    def test_baseline_mapping(self):
        # 0.0 mV maps to baseline (512)
        zero_mv = np.array([0.0])
        adc, clipped = voltage_to_simulated_adc(zero_mv, baseline_count=512.0, gain_counts_per_mv=200.0)
        assert adc[0] == 512
        assert clipped == 0

    def test_positive_and_negative_voltages(self):
        # +1.0 mV -> 512 + 200 = 712 counts
        # -1.0 mV -> 512 - 200 = 312 counts
        voltages = np.array([1.0, -1.0])
        adc, clipped = voltage_to_simulated_adc(voltages, baseline_count=512.0, gain_counts_per_mv=200.0)
        assert adc[0] == 712
        assert adc[1] == 312
        assert clipped == 0

    def test_clipping_detection(self):
        # +10.0 mV would be 512 + 2000 = 2512 -> clips to 1023
        # -10.0 mV would be 512 - 2000 = -1488 -> clips to 0
        extreme_voltages = np.array([10.0, -10.0])
        adc, clipped = voltage_to_simulated_adc(extreme_voltages, baseline_count=512.0, gain_counts_per_mv=200.0)
        assert adc[0] == 1023
        assert adc[1] == 0
        assert clipped == 2


class TestSimulationStreamPipeline:
    """Test full simulation generation and file output."""

    def test_simulate_from_mitbih_record_100(self):
        samples, prov = simulate_adc_stream(
            source_type="mitbih",
            record_id="100",
            duration_seconds=2.0,
            sampling_rate_hz=360,
        )
        assert len(samples) == 720  # 2.0 s * 360 Hz
        assert prov["source_type"] == "MITBIH"
        assert prov["mitbih_record_id"] == "100"
        assert "disclaimer" in prov

        # Validate sequence numbering
        seqs = [s.seq for s in samples]
        assert seqs == list(range(720))

    def test_simulate_lead_off_dropout(self):
        samples, prov = simulate_adc_stream(
            source_type="synthetic",
            duration_seconds=4.0,
            sampling_rate_hz=360,
            lead_off_start_s=1.0,
            lead_off_duration_s=1.0,
        )
        # Samples between 1.0s and 2.0s should have lead_off == True
        lead_off_samples = [s for s in samples if s.lead_off]
        expected_lo_count = int(1.0 * 360)  # 1.0s duration @ 360 Hz = 360 samples
        assert len(lead_off_samples) == expected_lo_count

    def test_roundtrip_simulated_file_to_parser(self, tmp_path):
        samples, prov = simulate_adc_stream(
            source_type="synthetic",
            duration_seconds=1.5,
            sampling_rate_hz=360,
        )
        out_csv, out_meta = write_simulated_stream_files(
            samples=samples,
            provenance=prov,
            output_base_path=tmp_path / "sim_test",
        )
        assert out_csv.is_file()
        assert out_meta.is_file()

        # Check metadata JSON content
        with open(out_meta, "r", encoding="utf-8") as f:
            meta_json = json.load(f)
        assert meta_json["sampling_rate_hz"] == 360
        assert "RAW SIMULATED ADC COUNTS" in meta_json["disclaimer"]

        # Parse CSV file back into SerialStreamParser
        parser = SerialStreamParser()
        with open(out_csv, "r", encoding="utf-8") as f:
            for line in f:
                parser.feed_line(line)

        assert parser.sample_count == len(samples)
        assert parser.corrupted_lines_count == 0
        df = parser.to_dataframe()
        assert len(df) == len(samples)
        assert "raw_adc" in df.columns
