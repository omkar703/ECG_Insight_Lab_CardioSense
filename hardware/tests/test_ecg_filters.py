"""Tests for ECG digital filtering utilities and streaming filter."""

import pytest
import numpy as np

from hardware.signal_processing.ecg_filters import (
    butter_bandpass_filter,
    notch_filter,
    preprocess_offline_ecg,
    StreamingECGFilter,
)


class TestButterBandpassFilter:
    """Test offline zero-phase Butterworth bandpass filter."""

    def test_filter_output_shape_and_finite(self):
        fs = 360.0
        t = np.arange(fs * 2) / fs
        raw = np.sin(2 * np.pi * 5.0 * t) + 0.5 * np.sin(2 * np.pi * 0.1 * t)

        filtered = butter_bandpass_filter(raw, lowcut=0.5, highcut=40.0, fs=fs)
        assert filtered.shape == raw.shape
        assert np.all(np.isfinite(filtered))

    def test_raw_data_immutability(self):
        fs = 360.0
        raw = np.ones(500, dtype=np.float64) * 512.0
        raw_copy = raw.copy()

        _ = butter_bandpass_filter(raw, lowcut=0.5, highcut=40.0, fs=fs)
        np.testing.assert_array_equal(raw, raw_copy)

    def test_attenuates_low_frequency_baseline_wander(self):
        fs = 360.0
        t = np.arange(fs * 5) / fs
        # 0.1 Hz baseline drift (below 0.5 Hz lowcut)
        low_freq = 2.0 * np.sin(2 * np.pi * 0.1 * t)
        # 10 Hz ECG band signal
        mid_freq = 1.0 * np.sin(2 * np.pi * 10.0 * t)
        raw = low_freq + mid_freq

        filtered = butter_bandpass_filter(raw, lowcut=0.5, highcut=40.0, fs=fs)

        # Baseline amplitude should be reduced by at least 80%
        # Check standard deviation ratio of filtered vs original low_freq
        assert np.std(filtered - mid_freq) < 0.3 * np.std(low_freq)

    def test_attenuates_high_frequency_noise(self):
        fs = 360.0
        t = np.arange(fs * 3) / fs
        # 120 Hz high frequency noise (well above 40 Hz highcut)
        high_noise = 1.5 * np.sin(2 * np.pi * 120.0 * t)
        mid_ecg = 1.0 * np.sin(2 * np.pi * 10.0 * t)
        raw = mid_ecg + high_noise

        filtered = butter_bandpass_filter(raw, lowcut=0.5, highcut=40.0, fs=fs)
        # High noise amplitude should be heavily attenuated
        assert np.std(filtered - mid_ecg) < 0.5 * np.std(high_noise)

    def test_invalid_parameters_raise_error(self):
        fs = 360.0
        data = np.ones(200)

        # Highcut >= Nyquist (180 Hz)
        with pytest.raises(ValueError, match="Nyquist"):
            butter_bandpass_filter(data, lowcut=0.5, highcut=180.0, fs=fs)

        # Lowcut >= Highcut
        with pytest.raises(ValueError, match="strictly less than highcut"):
            butter_bandpass_filter(data, lowcut=40.0, highcut=40.0, fs=fs)

        # Empty data
        with pytest.raises(ValueError, match="empty or None"):
            butter_bandpass_filter(np.array([]), lowcut=0.5, highcut=40.0, fs=fs)

        # NaN data
        with pytest.raises(ValueError, match="NaN or infinite"):
            nan_arr = np.ones(100)
            nan_arr[10] = np.nan
            butter_bandpass_filter(nan_arr, lowcut=0.5, highcut=40.0, fs=fs)


class TestNotchFilter:
    """Test 50 Hz powerline notch filter."""

    def test_notch_attenuates_50hz(self):
        fs = 360.0
        t = np.arange(fs * 4) / fs
        # Pure 50 Hz sinusoid
        mains_hum = 2.0 * np.sin(2 * np.pi * 50.0 * t)
        filtered = notch_filter(mains_hum, notch_freq=50.0, quality_factor=30.0, fs=fs)

        # The 50 Hz energy should be reduced by > 90% (excluding filter edge transients)
        center_samples = filtered[int(fs) : -int(fs)]
        assert np.std(center_samples) < 0.1 * np.std(mains_hum)

    def test_notch_nyquist_violation(self):
        with pytest.raises(ValueError, match="Nyquist"):
            notch_filter(np.ones(100), notch_freq=180.0, fs=360.0)


class TestStreamingECGFilter:
    """Test causal stateful streaming filter."""

    def test_streaming_consistency_chunk_vs_sample(self):
        fs = 360.0
        t = np.arange(360) / fs
        signal = np.sin(2 * np.pi * 5.0 * t)

        filter1 = StreamingECGFilter(lowcut=0.5, highcut=40.0, fs=fs)
        out_chunk = filter1.process_chunk(signal)

        filter2 = StreamingECGFilter(lowcut=0.5, highcut=40.0, fs=fs)
        out_sample_by_sample = np.array([filter2.process_sample(s) for s in signal])

        np.testing.assert_allclose(out_chunk, out_sample_by_sample, atol=1e-10)

    def test_streaming_is_causal(self):
        # A causal filter has delay; its output is not identical to zero-phase filtfilt
        fs = 360.0
        t = np.arange(720) / fs
        signal = np.sin(2 * np.pi * 5.0 * t)

        causal_filter = StreamingECGFilter(lowcut=0.5, highcut=40.0, fs=fs)
        out_causal = causal_filter.process_chunk(signal)
        out_zero_phase = butter_bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=fs)

        # Because causal filter introduces phase shift, outputs must not be identical
        assert not np.allclose(out_causal, out_zero_phase, atol=1e-2)
