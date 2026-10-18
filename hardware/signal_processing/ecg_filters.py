"""ECG digital filtering utilities for offline analysis and real-time streaming.

Provides:
1. Zero-phase bidirectional offline filtering (filtfilt) for research and batch analysis.
2. Causal stateful IIR filtering (StreamingECGFilter) for real-time sample/block processing.
3. Mains frequency notch filter (50 Hz / 60 Hz).

Notice:
- Preserves raw immutable input data without in-place modification.
- Explicitly documents filter delay / phase shift characteristics.
- Filtered hardware samples do not enter the classifier model pipeline by default.
"""

from typing import Optional, Tuple, Union
import numpy as np

try:
    from scipy.signal import butter, filtfilt, iirnotch, lfilter, lfilter_zi
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _check_scipy():
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "SciPy is required for ECG filtering functions. "
            "Please ensure scipy is installed in your Python environment."
        )


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    fs: float = 360.0,
    order: int = 2,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter to ECG data.

    Preserves raw data array immutably and returns a new filtered copy.
    Uses forward-backward filtering (filtfilt) for zero phase distortion.

    Args:
        data: 1D array of ECG voltage or ADC counts.
        lowcut: High-pass cutoff frequency in Hz (default: 0.5 Hz for baseline wander).
        highcut: Low-pass cutoff frequency in Hz (default: 40.0 Hz for high-frequency noise).
        fs: Sampling frequency in Hz (default: 360.0 Hz).
        order: Filter order (default: 2, yielding 4th order roll-off via filtfilt).

    Returns:
        Filtered 1D numpy array of float64 values with identical shape.

    Raises:
        ValueError: If cutoffs violate Nyquist, parameters are invalid, or data is too short.
    """
    _check_scipy()

    if data is None or len(data) == 0:
        raise ValueError("Cannot filter empty or None data array.")

    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Filter requires 1D array, got shape {arr.shape}")

    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError("Data contains NaN or infinite values, cannot compute filter.")

    nyquist = 0.5 * fs
    if lowcut <= 0:
        raise ValueError(f"Lowcut frequency must be positive, got {lowcut}")
    if highcut >= nyquist:
        raise ValueError(
            f"Highcut frequency ({highcut} Hz) must be strictly less than Nyquist frequency ({nyquist} Hz)."
        )
    if lowcut >= highcut:
        raise ValueError(f"Lowcut ({lowcut} Hz) must be strictly less than highcut ({highcut} Hz).")

    min_length = 3 * (order + 1)
    if len(arr) <= min_length:
        raise ValueError(
            f"Data length ({len(arr)}) is too short for order {order} filtfilt (requires > {min_length} samples)."
        )

    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="bandpass")
    filtered = filtfilt(b, a, arr)
    return filtered


def notch_filter(
    data: np.ndarray,
    notch_freq: float = 50.0,
    quality_factor: float = 30.0,
    fs: float = 360.0,
) -> np.ndarray:
    r"""Apply a zero-phase IIR notch filter to suppress powerline mains hum (50 Hz or 60 Hz).

    Args:
        data: 1D array of ECG voltage or ADC counts.
        notch_freq: Powerline frequency to reject in Hz (default: 50.0 Hz).
        quality_factor: Filter Q-factor ($f_0 / \Delta f$), default: 30.0.
        fs: Sampling frequency in Hz (default: 360.0 Hz).

    Returns:
        Filtered 1D numpy array.
    """
    _check_scipy()

    if data is None or len(data) == 0:
        raise ValueError("Cannot filter empty or None data array.")

    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Filter requires 1D array, got shape {arr.shape}")

    nyquist = 0.5 * fs
    if notch_freq >= nyquist:
        raise ValueError(
            f"Notch frequency ({notch_freq} Hz) must be strictly less than Nyquist frequency ({nyquist} Hz)."
        )

    b, a = iirnotch(w0=notch_freq, Q=quality_factor, fs=fs)
    min_length = 3 * max(len(a), len(b))
    if len(arr) <= min_length:
        raise ValueError(f"Data length ({len(arr)}) is too short for notch filtfilt.")

    filtered = filtfilt(b, a, arr)
    return filtered


def preprocess_offline_ecg(
    data: np.ndarray,
    fs: float = 360.0,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    mains_notch_hz: Optional[float] = 50.0,
) -> np.ndarray:
    """Convenience pipeline applying bandpass followed by optional mains notch filter.

    Leaves input array unchanged and returns new filtered array.
    """
    filtered = butter_bandpass_filter(data, lowcut=lowcut, highcut=highcut, fs=fs)
    if mains_notch_hz and mains_notch_hz < (0.5 * fs):
        filtered = notch_filter(filtered, notch_freq=mains_notch_hz, fs=fs)
    return filtered


class StreamingECGFilter:
    """Stateful, causal IIR filter for real-time sample-by-sample or chunk-by-chunk processing.

    Unlike zero-phase offline filtfilt, causal filtering introduces a realistic phase shift
    and time delay governed by the filter group delay.
    """

    def __init__(
        self,
        lowcut: float = 0.5,
        highcut: float = 40.0,
        fs: float = 360.0,
        order: int = 2,
        notch_freq: Optional[float] = 50.0,
    ):
        _check_scipy()
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.notch_freq = notch_freq

        nyquist = 0.5 * fs
        if highcut >= nyquist:
            raise ValueError(f"Highcut ({highcut} Hz) exceeds Nyquist ({nyquist} Hz).")

        # Design bandpass coefficients
        low = lowcut / nyquist
        high = highcut / nyquist
        self.b_band, self.a_band = butter(order, [low, high], btype="bandpass")
        self.zi_band = lfilter_zi(self.b_band, self.a_band)

        # Design notch coefficients if requested
        self.has_notch = notch_freq is not None and notch_freq < nyquist
        if self.has_notch:
            self.b_notch, self.a_notch = iirnotch(w0=notch_freq, Q=30.0, fs=fs)
            self.zi_notch = lfilter_zi(self.b_notch, self.a_notch)

        self.reset()

    def reset(self):
        """Reset internal filter delay state variables."""
        self._zi_band_state = self.zi_band.copy() * 0.0
        if self.has_notch:
            self._zi_notch_state = self.zi_notch.copy() * 0.0

    def process_sample(self, sample_value: float) -> float:
        """Process a single incoming scalar sample point causally."""
        chunk = np.array([sample_value], dtype=np.float64)
        out_chunk = self.process_chunk(chunk)
        return float(out_chunk[0])

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Process a 1D block of incoming samples causally, updating internal states."""
        arr = np.asarray(chunk, dtype=np.float64)
        if arr.ndim != 1 or len(arr) == 0:
            return np.array([], dtype=np.float64)

        # Apply causal bandpass
        bandpassed, self._zi_band_state = lfilter(
            self.b_band, self.a_band, arr, zi=self._zi_band_state
        )

        # Apply causal notch
        if self.has_notch:
            out, self._zi_notch_state = lfilter(
                self.b_notch, self.a_notch, bandpassed, zi=self._zi_notch_state
            )
            return out

        return bandpassed
