"""Signal-based R-peak detector and benchmark evaluator for single-lead ECG.

Crucial integrity rule: This detector operates strictly on raw ECG signals and does NOT
consume ground-truth annotation indices at inference.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.signal

from ecg_lab.single_lead_ml.config import NATIVE_SAMPLING_RATE_HZ


def pan_tompkins_qrs_detector(
    signal_mv: np.ndarray,
    sampling_rate_hz: int = NATIVE_SAMPLING_RATE_HZ,
    lowcut_hz: float = 5.0,
    highcut_hz: float = 15.0,
    integration_window_sec: float = 0.150,
    refractory_sec: float = 0.200,
) -> np.ndarray:
    """Detect R-peak sample indices from single-lead ECG signal using Pan-Tompkins algorithm.

    Steps:
        1. 2nd-order Butterworth bandpass filter (5-15 Hz).
        2. Discrete derivative to isolate steep QRS slopes.
        3. Squaring to enhance large amplitudes and eliminate polarity.
        4. Moving-window integration (~150 ms).
        5. Adaptive peak search with refractory lockout (>= 200 ms).
        6. Precise local alignment to raw signal maximum in search window.

    Returns:
        1D numpy array of detected R-peak sample indices in the original signal.
    """
    n_samples = len(signal_mv)
    if n_samples < int(sampling_rate_hz * 0.5):
        return np.zeros(0, dtype=int)

    # 1. Bandpass filter
    sos = scipy.signal.butter(
        2,
        [lowcut_hz, highcut_hz],
        btype="bandpass",
        fs=sampling_rate_hz,
        output="sos",
    )
    filtered = scipy.signal.sosfiltfilt(sos, signal_mv)

    # 2. Derivative filter: y[n] = (2*x[n] + x[n-1] - x[n-3] - 2*x[n-4]) / 8
    # Using np.gradient as robust discrete derivative
    differentiated = np.gradient(filtered)

    # 3. Squaring
    squared = differentiated ** 2

    # 4. Moving window integration
    window_samples = max(1, int(integration_window_sec * sampling_rate_hz))
    kernel = np.ones(window_samples) / float(window_samples)
    integrated = np.convolve(squared, kernel, mode="same")

    # 5. Peak finding on integrated signal
    refractory_samples = int(refractory_sec * sampling_rate_hz)
    
    # Adaptive threshold estimation
    # Start with initial percentile threshold
    sample_peak_threshold = float(np.percentile(integrated, 85)) * 0.5
    noise_threshold = float(np.percentile(integrated, 50)) * 0.5

    # Find local maxima in integrated signal
    raw_peak_locs, _ = scipy.signal.find_peaks(
        integrated,
        distance=refractory_samples,
        height=sample_peak_threshold,
    )

    # If too few peaks found (e.g. very low signal amplitude or high noise), lower threshold
    if len(raw_peak_locs) < max(1, int(n_samples / (sampling_rate_hz * 2.0))):
        fallback_threshold = float(np.mean(integrated) + 0.5 * np.std(integrated))
        raw_peak_locs, _ = scipy.signal.find_peaks(
            integrated,
            distance=refractory_samples,
            height=fallback_threshold,
        )

    # 6. Refine peak locations: lock onto highest amplitude in raw/filtered signal around integrated peak
    refined_peaks: List[int] = []
    search_radius = int(0.08 * sampling_rate_hz)  # +/- 80 ms (~29 samples)

    for p in raw_peak_locs:
        search_start = max(0, p - search_radius)
        search_end = min(n_samples, p + search_radius)
        if search_start >= search_end:
            continue
        # Find absolute max in filtered signal (accounts for both positive and negative R-peaks)
        local_segment = np.abs(filtered[search_start:search_end])
        local_offset = int(np.argmax(local_segment))
        exact_peak = search_start + local_offset

        # Check refractory period with previous refined peak
        if refined_peaks and (exact_peak - refined_peaks[-1]) < refractory_samples:
            # Keep the one with larger amplitude
            if abs(filtered[exact_peak]) > abs(filtered[refined_peaks[-1]]):
                refined_peaks[-1] = exact_peak
        else:
            refined_peaks.append(exact_peak)

    return np.array(refined_peaks, dtype=int)


def evaluate_r_peak_detector(
    detected_peaks: np.ndarray,
    reference_peaks: np.ndarray,
    sampling_rate_hz: int = NATIVE_SAMPLING_RATE_HZ,
    tolerance_ms: float = 50.0,
) -> Dict[str, Union[int, float]]:
    """Evaluate detected R-peaks against reference expert annotations per AAMI EC57.

    Standard matching tolerance is +/- 50 ms (+/- 18 samples at 360 Hz).

    Returns:
        Dictionary containing TP, FP, FN, Sensitivity, PPV, and F1.
    """
    tolerance_samples = int(np.round((tolerance_ms / 1000.0) * sampling_rate_hz))

    n_ref = len(reference_peaks)
    n_det = len(detected_peaks)

    if n_ref == 0:
        return {
            "n_reference": 0,
            "n_detected": n_det,
            "true_positives": 0,
            "false_positives": n_det,
            "false_negatives": 0,
            "sensitivity": 0.0,
            "positive_predictive_value": 0.0,
            "f1_score": 0.0,
            "tolerance_ms": tolerance_ms,
            "tolerance_samples": tolerance_samples,
        }

    if n_det == 0:
        return {
            "n_reference": n_ref,
            "n_detected": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": n_ref,
            "sensitivity": 0.0,
            "positive_predictive_value": 0.0,
            "f1_score": 0.0,
            "tolerance_ms": tolerance_ms,
            "tolerance_samples": tolerance_samples,
        }

    # Match each reference peak to nearest detected peak
    matched_det_indices = set()
    tp = 0

    for ref in reference_peaks:
        # Distance to all detected peaks
        dists = np.abs(detected_peaks - ref)
        min_idx = int(np.argmin(dists))
        min_dist = dists[min_idx]

        if min_dist <= tolerance_samples and min_idx not in matched_det_indices:
            tp += 1
            matched_det_indices.add(min_idx)

    fn = n_ref - tp
    fp = n_det - len(matched_det_indices)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2.0 * sensitivity * ppv) / (sensitivity + ppv) if (sensitivity + ppv) > 0 else 0.0

    return {
        "n_reference": int(n_ref),
        "n_detected": int(n_det),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "sensitivity": float(sensitivity),
        "positive_predictive_value": float(ppv),
        "f1_score": float(f1),
        "tolerance_ms": float(tolerance_ms),
        "tolerance_samples": int(tolerance_samples),
    }
