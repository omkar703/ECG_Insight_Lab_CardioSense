"""Deterministic feature extraction for single-lead ECG beat classification."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.stats

from ecg_lab.single_lead_ml.config import (
    BEAT_WINDOW_TOTAL_SAMPLES,
    NATIVE_SAMPLING_RATE_HZ,
    POST_R_SAMPLES,
    PRE_R_SAMPLES,
)
from ecg_lab.single_lead_ml.dataset import BeatAnnotation, SingleLeadRecord

FEATURE_NAMES: List[str] = [
    # Timing & RR interval features
    "pre_rr_sec",
    "post_rr_sec",
    "local_rr_sec",
    "rr_ratio_pre",
    "rr_ratio_post",
    # Global beat window morphology (200 samples)
    "amp_mean",
    "amp_std",
    "amp_min",
    "amp_max",
    "amp_peak_to_peak",
    "amp_skewness",
    "amp_kurtosis",
    "amp_rms",
    "amp_energy",
    "zero_crossings",
    # QRS core subsegment ([-30, +30] samples)
    "qrs_mean",
    "qrs_rms",
    "qrs_peak_to_peak",
    "qrs_energy",
    "qrs_energy_ratio",
    # Pre-QRS subsegment ([-90, -30] samples)
    "pre_mean",
    "pre_rms",
    "pre_energy",
    # Post-QRS subsegment ([+30, +110] samples)
    "post_mean",
    "post_rms",
    "post_energy",
    "post_peak_to_peak",
    # Subsampled morphology points (20 evenly-spaced samples across the 200-sample window)
    *[f"morph_sample_{i}" for i in range(20)],
]


def extract_beat_features(
    signal_mv: np.ndarray,
    peak_indices: np.ndarray,
    sampling_rate_hz: int = NATIVE_SAMPLING_RATE_HZ,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract deterministic feature vectors for detected/annotated peak indices.

    Args:
        signal_mv: 1D array of single-lead ECG signal in millivolts.
        peak_indices: 1D array of sample indices corresponding to R-peaks.
        sampling_rate_hz: Sampling frequency in Hz.

    Returns:
        (valid_mask, feature_matrix):
            valid_mask: boolean array of shape (len(peak_indices),) indicating whether the beat
                        was fully inside signal boundaries and features were extracted.
            feature_matrix: 2D array of shape (N_valid, len(FEATURE_NAMES)) with extracted features.
    """
    n_peaks = len(peak_indices)
    if n_peaks == 0:
        return np.zeros(0, dtype=bool), np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)

    sig_len = len(signal_mv)
    valid_mask = np.zeros(n_peaks, dtype=bool)

    # 1. Compute RR intervals (in seconds)
    rr_intervals_sec = np.diff(peak_indices) / float(sampling_rate_hz)
    # Default median RR for fallback edge imputation
    median_rr = float(np.median(rr_intervals_sec)) if len(rr_intervals_sec) > 0 else 0.8

    feature_rows: List[List[float]] = []

    for i, peak_idx in enumerate(peak_indices):
        start_idx = peak_idx - PRE_R_SAMPLES
        end_idx = peak_idx + POST_R_SAMPLES

        # Boundary check: reject beats that lie outside the recording bounds
        if start_idx < 0 or end_idx > sig_len:
            valid_mask[i] = False
            continue

        window = signal_mv[start_idx:end_idx]
        if len(window) != BEAT_WINDOW_TOTAL_SAMPLES:
            valid_mask[i] = False
            continue

        valid_mask[i] = True

        # Pre-RR
        if i == 0:
            pre_rr = rr_intervals_sec[0] if len(rr_intervals_sec) > 0 else median_rr
        else:
            pre_rr = (peak_idx - peak_indices[i - 1]) / float(sampling_rate_hz)

        # Post-RR
        if i >= n_peaks - 1:
            post_rr = rr_intervals_sec[-1] if len(rr_intervals_sec) > 0 else median_rr
        else:
            post_rr = (peak_indices[i + 1] - peak_idx) / float(sampling_rate_hz)

        # Local RR: average of up to 10 surrounding beats
        local_start = max(0, i - 5)
        local_end = min(n_peaks, i + 5)
        local_peaks = peak_indices[local_start:local_end]
        if len(local_peaks) > 1:
            local_rr = float(np.mean(np.diff(local_peaks) / float(sampling_rate_hz)))
        else:
            local_rr = median_rr

        local_rr = max(local_rr, 0.1)  # Prevent division by zero
        rr_ratio_pre = pre_rr / local_rr
        rr_ratio_post = post_rr / local_rr

        # Global window morphology
        amp_mean = float(np.mean(window))
        amp_std = float(np.std(window))
        amp_min = float(np.min(window))
        amp_max = float(np.max(window))
        amp_peak_to_peak = amp_max - amp_min
        amp_skewness = float(scipy.stats.skew(window))
        amp_kurtosis = float(scipy.stats.kurtosis(window))
        amp_rms = float(np.sqrt(np.mean(window ** 2)))
        amp_energy = float(np.sum(window ** 2) / float(BEAT_WINDOW_TOTAL_SAMPLES))
        zero_crossings = float(np.sum(np.diff(window > amp_mean) != 0))

        # QRS core subsegment ([-30, +30] samples relative to R-peak -> indices [60, 120] in window)
        qrs_sub = window[PRE_R_SAMPLES - 30 : PRE_R_SAMPLES + 30]
        qrs_mean = float(np.mean(qrs_sub))
        qrs_rms = float(np.sqrt(np.mean(qrs_sub ** 2)))
        qrs_peak_to_peak = float(np.max(qrs_sub) - np.min(qrs_sub))
        qrs_energy = float(np.sum(qrs_sub ** 2) / float(len(qrs_sub)))
        qrs_energy_ratio = qrs_energy / (amp_energy + 1e-8)

        # Pre-QRS subsegment ([-90, -30] samples -> indices [0, 60] in window)
        pre_sub = window[: PRE_R_SAMPLES - 30]
        pre_mean = float(np.mean(pre_sub))
        pre_rms = float(np.sqrt(np.mean(pre_sub ** 2)))
        pre_energy = float(np.sum(pre_sub ** 2) / float(len(pre_sub)))

        # Post-QRS subsegment ([+30, +110] samples -> indices [120, 200] in window)
        post_sub = window[PRE_R_SAMPLES + 30 :]
        post_mean = float(np.mean(post_sub))
        post_rms = float(np.sqrt(np.mean(post_sub ** 2)))
        post_energy = float(np.sum(post_sub ** 2) / float(len(post_sub)))
        post_peak_to_peak = float(np.max(post_sub) - np.min(post_sub))

        # Subsampled 20 points
        morph_subsampled = [float(window[idx]) for idx in np.linspace(0, len(window) - 1, 20, dtype=int)]

        row = [
            pre_rr,
            post_rr,
            local_rr,
            rr_ratio_pre,
            rr_ratio_post,
            amp_mean,
            amp_std,
            amp_min,
            amp_max,
            amp_peak_to_peak,
            amp_skewness,
            amp_kurtosis,
            amp_rms,
            amp_energy,
            zero_crossings,
            qrs_mean,
            qrs_rms,
            qrs_peak_to_peak,
            qrs_energy,
            qrs_energy_ratio,
            pre_mean,
            pre_rms,
            pre_energy,
            post_mean,
            post_rms,
            post_energy,
            post_peak_to_peak,
            *morph_subsampled,
        ]
        feature_rows.append(row)

    feature_matrix = np.array(feature_rows, dtype=np.float64)
    # Ensure no NaN or Inf
    np.nan_to_num(feature_matrix, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    return valid_mask, feature_matrix


def extract_annotated_record_features(
    record: SingleLeadRecord,
    annotations: List[BeatAnnotation],
) -> Tuple[np.ndarray, np.ndarray, List[BeatAnnotation]]:
    """Extract features and matched labels for all valid annotated beats in a record.

    Returns:
        (features, labels, valid_annotations):
            features: 2D array of shape (N_valid, n_features)
            labels: 1D string array of shape (N_valid,) containing AAMI classes ('N', 'S', 'V', 'Q')
            valid_annotations: list of BeatAnnotation objects for the valid beats.
    """
    if not annotations:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros(0, dtype=object), []

    peak_indices = np.array([ann.sample_index for ann in annotations], dtype=int)
    valid_mask, features = extract_beat_features(
        signal_mv=record.signal_mv,
        peak_indices=peak_indices,
        sampling_rate_hz=record.sampling_rate_hz,
    )

    valid_annotations = [ann for ann, is_valid in zip(annotations, valid_mask) if is_valid]
    labels = np.array([ann.aami_class for ann in valid_annotations], dtype=object)

    return features, labels, valid_annotations
