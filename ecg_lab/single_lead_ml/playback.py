"""Recorded ECG playback and offline inference simulator for MIT-BIH single-lead signals.

Crucial Disclaimer:
    RECORDED DATA REPLAY — NOT LIVE HARDWARE.
    This module simulates recorded Holter playback from the MIT-BIH Arrhythmia Database.
    No physical sensors (AD8232 / ESP32) are connected or acquired live.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ecg_lab.single_lead_ml.config import (
    AAMI_CLASS_DESCRIPTIONS,
    AAMI_TARGET_CLASSES,
    DEFAULT_ARTIFACT_DIR,
    DISCLAIMER_RECORDED_REPLAY,
    DISCLAIMER_SINGLE_LEAD_RESEARCH,
    MODEL_NAME,
    NATIVE_SAMPLING_RATE_HZ,
)
from ecg_lab.single_lead_ml.dataset import (
    BeatAnnotation,
    SingleLeadRecord,
    load_beat_annotations,
    load_record_with_annotations,
    load_single_lead_signal,
)
from ecg_lab.single_lead_ml.detector import pan_tompkins_qrs_detector
from ecg_lab.single_lead_ml.features import extract_beat_features
from ecg_lab.single_lead_ml.models import load_verified_single_lead_model


@dataclass
class PlaybackBeatResult:
    """Represents a classified beat detected from the signal."""
    peak_sample: int
    time_seconds: float
    predicted_class: str
    class_probabilities: Dict[str, float]
    nearest_reference_symbol: Optional[str] = None
    nearest_reference_class: Optional[str] = None


@dataclass
class PlaybackSegmentResult:
    """Represents a time slice of recorded ECG playback with detected and classified beats."""
    record_id: str
    lead_name: str
    start_second: float
    duration_seconds: float
    sampling_rate_hz: int
    time_axis: np.ndarray
    signal_mv: np.ndarray
    disclaimer: str
    detected_beats: List[PlaybackBeatResult]
    reference_annotations: List[BeatAnnotation]


def get_playback_slice(
    record_id: str,
    start_second: float = 0.0,
    duration_seconds: float = 10.0,
    model_dir: Path = DEFAULT_ARTIFACT_DIR,
    data_dir: Optional[Path] = None,
) -> PlaybackSegmentResult:
    """Retrieve a calibrated recorded ECG slice with signal-detected and classified beats.

    Notice: R-peak detection is executed directly on the raw signal, completely independent
    of ground-truth reference annotations.
    """
    rec, all_anns = load_record_with_annotations(record_id, data_dir=data_dir)
    fs = rec.sampling_rate_hz
    total_samples = len(rec.signal_mv)

    # Compute slice sample bounds
    start_sample = max(0, int(start_second * fs))
    n_slice_samples = int(duration_seconds * fs)
    end_sample = min(total_samples, start_sample + n_slice_samples)

    # Margin for peak detection to prevent edge truncation
    margin_samples = int(1.0 * fs)
    det_start = max(0, start_sample - margin_samples)
    det_end = min(total_samples, end_sample + margin_samples)

    sub_signal = rec.signal_mv[det_start:det_end]
    detected_local_peaks = pan_tompkins_qrs_detector(sub_signal, sampling_rate_hz=fs)
    detected_global_peaks = detected_local_peaks + det_start

    # Filter detected peaks strictly to the user-requested window [start_sample, end_sample)
    window_peaks = detected_global_peaks[
        (detected_global_peaks >= start_sample) & (detected_global_peaks < end_sample)
    ]

    # Extract slice signal and time axis
    slice_mv = rec.signal_mv[start_sample:end_sample]
    slice_time = np.arange(len(slice_mv)) / float(fs) + (start_sample / float(fs))

    # Reference annotations in this window
    window_anns = [
        ann for ann in all_anns
        if start_sample <= ann.sample_index < end_sample
    ]

    # Model inference on detected peaks
    model_path = Path(model_dir) / MODEL_NAME
    detected_beats: List[PlaybackBeatResult] = []

    if len(window_peaks) > 0 and model_path.is_file():
        try:
            model, manifest = load_verified_single_lead_model(model_path)
            # Extract features for all window peaks using the full recording context for RR intervals
            valid_mask, features = extract_beat_features(
                signal_mv=rec.signal_mv,
                peak_indices=window_peaks,
                sampling_rate_hz=fs,
            )

            if np.any(valid_mask):
                valid_features = features
                preds = model.predict(valid_features)
                probs = model.predict_proba(valid_features)

                model_classes = list(manifest.get("target_classes", AAMI_TARGET_CLASSES))

                v_idx = 0
                for i, peak_s in enumerate(window_peaks):
                    if not valid_mask[i]:
                        continue
                    p_cls = str(preds[v_idx])
                    p_prob = probs[v_idx]
                    prob_dict = {cls_name: float(p_prob[c_i]) for c_i, cls_name in enumerate(model_classes)}

                    # Find nearest reference annotation within 50 ms for diagnostic comparison
                    nearest_sym = None
                    nearest_cls = None
                    if window_anns:
                        dists = [abs(ann.sample_index - peak_s) for ann in window_anns]
                        min_dist_idx = int(np.argmin(dists))
                        if dists[min_dist_idx] <= int(0.050 * fs):
                            nearest_sym = window_anns[min_dist_idx].original_symbol
                            nearest_cls = window_anns[min_dist_idx].aami_class

                    detected_beats.append(PlaybackBeatResult(
                        peak_sample=int(peak_s),
                        time_seconds=peak_s / float(fs),
                        predicted_class=p_cls,
                        class_probabilities=prob_dict,
                        nearest_reference_symbol=nearest_sym,
                        nearest_reference_class=nearest_cls,
                    ))
                    v_idx += 1
        except Exception:
            # If model load fails, return peaks without classification
            for peak_s in window_peaks:
                detected_beats.append(PlaybackBeatResult(
                    peak_sample=int(peak_s),
                    time_seconds=peak_s / float(fs),
                    predicted_class="UNKNOWN",
                    class_probabilities={},
                ))
    else:
        for peak_s in window_peaks:
            detected_beats.append(PlaybackBeatResult(
                peak_sample=int(peak_s),
                time_seconds=peak_s / float(fs),
                predicted_class="DETECTED",
                class_probabilities={},
            ))

    return PlaybackSegmentResult(
        record_id=str(record_id),
        lead_name=rec.lead_name,
        start_second=start_second,
        duration_seconds=duration_seconds,
        sampling_rate_hz=fs,
        time_axis=slice_time,
        signal_mv=slice_mv,
        disclaimer=DISCLAIMER_RECORDED_REPLAY,
        detected_beats=detected_beats,
        reference_annotations=window_anns,
    )
