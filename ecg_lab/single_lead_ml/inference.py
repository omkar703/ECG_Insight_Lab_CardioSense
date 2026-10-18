"""Consolidated single-lead ECG inference and evaluation service for CardioSense.

Provides a unified interface for beat-by-beat inference across:
1. Recorded MIT-BIH waveform slices
2. Reconstructed offline AD8232-like simulation streams
3. (Future) Validated physical hardware acquisitions

Enforces strict integrity boundaries:
- Rejects uncalibrated or incompatible hardware inputs
- Operates peak detection and feature extraction in strict isolation from reference annotations
- Reference annotations are consulted solely in post-hoc evaluation
- Uses the frozen 4-class taxonomy: N, S, V, Q (F merged into Q)
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from ecg_lab.single_lead_ml.config import (
    AAMI_CLASS_DESCRIPTIONS,
    AAMI_TARGET_CLASSES,
    DEFAULT_ARTIFACT_DIR,
    DISCLAIMER_SINGLE_LEAD_RESEARCH,
    MODEL_NAME,
    NATIVE_SAMPLING_RATE_HZ,
)
from ecg_lab.single_lead_ml.dataset import BeatAnnotation
from ecg_lab.single_lead_ml.detector import (
    evaluate_r_peak_detector,
    pan_tompkins_qrs_detector,
)
from ecg_lab.single_lead_ml.evaluate import compute_beat_classification_metrics
from ecg_lab.single_lead_ml.features import extract_beat_features
from ecg_lab.single_lead_ml.models import load_verified_single_lead_model
from hardware.signal_processing.calibration import (
    IncompatibleSignalError,
    SignalUnits,
    UncalibratedHardwareError,
    validate_input_for_classifier,
)


@dataclass
class BeatPrediction:
    """Individual classified cardiac beat prediction."""
    beat_index: int
    peak_sample: int
    time_seconds: float
    predicted_class: str
    class_probabilities: Dict[str, float]
    confidence_score: float
    nearest_reference_symbol: Optional[str] = None
    nearest_reference_class: Optional[str] = None
    match_distance_ms: Optional[float] = None
    is_matched: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SingleLeadInferenceResult:
    """Structured result returned by run_single_lead_inference."""
    success: bool
    source_kind: str
    calibration_provenance: str
    sampling_rate_hz: int
    units: str
    total_samples: int
    duration_seconds: float
    detected_peaks_count: int
    classifiable_windows_count: int
    skipped_peaks_count: int
    predictions: List[BeatPrediction]
    predictions_summary: Dict[str, int]
    raw_signal_copy: np.ndarray
    detected_peak_samples: np.ndarray
    evaluation: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert prediction rows to a user-facing pandas DataFrame."""
        if not self.predictions:
            return pd.DataFrame()
        rows = []
        for p in self.predictions:
            row = {
                "Beat #": p.beat_index,
                "Time (s)": round(p.time_seconds, 3),
                "Sample Index": p.peak_sample,
                "Predicted Class": p.predicted_class,
                "Class Description": AAMI_CLASS_DESCRIPTIONS.get(p.predicted_class, p.predicted_class),
                "Confidence": round(p.confidence_score, 4),
            }
            # Add probability columns
            for cls_name in AAMI_TARGET_CLASSES:
                row[f"P({cls_name})"] = round(p.class_probabilities.get(cls_name, 0.0), 4)
            # Reference comparisons
            if p.nearest_reference_class is not None:
                row["Ref Class"] = p.nearest_reference_class
                row["Ref Symbol"] = p.nearest_reference_symbol or ""
                row["Match Dist (ms)"] = round(p.match_distance_ms, 1) if p.match_distance_ms is not None else ""
                row["Match Agreement"] = "Match" if p.predicted_class == p.nearest_reference_class else "Mismatch"
            rows.append(row)
        return pd.DataFrame(rows)


def run_single_lead_inference(
    signal: np.ndarray,
    sampling_rate_hz: int = NATIVE_SAMPLING_RATE_HZ,
    units: str = "mV",
    calibration_provenance: str = "MIT-BIH calibrated mV",
    source_kind: str = "MIT-BIH source waveform",
    model_path: Optional[Union[str, Path]] = None,
    reference_annotations: Optional[List[BeatAnnotation]] = None,
    lowcut_hz: float = 5.0,
    highcut_hz: float = 15.0,
    tolerance_ms: float = 50.0,
) -> SingleLeadInferenceResult:
    """Execute single-lead beat detection and machine-learning classification.

    Args:
        signal: 1D array of single-lead ECG signal in millivolts.
        sampling_rate_hz: Sampling frequency in Hz (must be 360 Hz for Workflow C).
        units: Signal unit string ('mV'). Raw ADC counts or uncalibrated volts are rejected.
        calibration_provenance: Audit tag specifying calibration origin.
        source_kind: Source description ('MIT-BIH source waveform',
                     'offline AD8232-like simulation', or 'physical Arduino/AD8232 stream').
        model_path: Optional custom path to model artifact.
        reference_annotations: Optional ground-truth annotations for post-hoc evaluation.
                               Ground-truth data is NEVER used during inference.
        lowcut_hz: Bandpass lower corner for Pan-Tompkins peak detector.
        highcut_hz: Bandpass upper corner for Pan-Tompkins peak detector.
        tolerance_ms: Tolerance window (ms) for post-hoc annotation matching.

    Returns:
        SingleLeadInferenceResult containing detections, classifications, and audit provenance.
    """
    warnings: List[str] = []

    # 1. Hardware Calibration & Safety Guards
    # Physical streams without hardware calibration must be rejected
    is_physical = "physical" in source_kind.lower() or "hardware" in source_kind.lower()
    is_simulation = "simulation" in calibration_provenance.lower() or "simulation" in source_kind.lower()

    if is_physical and not is_simulation:
        return SingleLeadInferenceResult(
            success=False,
            source_kind=source_kind,
            calibration_provenance=calibration_provenance,
            sampling_rate_hz=sampling_rate_hz,
            units=units,
            total_samples=len(signal) if signal is not None else 0,
            duration_seconds=0.0,
            detected_peaks_count=0,
            classifiable_windows_count=0,
            skipped_peaks_count=0,
            predictions=[],
            predictions_summary={},
            raw_signal_copy=np.zeros(0),
            detected_peak_samples=np.zeros(0, dtype=int),
            error_message="Inference blocked: hardware calibration unknown. Physical ADC stream requires verified calibration.",
        )

    # 2. Input Validation (units, finite, 1D, sampling rate)
    try:
        validate_input_for_classifier(
            signal=signal,
            sampling_rate_hz=sampling_rate_hz,
            unit_name=units,
            expected_sampling_rate_hz=NATIVE_SAMPLING_RATE_HZ,
        )
    except (IncompatibleSignalError, UncalibratedHardwareError, Exception) as exc:
        return SingleLeadInferenceResult(
            success=False,
            source_kind=source_kind,
            calibration_provenance=calibration_provenance,
            sampling_rate_hz=sampling_rate_hz,
            units=units,
            total_samples=len(signal) if signal is not None else 0,
            duration_seconds=0.0,
            detected_peaks_count=0,
            classifiable_windows_count=0,
            skipped_peaks_count=0,
            predictions=[],
            predictions_summary={},
            raw_signal_copy=np.zeros(0),
            detected_peak_samples=np.zeros(0, dtype=int),
            error_message=f"Signal validation failed: {exc}",
        )

    # Immutable copy of raw signal
    raw_copy = np.copy(np.asarray(signal, dtype=np.float64))
    total_samples = len(raw_copy)
    duration_sec = total_samples / float(sampling_rate_hz)

    # Minimum length requirement: at least 0.5 seconds required for QRS detector
    if duration_sec < 0.5:
        return SingleLeadInferenceResult(
            success=False,
            source_kind=source_kind,
            calibration_provenance=calibration_provenance,
            sampling_rate_hz=sampling_rate_hz,
            units=units,
            total_samples=total_samples,
            duration_seconds=duration_sec,
            detected_peaks_count=0,
            classifiable_windows_count=0,
            skipped_peaks_count=0,
            predictions=[],
            predictions_summary={},
            raw_signal_copy=raw_copy,
            detected_peak_samples=np.zeros(0, dtype=int),
            error_message=f"Signal duration ({duration_sec:.2f} s) too short for QRS detection. Minimum 0.5 s required.",
        )

    # 3. Autonomous Pan-Tompkins QRS Detection (NO reference annotation access)
    detected_peaks = pan_tompkins_qrs_detector(
        signal_mv=raw_copy,
        sampling_rate_hz=sampling_rate_hz,
        lowcut_hz=lowcut_hz,
        highcut_hz=highcut_hz,
    )
    detected_peaks_count = len(detected_peaks)

    if detected_peaks_count == 0:
        warnings.append("No candidate QRS peaks detected in the signal window.")
        return SingleLeadInferenceResult(
            success=True,
            source_kind=source_kind,
            calibration_provenance=calibration_provenance,
            sampling_rate_hz=sampling_rate_hz,
            units=units,
            total_samples=total_samples,
            duration_seconds=duration_sec,
            detected_peaks_count=0,
            classifiable_windows_count=0,
            skipped_peaks_count=0,
            predictions=[],
            predictions_summary={"N": 0, "S": 0, "V": 0, "Q": 0},
            raw_signal_copy=raw_copy,
            detected_peak_samples=np.zeros(0, dtype=int),
            warnings=warnings,
        )

    # 4. Feature Extraction using Frozen Preprocessing & Windowing
    valid_mask, features = extract_beat_features(
        signal_mv=raw_copy,
        peak_indices=detected_peaks,
        sampling_rate_hz=sampling_rate_hz,
    )

    classifiable_count = int(np.sum(valid_mask))
    skipped_count = detected_peaks_count - classifiable_count

    if skipped_count > 0:
        warnings.append(
            f"{skipped_count} detected peak(s) skipped due to signal boundary limits "
            f"([-90, +110] sample window required around each peak)."
        )

    if classifiable_count == 0:
        warnings.append("No valid beat windows inside boundary limits.")
        return SingleLeadInferenceResult(
            success=True,
            source_kind=source_kind,
            calibration_provenance=calibration_provenance,
            sampling_rate_hz=sampling_rate_hz,
            units=units,
            total_samples=total_samples,
            duration_seconds=duration_sec,
            detected_peaks_count=detected_peaks_count,
            classifiable_windows_count=0,
            skipped_peaks_count=skipped_count,
            predictions=[],
            predictions_summary={"N": 0, "S": 0, "V": 0, "Q": 0},
            raw_signal_copy=raw_copy,
            detected_peak_samples=detected_peaks,
            warnings=warnings,
        )

    # 5. Load Verified Frozen Model
    if model_path is None:
        model_path = DEFAULT_ARTIFACT_DIR / MODEL_NAME
    resolved_model_path = Path(model_path)

    try:
        model, manifest = load_verified_single_lead_model(resolved_model_path)
    except Exception as exc:
        return SingleLeadInferenceResult(
            success=False,
            source_kind=source_kind,
            calibration_provenance=calibration_provenance,
            sampling_rate_hz=sampling_rate_hz,
            units=units,
            total_samples=total_samples,
            duration_seconds=duration_sec,
            detected_peaks_count=detected_peaks_count,
            classifiable_windows_count=classifiable_count,
            skipped_peaks_count=skipped_count,
            predictions=[],
            predictions_summary={},
            raw_signal_copy=raw_copy,
            detected_peak_samples=detected_peaks,
            error_message=f"Model loading failed: {exc}",
        )

    # 6. Predict Classes and Probabilities
    preds = model.predict(features)
    probs = model.predict_proba(features)
    target_classes = list(manifest.get("target_classes", AAMI_TARGET_CLASSES))

    # 7. Construct Beat Predictions
    predictions: List[BeatPrediction] = []
    summary_counts: Dict[str, int] = {c: 0 for c in AAMI_TARGET_CLASSES}

    valid_idx = 0
    tolerance_samples = int(np.round((tolerance_ms / 1000.0) * sampling_rate_hz))

    for i, peak_s in enumerate(detected_peaks):
        if not valid_mask[i]:
            continue

        p_class = str(preds[valid_idx])
        p_probs = probs[valid_idx]
        prob_dict = {
            cls_name: float(p_probs[c_i]) for c_i, cls_name in enumerate(target_classes)
        }
        confidence = float(np.max(p_probs))

        summary_counts[p_class] = summary_counts.get(p_class, 0) + 1

        # Post-hoc annotation reference matching (if annotations provided)
        nearest_sym: Optional[str] = None
        nearest_cls: Optional[str] = None
        dist_ms: Optional[float] = None
        is_matched = False

        if reference_annotations:
            dists = [abs(ann.sample_index - peak_s) for ann in reference_annotations]
            if dists:
                min_idx = int(np.argmin(dists))
                min_samp_dist = dists[min_idx]
                dist_ms = (min_samp_dist / float(sampling_rate_hz)) * 1000.0
                if min_samp_dist <= tolerance_samples:
                    nearest_sym = reference_annotations[min_idx].original_symbol
                    nearest_cls = reference_annotations[min_idx].aami_class
                    is_matched = True

        predictions.append(
            BeatPrediction(
                beat_index=len(predictions) + 1,
                peak_sample=int(peak_s),
                time_seconds=float(peak_s / float(sampling_rate_hz)),
                predicted_class=p_class,
                class_probabilities=prob_dict,
                confidence_score=confidence,
                nearest_reference_symbol=nearest_sym,
                nearest_reference_class=nearest_cls,
                match_distance_ms=dist_ms,
                is_matched=is_matched,
            )
        )
        valid_idx += 1

    # 8. Post-Hoc Evaluation (Strictly decoupled from inference)
    eval_metrics: Optional[Dict[str, Any]] = None
    if reference_annotations and len(reference_annotations) > 0:
        ref_peak_samples = np.array([ann.sample_index for ann in reference_annotations], dtype=int)
        det_eval = evaluate_r_peak_detector(
            detected_peaks=detected_peaks,
            reference_peaks=ref_peak_samples,
            sampling_rate_hz=sampling_rate_hz,
            tolerance_ms=tolerance_ms,
        )

        # Matched conditional classification metrics
        y_true_matched: List[str] = []
        y_pred_matched: List[str] = []
        for p in predictions:
            if p.is_matched and p.nearest_reference_class in AAMI_TARGET_CLASSES:
                y_true_matched.append(p.nearest_reference_class)
                y_pred_matched.append(p.predicted_class)

        cls_eval: Optional[Dict[str, Any]] = None
        if y_true_matched:
            cls_eval = compute_beat_classification_metrics(
                y_true=np.array(y_true_matched),
                y_pred=np.array(y_pred_matched),
                target_classes=AAMI_TARGET_CLASSES,
            )

        eval_metrics = {
            "detector_evaluation": det_eval,
            "classifier_evaluation": cls_eval,
            "matched_beats_count": len(y_true_matched),
            "unmatched_reference_count": det_eval["false_negatives"],
            "unmatched_detection_count": det_eval["false_positives"],
            "evaluation_note": (
                "Post-hoc evaluation matched against reference annotations within ±50 ms. "
                "Ground truth annotations were NOT provided to detector or feature extractor."
            ),
        }

    return SingleLeadInferenceResult(
        success=True,
        source_kind=source_kind,
        calibration_provenance=calibration_provenance,
        sampling_rate_hz=sampling_rate_hz,
        units=units,
        total_samples=total_samples,
        duration_seconds=duration_sec,
        detected_peaks_count=detected_peaks_count,
        classifiable_windows_count=classifiable_count,
        skipped_peaks_count=skipped_count,
        predictions=predictions,
        predictions_summary=summary_counts,
        raw_signal_copy=raw_copy,
        detected_peak_samples=detected_peaks,
        evaluation=eval_metrics,
        warnings=warnings,
    )
