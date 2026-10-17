"""Classical morphological baseline digitizer for standard 12-lead paper ECG scans."""

import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image

from ecg_lab.models import ImageMetadata
from ecg_lab.digitization.types import (
    DigitizationResult,
    DigitizationStatus,
    CalibrationStatus,
    AmplitudeUnit,
    LeadSignal,
)
from ecg_lab.digitization.adapter import ECGDigitizer


# Standard 12-lead layout coordinates as proportional bounding box fractions (ymin, ymax, xmin, xmax)
# Configured for standard 3x4 clinical 12-lead grid layouts
STANDARD_12LEAD_LAYOUT_BOXES: Dict[str, Tuple[float, float, float, float]] = {
    # Column 1 (Limb Leads)
    "I":   (0.18, 0.38, 0.05, 0.28),
    "II":  (0.38, 0.58, 0.05, 0.28),
    "III": (0.58, 0.78, 0.05, 0.28),
    # Column 2 (Augmented Leads)
    "aVR": (0.18, 0.38, 0.28, 0.51),
    "aVL": (0.38, 0.58, 0.28, 0.51),
    "aVF": (0.58, 0.78, 0.28, 0.51),
    # Column 3 (Precordial V1-V3)
    "V1":  (0.18, 0.38, 0.51, 0.74),
    "V2":  (0.38, 0.58, 0.51, 0.74),
    "V3":  (0.58, 0.78, 0.51, 0.74),
    # Column 4 (Precordial V4-V6)
    "V4":  (0.18, 0.38, 0.74, 0.96),
    "V5":  (0.38, 0.58, 0.74, 0.96),
    "V6":  (0.58, 0.78, 0.74, 0.96),
}


class ClassicalBaselineDigitizer:
    """Deterministic, classical morphological digitizer for standard 12-lead ECG scans.

    This engine operates natively on CPU using NumPy and SciPy. It extracts continuous 1D
    traces via intensity thresholding and column-wise centroid tracking.

    LIMITATIONS & SCOPE:
    - Designed specifically for standard 3x4 12-lead grid layouts.
    - Requires reasonable contrast between ink waveform and paper background.
    - Does not correct for severe perspective skew (> 10 degrees) or crumpled paper.
    - Amplitudes are strictly uncalibrated relative units (CalibrationStatus.UNRECOVERED).
    - NEVER feeds output into a medical diagnostic model.
    """

    def __init__(self, target_samples_per_lead: int = 255):
        self.target_samples = target_samples_per_lead

    @property
    def name(self) -> str:
        return "Classical Morphological Baseline"

    @property
    def version(self) -> str:
        return "1.0.0-classical"

    def is_available(self) -> Tuple[bool, Optional[str]]:
        """Always available as it relies solely on NumPy and Pillow."""
        return True, None

    def digitize(
        self,
        image: Image.Image,
        metadata: ImageMetadata,
        options: Optional[Dict[str, Any]] = None,
    ) -> DigitizationResult:
        """Digitize the provided image using column-wise centroid tracking across lead windows."""
        start_time = time.time()
        options = options or {}
        samples_target = options.get("samples_per_lead", self.target_samples)

        # 1. Convert to grayscale numpy array
        gray_img = image.convert("L")
        img_arr = np.array(gray_img, dtype=np.float32)
        h, w = img_arr.shape

        # Global Contrast Check: Detect completely blank/flat or inverted images
        std_val = float(np.std(img_arr))
        if std_val < 3.0:
            return DigitizationResult(
                status=DigitizationStatus.REJECTED,
                method=self.name,
                method_version=self.version,
                calibration_status=CalibrationStatus.UNRECOVERED,
                amplitude_unit=AmplitudeUnit.RELATIVE,
                runtime_ms=(time.time() - start_time) * 1000.0,
                error_message="Image has almost zero contrast or is a solid color; no ECG trace detectable.",
                warnings=["Low image variance detected."],
            )

        # Dark pixels represent the waveform trace on light paper
        # Invert so trace has high intensity values
        max_val = np.max(img_arr)
        min_val = np.min(img_arr)
        if max_val > min_val:
            norm_inv = (max_val - img_arr) / (max_val - min_val)  # 1.0 for darkest ink, 0.0 for paper
        else:
            norm_inv = np.zeros_like(img_arr)

        leads: Dict[str, LeadSignal] = {}
        detected_names: List[str] = []
        missing_names: List[str] = []
        warnings: List[str] = []

        # 2. Extract each lead window
        for lead_name, (ymin_f, ymax_f, xmin_f, xmax_f) in STANDARD_12LEAD_LAYOUT_BOXES.items():
            y0, y1 = int(ymin_f * h), int(ymax_f * h)
            x0, x1 = int(xmin_f * w), int(xmax_f * w)

            lead_crop = norm_inv[y0:y1, x0:x1]
            crop_h, crop_w = lead_crop.shape

            if crop_h < 10 or crop_w < 10:
                missing_names.append(lead_name)
                continue

            lead_signal = self._extract_lead_signal(
                lead_crop,
                lead_name=lead_name,
                target_length=samples_target,
            )

            if lead_signal.is_valid:
                leads[lead_name] = lead_signal
                detected_names.append(lead_name)
            else:
                leads[lead_name] = lead_signal
                missing_names.append(lead_name)
                warnings.append(f"Lead {lead_name}: Unreliable or missing trace (confidence {lead_signal.confidence:.2f}).")

        # 3. Overall Execution Status Determination
        detected_count = len(detected_names)
        if detected_count == 0:
            status = DigitizationStatus.REJECTED
            error_msg = "No identifiable ECG traces were detected across the expected 12 lead regions."
        elif detected_count < 6:
            status = DigitizationStatus.PARTIAL
            error_msg = f"Only {detected_count} of 12 leads could be traced reliably."
        elif detected_count < 12:
            status = DigitizationStatus.PARTIAL
            error_msg = None
            warnings.append(f"Partial detection: {12 - detected_count} leads lacked sufficient contrast or continuity.")
        else:
            status = DigitizationStatus.SUCCESS
            error_msg = None

        return DigitizationResult(
            status=status,
            method=self.name,
            method_version=self.version,
            calibration_status=CalibrationStatus.UNRECOVERED,
            amplitude_unit=AmplitudeUnit.RELATIVE,
            leads=leads,
            detected_leads=detected_names,
            missing_leads=missing_names,
            warnings=warnings,
            error_message=error_msg,
            metadata={
                "target_samples_per_lead": samples_target,
                "input_dimensions": f"{w}x{h}",
                "grid_assumptions": "Standard 3x4 clinical 12-lead bounding boxes",
                "calibration_note": "Uncalibrated relative coordinates; voltages not claimed in mV.",
            },
            runtime_ms=(time.time() - start_time) * 1000.0,
        )

    def _extract_lead_signal(
        self,
        lead_crop: np.ndarray,
        lead_name: str,
        target_length: int,
    ) -> LeadSignal:
        """Extract a single 1D continuous waveform from a lead bounding box."""
        crop_h, crop_w = lead_crop.shape
        y_indices = np.arange(crop_h)

        # Threshold to isolate dark trace pixels (above 65th percentile in crop)
        p_thresh = np.percentile(lead_crop, 70)
        trace_mask = lead_crop > max(0.15, p_thresh)

        column_centers: List[Optional[float]] = []
        valid_cols = 0

        # Column-wise intensity-weighted centroid
        for x in range(crop_w):
            weights = lead_crop[:, x] * trace_mask[:, x]
            total_weight = np.sum(weights)

            if total_weight > 0.05:
                # Intensity-weighted vertical position
                centroid_y = float(np.sum(y_indices * weights) / total_weight)
                column_centers.append(centroid_y)
                valid_cols += 1
            else:
                column_centers.append(None)

        confidence = valid_cols / max(1, crop_w)
        flags: List[str] = []

        # If too few columns have trace pixels, mark as invalid
        if confidence < 0.25:
            return LeadSignal(
                lead_name=lead_name,
                samples=[0.0] * target_length,
                sample_indices=list(range(target_length)),
                confidence=confidence,
                is_valid=False,
                quality_flags=["INSUFFICIENT_TRACE_INK", "LOW_CONFIDENCE"],
            )

        # Interpolate missing column positions
        filled_y = self._interpolate_missing(column_centers)

        # Invert vertical coordinate so ECG peaks (R-waves) point upward
        # Image coordinates have y=0 at the top
        baseline_y = float(np.median(filled_y))
        raw_waveform = baseline_y - filled_y  # Upward deflection is now positive

        # Resample to target_length using linear interpolation
        x_orig = np.linspace(0, 1, len(raw_waveform))
        x_target = np.linspace(0, 1, target_length)
        resampled = np.interp(x_target, x_orig, raw_waveform)

        # Amplitude Normalization (Scale to relative range [-1.0, 1.0])
        amp_max = np.max(np.abs(resampled))
        if amp_max > 1e-4:
            normalized = resampled / amp_max
        else:
            normalized = resampled
            flags.append("FLAT_TRACE")

        # Smooth minor high-frequency column-to-column jitter with moving average
        kernel_size = 3
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(normalized, kernel, mode="same")

        return LeadSignal(
            lead_name=lead_name,
            samples=[float(val) for val in smoothed],
            sample_indices=list(range(target_length)),
            confidence=round(confidence, 3),
            is_valid=True,
            time_seconds=None,  # Explicitly None without calibrated grid timing
            quality_flags=flags,
        )

    def _interpolate_missing(self, values: List[Optional[float]]) -> np.ndarray:
        """Fill missing column centroid values via linear interpolation."""
        arr = np.array([v if v is not None else np.nan for v in values], dtype=np.float32)
        nans = np.isnan(arr)
        x = lambda z: z.nonzero()[0]

        if np.all(nans):
            return np.zeros_like(arr)

        arr[nans] = np.interp(x(nans), x(~nans), arr[~nans])
        return arr
