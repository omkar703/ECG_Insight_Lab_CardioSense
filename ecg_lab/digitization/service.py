"""Service layer orchestrating image validation, engine selection, and digitization."""

from io import BytesIO
from pathlib import Path
from typing import Union, Optional, Dict, Any
from PIL import Image

from ecg_lab.validation import validate_image_bytes, validate_image_file
from ecg_lab.digitization.types import (
    DigitizationResult,
    DigitizationStatus,
    CalibrationStatus,
    AmplitudeUnit,
    LeadSignal,
)
from ecg_lab.digitization.baseline import ClassicalBaselineDigitizer
from ecg_lab.digitization.adapter import OpenECGDigitizerAdapter, ECGDigitizer


def get_digitizer_engine(engine_name: str = "baseline") -> ECGDigitizer:
    """Factory creating the appropriate digitization engine adapter."""
    engine_lower = engine_name.lower().strip()
    if engine_lower in ("open-ecg", "open_ecg", "upstream", "unet"):
        return OpenECGDigitizerAdapter()
    return ClassicalBaselineDigitizer()


def digitize_ecg_image(
    image_input: Union[bytes, str, Path],
    engine: str = "baseline",
    options: Optional[Dict[str, Any]] = None,
) -> DigitizationResult:
    """Primary entry point for ECG image digitization.

    Validates the input using Phase 1 safeguards first, then delegates to the
    selected digitization engine. Never executes legacy pickle models or makes
    diagnostic claims.

    Args:
        image_input: Raw image bytes, or a filesystem Path / string path.
        engine: "baseline" (classical native CPU) or "open-ecg" (upstream adapter).
        options: Optional configuration dictionary (e.g. samples_per_lead).

    Returns:
        Structured DigitizationResult containing status, lead waveforms, explicit
        units, and metadata.
    """
    options = options or {}

    # 1. Front-Door Validation
    if isinstance(image_input, (str, Path)):
        val_result = validate_image_file(image_input)
        if not val_result.is_valid:
            return DigitizationResult(
                status=DigitizationStatus.REJECTED,
                method=engine,
                method_version="1.0.0",
                calibration_status=CalibrationStatus.UNRECOVERED,
                amplitude_unit=AmplitudeUnit.RELATIVE,
                error_message=val_result.message,
                warnings=val_result.warnings,
            )
        image = Image.open(image_input)
    elif isinstance(image_input, (bytes, bytearray)):
        val_result = validate_image_bytes(image_input, filename="upload.jpg")
        if not val_result.is_valid:
            return DigitizationResult(
                status=DigitizationStatus.REJECTED,
                method=engine,
                method_version="1.0.0",
                calibration_status=CalibrationStatus.UNRECOVERED,
                amplitude_unit=AmplitudeUnit.RELATIVE,
                error_message=val_result.message,
                warnings=val_result.warnings,
            )
        image = Image.open(BytesIO(image_input))
    else:
        return DigitizationResult(
            status=DigitizationStatus.ERROR,
            method=engine,
            method_version="1.0.0",
            calibration_status=CalibrationStatus.UNRECOVERED,
            amplitude_unit=AmplitudeUnit.RELATIVE,
            error_message=f"Unsupported image_input type: {type(image_input).__name__}.",
        )

    # 2. Engine Dispatch
    digitizer = get_digitizer_engine(engine)
    result = digitizer.digitize(image, val_result.metadata, options=options)

    # Attach validation warnings if present
    if val_result.warnings:
        result.warnings.extend([f"Input notice: {w}" for w in val_result.warnings])

    return result


def export_full_csv(
    result: DigitizationResult,
    output_path: Union[str, Path],
    format_type: str = "long",
) -> Path:
    """Save digitization waveform signals to CSV on disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format_type.lower() == "wide":
        df = result.to_wide_dataframe()
    else:
        df = result.to_long_dataframe()

    df.to_csv(path, index=False)
    return path


def export_metadata_json(
    result: DigitizationResult,
    output_path: Union[str, Path],
) -> Path:
    """Save digitization metadata sidecar to JSON on disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.to_json(indent=2))
    return path
