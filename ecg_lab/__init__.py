"""ECG Insight Lab core package."""

from ecg_lab.config import (
    ColorPalette,
    TypographySystem,
    ValidationLimits,
    DISCLAIMER_RESEARCH_ONLY,
    DISCLAIMER_HARDWARE_ABSENT,
)
from ecg_lab.models import (
    ValidationResult,
    ValidationStatus,
    ImageMetadata,
)
from ecg_lab.validation import (
    validate_image_file,
    validate_image_bytes,
)

__all__ = [
    "ColorPalette",
    "TypographySystem",
    "ValidationLimits",
    "DISCLAIMER_RESEARCH_ONLY",
    "DISCLAIMER_HARDWARE_ABSENT",
    "ValidationResult",
    "ValidationStatus",
    "ImageMetadata",
    "validate_image_file",
    "validate_image_bytes",
    "digitize_ecg_image",
    "DigitizationResult",
    "DigitizationStatus",
    "CalibrationStatus",
    "AmplitudeUnit",
    "LeadSignal",
]

from ecg_lab.digitization import (
    digitize_ecg_image,
    DigitizationResult,
    DigitizationStatus,
    CalibrationStatus,
    AmplitudeUnit,
    LeadSignal,
)
