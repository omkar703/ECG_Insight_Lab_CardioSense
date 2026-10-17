"""ECG digitization module exposing types, adapters, and service functions."""

from ecg_lab.digitization.types import (
    DigitizationStatus,
    CalibrationStatus,
    AmplitudeUnit,
    LeadSignal,
    DigitizationResult,
)
from ecg_lab.digitization.adapter import (
    ECGDigitizer,
    OpenECGDigitizerAdapter,
)
from ecg_lab.digitization.baseline import (
    ClassicalBaselineDigitizer,
    STANDARD_12LEAD_LAYOUT_BOXES,
)
from ecg_lab.digitization.service import (
    digitize_ecg_image,
    get_digitizer_engine,
    export_full_csv,
    export_metadata_json,
)

__all__ = [
    "DigitizationStatus",
    "CalibrationStatus",
    "AmplitudeUnit",
    "LeadSignal",
    "DigitizationResult",
    "ECGDigitizer",
    "OpenECGDigitizerAdapter",
    "ClassicalBaselineDigitizer",
    "STANDARD_12LEAD_LAYOUT_BOXES",
    "digitize_ecg_image",
    "get_digitizer_engine",
    "export_full_csv",
    "export_metadata_json",
]
