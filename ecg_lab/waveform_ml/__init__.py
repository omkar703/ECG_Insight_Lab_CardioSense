"""Digital Waveform ML Subsystem for PTB-XL v1.0.3.

Provides reproducible dataset handling, bounded-memory feature extraction,
patient-separated fold validation, lightweight ML pipelines, and evaluation metrics.
Independent of the image digitization subsystem.
"""

from ecg_lab.waveform_ml.config import (
    LEAD_NAMES,
    SAMPLING_RATE_HZ,
    TARGET_SUPERCLASSES,
    TEST_FOLDS,
    TRAIN_FOLDS,
    VAL_FOLDS,
)

__all__ = [
    "TARGET_SUPERCLASSES",
    "LEAD_NAMES",
    "SAMPLING_RATE_HZ",
    "TRAIN_FOLDS",
    "VAL_FOLDS",
    "TEST_FOLDS",
]
