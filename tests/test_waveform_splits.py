"""Unit tests for patient split integrity and patient leakage assertions."""

import pandas as pd
import pytest

from ecg_lab.waveform_ml.dataset import (
    PatientOverlapError,
    validate_split_integrity,
)


def test_split_integrity_valid():
    """Test that a cleanly separated dataset passes split validation."""
    df = pd.DataFrame({
        "patient_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "strat_fold": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })
    stats = validate_split_integrity(df)
    assert stats["split_integrity_verified"] is True
    assert stats["train_records"] == 8
    assert stats["val_records"] == 1
    assert stats["test_records"] == 1
    assert stats["train_val_overlap"] == 0
    assert stats["train_test_overlap"] == 0
    assert stats["val_test_overlap"] == 0


def test_split_integrity_train_val_leakage():
    """Test that sharing a patient between train (fold 1) and val (fold 9) raises PatientOverlapError."""
    df = pd.DataFrame({
        "patient_id": [1, 2, 3, 1],  # Patient 1 in both fold 1 and fold 9
        "strat_fold": [1, 2, 3, 9],
    })
    with pytest.raises(PatientOverlapError, match="Patient leakage detected"):
        validate_split_integrity(df)


def test_split_integrity_train_test_leakage():
    """Test that sharing a patient between train (fold 2) and test (fold 10) raises PatientOverlapError."""
    df = pd.DataFrame({
        "patient_id": [10, 20, 30, 20],  # Patient 20 in fold 2 and fold 10
        "strat_fold": [1, 2, 9, 10],
    })
    with pytest.raises(PatientOverlapError, match="Patient leakage detected"):
        validate_split_integrity(df)


def test_split_integrity_val_test_leakage():
    """Test that sharing a patient between val (fold 9) and test (fold 10) raises PatientOverlapError."""
    df = pd.DataFrame({
        "patient_id": [100, 200, 300, 300],  # Patient 300 in fold 9 and fold 10
        "strat_fold": [1, 2, 9, 10],
    })
    with pytest.raises(PatientOverlapError, match="Patient leakage detected"):
        validate_split_integrity(df)


def test_split_integrity_missing_columns():
    """Test that missing required columns raises ValueError."""
    df = pd.DataFrame({"some_col": [1, 2, 3]})
    with pytest.raises(ValueError, match="DataFrame must contain"):
        validate_split_integrity(df)
