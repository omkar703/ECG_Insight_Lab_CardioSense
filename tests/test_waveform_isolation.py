"""Unit tests enforcing strict isolation from legacy pickle models and verifying cryptographic provenance manifests."""

from pathlib import Path
import pytest
from sklearn.linear_model import LogisticRegression

from ecg_lab.waveform_ml.config import TARGET_SUPERCLASSES
from ecg_lab.waveform_ml.models import load_verified_model, save_trained_model


def test_legacy_model_load_rejection():
    """Verify that load_verified_model explicitly refuses to load quarantined legacy models."""
    legacy_paths = [
        "legacy/finalmodel.pkl",
        "legacy/Heart_Disease_Prediction_using_ECG (4).pkl",
        "some/dir/finalmodel.pkl",
    ]
    for lp in legacy_paths:
        with pytest.raises(PermissionError, match="Security policy violation"):
            load_verified_model(lp)


def test_unmanifested_model_rejection(tmp_path: Path):
    """Verify that a model file without a cryptographic manifest cannot be loaded."""
    unmanifested_model = tmp_path / "rogue_model.joblib"
    unmanifested_model.write_bytes(b"dummy_bytes")

    with pytest.raises(PermissionError, match="Provenance violation: No manifest found"):
        load_verified_model(unmanifested_model)


def test_tampered_model_rejection(tmp_path: Path):
    """Verify that a model whose SHA-256 does not match its manifest is rejected."""
    model = LogisticRegression()
    thresholds = [0.5] * len(TARGET_SUPERCLASSES)
    model_path, manifest_path = save_trained_model(
        model=model,
        thresholds=thresholds,
        target_classes=list(TARGET_SUPERCLASSES),
        train_metadata={"test": True},
        output_dir=tmp_path,
        model_filename="test_tamper.joblib",
    )

    # Tamper with the model file
    with open(model_path, "ab") as f:
        f.write(b"tamper_bytes_12345")

    with pytest.raises(ValueError, match="Integrity check failed: model SHA-256"):
        load_verified_model(model_path, manifest_path)


def test_verified_model_roundtrip(tmp_path: Path):
    """Verify that a legitimately trained and manifested model loads correctly."""
    model = LogisticRegression()
    thresholds = [0.42] * len(TARGET_SUPERCLASSES)
    model_path, manifest_path = save_trained_model(
        model=model,
        thresholds=thresholds,
        target_classes=list(TARGET_SUPERCLASSES),
        train_metadata={"trained_by": "test_suite"},
        output_dir=tmp_path,
        model_filename="test_legit.joblib",
    )

    loaded_model, manifest = load_verified_model(model_path, manifest_path)
    assert isinstance(loaded_model, LogisticRegression)
    assert manifest["train_metadata"]["trained_by"] == "test_suite"
    assert manifest["decision_thresholds"]["NORM"] == 0.42
