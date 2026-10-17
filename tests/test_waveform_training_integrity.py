"""Unit tests for Phase 3.2 full-fold training integrity, auditing, and cluster bootstrapping."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ecg_lab.waveform_ml.config import TARGET_SUPERCLASSES
from ecg_lab.waveform_ml.dataset import (
    PatientOverlapError,
    audit_dataset_completeness,
    validate_split_integrity,
)
from ecg_lab.waveform_ml.evaluate import (
    compute_patient_cluster_bootstrap_ci,
    format_pilot_vs_full_fold_comparison_table,
)
from ecg_lab.waveform_ml.models import (
    load_verified_model,
    tune_validation_thresholds,
)


def test_audit_dataset_completeness_mock(tmp_path: Path):
    """Test that audit_dataset_completeness distinguishes metadata rows from actual files on disk."""
    data_dir = tmp_path / "ptbxl"
    rec_dir = data_dir / "records100" / "00000"
    rec_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy metadata for 3 folds
    meta_df = pd.DataFrame({
        "patient_id": [1, 2, 3, 4],
        "strat_fold": [1, 1, 9, 10],
        "filename_lr": [
            "records100/00000/00001_lr",
            "records100/00000/00002_lr",
            "records100/00000/00003_lr",
            "records100/00000/00004_lr",
        ],
    }, index=[1, 2, 3, 4])

    # Record 1: fully present (.hea and .dat)
    (data_dir / "records100/00000/00001_lr.hea").write_text("dummy header")
    (data_dir / "records100/00000/00001_lr.dat").write_bytes(b"\x00" * 100)

    # Record 2: only .hea present (missing .dat)
    (data_dir / "records100/00000/00002_lr.hea").write_text("dummy header")

    # Record 3: missing both
    # Record 4: fully present
    (data_dir / "records100/00000/00004_lr.hea").write_text("dummy header")
    (data_dir / "records100/00000/00004_lr.dat").write_bytes(b"\x00" * 100)

    audit = audit_dataset_completeness(meta_df, data_dir=data_dir)

    assert audit["total_expected"] == 4
    assert audit["total_hea_present"] == 3
    assert audit["total_dat_present"] == 2
    assert audit["total_complete_records"] == 2
    assert audit["total_missing_records"] == 2
    assert audit["all_folds_complete"] is False

    # Fold 1: 1 complete, 1 incomplete
    f1 = audit["folds"][1]
    assert f1["expected_records"] == 2
    assert f1["complete_records"] == 1
    assert f1["missing_records"] == 1
    assert f1["is_complete"] is False

    # Fold 10: 1 complete, 0 missing
    f10 = audit["folds"][10]
    assert f10["expected_records"] == 1
    assert f10["complete_records"] == 1
    assert f10["missing_records"] == 0
    assert f10["is_complete"] is True


def test_patient_cluster_bootstrap_ci_bounds():
    """Verify that compute_patient_cluster_bootstrap_ci accounts for patient grouping and produces ordered bounds."""
    # 20 samples from 10 patients (each patient has 2 records)
    patient_ids = np.repeat(np.arange(1, 11), 2)
    y_true = np.array([
        [1, 0], [1, 0],
        [0, 1], [0, 1],
        [1, 1], [1, 1],
        [0, 0], [0, 0],
        [1, 0], [1, 0],
        [0, 1], [0, 1],
        [1, 0], [1, 0],
        [0, 0], [0, 0],
        [1, 1], [1, 1],
        [0, 1], [0, 1],
    ])
    y_pred = y_true.copy()
    # Add minor noise to prediction
    y_pred[0] = [0, 0]
    y_pred[19] = [1, 0]

    ci = compute_patient_cluster_bootstrap_ci(
        y_true, y_pred, patient_ids=patient_ids, target_classes=["C0", "C1"], n_bootstraps=100, random_seed=42
    )

    assert ci["cluster_type"] == "patient_cluster"
    assert ci["n_unique_patients"] == 10
    macro = ci["macro_f1"]
    assert 0.0 <= macro["ci_lower"] <= macro["mean"] <= macro["ci_upper"] <= 1.0

    for sc in ["C0", "C1"]:
        c_res = ci["per_class"][sc]
        assert 0.0 <= c_res["ci_lower"] <= c_res["mean"] <= c_res["ci_upper"] <= 1.0


def test_verified_model_manifest_strict_schema(tmp_path: Path):
    """Test that load_verified_model strictly rejects manifests with missing required schema keys."""
    model_file = tmp_path / "dummy_model.joblib"
    model_file.write_bytes(b"dummy serialized bytes")
    manifest_file = tmp_path / "dummy_model.joblib.manifest.json"

    # Incomplete manifest (missing decision_thresholds, train_metadata)
    incomplete_manifest = {
        "model_file": "dummy_model.joblib",
        "model_sha256": "fake_sha",
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(incomplete_manifest, f)

    with pytest.raises(ValueError, match="Manifest schema violation"):
        load_verified_model(model_file)


def test_format_pilot_vs_full_fold_comparison_table():
    """Verify markdown formatting for the pilot vs full-fold comparison table."""
    pilot_metrics = {
        "model_test_metrics": {
            "macro_f1": 0.4583,
            "per_label": {
                "NORM": {"f1": 0.5882},
                "MI": {"f1": 0.3883},
                "STTC": {"f1": 0.4909},
                "CD": {"f1": 0.4933},
                "HYP": {"f1": 0.3305},
            },
        }
    }
    full_metrics = {
        "n_samples": 2198,
        "macro_f1": 0.5100,
        "macro_pr_auc": 0.5200,
        "per_label": {
            "NORM": {"support": 963, "f1": 0.6200, "precision": 0.72, "recall": 0.55, "pr_auc": 0.71},
            "MI": {"support": 550, "f1": 0.4200, "precision": 0.35, "recall": 0.53, "pr_auc": 0.38},
            "STTC": {"support": 521, "f1": 0.5300, "precision": 0.39, "recall": 0.80, "pr_auc": 0.48},
            "CD": {"support": 496, "f1": 0.5200, "precision": 0.46, "recall": 0.60, "pr_auc": 0.50},
            "HYP": {"support": 262, "f1": 0.3600, "precision": 0.31, "recall": 0.43, "pr_auc": 0.30},
        },
    }

    table = format_pilot_vs_full_fold_comparison_table(pilot_metrics, full_metrics)
    assert "### Training Scale Comparison" in table
    assert "| **NORM** |" in table
    assert "| **Macro Aggregate** |" in table
    assert "+0.0517" in table  # delta macro F1
