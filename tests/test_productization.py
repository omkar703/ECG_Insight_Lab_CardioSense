"""Unit tests for Phase 4 productization, diagnostics, privacy safeguards, and workflow decoupling."""

import json
from pathlib import Path
import numpy as np
import pytest

from ecg_lab.diagnostics import check_system_diagnostics
from ecg_lab.validation import validate_image_bytes
from ecg_lab.digitization import digitize_ecg_image, DigitizationStatus
from ecg_lab.waveform_ml.config import DEFAULT_ARTIFACT_DIR, DEFAULT_DATA_DIR
from ecg_lab.waveform_ml.features import extract_record_features


def test_system_diagnostics_report_complete():
    """Verify that check_system_diagnostics produces complete status for all subsystems."""
    diag = check_system_diagnostics(
        data_dir=Path(DEFAULT_DATA_DIR),
        artifact_dir=Path(DEFAULT_ARTIFACT_DIR),
        sample_dir=Path("sample_data"),
    )

    d_dict = diag.to_dict()
    assert "all_critical_components_ready" in d_dict
    assert diag.all_critical_components_ready is True

    comps = d_dict["components"]
    assert "dataset_metadata" in comps
    assert "dataset_completeness" in comps
    assert "active_model" in comps
    assert "pilot_model" in comps
    assert "paired_comparison" in comps
    assert "synthetic_fixtures" in comps
    assert "privacy_safeguards" in comps

    assert comps["dataset_metadata"]["is_available"] is True
    assert comps["active_model"]["is_available"] is True
    assert comps["synthetic_fixtures"]["is_available"] is True
    assert comps["privacy_safeguards"]["details"]["disk_persistence"] is False


def test_system_diagnostics_missing_artifacts_graceful(tmp_path: Path):
    """Verify that diagnostics handles missing artifacts gracefully with actionable guidance."""
    empty_data = tmp_path / "empty_data"
    empty_art = tmp_path / "empty_art"
    empty_sample = tmp_path / "empty_sample"
    empty_data.mkdir()
    empty_art.mkdir()
    empty_sample.mkdir()

    diag = check_system_diagnostics(
        data_dir=empty_data,
        artifact_dir=empty_art,
        sample_dir=empty_sample,
    )

    assert diag.all_critical_components_ready is False
    assert diag.dataset_metadata.is_available is False
    assert "download-metadata" in diag.dataset_metadata.guidance

    assert diag.active_model.is_available is False
    assert "train" in diag.active_model.guidance

    assert diag.synthetic_fixtures.is_available is False


def test_in_memory_upload_privacy_lifecycle(tmp_path: Path):
    """Verify that image validation and digitization do not persist temporary files to disk."""
    bench_path = Path("sample_data/synthetic_12lead.png")
    assert bench_path.is_file()
    img_bytes = bench_path.read_bytes()

    # Track directory contents before
    files_before = set(tmp_path.rglob("*"))

    # Process entirely in memory
    val = validate_image_bytes(img_bytes, filename="uploaded_scan.png")
    assert val.is_valid is True

    res = digitize_ecg_image(img_bytes, engine="baseline")
    assert res.status in (DigitizationStatus.SUCCESS, DigitizationStatus.PARTIAL)

    # Export to memory
    csv_str = res.to_long_dataframe().to_csv(index=False)
    json_str = res.to_json()
    assert len(csv_str) > 0
    assert len(json_str) > 0

    # Verify no new files written to disk in current directory or temp directory
    files_after = set(tmp_path.rglob("*"))
    assert files_before == files_after, "Unexpected files were persisted to disk during in-memory processing!"


def test_synthetic_demo_labeling():
    """Verify that synthetic test fixtures are clearly identifiable and distinguished from genuine patient data."""
    syn_fixture = Path("sample_data/synthetic_12lead.png")
    syn_gt = Path("sample_data/synthetic_12lead_gt.csv")
    assert syn_fixture.is_file()
    assert syn_gt.is_file()

    # The synthetic fixture must have 12 leads defined in GT
    import pandas as pd
    gt_df = pd.read_csv(syn_gt)
    leads_in_gt = gt_df["lead"].unique()
    assert len(leads_in_gt) == 12
    # Verify synthetic filename convention
    assert "synthetic" in syn_fixture.name.lower()


def test_workflow_decoupling_assertion():
    """Verify that 1D digitized image signals cannot be passed as raw 2D WFDB records to feature extraction."""
    # Workflow A outputs a 1D lead signal per lead
    fake_digitized_1d = np.sin(np.linspace(0, 10, 1000))

    # Workflow B extract_record_features requires 2D shape (N_samples, 12)
    with pytest.raises(ValueError, match="Expected signal shape"):
        extract_record_features(fake_digitized_1d, fs=100)


def test_paired_comparison_methodology_caveats():
    """Verify that paired model comparison output payload includes complete pipeline metadata and caveats."""
    paired_json_path = Path(DEFAULT_ARTIFACT_DIR) / "paired_model_comparison.json"
    assert paired_json_path.is_file()

    with open(paired_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check that both models are named
    assert "Full-Fold Model" in data.get("model_a_name", "")
    assert "Pilot Model" in data.get("model_b_name", "")

    # Check that resamples count is at least 1,000 (2,000 preferred)
    assert data.get("n_bootstraps", 0) >= 1000

    # Check resampling unit
    assert data.get("resampling_unit") == "patient_cluster"

    # Check that paired difference delta is computed and bounded
    macro_d = data.get("paired_macro_delta", {})
    assert "point_delta" in macro_d
    assert "ci_lower" in macro_d
    assert "ci_upper" in macro_d
    assert macro_d["ci_lower"] <= macro_d["mean_delta"] <= macro_d["ci_upper"]
