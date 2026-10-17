"""Automated unit test suite for the ECG digitization subsystem (Phase 2)."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from io import BytesIO
from scipy.stats import pearsonr

from ecg_lab.digitization.types import (
    DigitizationStatus,
    CalibrationStatus,
    AmplitudeUnit,
    DigitizationResult,
)
from ecg_lab.digitization.service import (
    digitize_ecg_image,
    export_full_csv,
    export_metadata_json,
)
from ecg_lab.digitization.adapter import OpenECGDigitizerAdapter


SYNTHETIC_IMG = Path("sample_data/synthetic_12lead.png")
SYNTHETIC_GT = Path("sample_data/synthetic_12lead_gt.csv")
SYNTHETIC_META = Path("sample_data/synthetic_12lead_meta.json")


def test_synthetic_fixture_files_exist():
    """Verify that deterministic benchmark fixtures are present."""
    assert SYNTHETIC_IMG.is_file(), "Synthetic 12-lead image must exist."
    assert SYNTHETIC_GT.is_file(), "Ground-truth CSV must exist."
    assert SYNTHETIC_META.is_file(), "Metadata JSON must exist."


def test_digitization_schema_on_synthetic():
    """Verify full schema compliance and successful trace recovery on benchmark."""
    result = digitize_ecg_image(SYNTHETIC_IMG)

    assert isinstance(result, DigitizationResult)
    assert result.status == DigitizationStatus.SUCCESS
    assert result.method == "Classical Morphological Baseline"
    assert result.calibration_status == CalibrationStatus.UNRECOVERED
    assert result.amplitude_unit == AmplitudeUnit.RELATIVE
    assert result.lead_count == 12
    assert len(result.detected_leads) == 12
    assert len(result.missing_leads) == 0
    assert result.runtime_ms > 0

    # Inspect Lead I structure
    lead_i = result.leads["I"]
    assert lead_i.lead_name == "I"
    assert lead_i.sample_count == 255
    assert lead_i.is_valid is True
    assert lead_i.confidence >= 0.5
    assert lead_i.time_seconds is None  # Uncalibrated timing must be None


def test_synthetic_accuracy_against_ground_truth():
    """Measure empirical recovery accuracy (Pearson correlation and MAE) against known ground truth."""
    result = digitize_ecg_image(SYNTHETIC_IMG)
    gt_df = pd.read_csv(SYNTHETIC_GT)

    correlations = []
    maes = []

    for lead in result.detected_leads:
        sig = np.array(result.leads[lead].samples)
        gt = gt_df[gt_df["lead"] == lead]["amplitude"].values

        assert len(sig) == len(gt), f"Sample length mismatch for lead {lead}"
        r, _ = pearsonr(sig, gt)
        mae = float(np.mean(np.abs(sig - gt)))

        correlations.append(r)
        maes.append(mae)

    # Across all 12 leads, the average correlation must exceed 0.60 on this synthetic pattern
    avg_r = float(np.mean(correlations))
    avg_mae = float(np.mean(maes))

    assert avg_r > 0.60, f"Expected average correlation > 0.60, got {avg_r:.4f}"
    assert avg_mae < 0.15, f"Expected average MAE < 0.15, got {avg_mae:.4f}"


def test_zero_contrast_blank_image_rejected():
    """Ensure flat or solid-colored images are cleanly rejected without fabricating signals."""
    blank = Image.new("RGB", (1200, 800), color="white")
    buf = BytesIO()
    blank.save(buf, format="JPEG")

    result = digitize_ecg_image(buf.getvalue())
    assert result.status == DigitizationStatus.REJECTED
    assert result.lead_count == 0
    assert "contrast" in result.error_message.lower() or "trace" in result.error_message.lower()


def test_invalid_and_corrupt_inputs_rejected(tmp_path):
    """Ensure invalid formats, corrupt files, and missing paths fail safely."""
    # 1. Non-existent file
    res1 = digitize_ecg_image("does_not_exist_999.png")
    assert res1.status == DigitizationStatus.REJECTED

    # 2. Unsupported extension
    bad_ext = tmp_path / "notes.txt"
    bad_ext.write_text("Hello world")
    res2 = digitize_ecg_image(bad_ext)
    assert res2.status == DigitizationStatus.REJECTED

    # 3. Corrupted image bytes
    res3 = digitize_ecg_image(b"GARBAGE_BYTES_CORRUPTED" * 10)
    assert res3.status == DigitizationStatus.REJECTED


def test_upstream_adapter_unavailable_graceful():
    """Ensure requesting the upstream adapter when weights are absent yields UNAVAILABLE."""
    adapter = OpenECGDigitizerAdapter(repo_path=Path("non_existent_path"))
    available, reason = adapter.is_available()
    assert available is False

    result = digitize_ecg_image(SYNTHETIC_IMG, engine="open-ecg")
    assert result.status == DigitizationStatus.UNAVAILABLE
    assert "Open-ECG-Digitizer" in result.method
    assert len(result.warnings) > 0


def test_unit_and_calibration_honesty():
    """Verify that uncalibrated outputs never claim physical units or calibrated timing."""
    result = digitize_ecg_image(SYNTHETIC_IMG)
    assert result.calibration_status == CalibrationStatus.UNRECOVERED
    assert result.amplitude_unit == AmplitudeUnit.RELATIVE

    # Ensure no lead falsely reports time_seconds without calibration
    for lead in result.leads.values():
        assert lead.time_seconds is None


def test_long_form_csv_export(tmp_path):
    """Verify long-form CSV serialization and column schema."""
    result = digitize_ecg_image(SYNTHETIC_IMG)
    out_file = tmp_path / "digitized_signals.csv"

    export_full_csv(result, out_file, format_type="long")
    assert out_file.is_file()

    df = pd.read_csv(out_file)
    expected_cols = ["lead", "sample_index", "time_seconds", "amplitude", "amplitude_unit", "calibration_status", "method"]
    for col in expected_cols:
        assert col in df.columns

    assert len(df) == 12 * 255
    assert len(df["lead"].unique()) == 12


def test_wide_form_csv_export(tmp_path):
    """Verify wide-form CSV export where each lead is a distinct column."""
    result = digitize_ecg_image(SYNTHETIC_IMG)
    out_file = tmp_path / "wide_signals.csv"

    export_full_csv(result, out_file, format_type="wide")
    assert out_file.is_file()

    df = pd.read_csv(out_file)
    assert len(df) == 255
    assert "sample_index" in df.columns
    for lead_name in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]:
        assert lead_name in df.columns


def test_json_metadata_export(tmp_path):
    """Verify JSON metadata sidecar export."""
    result = digitize_ecg_image(SYNTHETIC_IMG)
    json_file = tmp_path / "metadata.json"

    export_metadata_json(result, json_file)
    assert json_file.is_file()

    with open(json_file) as f:
        data = json.load(f)

    assert data["status"] == "SUCCESS"
    assert data["lead_count"] == 12
    assert "leads" in data
    assert "I" in data["leads"]
    assert len(data["leads"]["I"]["samples"]) == 255


def test_custom_samples_per_lead():
    """Verify configurable sampling resolution per lead."""
    result = digitize_ecg_image(SYNTHETIC_IMG, options={"samples_per_lead": 128})
    assert result.status == DigitizationStatus.SUCCESS
    assert result.leads["I"].sample_count == 128
    long_df = result.to_long_dataframe()
    assert len(long_df) == 12 * 128
