"""Automated test verifying the reproducible artifacts and metrics of the controlled imbalance experiment."""

import json
from pathlib import Path
import pytest

EXPERIMENT_DIR = Path("artifacts/single_lead_ml/experiments/class_imbalance")
JSON_PATH = EXPERIMENT_DIR / "imbalance_experiment_results.json"
REPORT_PATH = EXPERIMENT_DIR / "imbalance_experiment_report.md"


def test_imbalance_experiment_artifacts_exist():
    """Verify that experiment outputs and documentation exist."""
    assert JSON_PATH.is_file(), f"Experiment JSON results missing: {JSON_PATH}"
    assert REPORT_PATH.is_file(), f"Experiment markdown report missing: {REPORT_PATH}"


def test_imbalance_experiment_protocol_and_baseline_integrity():
    """Verify that experiment adheres to rigorous protocol and reproduces production baseline."""
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    # Protocol validation
    protocol = data["protocol"]
    assert protocol["n_train_samples"] == 37419
    assert protocol["n_val_samples"] == 15725
    assert protocol["n_test_samples"] == 47561
    assert protocol["feature_count"] == 47
    assert protocol["classes"] == ["N", "S", "V", "Q"]

    # Baseline on test must reproduce frozen production model
    test_eval = data["test_evaluation"]
    baseline = test_eval["Baseline (Balanced RF)"]
    assert abs(baseline["macro_f1"] - 0.5396) < 1e-3
    assert abs(baseline["weighted_f1"] - 0.9036) < 1e-3

    # All candidate models must have evaluated all 4 target classes
    for name in [
        "Baseline (Balanced RF)",
        "Candidate 1 (Balanced Subsample RF)",
        "Candidate 2 (Balanced Logistic Regression)",
        "Candidate 3 (Minority-Boosted Cost-Sensitive RF)",
    ]:
        assert name in test_eval
        assert set(test_eval[name]["per_class"].keys()) == {"N", "S", "V", "Q"}


def test_imbalance_markdown_report_content():
    """Verify that the markdown report contains expected comparison sections."""
    content = REPORT_PATH.read_text(encoding="utf-8")
    assert "## 1. Research Question & Hypothesis" in content
    assert "## 3. Validation Set Model Selection" in content
    assert "## 4. Held-Out Test Evaluation" in content
    assert "## 5. Detailed Head-to-Head Comparison" in content
    assert "## 6. Key Scientific Findings & Trade-Offs" in content
