"""Unit tests for multi-label evaluation metrics, threshold tuning, and baseline predictors."""

import numpy as np
import pytest

from ecg_lab.waveform_ml.config import TARGET_SUPERCLASSES
from ecg_lab.waveform_ml.evaluate import evaluate_multilabel_predictions
from ecg_lab.waveform_ml.models import (
    PrevalenceBaselineClassifier,
    tune_validation_thresholds,
)


def test_evaluate_multilabel_metrics_exact_cases():
    """Test precision, recall, F1, and confusion matrix on deterministic labels."""
    # 4 samples, 2 classes
    y_true = np.array([
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 0],
    ])
    y_pred = np.array([
        [1, 0],
        [1, 0],  # FN for class 1
        [0, 1],
        [1, 0],  # FP for class 0
    ])
    y_probs = np.array([
        [0.9, 0.1],
        [0.8, 0.3],
        [0.2, 0.9],
        [0.6, 0.1],
    ])

    metrics = evaluate_multilabel_predictions(
        y_true, y_pred, y_probs, target_classes=["C0", "C1"], thresholds=[0.5, 0.5]
    )

    c0 = metrics["per_label"]["C0"]
    # C0: true positives: 2 (idx 0, 1), false positives: 1 (idx 3), false negatives: 0, true negatives: 1
    assert c0["support"] == 2
    assert c0["true_positives"] == 2
    assert c0["false_positives"] == 1
    assert c0["false_negatives"] == 0
    assert c0["true_negatives"] == 1
    assert pytest.approx(c0["precision"]) == 2 / 3
    assert pytest.approx(c0["recall"]) == 1.0
    assert pytest.approx(c0["f1"]) == (2 * (2/3) * 1.0) / (2/3 + 1.0)

    c1 = metrics["per_label"]["C1"]
    # C1: true positives: 1 (idx 2), false positives: 0, false negatives: 1 (idx 1), true negatives: 2
    assert c1["support"] == 2
    assert c1["true_positives"] == 1
    assert c1["false_positives"] == 0
    assert c1["false_negatives"] == 1
    assert c1["true_negatives"] == 2
    assert pytest.approx(c1["precision"]) == 1.0
    assert pytest.approx(c1["recall"]) == 0.5


def test_evaluate_multilabel_metrics_zero_support():
    """Test that classes with zero support in a split do not crash."""
    y_true = np.array([[0, 1], [0, 1]])
    y_pred = np.array([[0, 1], [0, 0]])

    metrics = evaluate_multilabel_predictions(
        y_true, y_pred, None, target_classes=["ZERO_SUPP", "HAS_SUPP"]
    )
    zs = metrics["per_label"]["ZERO_SUPP"]
    assert zs["support"] == 0
    assert zs["true_positives"] == 0
    assert zs["false_positives"] == 0
    assert zs["f1"] == 0.0


def test_prevalence_baseline_classifier():
    """Test that PrevalenceBaselineClassifier predicts empirical class frequencies."""
    X_train = np.zeros((10, 5))
    # 5 classes with prevalences 0.1, 0.2, 0.5, 0.8, 0.0
    y_train = np.array([
        [1, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])

    clf = PrevalenceBaselineClassifier()
    clf.fit(X_train, y_train)

    np.testing.assert_array_almost_equal(
        clf.prevalences_,
        [0.1, 0.2, 0.5, 0.8, 0.0]
    )

    X_test = np.zeros((3, 5))
    probs = clf.predict_proba(X_test)
    assert probs.shape == (3, 5)
    for row in probs:
        np.testing.assert_array_almost_equal(row, [0.1, 0.2, 0.5, 0.8, 0.0])


def test_tune_validation_thresholds():
    """Test that threshold tuning selects threshold uniquely maximizing F1."""
    y_true = np.array([[1], [1], [0], [0]])
    # Probabilities:
    # At t=0.25: pred=[1, 1, 1, 0], FP=1, F1=0.80
    # At t=0.35: pred=[1, 1, 0, 0], exact match, F1=1.00
    # At t=0.50: pred=[0, 0, 0, 0], F1=0.00
    y_probs = np.array([[0.40], [0.45], [0.30], [0.20]])

    thresholds = tune_validation_thresholds(y_true, y_probs, grid=np.array([0.25, 0.35, 0.50]))
    assert thresholds[0] == 0.35


def test_prevalence_baseline_mathematical_derivation():
    """Mathematically verify the exact derivation of NORM F1 = 38/49 = 0.7755 on 19/30 test support."""
    # Training set with 240 samples where NORM prevalence is 2/3 (0.6667) and other 4 classes are < 0.5
    X_train = np.zeros((240, 5))
    y_train = np.zeros((240, 5), dtype=int)
    y_train[:160, 0] = 1  # NORM: 160/240 = 0.6667
    y_train[:29, 1] = 1   # MI: 29/240 = 0.1208
    y_train[:34, 2] = 1   # STTC: 34/240 = 0.1417
    y_train[:37, 3] = 1   # CD: 37/240 = 0.1542
    y_train[:18, 4] = 1   # HYP: 18/240 = 0.0750

    clf = PrevalenceBaselineClassifier()
    clf.fit(X_train, y_train)

    # Assert training prevalences
    assert pytest.approx(clf.prevalences_[0]) == 160 / 240
    assert clf.prevalences_[0] >= 0.5
    for c in range(1, 5):
        assert clf.prevalences_[c] < 0.5

    # Test set with 30 samples where NORM has 19 positive cases
    X_test = np.zeros((30, 5))
    y_test = np.zeros((30, 5), dtype=int)
    y_test[:19, 0] = 1  # 19 positive NORM
    y_test[:3, 1] = 1   # 3 positive MI
    y_test[:5, 2] = 1   # 5 positive STTC
    y_test[:5, 3] = 1   # 5 positive CD
    y_test[:1, 4] = 1   # 1 positive HYP

    # Predictions at standard 0.5 threshold
    probs = clf.predict_proba(X_test)
    preds = clf.predict(X_test)

    # For NORM: prevalence 0.6667 >= 0.5 -> predicted positive for all 30 samples
    np.testing.assert_array_equal(preds[:, 0], np.ones(30))
    # For minority classes: prevalence < 0.5 -> predicted negative for all 30 samples
    for c in range(1, 5):
        np.testing.assert_array_equal(preds[:, c], np.zeros(30))

    metrics = evaluate_multilabel_predictions(y_test, preds, probs, target_classes=["NORM", "MI", "STTC", "CD", "HYP"])
    norm_m = metrics["per_label"]["NORM"]

    # Exact fractional proofs
    # TP = 19, FP = 11, FN = 0, TN = 0
    assert norm_m["true_positives"] == 19
    assert norm_m["false_positives"] == 11
    assert norm_m["false_negatives"] == 0
    assert norm_m["true_negatives"] == 0
    assert pytest.approx(norm_m["precision"]) == 19 / 30
    assert pytest.approx(norm_m["recall"]) == 19 / 19
    # F1 = 2 * (19/30) * 1 / (19/30 + 1) = 38 / 49 = 0.7755102...
    assert pytest.approx(norm_m["f1"]) == 38 / 49
    assert f"{norm_m['f1']:.4f}" == "0.7755"

    # Minority classes: TP = 0 -> F1 = 0.0
    for sc in ["MI", "STTC", "CD", "HYP"]:
        assert metrics["per_label"][sc]["f1"] == 0.0

    # Macro F1 = (38/49) / 5 = 38 / 245 = 0.155102...
    assert pytest.approx(metrics["macro_f1"]) == 38 / 245
    assert f"{metrics['macro_f1']:.4f}" == "0.1551"


def test_bootstrap_confidence_intervals_bounds():
    """Verify that bootstrap confidence intervals produce ordered bounds [lower <= mean <= upper]."""
    from ecg_lab.waveform_ml.evaluate import compute_bootstrap_confidence_intervals

    y_true = np.array([[1, 0], [1, 1], [0, 1], [0, 0]] * 10)
    y_pred = np.array([[1, 0], [1, 0], [0, 1], [0, 0]] * 10)

    ci = compute_bootstrap_confidence_intervals(
        y_true, y_pred, target_classes=["C0", "C1"], n_bootstraps=100, random_seed=42
    )

    assert "macro_f1" in ci
    macro = ci["macro_f1"]
    assert 0.0 <= macro["ci_lower"] <= macro["mean"] <= macro["ci_upper"] <= 1.0

    for sc in ["C0", "C1"]:
        c_data = ci["per_class"][sc]
        assert 0.0 <= c_data["ci_lower"] <= c_data["mean"] <= c_data["ci_upper"] <= 1.0


def test_format_ci_markdown_table():
    """Verify markdown table generation for bootstrap confidence intervals."""
    from ecg_lab.waveform_ml.evaluate import compute_bootstrap_confidence_intervals, format_ci_markdown_table

    y_true = np.array([[1, 0], [1, 1], [0, 1], [0, 0]] * 5)
    y_pred = np.array([[1, 0], [1, 0], [0, 1], [0, 0]] * 5)

    ci = compute_bootstrap_confidence_intervals(
        y_true, y_pred, target_classes=["C0", "C1"], n_bootstraps=50, random_seed=42
    )
    table = format_ci_markdown_table(ci, title="Test CI Table")
    assert "### Test CI Table" in table
    assert "| **C0** |" in table
    assert "| **C1** |" in table
    assert "95% Confidence Interval" in table
