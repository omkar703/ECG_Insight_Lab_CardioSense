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
