"""Evaluation utilities for single-lead ECG beat classification."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

from ecg_lab.single_lead_ml.config import AAMI_CLASS_DESCRIPTIONS, AAMI_TARGET_CLASSES


def compute_beat_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_classes: Tuple[str, ...] = AAMI_TARGET_CLASSES,
) -> Dict[str, Any]:
    """Compute per-class and aggregate classification metrics.

    Args:
        y_true: 1D array of ground truth AAMI class strings.
        y_pred: 1D array of predicted AAMI class strings.
        target_classes: Sequence of expected classes ('N', 'S', 'V', 'Q').

    Returns:
        Structured dictionary of evaluation metrics.
    """
    classes_list = list(target_classes)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes_list)

    # Per-class metrics
    per_class: Dict[str, Dict[str, Any]] = {}
    for i, cls_name in enumerate(classes_list):
        cls_true = (y_true == cls_name)
        cls_pred = (y_pred == cls_name)

        tp = int(np.sum(cls_true & cls_pred))
        fp = int(np.sum((~cls_true) & cls_pred))
        fn = int(np.sum(cls_true & (~cls_pred)))
        tn = int(np.sum((~cls_true) & (~cls_pred)))
        support = int(np.sum(cls_true))

        prec = float(precision_score(cls_true, cls_pred, zero_division=0))
        rec = float(recall_score(cls_true, cls_pred, zero_division=0))
        f1 = float(f1_score(cls_true, cls_pred, zero_division=0))

        per_class[cls_name] = {
            "description": AAMI_CLASS_DESCRIPTIONS.get(cls_name, cls_name),
            "support": support,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
        }

    macro_f1 = float(f1_score(y_true, y_pred, labels=classes_list, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=classes_list, average="weighted", zero_division=0))

    return {
        "target_classes": classes_list,
        "n_samples": int(len(y_true)),
        "confusion_matrix": cm.tolist(),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }
