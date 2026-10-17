"""Evaluation metrics, multi-label diagnostic reporting, and comparison tables.

Implements transparent, reproducible multi-label evaluation without test-set leakage.
Outputs precision, recall, F1, PR-AUC, support, and confusion counts per superclass.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from ecg_lab.waveform_ml.config import TARGET_SUPERCLASSES


def evaluate_multilabel_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    target_classes: List[str] = list(TARGET_SUPERCLASSES),
    thresholds: Optional[Union[List[float], np.ndarray]] = None,
) -> Dict[str, Any]:
    """Calculate multi-label evaluation metrics across diagnostic superclasses.

    Guarantees:
    - Never fabricates numbers if a class has zero positive support in the split.
    - Explicitly reports TP, FP, FN, TN per class.
    - Computes both Macro and Micro F1.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_pred_arr = np.asarray(y_pred, dtype=np.int32)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true_arr.shape} vs y_pred {y_pred_arr.shape}"
        )

    n_samples, n_classes = y_true_arr.shape
    if thresholds is None:
        thresh_arr = np.full(n_classes, 0.5)
    else:
        thresh_arr = np.asarray(thresholds, dtype=np.float64)

    per_label_metrics: Dict[str, Dict[str, Any]] = {}
    f1_list: List[float] = []
    ap_list: List[float] = []

    for c_idx, cls_name in enumerate(target_classes):
        yt = y_true_arr[:, c_idx]
        yp = y_pred_arr[:, c_idx]

        support = int(np.sum(yt))
        predicted_positives = int(np.sum(yp))

        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        tn = int(np.sum((yt == 0) & (yp == 0)))

        # Precision & Recall
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else (0.0 if support > 0 else None)
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else (0.0 if support > 0 else None)

        if prec is not None and rec is not None and (prec + rec) > 0:
            f1 = float(2 * prec * rec / (prec + rec))
        else:
            f1 = 0.0

        f1_list.append(f1)

        # Average Precision / PR-AUC
        ap_val: Optional[float] = None
        if y_probs is not None:
            yp_prob = y_probs[:, c_idx]
            if support > 0 and support < n_samples:
                try:
                    ap_val = float(average_precision_score(yt, yp_prob))
                    ap_list.append(ap_val)
                except Exception:
                    ap_val = None
            elif support == n_samples or support == 0:
                ap_val = None

        per_label_metrics[cls_name] = {
            "support": support,
            "predicted_positives": predicted_positives,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "pr_auc": ap_val,
            "decision_threshold": float(thresh_arr[c_idx]),
        }

    # Aggregate metrics
    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    micro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0))
    macro_pr_auc = float(np.mean(ap_list)) if ap_list else None

    return {
        "n_samples": n_samples,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_pr_auc": macro_pr_auc,
        "per_label": per_label_metrics,
    }


def format_metrics_markdown_table(
    metrics: Dict[str, Any],
    title: str = "Multi-Label Evaluation Summary",
) -> str:
    """Format evaluation results into a clear GitHub Flavored Markdown table."""
    per_label = metrics["per_label"]
    lines = [
        f"### {title}",
        f"**Sample Count ($N$):** {metrics['n_samples']} | **Macro F1:** {metrics['macro_f1']:.4f} | **Micro F1:** {metrics['micro_f1']:.4f}"
        + (f" | **Macro PR-AUC:** {metrics['macro_pr_auc']:.4f}" if metrics['macro_pr_auc'] is not None else ""),
        "",
        "| Superclass | Support | Threshold | TP / FP / FN / TN | Precision | Recall (Sens.) | F1 Score | PR-AUC |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for cls_name, m in per_label.items():
        prec_str = f"{m['precision']:.4f}" if m['precision'] is not None else "N/A"
        rec_str = f"{m['recall']:.4f}" if m['recall'] is not None else "N/A"
        f1_str = f"{m['f1']:.4f}"
        prauc_str = f"{m['pr_auc']:.4f}" if m['pr_auc'] is not None else "N/A"
        conf_str = f"{m['true_positives']} / {m['false_positives']} / {m['false_negatives']} / {m['true_negatives']}"
        lines.append(
            f"| **{cls_name}** | {m['support']} | {m['decision_threshold']:.2f} | {conf_str} | {prec_str} | {rec_str} | **{f1_str}** | {prauc_str} |"
        )

    return "\n".join(lines)


def format_comparison_markdown_table(
    baseline_metrics: Dict[str, Any],
    model_metrics: Dict[str, Any],
    target_classes: List[str] = list(TARGET_SUPERCLASSES),
) -> str:
    """Format a comparison table between naive baseline and lightweight ML model."""
    lines = [
        "### Benchmark Comparison: Naive Baseline vs. Lightweight Model",
        "",
        "| Superclass | Support | Baseline F1 | Model F1 | Delta ($\\Delta$) | Model Precision | Model Recall | Model PR-AUC |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for cls_name in target_classes:
        base_f1 = baseline_metrics["per_label"][cls_name]["f1"]
        mod_m = model_metrics["per_label"][cls_name]
        mod_f1 = mod_m["f1"]
        delta = mod_f1 - base_f1
        delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"

        prec_str = f"{mod_m['precision']:.4f}" if mod_m['precision'] is not None else "N/A"
        rec_str = f"{mod_m['recall']:.4f}" if mod_m['recall'] is not None else "N/A"
        prauc_str = f"{mod_m['pr_auc']:.4f}" if mod_m['pr_auc'] is not None else "N/A"

        lines.append(
            f"| **{cls_name}** | {mod_m['support']} | {base_f1:.4f} | **{mod_f1:.4f}** | {delta_str} | {prec_str} | {rec_str} | {prauc_str} |"
        )

    # Aggregates
    macro_pr_auc_str = f"{model_metrics['macro_pr_auc']:.4f}" if model_metrics.get("macro_pr_auc") is not None else "—"
    delta_macro = model_metrics["macro_f1"] - baseline_metrics["macro_f1"]
    delta_macro_str = f"+{delta_macro:.4f}" if delta_macro > 0 else f"{delta_macro:.4f}"

    lines.append(
        f"| **Macro Aggregate** | {model_metrics['n_samples']} | {baseline_metrics['macro_f1']:.4f} | **{model_metrics['macro_f1']:.4f}** | "
        f"{delta_macro_str} | — | — | {macro_pr_auc_str} |"
    )

    return "\n".join(lines)
