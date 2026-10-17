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


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_classes: List[str] = list(TARGET_SUPERCLASSES),
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Calculate non-parametric bootstrap confidence intervals for per-class and macro F1."""
    rng = np.random.RandomState(random_seed)
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_pred_arr = np.asarray(y_pred, dtype=np.int32)
    n_samples = len(y_true_arr)
    alpha = (1.0 - confidence_level) / 2.0

    boot_per_class: Dict[str, List[float]] = {sc: [] for sc in target_classes}
    boot_macro: List[float] = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        yt_boot = y_true_arr[idx]
        yp_boot = y_pred_arr[idx]

        class_f1s = []
        for c_idx, sc in enumerate(target_classes):
            f = float(f1_score(yt_boot[:, c_idx], yp_boot[:, c_idx], zero_division=0))
            boot_per_class[sc].append(f)
            class_f1s.append(f)
        boot_macro.append(float(np.mean(class_f1s)))

    ci_results: Dict[str, Any] = {
        "n_bootstraps": n_bootstraps,
        "confidence_level": confidence_level,
        "macro_f1": {
            "mean": float(np.mean(boot_macro)),
            "ci_lower": float(np.percentile(boot_macro, 100.0 * alpha)),
            "ci_upper": float(np.percentile(boot_macro, 100.0 * (1.0 - alpha))),
        },
        "per_class": {},
    }

    for sc in target_classes:
        scores = boot_per_class[sc]
        ci_results["per_class"][sc] = {
            "mean": float(np.mean(scores)),
            "ci_lower": float(np.percentile(scores, 100.0 * alpha)),
            "ci_upper": float(np.percentile(scores, 100.0 * (1.0 - alpha))),
        }

    return ci_results


def compute_patient_cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: Union[np.ndarray, List[Any]],
    target_classes: List[str] = list(TARGET_SUPERCLASSES),
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Calculate patient-clustered bootstrap confidence intervals for per-class and macro F1.

    Samples unique patient IDs with replacement and aggregates all recordings
    belonging to sampled patients in each bootstrap replication. Properly accounts
    for intra-patient correlation across repeated ECG recordings.
    """
    rng = np.random.RandomState(random_seed)
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_pred_arr = np.asarray(y_pred, dtype=np.int32)
    pts_arr = np.asarray(patient_ids)

    unique_patients = np.unique(pts_arr)
    n_unique_patients = len(unique_patients)
    alpha = (1.0 - confidence_level) / 2.0

    # Build patient-to-indices lookup for fast cluster retrieval
    patient_to_indices: Dict[Any, np.ndarray] = {}
    for idx, p_id in enumerate(pts_arr):
        if p_id not in patient_to_indices:
            patient_to_indices[p_id] = []
        patient_to_indices[p_id].append(idx)
    for p_id in patient_to_indices:
        patient_to_indices[p_id] = np.array(patient_to_indices[p_id], dtype=np.int64)

    boot_per_class: Dict[str, List[float]] = {sc: [] for sc in target_classes}
    boot_macro: List[float] = []

    for _ in range(n_bootstraps):
        sampled_pts = rng.choice(unique_patients, size=n_unique_patients, replace=True)
        # Gather all record indices for the sampled patient clusters
        sampled_indices_list = [patient_to_indices[p] for p in sampled_pts]
        sampled_indices = np.concatenate(sampled_indices_list)

        yt_boot = y_true_arr[sampled_indices]
        yp_boot = y_pred_arr[sampled_indices]

        class_f1s = []
        for c_idx, sc in enumerate(target_classes):
            f = float(f1_score(yt_boot[:, c_idx], yp_boot[:, c_idx], zero_division=0))
            boot_per_class[sc].append(f)
            class_f1s.append(f)
        boot_macro.append(float(np.mean(class_f1s)))

    ci_results: Dict[str, Any] = {
        "n_bootstraps": n_bootstraps,
        "n_unique_patients": int(n_unique_patients),
        "cluster_type": "patient_cluster",
        "confidence_level": confidence_level,
        "macro_f1": {
            "mean": float(np.mean(boot_macro)),
            "ci_lower": float(np.percentile(boot_macro, 100.0 * alpha)),
            "ci_upper": float(np.percentile(boot_macro, 100.0 * (1.0 - alpha))),
        },
        "per_class": {},
    }

    for sc in target_classes:
        scores = boot_per_class[sc]
        ci_results["per_class"][sc] = {
            "mean": float(np.mean(scores)),
            "ci_lower": float(np.percentile(scores, 100.0 * alpha)),
            "ci_upper": float(np.percentile(scores, 100.0 * (1.0 - alpha))),
        }

    return ci_results


def format_ci_markdown_table(
    ci_results: Dict[str, Any],
    title: str = "95% Bootstrap Confidence Intervals (Fold 10)",
) -> str:
    """Format bootstrap confidence interval results into a clear GitHub Flavored Markdown table."""
    c_type = ci_results.get("cluster_type", "record_bootstrap")
    header_info = (
        f"**Bootstrap Resamples:** {ci_results['n_bootstraps']} ({c_type}, {ci_results.get('n_unique_patients', 'N/A')} patients) | "
        f"**Macro F1 (95% CI):** {ci_results['macro_f1']['mean']:.4f} "
        f"[{ci_results['macro_f1']['ci_lower']:.4f}, {ci_results['macro_f1']['ci_upper']:.4f}]"
    )
    lines = [
        f"### {title}",
        header_info,
        "",
        "| Superclass | Bootstrap Mean F1 | 95% Confidence Interval | CI Width |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for sc, data in ci_results["per_class"].items():
        width = data["ci_upper"] - data["ci_lower"]
        lines.append(
            f"| **{sc}** | {data['mean']:.4f} | [{data['ci_lower']:.4f}, {data['ci_upper']:.4f}] | {width:.4f} |"
        )
    return "\n".join(lines)


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


def format_pilot_vs_full_fold_comparison_table(
    pilot_metrics: Optional[Dict[str, Any]],
    full_fold_metrics: Dict[str, Any],
    target_classes: List[str] = list(TARGET_SUPERCLASSES),
) -> str:
    """Format comparison between Phase 3.1 Pilot Model (N_tr=240) and Phase 3.2 Full-Fold Model (N_tr=17,418)."""
    lines = [
        "### Training Scale Comparison: Phase 3.1 Pilot Model vs. Phase 3.2 Full-Fold Model (Fold 10)",
        "",
        "> [!NOTE]",
        "> **Methodological Disclosure:** The Phase 3.1 pilot model was trained on 240 records (30 per fold, Folds 1–8) "
        "and thresholded on 30 records (Fold 9). The Phase 3.2 model is trained on the full permitted training partition "
        "and thresholded on the full validation partition. Both models are evaluated on the exact same Fold 10 test cohort.",
        "",
        "| Superclass | Test Support | Pilot Model F1 (N_tr=240) | Full-Fold Model F1 | Delta ($\\Delta\\text{ F1}$) | Full Precision | Full Recall | Full PR-AUC |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    if not pilot_metrics:
        lines.append("| — | — | N/A | — | — | — | — | — |")
        return "\n".join(lines)

    p_per = pilot_metrics.get("model_test_metrics", {}).get("per_label", {})
    f_per = full_fold_metrics["per_label"]

    for sc in target_classes:
        p_f1 = p_per.get(sc, {}).get("f1", 0.0)
        f_m = f_per[sc]
        f_f1 = f_m["f1"]
        delta = f_f1 - p_f1
        delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
        prec_str = f"{f_m['precision']:.4f}" if f_m['precision'] is not None else "N/A"
        rec_str = f"{f_m['recall']:.4f}" if f_m['recall'] is not None else "N/A"
        prauc_str = f"{f_m['pr_auc']:.4f}" if f_m['pr_auc'] is not None else "N/A"

        lines.append(
            f"| **{sc}** | {f_m['support']} | {p_f1:.4f} | **{f_f1:.4f}** | {delta_str} | {prec_str} | {rec_str} | {prauc_str} |"
        )

    p_macro = pilot_metrics.get("model_test_metrics", {}).get("macro_f1", 0.0)
    f_macro = full_fold_metrics["macro_f1"]
    delta_macro = f_macro - p_macro
    delta_macro_str = f"+{delta_macro:.4f}" if delta_macro > 0 else f"{delta_macro:.4f}"
    prauc_macro = f"{full_fold_metrics['macro_pr_auc']:.4f}" if full_fold_metrics.get("macro_pr_auc") is not None else "—"

    lines.append(
        f"| **Macro Aggregate** | {full_fold_metrics['n_samples']} | {p_macro:.4f} | **{f_macro:.4f}** | "
        f"{delta_macro_str} | — | — | {prauc_macro} |"
    )

    return "\n".join(lines)


def compute_paired_patient_cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    patient_ids: Union[np.ndarray, List[Any]],
    target_classes: List[str] = list(TARGET_SUPERCLASSES),
    n_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    model_a_name: str = "Full-Fold Model (N_tr=17,418)",
    model_b_name: str = "Pilot Model (N_tr=240)",
) -> Dict[str, Any]:
    """Calculate paired patient-clustered bootstrap confidence intervals for model comparison.

    Resamples unique patient clusters with replacement (preserving intra-patient
    correlation) and evaluates both models simultaneously on the exact same patient
    cluster draws. Computes paired differences (Model A - Model B) across all replicates.

    Returns point estimates, percentile 95% CIs, bootstrap SE, and empirical p-values.
    """
    rng = np.random.RandomState(random_seed)
    yt_arr = np.asarray(y_true, dtype=np.int32)
    ya_arr = np.asarray(y_pred_a, dtype=np.int32)
    yb_arr = np.asarray(y_pred_b, dtype=np.int32)
    pts_arr = np.asarray(patient_ids)

    n_samples, n_classes = yt_arr.shape
    unique_patients = np.unique(pts_arr)
    n_unique_patients = len(unique_patients)
    alpha = (1.0 - confidence_level) / 2.0

    # Build patient-to-indices lookup for fast cluster retrieval
    patient_to_indices: Dict[Any, np.ndarray] = {}
    for idx, p_id in enumerate(pts_arr):
        if p_id not in patient_to_indices:
            patient_to_indices[p_id] = []
        patient_to_indices[p_id].append(idx)
    for p_id in patient_to_indices:
        patient_to_indices[p_id] = np.array(patient_to_indices[p_id], dtype=np.int64)

    # Point estimates on full cohort
    def _compute_f1_vector(yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
        tp = np.sum((yt == 1) & (yp == 1), axis=0)
        denom = np.sum(yt, axis=0) + np.sum(yp, axis=0)
        return np.where(denom > 0, 2.0 * tp / denom, 0.0)

    f1_a_point = _compute_f1_vector(yt_arr, ya_arr)
    f1_b_point = _compute_f1_vector(yt_arr, yb_arr)
    macro_a_point = float(np.mean(f1_a_point))
    macro_b_point = float(np.mean(f1_b_point))
    delta_macro_point = float(macro_a_point - macro_b_point)
    delta_class_point = {
        sc: float(f1_a_point[c] - f1_b_point[c]) for c, sc in enumerate(target_classes)
    }

    # Replicate tracking
    boot_delta_macro: List[float] = []
    boot_macro_a: List[float] = []
    boot_macro_b: List[float] = []
    boot_delta_per_class: Dict[str, List[float]] = {sc: [] for sc in target_classes}
    boot_class_a: Dict[str, List[float]] = {sc: [] for sc in target_classes}
    boot_class_b: Dict[str, List[float]] = {sc: [] for sc in target_classes}

    for _ in range(n_bootstraps):
        sampled_pts = rng.choice(unique_patients, size=n_unique_patients, replace=True)
        sampled_indices = np.concatenate([patient_to_indices[p] for p in sampled_pts])

        yt_boot = yt_arr[sampled_indices]
        ya_boot = ya_arr[sampled_indices]
        yb_boot = yb_arr[sampled_indices]

        f1_a_boot = _compute_f1_vector(yt_boot, ya_boot)
        f1_b_boot = _compute_f1_vector(yt_boot, yb_boot)

        m_a = float(np.mean(f1_a_boot))
        m_b = float(np.mean(f1_b_boot))
        d_m = float(m_a - m_b)

        boot_macro_a.append(m_a)
        boot_macro_b.append(m_b)
        boot_delta_macro.append(d_m)

        for c, sc in enumerate(target_classes):
            boot_class_a[sc].append(float(f1_a_boot[c]))
            boot_class_b[sc].append(float(f1_b_boot[c]))
            boot_delta_per_class[sc].append(float(f1_a_boot[c] - f1_b_boot[c]))

    delta_macro_arr = np.array(boot_delta_macro)
    se_delta_macro = float(np.std(delta_macro_arr, ddof=1))
    ci_lower_macro = float(np.percentile(delta_macro_arr, 100.0 * alpha))
    ci_upper_macro = float(np.percentile(delta_macro_arr, 100.0 * (1.0 - alpha)))
    p_val_nonpos = float(np.mean(delta_macro_arr <= 0.0))

    per_class_summary: Dict[str, Dict[str, Any]] = {}
    for sc in target_classes:
        d_arr = np.array(boot_delta_per_class[sc])
        ci_l = float(np.percentile(d_arr, 100.0 * alpha))
        ci_u = float(np.percentile(d_arr, 100.0 * (1.0 - alpha)))
        per_class_summary[sc] = {
            "point_f1_a": float(f1_a_point[target_classes.index(sc)]),
            "point_f1_b": float(f1_b_point[target_classes.index(sc)]),
            "point_delta": delta_class_point[sc],
            "mean_delta": float(np.mean(d_arr)),
            "se_delta": float(np.std(d_arr, ddof=1)),
            "ci_lower": ci_l,
            "ci_upper": ci_u,
            "is_significantly_positive": bool(ci_l > 0.0),
        }

    return {
        "n_bootstraps": int(n_bootstraps),
        "n_unique_patients": int(n_unique_patients),
        "n_records": int(n_samples),
        "random_seed": int(random_seed),
        "confidence_level": float(confidence_level),
        "resampling_unit": "patient_cluster",
        "model_a_name": model_a_name,
        "model_b_name": model_b_name,
        "paired_macro_delta": {
            "point_f1_a": macro_a_point,
            "point_f1_b": macro_b_point,
            "point_delta": delta_macro_point,
            "mean_delta": float(np.mean(delta_macro_arr)),
            "se_delta": se_delta_macro,
            "ci_lower": ci_lower_macro,
            "ci_upper": ci_upper_macro,
            "empirical_p_value_nonpos": p_val_nonpos,
            "is_significantly_positive": bool(ci_lower_macro > 0.0),
        },
        "per_class": per_class_summary,
    }


def format_paired_comparison_markdown_table(
    paired_res: Dict[str, Any],
    title: str = "Paired Patient-Clustered Bootstrap Comparison (Fold 10)",
) -> str:
    """Format paired patient-clustered bootstrap results into a clear GitHub Flavored Markdown table."""
    m_info = paired_res["paired_macro_delta"]
    b_count = paired_res["n_bootstraps"]
    n_pts = paired_res["n_unique_patients"]
    n_rec = paired_res["n_records"]
    c_lvl = int(paired_res["confidence_level"] * 100)

    delta_sign = "+" if m_info["point_delta"] > 0 else ""
    macro_sig_badge = "**Significant (+)**" if m_info["is_significantly_positive"] else "Not Significant"

    lines = [
        f"### {title}",
        "",
        "> [!NOTE]",
        f"> **Paired Bootstrap Design:** Evaluated across {n_rec:,} Fold 10 records sharing {n_pts:,} unique patient clusters. "
        f"In each of {b_count:,} bootstrap iterations (seed: {paired_res['random_seed']}), identical patient clusters were resampled "
        f"with replacement for both `{paired_res['model_a_name']}` and `{paired_res['model_b_name']}`. "
        f"The paired difference $\\Delta = \\text{{Macro F1}}_A - \\text{{Macro F1}}_B$ directly measures paired gain while controlling for intra-patient correlation.",
        "",
        f"**Macro F1 Point Delta:** {delta_sign}{m_info['point_delta']:.4f} | "
        f"**{c_lvl}% Paired CI:** [{m_info['ci_lower']:.4f}, {m_info['ci_upper']:.4f}] | "
        f"**Bootstrap SE:** {m_info['se_delta']:.4f} | "
        f"**Outcome:** {macro_sig_badge}",
        "",
        f"| Superclass | {paired_res['model_b_name']} F1 | {paired_res['model_a_name']} F1 | Paired Delta ($\\Delta$) | Bootstrap SE | {c_lvl}% Paired CI | 95% Significant? |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for sc, p_info in paired_res["per_class"].items():
        d_str = f"+{p_info['point_delta']:.4f}" if p_info['point_delta'] > 0 else f"{p_info['point_delta']:.4f}"
        sig_str = "Yes (Positive)" if p_info["is_significantly_positive"] else "No"
        lines.append(
            f"| **{sc}** | {p_info['point_f1_b']:.4f} | **{p_info['point_f1_a']:.4f}** | **{d_str}** | "
            f"{p_info['se_delta']:.4f} | [{p_info['ci_lower']:.4f}, {p_info['ci_upper']:.4f}] | {sig_str} |"
        )

    lines.append(
        f"| **Macro Aggregate** | {m_info['point_f1_b']:.4f} | **{m_info['point_f1_a']:.4f}** | **{delta_sign}{m_info['point_delta']:.4f}** | "
        f"{m_info['se_delta']:.4f} | **[{m_info['ci_lower']:.4f}, {m_info['ci_upper']:.4f}]** | **{sig_str}** |"
    )

    return "\n".join(lines)


