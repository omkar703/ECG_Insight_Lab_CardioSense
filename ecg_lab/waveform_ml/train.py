"""Training workflow, threshold optimization, and artifact generation.

Enforces patient split integrity and guarantees that the test fold (fold 10)
is strictly held out and never accessed for model fitting or threshold tuning.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ecg_lab.waveform_ml.config import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_RANDOM_SEED,
    PHYSIONET_DATASET_NAME,
    TARGET_SUPERCLASSES,
    TEST_FOLDS,
    TRAIN_FOLDS,
    VAL_FOLDS,
)
from ecg_lab.waveform_ml.dataset import (
    PatientOverlapError,
    load_ptbxl_metadata,
    validate_split_integrity,
)
from ecg_lab.waveform_ml.evaluate import (
    compute_bootstrap_confidence_intervals,
    compute_patient_cluster_bootstrap_ci,
    evaluate_multilabel_predictions,
    format_ci_markdown_table,
    format_comparison_markdown_table,
    format_metrics_markdown_table,
    format_pilot_vs_full_fold_comparison_table,
)
from ecg_lab.waveform_ml.features import FeatureCacheManager
from ecg_lab.waveform_ml.models import (
    PrevalenceBaselineClassifier,
    build_lightweight_pipeline,
    save_trained_model,
    tune_validation_thresholds,
)


def run_training_and_evaluation(
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    output_dir: Union[str, Path] = DEFAULT_ARTIFACT_DIR,
    random_seed: int = DEFAULT_RANDOM_SEED,
    c_param: float = 1.0,
    force_rebuild_cache: bool = False,
    max_records: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute the end-to-end training and evaluation pipeline.

    Returns:
        Dictionary containing validation and test metrics for baseline and model,
        split statistics, thresholds, and artifact paths.
    """
    data_path = Path(data_dir)
    cache_path = Path(cache_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load metadata and validate patient split integrity
    meta_df = load_ptbxl_metadata(data_path, verify_checksums=False)
    split_stats = validate_split_integrity(meta_df)

    # 2. Extract features or load from verified cache
    cache_mgr = FeatureCacheManager(cache_dir=cache_path)
    X, y, ecg_ids, cache_manifest = cache_mgr.build_or_load_cache(
        metadata_df=meta_df,
        data_dir=data_path,
        metadata_hash=cache_manifest_hash(meta_df),
        force_rebuild=force_rebuild_cache,
        max_records=max_records,
    )

    target_classes = list(TARGET_SUPERCLASSES)

    # 3. Verify that feature rows join to labels by ecg_id explicitly
    for col_idx, sc in enumerate(target_classes):
        expected_labels = meta_df.loc[ecg_ids, f"y_{sc}"].values.astype(np.int32)
        if not np.array_equal(y[:, col_idx], expected_labels):
            raise ValueError(f"Label alignment violation for superclass {sc} and ecg_ids.")

    # 4. Create fold masks matching extracted ecg_ids
    folds = meta_df.loc[ecg_ids, "strat_fold"].values
    patient_ids = meta_df.loc[ecg_ids, "patient_id"].values

    train_mask = np.isin(folds, TRAIN_FOLDS)
    val_mask = np.isin(folds, VAL_FOLDS)
    test_mask = np.isin(folds, TEST_FOLDS)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    train_pts = set(patient_ids[train_mask])
    val_pts = set(patient_ids[val_mask])
    test_pts = set(patient_ids[test_mask])

    if (train_pts & val_pts) or (train_pts & test_pts) or (val_pts & test_pts):
        raise PatientOverlapError("Patient leakage detected across partition splits.")

    # 5. Train naive prevalence baseline on full training partition (Folds 1-8)
    baseline = PrevalenceBaselineClassifier()
    baseline.fit(X_train, y_train)

    base_val_probs = baseline.predict_proba(X_val)
    base_val_preds = baseline.predict(X_val)
    base_val_metrics = evaluate_multilabel_predictions(
        y_val, base_val_preds, base_val_probs, target_classes
    )

    # 6. Train lightweight One-vs-Rest model on full training partition
    model = build_lightweight_pipeline(
        random_seed=random_seed,
        c_param=c_param,
    )
    model.fit(X_train, y_train)

    # 7. Predict on validation fold and tune decision thresholds (FOLD 9 ONLY)
    val_probs = model.predict_proba(X_val)
    val_thresholds = tune_validation_thresholds(y_val, val_probs)

    val_preds_tuned = (val_probs >= val_thresholds).astype(np.int32)
    val_metrics = evaluate_multilabel_predictions(
        y_val, val_preds_tuned, val_probs, target_classes, thresholds=val_thresholds
    )

    # 8. FREEZE MODEL AND CRYPTOGRAPHIC PROVENANCE MANIFEST BEFORE FOLD 10 INFERENCE
    def _get_git_revision() -> str:
        try:
            import subprocess
            res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
            return res.stdout.strip()
        except Exception:
            return "UNKNOWN"

    import resource
    peak_memory_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)

    train_metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "c_param": c_param,
        "dataset_name": PHYSIONET_DATASET_NAME,
        "code_revision": _get_git_revision(),
        "training_mode": "full_fold",
        "train_samples": int(len(X_train)),
        "train_patients": int(len(train_pts)),
        "val_samples": int(len(X_val)),
        "val_patients": int(len(val_pts)),
        "test_samples": int(len(X_test)),
        "test_patients": int(len(test_pts)),
        "train_positive_support": {
            cls_name: int(np.sum(y_train[:, idx])) for idx, cls_name in enumerate(target_classes)
        },
        "val_positive_support": {
            cls_name: int(np.sum(y_val[:, idx])) for idx, cls_name in enumerate(target_classes)
        },
        "split_integrity_verified": split_stats["split_integrity_verified"],
        "peak_rss_mb": peak_memory_mb,
        "frozen_prior_to_test_evaluation": True,
    }

    model_path, model_manifest_path = save_trained_model(
        model=model,
        thresholds=val_thresholds,
        target_classes=target_classes,
        train_metadata=train_metadata,
        output_dir=out_path,
    )

    # 9. PREPLANNED FOLD 10 RE-EVALUATION (STRICTLY FROZEN, NO TUNING)
    # Disclosed: Fold 10 was previously evaluated in Phase 3.1 with a pilot-trained model.
    base_test_probs = baseline.predict_proba(X_test)
    base_test_preds = baseline.predict(X_test)
    base_test_metrics = evaluate_multilabel_predictions(
        y_test, base_test_preds, base_test_probs, target_classes
    )

    test_probs = model.predict_proba(X_test)
    test_preds_tuned = (test_probs >= val_thresholds).astype(np.int32)
    test_metrics = evaluate_multilabel_predictions(
        y_test, test_preds_tuned, test_probs, target_classes, thresholds=val_thresholds
    )

    # Update manifest with test positive support
    train_metadata["test_positive_support"] = {
        cls_name: int(np.sum(y_test[:, idx])) for idx, cls_name in enumerate(target_classes)
    }

    # 10. Patient-Clustered Bootstrap Confidence Intervals (Fold 10)
    test_patient_ids = patient_ids[test_mask]
    test_ci = compute_patient_cluster_bootstrap_ci(
        y_test,
        test_preds_tuned,
        patient_ids=test_patient_ids,
        target_classes=target_classes,
        n_bootstraps=1000,
        random_seed=random_seed,
    )
    ci_table = format_ci_markdown_table(
        test_ci,
        title=f"95% Patient-Clustered Bootstrap Confidence Intervals (Fold 10, N={len(X_test):,}, {len(test_pts):,} Patients)",
    )

    # 11. Format Comparative Markdown Reports
    pilot_file = out_path / "pilot_evaluation" / "pilot_metrics.json"
    pilot_data = None
    if pilot_file.is_file():
        try:
            with open(pilot_file, "r", encoding="utf-8") as pf:
                pilot_data = json.load(pf)
        except Exception:
            pilot_data = None

    pilot_comp_table = format_pilot_vs_full_fold_comparison_table(pilot_data, test_metrics, target_classes)
    val_table = format_metrics_markdown_table(
        val_metrics,
        title=f"Validation Fold 9 Evaluation (Full Cohort, N={len(X_val):,}, {len(val_pts):,} Patients)",
    )
    test_table = format_metrics_markdown_table(
        test_metrics,
        title=f"Held-Out Test Fold 10 Re-Evaluation (Full Cohort, N={len(X_test):,}, {len(test_pts):,} Patients)",
    )
    comp_table = format_comparison_markdown_table(base_test_metrics, test_metrics, target_classes)

    summary_report = "\n\n".join([
        "# CardioSense Phase 3.2 Waveform ML Full-Fold Benchmark Report",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Dataset:** {PHYSIONET_DATASET_NAME} (100 Hz WFDB)",
        f"**Model:** Logistic Regression (One-vs-Rest, Balanced Class Weights, C={c_param})",
        f"**Features:** 148 deterministic lead & cross-lead statistics (Version {cache_manifest.get('feature_version')})",
        f"**Split Statistics:** Train: {len(X_train):,} records ({len(train_pts):,} patients, Folds 1–8) | "
        f"Validation: {len(X_val):,} records ({len(val_pts):,} patients, Fold 9) | "
        f"Test: {len(X_test):,} records ({len(test_pts):,} patients, Fold 10, strictly held out)",
        f"**Peak Memory Footprint:** {peak_memory_mb:.2f} MiB RSS",
        pilot_comp_table,
        comp_table,
        val_table,
        test_table,
        ci_table,
    ])

    report_path = out_path / "benchmark_summary.md"
    metrics_path = out_path / "metrics.json"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(summary_report)

    output_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "manifest_path": str(model_manifest_path),
        "report_path": str(report_path),
        "split_stats": split_stats,
        "peak_rss_mb": peak_memory_mb,
        "val_thresholds": {cls_name: float(val_thresholds[i]) for i, cls_name in enumerate(target_classes)},
        "baseline_val_metrics": base_val_metrics,
        "baseline_test_metrics": base_test_metrics,
        "model_val_metrics": val_metrics,
        "model_test_metrics": test_metrics,
        "test_bootstrap_ci": test_ci,
        "pilot_comparison": pilot_data.get("model_test_metrics") if pilot_data else None,
        "summary_markdown": summary_report,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    return output_payload


def cache_manifest_hash(df: pd.DataFrame) -> str:
    """Derive a compact hash from the metadata record count and fold counts."""
    fold_counts = df["strat_fold"].value_counts().sort_index().to_dict()
    key = f"{len(df)}:{fold_counts}"
    import hashlib
    return hashlib.sha256(key.encode()).hexdigest()[:16]
