"""End-to-end training and evaluation script for single-lead ECG beat classification."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

from ecg_lab.single_lead_ml.config import (
    AAMI_TARGET_CLASSES,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_RANDOM_SEED,
    FEATURE_SCHEMA_VERSION,
    MODEL_NAME,
    MODEL_VERSION,
    PREFERRED_LEADS_ORDER,
    TEST_RECORDS,
    TRAIN_RECORDS,
    VAL_RECORDS,
    get_subject_id,
)
from ecg_lab.single_lead_ml.dataset import (
    get_available_records,
    load_record_with_annotations,
    verify_split_integrity,
)
from ecg_lab.single_lead_ml.detector import evaluate_r_peak_detector, pan_tompkins_qrs_detector
from ecg_lab.single_lead_ml.evaluate import compute_beat_classification_metrics
from ecg_lab.single_lead_ml.features import FEATURE_NAMES, extract_annotated_record_features
from ecg_lab.single_lead_ml.models import (
    SingleLeadBaselineClassifier,
    build_single_lead_pipeline,
    save_single_lead_model,
)


def load_partition_dataset(
    record_ids: Tuple[str, ...],
    data_dir: Path = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Extract features and labels for all records in a partition."""
    all_features: List[np.ndarray] = []
    all_labels: List[str] = []
    record_summaries: Dict[str, Any] = {}

    for rec_id in record_ids:
        rec, anns = load_record_with_annotations(rec_id, data_dir=data_dir)
        feats, labels, v_anns = extract_annotated_record_features(rec, anns)
        all_features.append(feats)
        all_labels.extend(labels)

        class_counts = {c: int(np.sum(labels == c)) for c in AAMI_TARGET_CLASSES}
        record_summaries[rec_id] = {
            "lead": rec.lead_name,
            "total_annotations": len(anns),
            "valid_beats": len(v_anns),
            "class_counts": class_counts,
        }

    X = np.vstack(all_features) if all_features else np.zeros((0, len(FEATURE_NAMES)))
    y = np.array(all_labels, dtype=object)
    return X, y, record_summaries


def run_training_pipeline(
    data_dir: Path = None,
    output_dir: Path = DEFAULT_ARTIFACT_DIR,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """Execute end-to-end model training, validation, and held-out test evaluation."""
    print("=" * 70)
    print("CardioSense Phase 6: MIT-BIH Single-Lead Beat ML Pipeline")
    print("=" * 70)

    # 1. Verify split integrity
    split_info = verify_split_integrity()
    print(f"Split integrity verified: {split_info['total_records']} records across {split_info['total_subjects']} subjects.")
    print(f"Records 201 & 202 forced to Train (subject: sub_201_202). Zero subject overlap.")

    # 2. Ingest Train partition
    print(f"\n[1/5] Extracting features for Train partition ({len(TRAIN_RECORDS)} records)...")
    X_train, y_train, train_meta = load_partition_dataset(TRAIN_RECORDS, data_dir=data_dir)
    print(f"  Train samples: {X_train.shape[0]} beats, {X_train.shape[1]} features")
    for c in AAMI_TARGET_CLASSES:
        print(f"    Class {c}: {np.sum(y_train == c)}")

    # 3. Ingest Validation partition
    print(f"\n[2/5] Extracting features for Validation partition ({len(VAL_RECORDS)} records)...")
    X_val, y_val, val_meta = load_partition_dataset(VAL_RECORDS, data_dir=data_dir)
    print(f"  Validation samples: {X_val.shape[0]} beats")
    for c in AAMI_TARGET_CLASSES:
        print(f"    Class {c}: {np.sum(y_val == c)}")

    # 4. Train baseline & Random Forest model
    print("\n[3/5] Fitting Empirical Baseline & Balanced Random Forest...")
    baseline = SingleLeadBaselineClassifier(target_classes=AAMI_TARGET_CLASSES)
    baseline.fit(X_train, y_train)

    rf_pipeline = build_single_lead_pipeline(random_seed=random_seed, n_estimators=100, max_depth=12)
    rf_pipeline.fit(X_train, y_train)

    # Evaluate on Validation split
    val_pred_base = baseline.predict(X_val)
    val_metrics_base = compute_beat_classification_metrics(y_val, val_pred_base)

    val_pred_rf = rf_pipeline.predict(X_val)
    val_metrics_rf = compute_beat_classification_metrics(y_val, val_pred_rf)
    print(f"  Validation Baseline Macro F1: {val_metrics_base['macro_f1']:.4f}")
    print(f"  Validation Random Forest Macro F1: {val_metrics_rf['macro_f1']:.4f}")

    # 5. Held-Out Test Evaluation (Strictly frozen model)
    print(f"\n[4/5] Evaluating Frozen Model on Held-Out Test partition ({len(TEST_RECORDS)} records)...")
    X_test, y_test, test_meta = load_partition_dataset(TEST_RECORDS, data_dir=data_dir)
    print(f"  Test samples: {X_test.shape[0]} beats")
    for c in AAMI_TARGET_CLASSES:
        print(f"    Class {c}: {np.sum(y_test == c)}")

    test_pred_base = baseline.predict(X_test)
    test_metrics_base = compute_beat_classification_metrics(y_test, test_pred_base)

    test_pred_rf = rf_pipeline.predict(X_test)
    test_metrics_rf = compute_beat_classification_metrics(y_test, test_pred_rf)

    print("\n" + "-" * 50)
    print("HELD-OUT TEST RESULTS (FROZEN MODEL):")
    print(f"Baseline Macro F1: {test_metrics_base['macro_f1']:.4f} | Weighted F1: {test_metrics_base['weighted_f1']:.4f}")
    print(f"Random Forest Macro F1: {test_metrics_rf['macro_f1']:.4f} | Weighted F1: {test_metrics_rf['weighted_f1']:.4f}")
    print("-" * 50)
    for c in AAMI_TARGET_CLASSES:
        p_info = test_metrics_rf["per_class"][c]
        print(f"  Class {c} ({p_info['description']}): Support={p_info['support']}, Prec={p_info['precision']:.4f}, Rec={p_info['recall']:.4f}, F1={p_info['f1_score']:.4f}")

    # 6. Evaluate Signal-Based R-Peak Detector on Test Set
    print("\n[5/5] Evaluating Signal-Based R-Peak Detector on Test Set (strictly raw signal, no annotations)...")
    detector_results: Dict[str, Any] = {}
    total_ref = 0
    total_det = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for rec_id in TEST_RECORDS:
        rec, anns = load_record_with_annotations(rec_id, data_dir=data_dir)
        ref_peaks = np.array([a.sample_index for a in anns])
        det_peaks = pan_tompkins_qrs_detector(rec.signal_mv, sampling_rate_hz=rec.sampling_rate_hz)
        res = evaluate_r_peak_detector(det_peaks, ref_peaks, tolerance_ms=50.0)
        detector_results[rec_id] = res
        total_ref += res["n_reference"]
        total_det += res["n_detected"]
        total_tp += res["true_positives"]
        total_fp += res["false_positives"]
        total_fn += res["false_negatives"]

    det_se = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    det_ppv = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    det_f1 = (2 * det_se * det_ppv) / (det_se + det_ppv) if (det_se + det_ppv) > 0 else 0.0

    print(f"  Aggregate Test Set Detector: Ref={total_ref}, Det={total_det}, TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"  Sensitivity: {det_se:.4f} | PPV: {det_ppv:.4f} | F1: {det_f1:.4f}")

    # 7. Persist Model and Cryptographic Manifest
    out_dir = Path(output_dir)
    train_metadata = {
        "dataset_name": "MIT-BIH Arrhythmia Database v1.0.0",
        "dataset_citation": "Moody G, Mark R. PhysioNet, 2005. DOI: 10.13026/C2F305",
        "dataset_license": "Open Data Commons Attribution License v1.0 (ODC-By)",
        "train_records": list(TRAIN_RECORDS),
        "val_records": list(VAL_RECORDS),
        "test_records": list(TEST_RECORDS),
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "preferred_leads": list(PREFERRED_LEADS_ORDER),
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "random_seed": random_seed,
        "n_estimators": 100,
        "max_depth": 12,
        "validation_metrics": val_metrics_rf,
        "test_metrics": test_metrics_rf,
        "r_peak_detector_test_metrics": {
            "total_reference_beats": total_ref,
            "total_detected_peaks": total_det,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "sensitivity": det_se,
            "positive_predictive_value": det_ppv,
            "f1_score": det_f1,
            "tolerance_ms": 50.0,
        },
    }

    model_path, manifest_path = save_single_lead_model(
        model=rf_pipeline,
        target_classes=list(AAMI_TARGET_CLASSES),
        train_metadata=train_metadata,
        output_dir=out_dir,
        model_filename=MODEL_NAME,
    )
    print(f"\nModel saved to: {model_path}")
    print(f"Manifest saved to: {manifest_path}")

    # Save benchmark summary
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "split_info": split_info,
        "baseline_test_metrics": test_metrics_base,
        "random_forest_test_metrics": test_metrics_rf,
        "r_peak_detector_test_metrics": train_metadata["r_peak_detector_test_metrics"],
        "records_train": list(TRAIN_RECORDS),
        "records_val": list(VAL_RECORDS),
        "records_test": list(TEST_RECORDS),
    }

    summary_path = out_dir / "single_lead_benchmark_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Benchmark summary saved to: {summary_path}")

    return summary


if __name__ == "__main__":
    run_training_pipeline()
