"""Unified CLI for the Digital Waveform ML Subsystem.

Provides commands for dataset validation, feature extraction, training,
evaluation, and sample downloading.
"""

import argparse
import json
import sys
from pathlib import Path

from ecg_lab.waveform_ml.config import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_RANDOM_SEED,
    OFFICIAL_CHECKSUMS,
    TARGET_SUPERCLASSES,
)
from ecg_lab.waveform_ml.dataset import (
    audit_dataset_completeness,
    download_missing_records,
    download_ptbxl_metadata,
    download_single_record,
    load_ptbxl_metadata,
    validate_split_integrity,
)
from ecg_lab.waveform_ml.evaluate import (
    format_comparison_markdown_table,
    format_metrics_markdown_table,
)
from ecg_lab.waveform_ml.features import FeatureCacheManager
from ecg_lab.waveform_ml.models import load_verified_model
from ecg_lab.waveform_ml.train import run_training_and_evaluation


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate dataset files, metadata, and patient split integrity."""
    data_dir = Path(args.data_dir)
    print(f"[*] Validating PTB-XL dataset in: {data_dir.resolve()}")

    if not data_dir.is_dir():
        print(f"[ERROR] Directory does not exist: {data_dir}")
        return 1

    try:
        df = load_ptbxl_metadata(data_dir, verify_checksums=args.verify_checksums)
        print(f"[OK] Metadata loaded successfully: {len(df)} records.")
    except Exception as e:
        print(f"[ERROR] Failed to load metadata: {e}")
        return 1

    # Check superclass distribution
    print("\n--- Diagnostic Superclass Distribution ---")
    for sc in TARGET_SUPERCLASSES:
        count = int(df[f"y_{sc}"].sum())
        pct = (count / len(df)) * 100
        print(f"  {sc:6s}: {count:5d} records ({pct:5.2f}%)")

    unmapped = int((df["num_diagnostic_labels"] == 0).sum())
    print(f"  Unmapped/Non-diagnostic: {unmapped} records ({(unmapped/len(df))*100:.2f}%)")

    # Patient split integrity
    print("\n--- Patient Split Integrity Check ---")
    try:
        stats = validate_split_integrity(df)
        print(f"  Train records (folds 1-8): {stats['train_records']} ({stats['train_patients']} patients)")
        print(f"  Val records   (fold 9):    {stats['val_records']} ({stats['val_patients']} patients)")
        print(f"  Test records  (fold 10):   {stats['test_records']} ({stats['test_patients']} patients)")
        print("  Overlap Train & Val:  0 patients")
        print("  Overlap Train & Test: 0 patients")
        print("  Overlap Val & Test:   0 patients")
        print("[SUCCESS] Zero patient leakage across splits! Split integrity verified.")
    except Exception as e:
        print(f"[FATAL] Split integrity violation: {e}")
        return 1

    return 0


def cmd_download_metadata(args: argparse.Namespace) -> int:
    """Download official metadata from PhysioNet/AWS Open Data."""
    data_dir = Path(args.data_dir)
    print(f"[*] Downloading PTB-XL metadata to {data_dir.resolve()}...")
    try:
        saved = download_ptbxl_metadata(data_dir)
        for name, path in saved.items():
            print(f"  [OK] Saved {name} ({path.stat().st_size:,} bytes)")
        print("[SUCCESS] Official metadata downloaded and checksum-verified.")
        return 0
    except Exception as e:
        print(f"[ERROR] Metadata download failed: {e}")
        return 1


def cmd_download_samples(args: argparse.Namespace) -> int:
    """Download a stratified set of real sample records for offline evaluation or UI inspection."""
    data_dir = Path(args.data_dir)
    count_per_fold = args.count_per_fold
    print(f"[*] Downloading {count_per_fold} sample records per fold into {data_dir.resolve()}...")

    try:
        df = load_ptbxl_metadata(data_dir)
    except Exception as e:
        print(f"[ERROR] Could not load metadata: {e}. Run download-metadata first.")
        return 1

    downloaded = 0
    errors = 0
    for fold in range(1, 11):
        fold_records = df[df["strat_fold"] == fold].head(count_per_fold)
        print(f"  Downloading fold {fold} samples ({len(fold_records)} records)...")
        for ecg_id, row in fold_records.iterrows():
            rel_path = row["filename_lr"]
            try:
                download_single_record(rel_path, dest_dir=data_dir)
                downloaded += 1
            except Exception as e:
                print(f"    [WARN] Failed to download {rel_path}: {e}")
                errors += 1

    print(f"[DONE] Downloaded {downloaded} records ({errors} errors).")
    return 0 if errors == 0 else 2


def cmd_audit_completeness(args: argparse.Namespace) -> int:
    """Audit actual on-disk WFDB availability for every record by fold."""
    data_dir = Path(args.data_dir)
    print(f"[*] Auditing PTB-XL completeness in: {data_dir.resolve()}")
    try:
        import pandas as pd
        df = load_ptbxl_metadata(data_dir)
        audit_res = audit_dataset_completeness(df, data_dir=data_dir)
        print("\n--- Partition Fold Completeness Audit ---")
        print(f"{'Fold':<6} | {'Expected':<10} | {'.hea Present':<14} | {'.dat Present':<14} | {'Complete':<10} | {'Missing':<10} | {'Status'}")
        print("-" * 88)
        for fold, f_info in audit_res["folds"].items():
            status = "[OK] Complete" if f_info["is_complete"] else "[WARN] Incomplete"
            print(
                f"Fold {fold:<2} | {f_info['expected_records']:<10} | {f_info['present_hea']:<14} | "
                f"{f_info['present_dat']:<14} | {f_info['complete_records']:<10} | {f_info['missing_records']:<10} | {status}"
            )
        print("-" * 88)
        print(f"Total Expected: {audit_res['total_expected']}")
        print(f"Total Complete: {audit_res['total_complete_records']}")
        print(f"Total Missing:  {audit_res['total_missing_records']}")
        print(f"All Folds Complete: {audit_res['all_folds_complete']}")

        # Save machine-readable manifest and CSV
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "dataset_completeness_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(audit_res, f, indent=2)
        print(f"\n[SUCCESS] Manifest saved to: {manifest_path}")

        # Save docs CSV
        csv_path = Path("docs/dataset_completeness_audit.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_rows = []
        for fold, f_info in audit_res["folds"].items():
            csv_rows.append(f_info)
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"[SUCCESS] Audit CSV saved to: {csv_path}")

        return 0
    except Exception as e:
        print(f"[ERROR] Completeness audit failed: {e}")
        return 1


def cmd_download_missing(args: argparse.Namespace) -> int:
    """Download all missing PTB-XL records concurrently."""
    data_dir = Path(args.data_dir)
    target_folds = [int(f.strip()) for f in args.folds.split(",")] if args.folds else None
    print(f"[*] Downloading missing records to {data_dir.resolve()} (target folds: {target_folds or 'all'})...")
    try:
        df = load_ptbxl_metadata(data_dir)
        res = download_missing_records(
            metadata_df=df,
            data_dir=data_dir,
            concurrency=args.concurrency,
            target_folds=target_folds,
        )
        print(f"[SUCCESS] Download completed: {res['downloaded']} downloaded, {res['failed']} failed (attempted {res['attempted']}).")
        return 0 if res["failed"] == 0 else 2
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return 1



def cmd_extract_features(args: argparse.Namespace) -> int:
    """Extract features and build/refresh cache."""
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    print(f"[*] Extracting features from {data_dir} -> {cache_dir}...")

    try:
        df = load_ptbxl_metadata(data_dir)
        cache_mgr = FeatureCacheManager(cache_dir=cache_dir)
        from ecg_lab.waveform_ml.train import cache_manifest_hash
        X, y, ecg_ids, manifest = cache_mgr.build_or_load_cache(
            metadata_df=df,
            data_dir=data_dir,
            metadata_hash=cache_manifest_hash(df),
            force_rebuild=args.force_rebuild,
            max_records=args.max_records,
        )
        print(f"[SUCCESS] Feature cache ready.")
        print(f"  Records extracted: {len(X)}")
        print(f"  Feature dimensions: {X.shape[1]}")
        print(f"  Target classes: {y.shape[1]}")
        print(f"  Success count: {manifest['success_count']}, Missing: {manifest['missing_count']}")
        return 0
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return 1


def cmd_train(args: argparse.Namespace) -> int:
    """Train naive baseline and lightweight ML model; tune thresholds on fold 9 only."""
    print("[*] Running CardioSense Phase 3 Training Pipeline...")
    try:
        results = run_training_and_evaluation(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            random_seed=args.random_seed,
            c_param=args.c_param,
            force_rebuild_cache=args.force_rebuild_cache,
            max_records=args.max_records,
        )
        print("\n" + results["summary_markdown"])
        print(f"\n[SUCCESS] Model saved to: {results['model_path']}")
        print(f"[SUCCESS] Manifest saved to: {results['manifest_path']}")
        print(f"[SUCCESS] Metrics saved to: {results['report_path']}")
        return 0
    except Exception as e:
        print(f"[FATAL] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ecg_lab.waveform_ml",
        description="CardioSense Phase 3: PTB-XL Digital Waveform ML Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: validate
    sub_val = subparsers.add_parser("validate", help="Validate dataset and split integrity")
    sub_val.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Dataset root dir")
    sub_val.add_argument("--verify-checksums", action="store_true", help="Verify SHA256 hashes")

    # Subcommand: download-metadata
    sub_dl_m = subparsers.add_parser("download-metadata", help="Download official metadata files")
    sub_dl_m.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Dataset root dir")

    # Subcommand: download-samples
    sub_dl_s = subparsers.add_parser("download-samples", help="Download stratified sample records")
    sub_dl_s.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Dataset root dir")
    sub_dl_s.add_argument("--count-per-fold", type=int, default=10, help="Records to fetch per fold")

    # Subcommand: extract-features
    sub_feat = subparsers.add_parser("extract-features", help="Extract features and build cache")
    sub_feat.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Dataset root dir")
    sub_feat.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Cache directory")
    sub_feat.add_argument("--force-rebuild", action="store_true", help="Force rebuild cache")
    sub_feat.add_argument("--max-records", type=int, default=None, help="Limit records")

    # Subcommand: train
    sub_tr = subparsers.add_parser("train", help="Train baseline and lightweight ML model")
    sub_tr.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Dataset root dir")
    sub_tr.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Cache directory")
    sub_tr.add_argument("--output-dir", default=str(DEFAULT_ARTIFACT_DIR), help="Artifacts directory")
    sub_tr.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed")
    sub_tr.add_argument("--c-param", type=float, default=1.0, help="Logistic Regression C parameter")
    sub_tr.add_argument("--force-rebuild-cache", action="store_true", help="Rebuild feature cache")
    sub_tr.add_argument("--max-records", type=int, default=None, help="Limit records for testing")

    # Subcommand: audit-completeness
    sub_audit = subparsers.add_parser("audit-completeness", help="Audit on-disk WFDB record availability")
    sub_audit.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Dataset root dir")
    sub_audit.add_argument("--output-dir", default=str(DEFAULT_ARTIFACT_DIR), help="Artifacts directory")

    # Subcommand: download-missing
    sub_dl_miss = subparsers.add_parser("download-missing", help="Download all missing PTB-XL records")
    sub_dl_miss.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Dataset root dir")
    sub_dl_miss.add_argument("--concurrency", type=int, default=40, help="Async download concurrency")
    sub_dl_miss.add_argument("--folds", default=None, help="Comma-separated folds to download (e.g. 1,2,3)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    if args.command == "validate":
        return cmd_validate(args)
    elif args.command == "download-metadata":
        return cmd_download_metadata(args)
    elif args.command == "download-samples":
        return cmd_download_samples(args)
    elif args.command == "audit-completeness":
        return cmd_audit_completeness(args)
    elif args.command == "download-missing":
        return cmd_download_missing(args)
    elif args.command == "extract-features":
        return cmd_extract_features(args)
    elif args.command == "train":
        return cmd_train(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
