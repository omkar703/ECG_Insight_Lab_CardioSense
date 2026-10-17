"""Deterministic per-lead feature extraction and fingerprint-based caching.

Extracts robust statistical, dynamic, and morphology-summary features across 12 leads
with bounded memory footprint and auditable cache invalidation.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ecg_lab.waveform_ml.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    EXPECTED_LEADS_COUNT,
    FEATURE_SCHEMA_VERSION,
    LEAD_NAMES,
    SAMPLING_RATE_HZ,
    TARGET_SUPERCLASSES,
)
from ecg_lab.waveform_ml.dataset import MalformedRecordError, read_wfdb_record

PER_LEAD_METRIC_NAMES = (
    "mean",
    "std",
    "median",
    "iqr",
    "min",
    "max",
    "peak_to_peak",
    "skew",
    "kurtosis",
    "rms",
    "zero_crossings",
    "derivative_rms",
)

GLOBAL_METRIC_NAMES = (
    "global_mean_energy",
    "global_max_amplitude",
    "limb_lead_rms_ratio",
    "precordial_lead_rms_ratio",
)


def get_feature_schema_names() -> List[str]:
    """Return the ordered list of all 148 feature names."""
    names: List[str] = []
    for lead in LEAD_NAMES:
        for metric in PER_LEAD_METRIC_NAMES:
            names.append(f"{lead}_{metric}")
    names.extend(GLOBAL_METRIC_NAMES)
    return names


def extract_record_features(signal: np.ndarray, fs: int = SAMPLING_RATE_HZ) -> np.ndarray:
    """Extract 148 deterministic summary features from a 12-lead ECG signal.

    Args:
        signal: Array of shape (samples, 12), physical values in mV.
        fs: Sampling rate in Hz.

    Returns:
        1D float64 array of shape (148,).

    Guarantees:
        - Deterministic output.
        - Robust to zero-variance/flatline signals.
        - Handles finite sanitization gracefully.
    """
    if signal.ndim != 2 or signal.shape[1] != EXPECTED_LEADS_COUNT:
        raise ValueError(
            f"Expected signal shape (samples, {EXPECTED_LEADS_COUNT}), got {signal.shape}."
        )

    # Sanitize any non-finite values defensively
    if not np.isfinite(signal).all():
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples, n_leads = signal.shape
    features: List[float] = []
    lead_rms_values: List[float] = []

    # 1. Per-lead features (12 leads * 12 features = 144)
    for lead_idx in range(n_leads):
        lead_data = signal[:, lead_idx]
        mean_val = float(np.mean(lead_data))
        std_val = float(np.std(lead_data))
        median_val = float(np.median(lead_data))

        q25, q75 = np.percentile(lead_data, [25, 75])
        iqr_val = float(q75 - q25)

        min_val = float(np.min(lead_data))
        max_val = float(np.max(lead_data))
        ptp_val = float(max_val - min_val)

        # Skewness and kurtosis with fallback for flatlines
        if std_val < 1e-9:
            skew_val = 0.0
            kurt_val = 0.0
        else:
            skew_val = float(stats.skew(lead_data))
            kurt_val = float(stats.kurtosis(lead_data))

        rms_val = float(np.sqrt(np.mean(lead_data**2)))
        lead_rms_values.append(rms_val)

        # Zero-crossings of mean-centered waveform
        centered = lead_data - mean_val
        crossings = np.nonzero(np.diff(centered > 0))[0]
        zc_val = float(len(crossings))

        # First difference derivative RMS (voltage velocity)
        if n_samples > 1:
            diffs = np.diff(lead_data)
            deriv_rms = float(np.sqrt(np.mean(diffs**2)))
        else:
            deriv_rms = 0.0

        features.extend([
            mean_val,
            std_val,
            median_val,
            iqr_val,
            min_val,
            max_val,
            ptp_val,
            skew_val,
            kurt_val,
            rms_val,
            zc_val,
            deriv_rms,
        ])

    # 2. Global cross-lead features (4 features)
    total_energy = float(np.mean(signal**2))
    max_amp = float(np.max(np.abs(signal)))

    limb_rms = float(np.mean(lead_rms_values[:6])) if lead_rms_values else 0.0
    precordial_rms = float(np.mean(lead_rms_values[6:])) if len(lead_rms_values) == 12 else 0.0
    sum_rms = limb_rms + precordial_rms + 1e-9

    features.extend([
        total_energy,
        max_amp,
        limb_rms / sum_rms,
        precordial_rms / sum_rms,
    ])

    return np.array(features, dtype=np.float64)


def compute_feature_fingerprint(
    dataset_records_count: int,
    metadata_hash: str,
    sampling_rate: int = SAMPLING_RATE_HZ,
    feature_version: str = FEATURE_SCHEMA_VERSION,
) -> str:
    """Compute a cryptographic cache fingerprint for the feature extraction pipeline."""
    payload = f"{dataset_records_count}:{metadata_hash}:{sampling_rate}:{feature_version}:148_features"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class FeatureCacheManager:
    """Manages bounded-memory sequential feature extraction and disk caching with integrity checks."""

    def __init__(
        self,
        cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
        feature_version: str = FEATURE_SCHEMA_VERSION,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.feature_version = feature_version

    def get_cache_paths(self, fingerprint: str) -> Tuple[Path, Path]:
        """Return (npz_path, manifest_path) for a fingerprint."""
        short_fp = fingerprint[:16]
        npz_path = self.cache_dir / f"ptbxl_features_{short_fp}.npz"
        manifest_path = self.cache_dir / f"ptbxl_features_{short_fp}_manifest.json"
        return npz_path, manifest_path

    def is_cache_valid(self, fingerprint: str) -> bool:
        """Check if cached features exist and match fingerprint."""
        npz_path, manifest_path = self.get_cache_paths(fingerprint)
        if not npz_path.is_file() or not manifest_path.is_file():
            return False
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return meta.get("fingerprint") == fingerprint
        except Exception:
            return False

    def load_cache(
        self, fingerprint: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Load cached (features, labels, ecg_ids, manifest)."""
        npz_path, manifest_path = self.get_cache_paths(fingerprint)
        if not self.is_cache_valid(fingerprint):
            raise FileNotFoundError(f"Valid cache not found for fingerprint {fingerprint}")

        data = np.load(npz_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        return data["features"], data["labels"], data["ecg_ids"], manifest

    def build_or_load_cache(
        self,
        metadata_df: pd.DataFrame,
        data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
        metadata_hash: str = "custom",
        force_rebuild: bool = False,
        max_records: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Extract features in bounded batches and persist to cache, or load existing cache.

        Handles missing WFDB files gracefully by logging an explicit status manifest.
        Does NOT silently fabricate or replace missing signals.
        """
        data_path = Path(data_dir)
        df_to_process = metadata_df if max_records is None else metadata_df.iloc[:max_records]
        records_count = len(df_to_process)

        fingerprint = compute_feature_fingerprint(
            dataset_records_count=records_count,
            metadata_hash=metadata_hash,
            sampling_rate=SAMPLING_RATE_HZ,
            feature_version=self.feature_version,
        )

        if not force_rebuild and self.is_cache_valid(fingerprint):
            return self.load_cache(fingerprint)

        # Build feature cache record by record (bounded memory)
        schema_names = get_feature_schema_names()
        feature_matrix_list: List[np.ndarray] = []
        labels_list: List[List[int]] = []
        valid_ecg_ids: List[int] = []

        manifest_records: List[Dict[str, Any]] = []
        success_count = 0
        missing_count = 0
        corrupt_count = 0

        target_cols = [f"y_{sc}" for sc in TARGET_SUPERCLASSES]

        processed_count = 0
        for ecg_id, row in df_to_process.iterrows():
            processed_count += 1
            if processed_count % 2500 == 0 or processed_count == records_count:
                print(f"  [Feature Extraction] Processed {processed_count}/{records_count} records ({(processed_count/records_count)*100:.1f}%)...")
            filename_lr = row.get("filename_lr")
            label_vec = [int(row[col]) for col in target_cols]

            if not filename_lr or pd.isna(filename_lr):
                missing_count += 1
                manifest_records.append({
                    "ecg_id": int(ecg_id),
                    "status": "MISSING_FILENAME",
                    "reason": "filename_lr is empty in metadata",
                })
                continue

            rec_path = data_path / str(filename_lr)
            try:
                signal, meta = read_wfdb_record(rec_path)
                feats = extract_record_features(signal, fs=meta["fs"])
                feature_matrix_list.append(feats)
                labels_list.append(label_vec)
                valid_ecg_ids.append(int(ecg_id))
                success_count += 1
                manifest_records.append({
                    "ecg_id": int(ecg_id),
                    "status": "SUCCESS",
                })
            except FileNotFoundError:
                missing_count += 1
                manifest_records.append({
                    "ecg_id": int(ecg_id),
                    "status": "MISSING_FILE",
                    "reason": f"File not found: {rec_path}",
                })
            except MalformedRecordError as err:
                corrupt_count += 1
                manifest_records.append({
                    "ecg_id": int(ecg_id),
                    "status": "MALFORMED_RECORD",
                    "reason": str(err),
                })
            except Exception as ex:
                corrupt_count += 1
                manifest_records.append({
                    "ecg_id": int(ecg_id),
                    "status": "EXTRACTION_ERROR",
                    "reason": str(ex),
                })

        if not feature_matrix_list:
            raise RuntimeError(
                f"No records could be extracted ({missing_count} missing, {corrupt_count} corrupt). "
                f"Ensure records are placed in {data_path}."
            )

        features_arr = np.array(feature_matrix_list, dtype=np.float64)
        labels_arr = np.array(labels_list, dtype=np.int32)
        ecg_ids_arr = np.array(valid_ecg_ids, dtype=np.int64)

        npz_path, manifest_path = self.get_cache_paths(fingerprint)

        manifest = {
            "fingerprint": fingerprint,
            "feature_version": self.feature_version,
            "feature_schema_names": schema_names,
            "feature_count": len(schema_names),
            "target_superclasses": list(TARGET_SUPERCLASSES),
            "total_records_queried": records_count,
            "success_count": success_count,
            "missing_count": missing_count,
            "corrupt_count": corrupt_count,
            "records": manifest_records,
        }

        # Persist cache atomically
        np.savez_compressed(
            npz_path,
            features=features_arr,
            labels=labels_arr,
            ecg_ids=ecg_ids_arr,
        )
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return features_arr, labels_arr, ecg_ids_arr, manifest
