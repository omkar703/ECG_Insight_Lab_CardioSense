"""Lightweight single-lead ECG beat classification models, baselines, and verified persistence."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ecg_lab.single_lead_ml.config import (
    AAMI_TARGET_CLASSES,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_RANDOM_SEED,
    FEATURE_SCHEMA_VERSION,
    MODEL_NAME,
    MODEL_VERSION,
)


class SingleLeadBaselineClassifier(BaseEstimator, ClassifierMixin):
    """Empirical class-prevalence baseline classifier for single-lead beat classification.

    Outputs training class prevalence distributions as benchmark probabilities.
    """

    def __init__(self, target_classes: Tuple[str, ...] = AAMI_TARGET_CLASSES):
        self.target_classes = list(target_classes)
        self.classes_ = np.array(self.target_classes)
        self.prevalences_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SingleLeadBaselineClassifier":
        y_arr = np.asarray(y)
        counts = [np.sum(y_arr == c) for c in self.target_classes]
        total = max(1, len(y_arr))
        self.prevalences_ = np.array(counts, dtype=np.float64) / float(total)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.prevalences_ is None:
            raise RuntimeError("Classifier has not been fitted.")
        n_samples = X.shape[0]
        return np.tile(self.prevalences_, (n_samples, 1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        max_indices = np.argmax(probs, axis=1)
        return np.array([self.target_classes[idx] for idx in max_indices])


def build_single_lead_pipeline(
    random_seed: int = DEFAULT_RANDOM_SEED,
    n_estimators: int = 100,
    max_depth: int = 12,
) -> Pipeline:
    """Build a scikit-learn Pipeline with feature scaling and balanced Random Forest classifier."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=-1,
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", rf),
    ])
    return pipeline


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Compute cryptographic SHA-256 digest of artifact."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def save_single_lead_model(
    model: Any,
    target_classes: List[str],
    train_metadata: Dict[str, Any],
    output_dir: Union[str, Path] = DEFAULT_ARTIFACT_DIR,
    model_filename: str = MODEL_NAME,
) -> Tuple[Path, Path]:
    """Persist a trained single-lead model alongside a cryptographically bound SHA-256 manifest."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_path = out_path / model_filename
    manifest_path = out_path / f"{model_filename}.manifest.json"

    # Persist model binary
    joblib.dump(model, model_path, compress=3)
    sha256_digest = compute_file_sha256(model_path)

    manifest = {
        "model_file": model_filename,
        "model_sha256": sha256_digest,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "target_classes": list(target_classes),
        "train_metadata": train_metadata,
        "trust_boundary_disclaimer": (
            "This model artifact was produced by the documented CardioSense Phase 6 "
            "reproducible training script on MIT-BIH Arrhythmia Database v1.0.0. "
            "It is an academic single-channel beat classifier and does NOT connect "
            "to physical AD8232 hardware or predict patient diagnoses (MI/STTC/CD/HYP)."
        ),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return model_path, manifest_path


def load_verified_single_lead_model(
    model_path: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Load a single-lead model artifact after verifying its cryptographic SHA-256 manifest.

    Security & Provenance Checks:
        1. Explicitly rejects quarantined legacy models or untrusted filenames.
        2. Validates JSON schema of the cryptographic manifest.
        3. Computes bit-for-bit SHA-256 digest of the serialized model on disk.
        4. Rejects deserialization if digest mismatch is detected.
    """
    m_path = Path(model_path)
    # Strictly reject legacy models
    filename_lower = m_path.name.lower()
    path_str = str(m_path).lower()
    if "legacy" in path_str or "finalmodel" in filename_lower or "heart_disease" in filename_lower:
        raise PermissionError(
            f"Security policy violation: attempted to load quarantined legacy artifact: {m_path}"
        )

    if not m_path.is_file():
        raise FileNotFoundError(f"Single-lead model artifact not found: {m_path}")

    if manifest_path is None:
        manifest_path = m_path.with_name(f"{m_path.name}.manifest.json")
    man_path = Path(manifest_path)

    if not man_path.is_file():
        raise PermissionError(
            f"Provenance violation: No manifest found for model at {man_path}. Artifact cannot be trusted."
        )

    try:
        with open(man_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        raise ValueError(f"Manifest corruption: unable to parse JSON manifest at {man_path}: {e}")

    required_keys = {
        "model_file",
        "model_sha256",
        "created_at_utc",
        "architecture_version",
        "target_classes",
        "train_metadata",
    }
    missing_keys = required_keys - set(manifest.keys())
    if missing_keys:
        raise ValueError(f"Manifest schema violation: missing required keys {missing_keys} in {man_path}")

    if manifest["model_file"] != m_path.name:
        raise ValueError(
            f"Manifest name mismatch: manifest describes '{manifest['model_file']}' but loaded from '{m_path.name}'"
        )

    expected_sha = manifest["model_sha256"]
    actual_sha = compute_file_sha256(m_path)

    if actual_sha != expected_sha:
        raise ValueError(
            f"Integrity check failed: model SHA-256 {actual_sha} does not match manifest {expected_sha}."
        )

    model = joblib.load(m_path)
    return model, manifest
