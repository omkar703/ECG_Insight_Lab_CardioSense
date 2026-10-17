"""Unit tests for deterministic waveform feature extraction and caching."""

from pathlib import Path
import numpy as np
import pytest

from ecg_lab.waveform_ml.config import EXPECTED_LEADS_COUNT
from ecg_lab.waveform_ml.features import (
    FeatureCacheManager,
    compute_feature_fingerprint,
    extract_record_features,
    get_feature_schema_names,
)


@pytest.fixture
def synthetic_12lead_signal() -> np.ndarray:
    """Generate a deterministic synthetic 12-lead ECG signal (1000 samples, 12 leads)."""
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    # Sine wave with varying lead amplitudes
    leads = []
    for l_idx in range(12):
        lead_signal = np.sin(2 * np.pi * 1.2 * t + (l_idx * 0.2)) * (0.5 + 0.1 * l_idx)
        leads.append(lead_signal)
    return np.column_stack(leads)


def test_feature_schema_length():
    """Verify that feature schema names list contains exactly 148 features."""
    names = get_feature_schema_names()
    assert len(names) == 148
    assert "I_mean" in names
    assert "V6_derivative_rms" in names
    assert "global_mean_energy" in names
    assert "precordial_lead_rms_ratio" in names


def test_extract_record_features_shape_and_determinism(synthetic_12lead_signal: np.ndarray):
    """Test that feature extraction produces 148 finite numbers deterministically."""
    feats1 = extract_record_features(synthetic_12lead_signal, fs=100)
    feats2 = extract_record_features(synthetic_12lead_signal, fs=100)

    assert feats1.shape == (148,)
    assert np.isfinite(feats1).all()
    # Determinism assertion
    np.testing.assert_array_equal(feats1, feats2)


def test_extract_record_features_flatline():
    """Test that zero-variance flatline signal does not crash or produce NaN/Inf."""
    flatline = np.zeros((1000, 12), dtype=np.float64)
    feats = extract_record_features(flatline, fs=100)

    assert feats.shape == (148,)
    assert np.isfinite(feats).all()
    # All means, stds, and derivatives should be 0
    assert feats[0] == 0.0  # I_mean
    assert feats[1] == 0.0  # I_std


def test_extract_record_features_nan_sanitization():
    """Test that signals containing isolated NaN or Inf are sanitized gracefully."""
    signal_with_nan = np.ones((1000, 12), dtype=np.float64)
    signal_with_nan[10, 0] = np.nan
    signal_with_nan[20, 1] = np.inf

    feats = extract_record_features(signal_with_nan, fs=100)
    assert feats.shape == (148,)
    assert np.isfinite(feats).all()


def test_extract_record_features_invalid_shape():
    """Test that wrong number of leads raises ValueError."""
    invalid_signal = np.zeros((1000, 8))  # 8 leads instead of 12
    with pytest.raises(ValueError, match="Expected signal shape"):
        extract_record_features(invalid_signal)


def test_cache_fingerprint_invalidation():
    """Test that changing record count or feature version alters the cache fingerprint."""
    fp1 = compute_feature_fingerprint(100, "hash_a", 100, "1.0.0")
    fp2 = compute_feature_fingerprint(100, "hash_a", 100, "1.0.0")
    fp3 = compute_feature_fingerprint(200, "hash_a", 100, "1.0.0")
    fp4 = compute_feature_fingerprint(100, "hash_b", 100, "1.0.0")

    assert fp1 == fp2
    assert fp1 != fp3
    assert fp1 != fp4


def test_feature_cache_manager_save_and_load(tmp_path: Path):
    """Test saving and loading cache files using FeatureCacheManager."""
    mgr = FeatureCacheManager(cache_dir=tmp_path)
    fp = "test_fingerprint_12345"

    npz_path, manifest_path = mgr.get_cache_paths(fp)
    assert not mgr.is_cache_valid(fp)

    features = np.random.randn(5, 148)
    labels = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    ecg_ids = np.array([1, 2, 3, 4, 5])

    np.savez_compressed(npz_path, features=features, labels=labels, ecg_ids=ecg_ids)
    import json
    with open(manifest_path, "w") as f:
        json.dump({"fingerprint": fp, "feature_version": "1.0.0"}, f)

    assert mgr.is_cache_valid(fp)
    loaded_f, loaded_l, loaded_ids, manifest = mgr.load_cache(fp)
    np.testing.assert_array_almost_equal(loaded_f, features)
    np.testing.assert_array_equal(loaded_l, labels)
    np.testing.assert_array_equal(loaded_ids, ecg_ids)
