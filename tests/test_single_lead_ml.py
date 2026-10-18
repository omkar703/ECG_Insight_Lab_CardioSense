"""Automated tests for MIT-BIH Single-Lead Beat ML (Phase 6).

Verifies:
- Dataset provenance and license metadata
- Record list completeness and missing file handling
- Channel selection policy (MLII preferred, V5 fallback)
- Official annotation mapping to AAMI EC57 classes
- Quarantine of paced records
- Record 201/202 grouping (same subject sub_201_202)
- Subject-disjoint split verification (zero overlap)
- Deterministic feature extraction and window dimensions
- Signal-based R-peak detector (no ground-truth annotations used at inference)
- Detector accuracy (Sensitivity & PPV >= 0.95 on benchmark record)
- Cryptographic SHA-256 model manifest verification
- Rejection of tampered models and quarantined legacy pickles
- UI playback disclaimer integrity (RECORDED DATA REPLAY — NOT LIVE HARDWARE)
- Strict isolation from Workflow A (digitizer) and Workflow B (PTB-XL 12-lead)
"""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

from ecg_lab.single_lead_ml.config import (
    AAMI_CLASS_DESCRIPTIONS,
    AAMI_TARGET_CLASSES,
    ADC_GAIN_UNITS_PER_MV,
    ADC_ZERO_BASELINE,
    BEAT_WINDOW_TOTAL_SAMPLES,
    DEFAULT_ARTIFACT_DIR,
    DISCLAIMER_RECORDED_REPLAY,
    DISCLAIMER_SINGLE_LEAD_RESEARCH,
    MODEL_NAME,
    NATIVE_SAMPLING_RATE_HZ,
    NON_BEAT_SYMBOLS,
    PACED_RECORDS,
    POST_R_SAMPLES,
    PRE_R_SAMPLES,
    PREFERRED_LEADS_ORDER,
    SYMBOL_TO_AAMI,
    TEST_RECORDS,
    TRAIN_RECORDS,
    VAL_RECORDS,
    get_subject_id,
)
from ecg_lab.single_lead_ml.dataset import (
    get_available_records,
    load_beat_annotations,
    load_record_with_annotations,
    load_single_lead_signal,
    resolve_mitdb_dir,
    verify_split_integrity,
)
from ecg_lab.single_lead_ml.detector import evaluate_r_peak_detector, pan_tompkins_qrs_detector
from ecg_lab.single_lead_ml.features import (
    FEATURE_NAMES,
    extract_annotated_record_features,
    extract_beat_features,
)
from ecg_lab.single_lead_ml.models import (
    SingleLeadBaselineClassifier,
    build_single_lead_pipeline,
    compute_file_sha256,
    load_verified_single_lead_model,
    save_single_lead_model,
)
from ecg_lab.single_lead_ml.playback import get_playback_slice


def is_mitdb_available() -> bool:
    try:
        resolve_mitdb_dir()
        return True
    except FileNotFoundError:
        return False


mitdb_required = pytest.mark.skipif(
    not is_mitdb_available(),
    reason="MIT-BIH Arrhythmia Database files not present on disk (optional dataset integration test)",
)


@mitdb_required
class TestDatasetProvenanceAndIntegrity:
    """Stage 1: Ingestion, provenance, channel policy, and split verification."""

    def test_dataset_directory_resolves(self):
        root = resolve_mitdb_dir()
        assert root.is_dir(), f"Resolved directory must exist: {root}"
        records = get_available_records()
        assert len(records) == 48, f"Expected 48 MIT-BIH records, got {len(records)}"

    def test_missing_record_raises_error(self):
        with pytest.raises(FileNotFoundError):
            load_single_lead_signal("non_existent_record_999")

    def test_channel_selection_prefers_mlii(self):
        # Record 100 has MLII and V5
        rec = load_single_lead_signal("100")
        assert rec.lead_name == "MLII"
        assert "MLII" in rec.available_leads
        assert rec.sampling_rate_hz == NATIVE_SAMPLING_RATE_HZ

    def test_channel_selection_fallback_when_mlii_absent(self):
        # Records 102 and 104 have V5 and V2 (MLII is absent)
        rec = load_single_lead_signal("102")
        assert rec.lead_name == "V5"
        assert "MLII" not in rec.available_leads

    def test_adc_to_physical_millivolts_conversion(self):
        rec = load_single_lead_signal("100")
        # Physical formula: (ADC - 1024) / 200 mV
        expected_mv = (rec.raw_adc - ADC_ZERO_BASELINE) / ADC_GAIN_UNITS_PER_MV
        np.testing.assert_allclose(rec.signal_mv, expected_mv, atol=1e-6)

    def test_annotation_parsing_and_aami_mapping(self):
        anns = load_beat_annotations("100")
        assert len(anns) > 2000
        # Assert mapped classes belong to AAMI target classes
        for ann in anns:
            assert ann.aami_class in AAMI_TARGET_CLASSES

    def test_official_symbol_mapping_integrity(self):
        # Normal beats
        for sym in ["N", "L", "R", "e", "j"]:
            assert SYMBOL_TO_AAMI[sym] == "N"
        # SVEBs
        for sym in ["A", "a", "J", "S"]:
            assert SYMBOL_TO_AAMI[sym] == "S"
        # VEBs
        for sym in ["V", "E"]:
            assert SYMBOL_TO_AAMI[sym] == "V"
        # Fusion / Paced / Unknown
        for sym in ["F", "/", "f", "Q"]:
            assert SYMBOL_TO_AAMI[sym] == "Q"

        # Non-beat events must not be in SYMBOL_TO_AAMI
        for non_beat in NON_BEAT_SYMBOLS:
            assert non_beat not in SYMBOL_TO_AAMI

    def test_quarantine_of_paced_records(self):
        for paced in ["102", "104", "107", "217"]:
            assert paced in PACED_RECORDS
            assert paced not in TRAIN_RECORDS
            assert paced not in VAL_RECORDS
            assert paced not in TEST_RECORDS

    def test_record_201_and_202_same_subject(self):
        sub_201 = get_subject_id("201")
        sub_202 = get_subject_id("202")
        assert sub_201 == sub_202 == "sub_201_202"
        assert "201" in TRAIN_RECORDS
        assert "202" in TRAIN_RECORDS

    def test_patient_disjoint_partitions(self):
        res = verify_split_integrity()
        assert res["is_valid"] is True
        assert res["total_records"] == 48
        assert res["total_subjects"] == 47

    def test_count_reconciliation_reference_vs_boundary_drops(self):
        # Audit A: Reconcile 47,576 reference beats - 15 boundary drops = 47,561 evaluated beats
        total_anns = 0
        boundary_drops = 0
        for r in TEST_RECORDS:
            rec, anns = load_record_with_annotations(r)
            total_anns += len(anns)
            sig_len = len(rec.signal_mv)
            for a in anns:
                if (a.sample_index - PRE_R_SAMPLES < 0) or (a.sample_index + POST_R_SAMPLES > sig_len):
                    boundary_drops += 1
        assert total_anns == 47576
        assert boundary_drops == 15
        assert total_anns - boundary_drops == 47561

    def test_aami_mapping_fusion_f_merged_into_q_audit(self):
        # Audit B: Document and assert that fusion beats ('F') are mapped to 'Q'
        assert SYMBOL_TO_AAMI["F"] == "Q"
        assert "F" not in AAMI_TARGET_CLASSES
        assert len(AAMI_TARGET_CLASSES) == 4  # N, S, V, Q

    def test_adc_conversion_parameters_across_channels(self):
        # Audit C: Assert ADC constants match MIT-BIH 200 ADC/mV and 1024 baseline
        assert ADC_ZERO_BASELINE == 1024.0
        assert ADC_GAIN_UNITS_PER_MV == 200.0
        rec_mlii = load_single_lead_signal("100")
        rec_v5 = load_single_lead_signal("102")
        assert rec_mlii.sampling_rate_hz == 360
        assert rec_v5.sampling_rate_hz == 360



class TestFeatureExtractionAndDetector:
    """Stage 2 & 3: Feature engineering and signal-based R-peak detector."""

    def test_window_geometry_constants(self):
        assert PRE_R_SAMPLES == 90
        assert POST_R_SAMPLES == 110
        assert BEAT_WINDOW_TOTAL_SAMPLES == 200

    @mitdb_required
    def test_feature_extraction_dimensions_and_finiteness(self):
        rec, anns = load_record_with_annotations("100")
        feats, labels, v_anns = extract_annotated_record_features(rec, anns)
        assert feats.ndim == 2
        assert len(FEATURE_NAMES) == 47, "Workflow C feature schema must contain exactly 47 features (5 + 10 + 12 + 20)"
        assert feats.shape[1] == 47
        assert len(labels) == len(v_anns) == feats.shape[0]
        assert np.all(np.isfinite(feats))

    def test_feature_schema_breakdown_reconciliation(self):
        """Verify the exact count breakdown of all 47 deterministic features."""
        rr_features = [f for f in FEATURE_NAMES if "rr" in f]
        global_morph = ["amp_mean", "amp_std", "amp_min", "amp_max", "amp_peak_to_peak", "amp_skewness", "amp_kurtosis", "amp_rms", "amp_energy", "zero_crossings"]
        qrs_sub = [f for f in FEATURE_NAMES if f.startswith("qrs_")]
        pre_sub = [f for f in FEATURE_NAMES if f.startswith("pre_") and not f.endswith("_rr_sec") and not f.startswith("pre_rr")]
        post_sub = [f for f in FEATURE_NAMES if f.startswith("post_") and not f.endswith("_rr_sec") and not f.startswith("post_rr")]
        morph_points = [f for f in FEATURE_NAMES if f.startswith("morph_sample_")]

        assert len(rr_features) == 5, f"Expected 5 RR features, got {len(rr_features)}"
        assert len(global_morph) == 10, f"Expected 10 global features, got {len(global_morph)}"
        assert len(qrs_sub) == 5, f"Expected 5 QRS core features, got {len(qrs_sub)}"
        assert len(pre_sub) == 3, f"Expected 3 Pre-QRS features, got {len(pre_sub)}"
        assert len(post_sub) == 4, f"Expected 4 Post-QRS features, got {len(post_sub)}"
        assert len(morph_points) == 20, f"Expected 20 morphology points, got {len(morph_points)}"
        assert (len(rr_features) + len(global_morph) + len(qrs_sub) + len(pre_sub) + len(post_sub) + len(morph_points)) == 47

    def test_detector_operates_without_annotations(self):
        # Generate synthetic 10-second signal with 10 simulated QRS spikes
        fs = 360
        t = np.arange(10 * fs) / float(fs)
        sig = 0.05 * np.sin(2 * np.pi * 1.0 * t)  # baseline
        # Add 10 spikes at 1.0s intervals
        for sec in range(1, 10):
            idx = int(sec * fs)
            sig[idx - 5 : idx + 5] += 1.5

        # Pan-Tompkins takes only (sig, fs) - zero annotations provided!
        detected_peaks = pan_tompkins_qrs_detector(sig, sampling_rate_hz=fs)
        assert len(detected_peaks) >= 8

    def test_detector_accuracy_on_reference_record(self):
        rec, anns = load_record_with_annotations("100")
        ref_peaks = np.array([a.sample_index for a in anns])
        det_peaks = pan_tompkins_qrs_detector(rec.signal_mv, sampling_rate_hz=rec.sampling_rate_hz)
        eval_res = evaluate_r_peak_detector(det_peaks, ref_peaks, tolerance_ms=50.0)

        # Record 100 has very clear MLII signal; Pan-Tompkins should achieve > 98% Se and PPV
        assert eval_res["sensitivity"] >= 0.98
        assert eval_res["positive_predictive_value"] >= 0.98
        assert eval_res["f1_score"] >= 0.98


class TestModelArtifactsAndSecurity:
    """Stage 4 & 5: Verified loader, manifests, and playback simulation."""

    def test_baseline_classifier_fit_predict(self):
        base = SingleLeadBaselineClassifier()
        X_mock = np.zeros((10, len(FEATURE_NAMES)))
        y_mock = np.array(["N", "N", "N", "S", "V", "Q", "N", "N", "V", "N"])
        base.fit(X_mock, y_mock)
        preds = base.predict(X_mock)
        assert len(preds) == 10
        # Majority class is N
        assert np.all(preds == "N")

    def test_reject_quarantined_legacy_models(self):
        with pytest.raises(PermissionError, match="Security policy violation"):
            load_verified_single_lead_model("legacy_models/finalmodel.pkl")

    def test_sha256_manifest_verification(self):
        # Build and save a tiny model in a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model = build_single_lead_pipeline(n_estimators=5, max_depth=3)
            X = np.random.randn(20, len(FEATURE_NAMES))
            y = np.array(["N"] * 10 + ["S"] * 3 + ["V"] * 5 + ["Q"] * 2)
            model.fit(X, y)

            m_path, man_path = save_single_lead_model(
                model=model,
                target_classes=list(AAMI_TARGET_CLASSES),
                train_metadata={"test": True},
                output_dir=tmp_path,
                model_filename="test_model.joblib",
            )
            assert m_path.is_file()
            assert man_path.is_file()

            # Load verified model
            loaded_model, manifest = load_verified_single_lead_model(m_path)
            assert manifest["architecture_version"] == "1.0.0"
            preds = loaded_model.predict(X)
            assert len(preds) == 20

            # Tamper with file to verify hash protection
            with open(m_path, "ab") as f:
                f.write(b"corrupted_bytes")
            with pytest.raises(ValueError, match="Integrity check failed"):
                load_verified_single_lead_model(m_path)

    def test_disclaimer_strings_and_replay_banner(self):
        assert DISCLAIMER_RECORDED_REPLAY == "RECORDED DATA REPLAY — NOT LIVE HARDWARE"
        assert "Not a medical device" in DISCLAIMER_SINGLE_LEAD_RESEARCH
        assert "No physical sensors" in DISCLAIMER_SINGLE_LEAD_RESEARCH

    def test_playback_slice_execution(self):
        res = get_playback_slice(record_id="100", start_second=0.0, duration_seconds=5.0)
        assert res.record_id == "100"
        assert res.lead_name == "MLII"
        assert len(res.signal_mv) == 5 * NATIVE_SAMPLING_RATE_HZ
        assert res.disclaimer == "RECORDED DATA REPLAY — NOT LIVE HARDWARE"
        assert len(res.detected_beats) > 0
        assert len(res.reference_annotations) > 0

    def test_app_ui_contains_workflow_c_and_disclaimer(self):
        app_path = Path("app.py")
        assert app_path.is_file()
        content = app_path.read_text(encoding="utf-8")
        assert "Single-Lead Lab" in content
        assert "RECORDED DATA REPLAY — NOT LIVE HARDWARE" in content
        assert "DISCLAIMER_SINGLE_LEAD_RESEARCH" in content


class TestWorkflowIsolation:
    """Verify isolation: Workflow A and Workflow B must remain completely intact."""

    def test_ptbxl_12_lead_imports_and_workflow_b_untouched(self):
        from ecg_lab.waveform_ml.config import TARGET_SUPERCLASSES, LEAD_NAMES
        from ecg_lab.waveform_ml.models import load_verified_model
        assert len(LEAD_NAMES) == 12
        assert list(TARGET_SUPERCLASSES) == ["NORM", "MI", "STTC", "CD", "HYP"]

    def test_image_digitizer_workflow_a_untouched(self):
        from ecg_lab.digitization import digitize_ecg_image, DigitizationStatus
        assert DigitizationStatus.SUCCESS.value == "SUCCESS"


class TestPhase62TaxonomyDisclosures:
    """Phase 6.2 & 6.4: Verify transparent taxonomy disclosure across UI and documentation."""

    TAXONOMY_DISCLOSURE_PHRASE = (
        "Project-defined 4-class beat grouping: N, S, V, Q, with ventricular fusion beats (F) merged into Q. "
        "This is not a standard five-class AAMI EC57 implementation."
    )

    PRIMARY_BENCHMARK_TITLE = "Matched-Detection Conditional Beat-Classification Metrics (N = 47,034)"

    def test_app_ui_contains_taxonomy_disclosure(self):
        content = Path("app.py").read_text(encoding="utf-8")
        assert self.TAXONOMY_DISCLOSURE_PHRASE in content
        assert "Class Q (Fusion & Other)" in content
        assert "Ventricular Fusion (F) & Unclassifiable (Q)" in content
        assert self.PRIMARY_BENCHMARK_TITLE in content
        assert "Secondary Benchmark: Annotation-Centred (Oracle) Classifier Reference" in content

    def test_readme_contains_taxonomy_disclosure(self):
        content = Path("README.md").read_text(encoding="utf-8")
        assert self.TAXONOMY_DISCLOSURE_PHRASE in content
        assert self.PRIMARY_BENCHMARK_TITLE in content
        assert "Secondary Benchmark: Annotation-Centred (Oracle) Classifier Reference" in content
        assert "0.5274" in content  # Macro F1 for Matched-Detection Conditional
        assert "0.5396" in content  # Macro F1 for Secondary Oracle

    def test_model_card_contains_taxonomy_disclosure(self):
        content = Path("docs/single_lead_model_card.md").read_text(encoding="utf-8")
        assert self.TAXONOMY_DISCLOSURE_PHRASE in content
        assert self.PRIMARY_BENCHMARK_TITLE in content
        assert "Secondary Benchmark: Annotation-Centred (Oracle) Classifier Reference" in content

    def test_phase6_report_contains_taxonomy_disclosure(self):
        content = Path("docs/reports/mitbih_arrhythmia_classification.md").read_text(encoding="utf-8")
        assert self.TAXONOMY_DISCLOSURE_PHRASE in content
        assert self.PRIMARY_BENCHMARK_TITLE in content
        assert "47,034" in content
        assert "0.5274" in content
        assert "0.5396" in content


class TestPhase64BenchmarkCorrections:
    """Phase 6.4: Verify audited benchmark support numbers, caption note, and decomposition arithmetic."""

    MANDATORY_EXPLANATORY_NOTE = (
        "Matched-detection conditional beat-classification metrics — classification F1 is calculated only "
        "for detector peaks matched to a reference annotation and retained after feature-window boundary checks. "
        "Detector false positives (115 peaks) and missed reference beats (542 beats, including boundary-dropped "
        "matched beats) are reported separately in detector/count-reconciliation metrics and are not included "
        "as errors in this classifier confusion matrix."
    )

    EXPECTED_MODEL_SHA256 = "3889a5f5ab97803da339710d4ae9fde8ea50f8e5a5ff76ed2015a0fb0b6ec8e9"

    def test_model_artifact_hash_remains_frozen(self):
        import hashlib
        model_path = Path("artifacts/single_lead_ml/single_lead_rf_model.joblib")
        assert model_path.is_file()
        h = hashlib.sha256(model_path.read_bytes()).hexdigest()
        assert h == self.EXPECTED_MODEL_SHA256

    def test_mandatory_caption_note_across_deliverables(self):
        files_to_check = [
            Path("app.py"),
            Path("README.md"),
            Path("docs/single_lead_model_card.md"),
            Path("docs/reports/mitbih_arrhythmia_classification.md"),
            Path("docs/reports/aami_taxonomy_and_beat_mapping.md"),
        ]
        for f in files_to_check:
            assert f.is_file(), f"Missing file {f}"
            content = f.read_text(encoding="utf-8")
            normalized_content = " ".join(content.replace('"', ' ').replace("'", " ").replace("*", " ").replace(">", " ").split())
            normalized_note = " ".join(self.MANDATORY_EXPLANATORY_NOTE.split())
            assert normalized_note in normalized_content, f"Missing mandatory note in {f}"

    def test_audited_supports_and_metrics_in_docs(self):
        # Check supports in README.md, single_lead_model_card.md, app.py
        for path in [Path("README.md"), Path("docs/single_lead_model_card.md"), Path("app.py")]:
            content = path.read_text(encoding="utf-8")
            assert "41,883" in content
            assert "1,757" in content
            assert "3,018" in content
            assert "376" in content
            assert "0.9592" in content
            assert "0.9218" in content
            assert "0.3555" in content
            assert "0.1793" in content
            assert "0.7781" in content
            assert "0.9298" in content
            assert "0.0489" in content
            assert "0.2979" in content

    def test_decomposition_arithmetic_consistency(self):
        # 527 non-boundary FNs + 1 boundary FN + 14 boundary drops = 542 unrepresented beats
        # N: 301 + 1 + 13 = 315
        # S: 25 + 0 + 0 = 25
        # V: 183 + 0 + 1 = 184
        # Q: 18 + 0 + 0 = 18
        non_boundary_fn = {"N": 301, "S": 25, "V": 183, "Q": 18}
        boundary_fn = {"N": 1, "S": 0, "V": 0, "Q": 0}
        boundary_drops = {"N": 13, "S": 0, "V": 1, "Q": 0}
        unrepresented = {c: non_boundary_fn[c] + boundary_fn[c] + boundary_drops[c] for c in non_boundary_fn}

        assert sum(non_boundary_fn.values()) == 527
        assert sum(boundary_fn.values()) == 1
        assert sum(boundary_drops.values()) == 14
        assert sum(unrepresented.values()) == 542
        assert unrepresented == {"N": 315, "S": 25, "V": 184, "Q": 18}

        ref_support = {"N": 42198, "S": 1782, "V": 3202, "Q": 394}
        matched_support = {c: ref_support[c] - unrepresented[c] for c in ref_support}
        assert matched_support == {"N": 41883, "S": 1757, "V": 3018, "Q": 376}
        assert sum(matched_support.values()) == 47034

