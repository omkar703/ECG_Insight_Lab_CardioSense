# Phase 6 Implementation & Verification Report

**ECG Insight Lab / CardioSense — Phase 6: MIT-BIH Single-Lead Beat ML & Recorded Playback Demo**  
*Ingestion and channel auditing of the MIT-BIH Arrhythmia Database v1.0.0, patient-disjoint partition integrity (records 201 and 202 co-located), autonomous signal-based R-peak detection, AAMI EC57 beat classification, recorded Holter replay simulation, and strict subsystem isolation.*

---

## 1. Executive Summary & Phase Outcome

Following `CardioSense_Phase_6_MITBIH_Single_Lead_ML_Prompt.md`, Phase 6 establishes an independent, third research subsystem: **Workflow C (Single-Lead Beat Lab)**. This phase evaluates the feasibility and performance of automated single-channel beat-type classification on standardized Holter recordings while enforcing scientific boundaries against live hardware claims, single-lead infarction diagnoses, and ground-truth annotation leakage.

### Core Deliverables Completed
1. **Dataset Acquisition & Provenance Gate:**
   - Acquired and validated the complete **MIT-BIH Arrhythmia Database v1.0.0** (PhysioNet; DOI: [10.13026/C2F305](https://doi.org/10.13026/C2F305)).
   - Ingested all 48 half-hour two-channel records (650,000 samples per channel @ 360 Hz, ~109,494 cardiac beats).
   - Documented ODC-By v1.0 license compliance and formal academic attribution in `docs/PHASE6_DATASET_FEASIBILITY.md` and `docs/DATA_SOURCES_AND_LICENSES.md`.
2. **Channel Selection Policy & Physical Calibration:**
   - Evaluated per-record lead configurations: 46 of 48 records contain `MLII` (Modified Limb Lead II), and 2 records (`102`, `104`) contain `V5`.
   - Established deterministic lead selection policy: Primary `MLII`, fallback `V5`.
   - Converted 11-bit ADC integers into physical millivolts using the official system transfer function: $\text{mV} = (\text{ADC} - 1024.0) / 200.0$.
3. **Leak-Free Patient-Disjoint Partition Strategy:**
   - Identified that records **201** and **202** originate from the same patient. Both are strictly forced into the Training partition under subject ID `sub_201_202`.
   - Quarantined 4 paced recordings (`102`, `104`, `107`, `217`) per ANSI/AAMI EC57 standards.
   - Partitions: Training (16 records / 15 subjects, 37,419 beats); Validation (7 records / 7 subjects, 15,725 beats); Held-Out Test (21 records / 21 subjects, 47,561 beats).
   - Confirmed zero subject overlap across partitions ($\text{Leakage} = 0$).
4. **Autonomous Signal-Based R-Peak Detector (Pan-Tompkins):**
   - Implemented a decoupled QRS detector (`ecg_lab/single_lead_ml/detector.py`) operating strictly on raw ECG signals (Butterworth 5–15 Hz bandpass, differentiation, squaring, moving-window integration, adaptive peak detection, refractory lockout $\ge 200\,\text{ms}$).
   - **Crucial Leakage Guard:** Ground-truth annotation indices are **never** consumed at inference.
   - Evaluated across all 21 held-out test patients (47,576 reference beats): Sensitivity = **98.89%**, PPV = **99.76%**, F1 = **99.32%** at $\pm 50\,\text{ms}$ tolerance.
5. **AAMI EC57 Beat Classification Pipeline:**
   - Implemented 47 deterministic timing and morphological features (`ecg_lab/single_lead_ml/features.py`) over a 200-sample window ($-90$ to $+110$ samples @ 360 Hz).
   - Trained a balanced Random Forest classifier (`ecg_lab/single_lead_ml/models.py`) on 37,419 training beats.
   - Evaluated frozen model on 47,561 held-out test beats: **Macro F1 = 0.5396** (vs. baseline 0.2350), **Weighted F1 = 0.9036** (vs. baseline 0.8338).
   - Persisted model with SHA-256 sidecar manifest (`artifacts/single_lead_ml/single_lead_rf_model.joblib.manifest.json`).
6. **Streamlit Single-Lead Lab & Recorded Playback Simulator:**
   - Added Workflow C navigation tab in `app.py`.
   - Embedded mandatory warning banner: `⚠️ RECORDED DATA REPLAY — NOT LIVE HARDWARE`.
   - Interactive calibrated waveform viewer, detected peak markers, AAMI class badges, heart rate estimation, and reference annotation overlay.
7. **Automated Test Suite Expansion:**
   - Created `tests/test_single_lead_ml.py` adding 22 focused test cases.
   - Full test suite expanded from 64 to **86 passed tests** with 100% pass rate.

---

## 2. Strict Boundary Confirmations

- [x] **No Icentia11k Download:** Excluded per prompt instructions; focused entirely on MIT-BIH v1.0.0.
- [x] **No Hardware or Sensor Equivalence Claims:** Prominently documented that MIT-BIH recordings are digitized reel-to-reel Holter cassette tapes from 1975–1979, and do not represent data from absent AD8232 or ESP32 hardware modules.
- [x] **No Acute Coronary Syndrome or Diagnostic Infarction Claims:** Single-lead signals are strictly bounded to beat-level morphological classification (AAMI classes N, S, V, Q); no MI, STTC, CD, or HYP diagnostic claims are made.
- [x] **No Inference Leakage:** R-peak detection in playback and evaluation runs autonomously on the raw signal without referencing ground-truth annotation indices.
- [x] **No Paper-to-Digital Bridge:** Workflow A (optical digitizer) remains strictly isolated from Workflow B and Workflow C.
- [x] **No Retraining or Modification of Workflow B:** The PTB-XL 12-lead Logistic Regression model, 148 feature extractors, and Fold 10 benchmark results remain completely untouched.
- [x] **No Untrusted Deserialization:** Quarantined legacy pickles (`finalmodel.pkl`, etc.) remain blocked by code policy. Single-lead model is verified via bit-for-bit SHA-256 digest.

---

## 3. Detailed Acceptance Criteria Evaluation

| Criterion | Requirement Summary | Status | Verification Evidence & Actual Metrics |
| :--- | :--- | :---: | :--- |
| **Criterion 1** | MIT-BIH acquired from official source, validated, license & citation documented | **PASS** | 48 records validated in `data/mitdb/raw`; documented in `docs/PHASE6_DATASET_FEASIBILITY.md` and `docs/DATA_SOURCES_AND_LICENSES.md` under ODC-By v1.0. |
| **Criterion 2** | Actual per-record channel, sampling rate, and annotation structure inspected; lead policy recorded | **PASS** | Ingested 360 Hz signals; 46 records MLII, 2 records V5. Channel policy: MLII primary, V5 fallback. Physical conversion $(ADC - 1024)/200\,\text{mV}$ verified. |
| **Criterion 3** | Annotation labels mapped from official definitions; no silent relabeling | **PASS** | Mapped standard symbols to AAMI EC57 (`N`, `S`, `V`, `Q`) in `SYMBOL_TO_AAMI`. Non-beat symbols filtered; 4 paced records quarantined. Confirmed in `tests/test_single_lead_ml.py`. |
| **Criterion 4** | Train/val/test partitions subject-disjoint; records 201 and 202 kept together | **PASS** | Records 201 & 202 forced to Train under `sub_201_202`. Train (15 subjects), Val (7 subjects), Test (21 subjects) have exactly 0 overlapping subjects. |
| **Criterion 5** | CPU-friendly model trained and evaluated honestly on held-out test partition | **PASS** | Balanced Random Forest evaluated on 47,561 test beats: Macro F1 = 0.5396, Weighted F1 = 0.9036 across N (0.9418), V (0.8517), S (0.2712), Q (0.0936). |
| **Criterion 6** | Inference does not consume ground-truth annotations; signal detector evaluated | **PASS** | Pan-Tompkins QRS detector operates strictly on raw ECG signal. On 47,576 test beats: Sensitivity = 98.89%, PPV = 99.76%, F1 = 99.32% ($\pm 50\,\text{ms}$ tolerance). |
| **Criterion 7** | Recorded-signal replay is authentic and prominently not-live | **PASS** | Rendered in Streamlit with `RECORDED DATA REPLAY — NOT LIVE HARDWARE` banner; asserted in `tests/test_single_lead_ml.py::test_app_ui_contains_workflow_c_and_disclaimer`. |
| **Criterion 8** | Existing Workflow A and Workflow B code and model results remain unchanged | **PASS** | PTB-XL 12-lead model, 148 features, and optical digitizer code untouched; all 64 legacy unit/integration tests pass identically. |
| **Criterion 9** | All old and new tests pass, with exact outputs recorded | **PASS** | **86 of 86 tests passed** in 9.66 seconds (`pytest tests/ -v`). |
| **Criterion 10** | `docs/PHASE6_REPORT.md` includes checklist, real metrics, hash, manifest, git status | **PASS** | This document, incorporating actual metrics, model SHA-256, git status, and audit trail. |

---

## 4. Quantitative Results & Evaluation Benchmark

### 4.1 Signal-Based R-Peak Detector (Pan-Tompkins) on Held-Out Test Set
Evaluated across all 21 held-out test records against expert reference annotations ($\pm 50\,\text{ms}$ tolerance, $\pm 18$ samples @ 360 Hz):
- **Total Reference Beats:** 47,576
- **Total Detected Peaks:** 47,163
- **True Positives (TP):** 47,048
- **False Positives (FP):** 115
- **False Negatives (FN):** 528
- **Sensitivity:** **98.89%** ($0.9889$)
- **Positive Predictive Value (PPV):** **99.76%** ($0.9976$)
- **Detector F1-Score:** **99.32%** ($0.9932$)

### 4.2 Beat Classification on Held-Out Test Partition (Annotation-Centred Oracle Reference: 21 Patients, 47,561 Beats)

> [!NOTE]
> **Evaluation Protocol Clarification (Phase 6.1 & 6.4 Audits):**  
> The metrics below represent **Protocol 1: Annotation-Centred (Oracle) Classifier Reference**, where beat windows are extracted centred on expert ground-truth annotation indices. For the **Primary Matched-Detection Conditional Beat-Classification Metrics** ($N=47,034$, Macro F1 = 0.5274) and transparent taxonomy disclosures, see **Section 9**.

| Metric | Empirical Prevalence Baseline | Balanced Random Forest Model | Absolute Gain ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Macro F1** | 0.2350 | **0.5396** | **+0.3046** |
| **Weighted F1** | 0.8338 | **0.9036** | **+0.0698** |

### 4.3 Detailed Per-Class Breakdown (Held-Out Test Partition)

| AAMI Class | Description | Test Support | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **N** | Normal / Bundle Branch Block | 42,184 | 0.9595 | 0.9247 | **0.9418** |
| **S** | Supraventricular Ectopic (SVEB/PAC) | 1,782 | 0.4035 | 0.2043 | **0.2712** |
| **V** | Ventricular Ectopic (VEB/PVC) | 3,201 | 0.7912 | 0.9222 | **0.8517** |
| **Q** | Fusion / Paced / Unclassifiable | 394 | 0.0549 | 0.3173 | **0.0936** |
| **Macro Average** | — | **47,561** | **0.5523** | **0.5921** | **0.5396** |
| **Weighted Average** | — | **47,561** | **0.9192** | **0.9002** | **0.9036** |

### 4.4 Held-Out Test Confusion Matrix

| True \ Pred | Predicted N | Predicted S | Predicted V | Predicted Q | Total True |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **True N** | 39,006 | 492 | 535 | 2,151 | 42,184 |
| **True S** | 1,380 | 364 | 38 | 0 | 1,782 |
| **True V** | 203 | 45 | 2,952 | 1 | 3,201 |
| **True Q** | 62 | 1 | 206 | 125 | 394 |
| **Total Pred** | 40,651 | 902 | 3,731 | 2,277 | 47,561 |

*Diagnostic Analysis of Results:*
- **Ventricular Ectopic Beats (VEBs / PVCs):** The model demonstrates high diagnostic sensitivity (**92.22%**) and strong F1 (**0.8517**) on VEBs, driven by the pronounced widening of the QRS core subsegment and elevated local energy ratios.
- **Supraventricular Ectopic Beats (SVEBs / PACs):** Detection of PACs is inherently more challenging in single-lead signals (F1 = **0.2712**, Recall = **20.43%**), because PACs frequently share identical QRS morphology with normal beats and differ primarily in subtle P-wave morphology and shortened pre-RR intervals without full compensatory pauses.
- **Fusion / Unclassifiable Beats (Q):** Low precision (**5.49%**) reflects the extreme class imbalance ($N=394$ in test) and morphological overlap with wide-complex ventricular and paced beats.

---

## 5. Artifact Provenance & Cryptographic Digests

| Artifact | File Path | File Size | Cryptographic SHA-256 Digest |
| :--- | :--- | :--- | :--- |
| **Trained Single-Lead Model** | `artifacts/single_lead_ml/single_lead_rf_model.joblib` | 13.9 MB | `3889a5f5ab97803da339710d4ae9fde8ea50f8e5a5ff76ed2015a0fb0b6ec8e9` |
| **Cryptographic Manifest** | `artifacts/single_lead_ml/single_lead_rf_model.joblib.manifest.json` | 7.4 KB | Verified sidecar binding architecture, schema, splits, and training parameters |
| **Benchmark Summary JSON** | `artifacts/single_lead_ml/single_lead_benchmark_results.json` | 5.1 KB | Full per-class metrics, confusion matrix, and detector evaluations |

---

## 6. Complete Automated Test Suite Execution

```text
$ .venv/bin/pytest tests/ -v
============================= test session starts ==============================
platform linux -- Python 3.14.4, pytest-9.1.1, pluggy-1.7.0 -- /run/media/op/DATA/Omkar/VIT/2y/3 sem/Project_expo/code/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /run/media/op/DATA/Omkar/VIT/2y/3 sem/Project_expo/code
plugins: anyio-4.15.1, platformdirs-4.13.3
collected 86 items

tests/test_digitization.py::test_synthetic_fixture_files_exist PASSED    [  1%]
tests/test_digitization.py::test_digitization_schema_on_synthetic PASSED [  2%]
tests/test_digitization.py::test_synthetic_accuracy_against_ground_truth PASSED [  3%]
tests/test_digitization.py::test_digitization_service_json_export PASSED [  4%]
tests/test_digitization.py::test_digitization_service_csv_export PASSED  [  5%]
tests/test_digitization.py::test_adapter_delegation_fallback PASSED      [  6%]
tests/test_digitization.py::test_calibration_status_values PASSED        [  8%]
tests/test_digitization.py::test_corrupted_image_handling PASSED         [  9%]
tests/test_digitization.py::test_unsupported_format_rejection PASSED     [ 10%]
tests/test_digitization.py::test_zero_persistence_behavior PASSED        [ 11%]
tests/test_digitization.py::test_benchmark_metrics_deterministic PASSED   [ 12%]
tests/test_productization.py::test_system_diagnostics_operational PASSED [ 13%]
tests/test_productization.py::test_diagnostics_fails_on_missing_model PASSED [ 15%]
tests/test_productization.py::test_privacy_in_memory_buffers PASSED      [ 16%]
tests/test_productization.py::test_synthetic_demo_label_in_sample_data PASSED [ 17%]
tests/test_productization.py::test_benchmark_wording_caveats_in_readme PASSED [ 18%]
tests/test_productization.py::test_benchmark_wording_caveats_in_app PASSED [ 19%]
tests/test_single_lead_ml.py::TestDatasetProvenanceAndIntegrity::test_dataset_directory_resolves PASSED [ 20%]
tests/test_single_lead_ml.py::TestDatasetProvenanceAndIntegrity::test_missing_record_raises_error PASSED [ 22%]
tests/test_single_lead_ml.py::TestDatasetProvenanceAndIntegrity::test_channel_selection_prefers_mlii PASSED [ 23%]
tests/test_single_lead_ml.py::TestDatasetProvenanceAndIntegrity::test_channel_selection_fallback_when_mlii_absent PASSED [ 24%]
tests/test_single_lead_ml.py::TestDatasetProvenanceAndIntegrity::test_adc_to_physical_millivolts_conversion PASSED [ 25%]
tests/test_single_lead_ml.py::TestDatasetProvenanceAndIntegrity::test_annotation_parsing_and_aami_mapping PASSED [ 26%]
tests/test_single_lead_ml.py::TestDatasetProvenanceAndIntegrity::test_official_symbol_mapping_integrity PASSED [ 27%]
tests/test_single_lead_ml.py::TestDatasetProvenanceAndIntegrity::test_quarantine_of_paced_records PASSED [ 29%]
tests/test_single_lead_ml.py::TestDatasetProvenanceAndIntegrity::test_record_201_and_202_same_subject PASSED [ 30%]
tests/test_single_lead_ml.py::TestDatasetProvenanceAndIntegrity::test_patient_disjoint_partitions PASSED [ 31%]
tests/test_single_lead_ml.py::TestFeatureExtractionAndDetector::test_window_geometry_constants PASSED [ 32%]
tests/test_single_lead_ml.py::TestFeatureExtractionAndDetector::test_feature_extraction_dimensions_and_finiteness PASSED [ 33%]
tests/test_single_lead_ml.py::TestFeatureExtractionAndDetector::test_detector_operates_without_annotations PASSED [ 34%]
tests/test_single_lead_ml.py::TestFeatureExtractionAndDetector::test_detector_accuracy_on_reference_record PASSED [ 36%]
tests/test_single_lead_ml.py::TestModelArtifactsAndSecurity::test_baseline_classifier_fit_predict PASSED [ 37%]
tests/test_single_lead_ml.py::TestModelArtifactsAndSecurity::test_reject_quarantined_legacy_models PASSED [ 38%]
tests/test_single_lead_ml.py::TestModelArtifactsAndSecurity::test_sha256_manifest_verification PASSED [ 39%]
tests/test_single_lead_ml.py::TestModelArtifactsAndSecurity::test_disclaimer_strings_and_replay_banner PASSED [ 40%]
tests/test_single_lead_ml.py::TestModelArtifactsAndSecurity::test_playback_slice_execution PASSED [ 41%]
tests/test_single_lead_ml.py::TestModelArtifactsAndSecurity::test_app_ui_contains_workflow_c_and_disclaimer PASSED [ 43%]
tests/test_single_lead_ml.py::TestWorkflowIsolation::test_ptbxl_12_lead_imports_and_workflow_b_untouched PASSED [ 44%]
tests/test_single_lead_ml.py::TestWorkflowIsolation::test_image_digitizer_workflow_a_untouched PASSED [ 45%]
tests/test_validation.py::test_valid_png_stream PASSED                   [ 46%]
tests/test_validation.py::test_valid_jpeg_stream PASSED                  [ 47%]
tests/test_validation.py::test_tiff_rejected PASSED                     [ 48%]
tests/test_validation.py::test_bmp_rejected PASSED                      [ 50%]
tests/test_validation.py::test_webp_rejected PASSED                     [ 51%]
tests/test_validation.py::test_file_size_exceeded PASSED                [ 52%]
tests/test_validation.py::test_resolution_too_low PASSED                [ 53%]
tests/test_validation.py::test_resolution_too_high PASSED               [ 54%]
tests/test_validation.py::test_aspect_ratio_too_extreme PASSED          [ 55%]
tests/test_validation.py::test_corrupted_payload PASSED                 [ 56%]
tests/test_validation.py::test_palette_tokens_structure PASSED          [ 58%]
tests/test_validation.py::test_typography_tokens_structure PASSED       [ 59%]
tests/test_waveform_dataset.py::test_resolve_data_dir_default PASSED    [ 60%]
tests/test_waveform_dataset.py::test_metadata_parsing PASSED             [ 61%]
tests/test_waveform_dataset.py::test_record_reading PASSED              [ 62%]
tests/test_waveform_dataset.py::test_scp_statement_mapping PASSED       [ 63%]
tests/test_waveform_dataset.py::test_dataset_completeness_manifest PASSED [ 65%]
tests/test_waveform_dataset.py::test_missing_file_diagnostics PASSED    [ 66%]
tests/test_waveform_dataset.py::test_strat_fold_counts PASSED           [ 67%]
tests/test_waveform_features.py::test_feature_names_count PASSED        [ 68%]
tests/test_waveform_features.py::test_extract_record_features_shape PASSED [ 69%]
tests/test_waveform_features.py::test_feature_values_finite PASSED      [ 70%]
tests/test_waveform_features.py::test_spectral_power_bands PASSED       [ 72%]
tests/test_waveform_features.py::test_deterministic_extraction PASSED   [ 73%]
tests/test_waveform_features.py::test_feature_cache_save_and_load PASSED [ 74%]
tests/test_waveform_features.py::test_cache_invalidation_on_schema_change PASSED [ 75%]
tests/test_waveform_isolation.py::test_rejects_legacy_finalmodel PASSED [ 76%]
tests/test_waveform_isolation.py::test_rejects_arbitrary_untrusted_pickle PASSED [ 77%]
tests/test_waveform_isolation.py::test_rejects_model_without_manifest PASSED [ 79%]
tests/test_waveform_isolation.py::test_verified_model_loader_success PASSED [ 80%]
tests/test_waveform_metrics.py::test_macro_f1_score PASSED              [ 81%]
tests/test_waveform_metrics.py::test_pr_auc_calculation PASSED          [ 82%]
tests/test_waveform_metrics.py::test_evaluation_report_structure PASSED [ 83%]
tests/test_waveform_metrics.py::test_threshold_tuning_improves_or_matches_f1 PASSED [ 84%]
tests/test_waveform_metrics.py::test_threshold_grid_boundaries PASSED   [ 86%]
tests/test_waveform_metrics.py::test_full_fold_evaluator_execution PASSED [ 87%]
tests/test_waveform_metrics.py::test_confidence_intervals_structure PASSED [ 88%]
tests/test_waveform_splits.py::test_ten_strat_folds_present PASSED      [ 89%]
tests/test_waveform_splits.py::test_official_fold_assignments PASSED   [ 90%]
tests/test_waveform_splits.py::test_patient_disjointness PASSED         [ 91%]
tests/test_waveform_splits.py::test_label_coverage_across_splits PASSED [ 93%]
tests/test_waveform_splits.py::test_split_indices_reproducibility PASSED [ 94%]
tests/test_waveform_training_integrity.py::test_full_fold_split_counts PASSED [ 95%]
tests/test_waveform_training_integrity.py::test_model_artifact_sha256_match PASSED [ 96%]
tests/test_waveform_training_integrity.py::test_paired_bootstrap_reproducibility PASSED [ 97%]
tests/test_waveform_training_integrity.py::test_paired_bootstrap_confidence_interval_bounds PASSED [ 98%]
tests/test_waveform_training_integrity.py::test_evaluation_report_contains_caveats PASSED [100%]

============================== 86 passed in 9.66s ==============================
```

---

## 7. Git Working Tree & Unresolved Work

### Git Working Tree Status
- **Current Branch:** `main`
- **Head Revision at Start:** `23eaf8a`
- **Clean Architectural Segregation:**
  - All large raw datasets (`data/mitdb/`, `.data/MIT-BHI`) and caches remain strictly gitignored.
  - Model artifacts (`artifacts/single_lead_ml/`) tracked with cryptographic SHA-256 manifests.
  - No legacy pickle models or unapproved binaries committed.

### Unresolved Work & Future Scope
- **Hardware Integration (Future Phase):** If an AD8232 or ESP32 hardware module is acquired in the future, it must be validated with its own front-end ADC calibration, analog bandpass characteristics, and live serial interface—not conflated with MIT-BIH Holter tape data.
- **SVEB Feature Engineering:** Further research could explore specialized P-wave onset/offset segmenters or local RR dynamic modeling to improve Supraventricular Ectopic Beat (PAC) recall beyond 20.43%.

---

## 8. Conclusion & Sign-Off Request

Phase 6 is **fully implemented and verified**. All 10 acceptance criteria have achieved **PASS** status. Zero regressions were introduced into Workflow A or Workflow B, all 86 unit and integration tests pass cleanly, and recorded replay is clearly distinguished from live hardware.

---

## 9. Phase 6.1 Audit Resolution & Phase 6.2 Transparent Taxonomy Update

Following the formal audit conducted in `docs/PHASE6_1_EVALUATION_PROTOCOL_AUDIT.md`, two key methodological clarifications were resolved and disclosed in Phase 6.2:

### 9.1 Transparent Taxonomy Disclosure
- **Taxonomy Disclosure:** Project-defined 4-class beat grouping: N, S, V, Q, with ventricular fusion beats (F) merged into Q. This is not a standard five-class AAMI EC57 implementation.
- **Composition of Class Q in Non-Paced Records:** In the evaluated non-paced records, **98.2% of Class Q beats are ventricular fusion beats (`F`)** rather than unclassifiable beats. Paced beats (`/` and `f`) reside almost exclusively in the four quarantined records (`102`, `104`, `107`, `217`).

### 9.2 Separation of Evaluation Protocols

#### Primary Benchmark: Matched-Detection Conditional Beat-Classification Metrics (N = 47,034)
Evaluates continuous raw ECG signals without ground-truth annotations. R-peaks are detected by the Pan-Tompkins algorithm, matched to reference annotations within $\pm 50\,\text{ms}$, and classified after feature extraction:

> [!NOTE]
> **Matched-detection conditional beat-classification metrics** — classification F1 is calculated only for detector peaks matched to a reference annotation and retained after feature-window boundary checks. Detector false positives (115 peaks) and missed reference beats (542 beats, including boundary-dropped matched beats) are reported separately in detector/count-reconciliation metrics and are not included as errors in this classifier confusion matrix.

| Class | True Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **N** | 41,883 | 0.9592 | 0.9218 | **0.9401** |
| **S** | 1,757 | 0.3555 | 0.1793 | **0.2384** |
| **V** | 3,018 | 0.7781 | 0.9298 | **0.8472** |
| **Q** | 376 | 0.0489 | 0.2979 | **0.0840** |
| **Macro Average** | **47,034** | **0.5354** | **0.5822** | **0.5274** |
| **Weighted Average** | **47,034** | **0.9178** | **0.8896** | **0.9011** |

- **Detector Performance:** 47,576 reference beats, 47,163 candidate peaks, 47,048 TP, 115 FP (unmatched artifacts), 528 FN. Sensitivity = **98.89%**, PPV = **99.76%**, F1 = **99.32%**.
- **Pipeline Loss Accounting:** 542 reference beats unrepresented (315 N, 25 S, 184 V, 18 Q) due to 527 non-boundary detector misses + 1 boundary detector miss + 14 boundary-dropped matched detections.

#### Secondary Benchmark: Annotation-Centred (Oracle) Classifier Reference
Diagnostic reference isolating classifier feature discrimination assuming ideal QRS localization:
- **Evaluated Windows:** **47,561 beats** (47,576 reference minus 15 boundary truncations)
- **Macro F1:** **0.5396** | **Weighted F1:** **0.9036**
- **Per-Class F1:** N: **0.9418**, S: **0.2712**, V: **0.8517**, Q: **0.0936**

### 9.3 Exact Beat Count Reconciliation
- $47,576$ total reference annotations across the 21 test records.
- $15$ boundary reference beats dropped ($<90$ samples from start or $>649,890$ from end: 14 Class N, 1 Class V in Record 233) $\to \mathbf{47,561}$ oracle beats.
- Pan-Tompkins detector finds $47,163$ peaks: $47,048$ match reference beats within $\pm 50\,\text{ms}$ (TP). $14$ of these match boundary beats (13 Class N, 1 Class V in Record 233) where feature extraction cannot yield a full 200-sample window $\to \mathbf{47,034}$ evaluated matched detections.
- $528$ detector false negatives: $527$ non-boundary misses ($301\text{ N} + 25\text{ S} + 183\text{ V} + 18\text{ Q}$) + $1$ boundary miss (Class N, Record 213, sample 649,992).
- $47,576 - 47,034 = \mathbf{542}$ unrepresented reference beats: $527$ non-boundary misses + $1$ boundary miss + $14$ boundary-dropped matched detections ($315\text{ N} + 25\text{ S} + 184\text{ V} + 18\text{ Q}$).

### 9.4 Zero-Retraining Boundary Integrity
- Model weights, feature schemas, thresholds, partitions, and serialization remained 100% frozen.
- SHA-256 digest of `artifacts/single_lead_ml/single_lead_rf_model.joblib` confirmed unchanged: `3889a5f5ab97803da339710d4ae9fde8ea50f8e5a5ff76ed2015a0fb0b6ec8e9`.
- Phase 6.2 is strictly a transparent documentation, UI disclosure, and audit reconciliation update.

