# Project File Schema & Repository Taxonomy

**ECG Insight Lab / CardioSense — Phase 5 Portfolio Deliverable**  
*Comprehensive inventory, structural taxonomy, tracking policy, and artifact provenance for the repository.*

---

## 1. High-Level Architectural Layout

```text
code/
├── app.py                             # Streamlit research web application (Quiet Academic Editorial)
├── requirements.txt                   # Pinned, reproducible production Python dependencies
├── requirements-notebooks.txt         # Optional Jupyter & teaching notebook dependencies
├── README.md                          # Main project portfolio landing page
├── ecg_lab/                           # Core modular Python library
│   ├── config.py                      # Design tokens, color palettes, upload limits
│   ├── models.py                      # ValidationResult, ImageMetadata data structures
│   ├── validation.py                  # In-memory image safety, resolution & format validation
│   ├── diagnostics.py                 # 7-point automated system self-test suite
│   ├── digitization/                  # Subsystem for optical ECG recovery (Workflow A)
│   │   ├── types.py                   # DigitizationResult, LeadSignal, CalibrationStatus
│   │   ├── adapter.py                 # ECGDigitizer protocol & Open-ECG adapter interface
│   │   ├── baseline.py                # Classical morphological centroid tracking digitizer
│   │   └── service.py                 # Ingestion dispatch & CSV/JSON export service
│   ├── waveform_ml/                   # Subsystem for digital waveform ML (Workflow B)
│   │   ├── config.py                  # PTB-XL constants, fold definitions, SHA-256 checksums
│   │   ├── dataset.py                 # Metadata parser, SCP-ECG mapping, patient split integrity
│   │   ├── features.py                # 148 deterministic lead & rhythm feature extractors
│   │   ├── models.py                  # OvR Logistic Regression, threshold tuner, verified loader
│   │   ├── train.py                   # Full-fold model training & validation threshold pipeline
│   │   ├── evaluate.py                # Multi-label metrics & patient-clustered bootstrap
│   │   └── cli.py                     # Unified CLI command suite
│   └── single_lead_ml/                # Subsystem for single-channel beat classification (Workflow C)
│       ├── config.py                  # AAMI mappings, 48-record split definitions, disclaimers
│       ├── dataset.py                 # MIT-BIH CSV & annotation reader, channel selector, ADC-to-mV
│       ├── features.py                # 47 deterministic timing & morphological beat descriptors
│       ├── detector.py                # Signal-based Pan-Tompkins QRS detector (no annotations at inference)
│       ├── models.py                  # Single-lead Random Forest & empirical baseline, verified loader
│       ├── train.py                   # End-to-end training & frozen test benchmark pipeline
│       ├── evaluate.py                # Per-class precision, recall, F1, confusion matrix
│       └── playback.py                # Recorded Holter replay simulator & peak inference
├── notebooks/                         # Interactive educational Jupyter notebooks
│   ├── README.md                      # Notebook guide, execution order & defense script
│   ├── 00_system_architecture_and_diagnostics.ipynb # Narrative, baseline audit trail, decoupled architecture
│   ├── 01_ecg_image_digitization.ipynb         # Workflow A: Input validation, centroid digitizer, accuracy
│   ├── 02_ptbxl_data_and_features.ipynb        # Workflow B: PTB-XL composition, zero leakage, 148 features
│   └── 03_model_predictions_and_evaluation.ipynb # Workflow B: Verified model inference, Fold 10 benchmark
├── tests/                             # Comprehensive automated test suite (146 tests)
│   ├── test_validation.py             # Image security limits, format verification & design tokens
│   ├── test_digitization.py           # Morphological digitizer, synthetic benchmark, CSV/JSON export
│   ├── test_single_lead_ml.py         # MIT-BIH provenance, split integrity, detector, model & taxonomy tests
│   ├── test_waveform_dataset.py       # PTB-XL metadata, SCP-statement mapping, counts reconciliation
│   ├── test_waveform_splits.py        # Patient split integrity & zero fold leakage assertions
│   ├── test_waveform_features.py      # 148 deterministic signal features & cache invalidation
│   ├── test_waveform_isolation.py     # SHA-256 manifest checks & legacy pickle quarantine
│   ├── test_waveform_metrics.py       # Multi-label metrics & validation threshold tuning
│   ├── test_waveform_training_integrity.py # Full-fold training, patient-cluster bootstrap & delta CI
│   ├── test_productization.py         # System diagnostics, in-memory privacy, safe fixtures
│   ├── test_hardware_integration.py   # Hardware prototype safety gates & model isolation
│   ├── test_simulation_inference.py   # AD8232 simulation, protocol inversion & AppTest checks
│   ├── test_clinical_explanations.py  # Groq clinical explanation service & privacy guards
│   └── test_dashboard_helpers.py      # UI state navigation & callback helpers
├── sample_data/                       # Small, licensed test assets & synthetic benchmarks
│   ├── README.md                      # Sample data documentation & synthetic labeling
│   ├── synthetic_12lead.png           # 12-lead synthetic benchmark (SYNTHETIC DEMO — NOT PATIENT DATA)
│   ├── synthetic_12lead_gt.csv        # Mathematical ground-truth waveform values
│   ├── synthetic_12lead_meta.json     # Synthetic fixture bounding boxes & layout metadata
│   ├── sample_12lead_ecg_paper.png    # High-contrast 12-lead digitized paper sample image
│   ├── mitbih_demo_record_100.json    # Calibrated MIT-BIH Record 100 sample fixture
│   └── ptbxl_demo_records.json        # Calibrated 12-lead PTB-XL demo waveforms
├── hardware/                          # AD8232 + Arduino Uno hardware prototype
│   ├── README.md                      # Hardware architecture, BOM, and simulation guide
│   ├── PINOUT_AND_SAFETY.md           # Electrical safety rules & biomedical isolation warning
│   ├── validate_simulation_pipeline.py # Standalone simulation & inference validation script
│   ├── diagrams/                      # Editable SVG wiring diagram and pin maps
│   │   ├── ad8232_arduino_uno_wiring.svg # Editable vector wiring schematic with safety banners
│   │   └── pin_map.md                 # Pin tables, electrical levels, and wire color codes
│   ├── firmware/                      # Microcontroller sketches & upload guides
│   │   └── ad8232_arduino_uno/        # 360 Hz deterministic acquisition sketch (.ino)
│   ├── host/                          # Serial protocol parser, capture, replay & ADC simulator
│   ├── signal_processing/             # Digital filters (zero-phase & causal) and calibration guards
│   ├── simulation/                    # Simulation profiles & sample configurations
│   ├── tests/                         # Hardware-specific test suite (46 automated tests)
│   └── reports/                       # Formal hardware validation reports
│       ├── hardware_offline_validation.md     # Offline hardware simulation validation report
│       ├── simulation_inference_validation.md # Streamlit simulation and inference benchmark report
│       └── simulation_validation_results.json # Structured benchmark metrics output
├── docs/                              # Technical reports, audits & academic documentation
│   ├── audit_report.md                # Technical baseline audit report (monolithic legacy defects)
│   ├── model_card.md                  # Standardized 12-lead PTB-XL model card
│   ├── single_lead_model_card.md      # Standardized MIT-BIH single-lead beat model card
│   ├── data_sources_and_licenses.md   # Dataset citations, SCP/AAMI taxonomies & licenses
│   ├── deployment_guide.md            # Deployment guide, containerization & cloud hosting
│   ├── project_file_schema.md         # This taxonomy and schema document
│   ├── ui_capture_guide.md            # Visual walkthrough & manual screenshot guide
│   ├── reports/                       # Formal scientific milestone reports
│   │   ├── optical_digitization_feasibility.md
│   │   ├── optical_digitization_baseline.md
│   │   ├── ptbxl_waveform_ml_baseline.md
│   │   ├── ptbxl_evaluation_audit.md
│   │   ├── ptbxl_full_fold_training.md
│   │   ├── ptbxl_paired_bootstrap_comparison.md
│   │   ├── deployment_feasibility_assessment.md
│   │   ├── system_performance_and_optimization.md
│   │   ├── pedagogical_notebooks_deliverable.md
│   │   ├── mitbih_dataset_feasibility.md
│   │   ├── mitbih_arrhythmia_classification.md
│   │   ├── pan_tompkins_evaluation_audit.md
│   │   ├── aami_taxonomy_and_beat_mapping.md
│   │   ├── detector_count_reconciliation_audit.md
│   │   ├── single_lead_benchmark_corrections.md
│   │   └── groq_clinical_explanations_integration.md
│   └── assets/                        # Publication-grade authentic scientific figures
│       ├── favicon.png                # Platform cardiac vector icon
│       ├── cardiac_pipeline_overview.jpg     # Complete pipeline schematic
│       ├── decoupled_ml_architectures.jpg    # Dual-model comparison
│       ├── five_stage_pipeline_infographic.jpg # 5-stage clinical infographic
│       ├── model_evaluation_metrics.png      # Fold 10 benchmark performance bar chart
│       └── paired_comparison_bootstrap.png   # Paired bootstrap delta distribution & CIs
├── scripts/                           # Reproducible developer & build automation scripts
│   ├── build_notebooks.py             # Programmatic notebook builder & nbclient executor
│   ├── download_mitbih.py             # MIT-BIH dataset retrieval script
│   └── generate_scientific_assets.py  # Scientific figure generation script
├── artifacts/                         # [Tracked / Ignored split] Auditable model artifacts
│   ├── waveform_ml/
│   │   ├── waveform_lr_model.joblib   # Tracked: 10.5 KB trained Logistic Regression model
│   │   ├── waveform_lr_model.joblib.manifest.json # Tracked: Cryptographic SHA-256 sidecar manifest
│   │   ├── benchmark_summary.md       # Tracked: Full Fold 10 evaluation benchmark markdown
│   │   ├── metrics.json               # Tracked: Complete evaluation metrics dictionary
│   │   ├── paired_model_comparison.json # Tracked: Paired bootstrap comparison results
│   │   └── paired_model_comparison.md # Tracked: Paired comparison summary report
│   └── single_lead_ml/
│       ├── single_lead_rf_model.joblib # Tracked: Balanced Random Forest model (SHA-256 verified)
│       ├── single_lead_rf_model.joblib.manifest.json # Tracked: Cryptographic SHA-256 sidecar manifest
│       └── single_lead_benchmark_results.json # Tracked: Complete held-out test evaluation results
├── data/                              # [GITIGNORED] Large raw physiological data
│   ├── ptb-xl/                        # PhysioNet PTB-XL v1.0.3 (21,799 WFDB records, ~520 MB)
│   ├── mitdb/                         # PhysioNet MIT-BIH Arrhythmia Database v1.0.0
│   └── cache/                         # [GITIGNORED] Extracted feature vectors cache
└── legacy/                            # [TRACKED - QUARANTINED] Historical student artifacts
    ├── finalmodel.pkl                 # Quarantined: Untrusted pickle (loading blocked by code)
    ├── Heart_Disease_Prediction_using_ECG (4).pkl # Quarantined: Legacy model
    ├── test_image.jpg                 # Sample paper ECG photograph from legacy work
    └── README.md                      # Quarantine warning and historical context
```

---

## 2. Directory Classification & Tracking Policy

| Directory / File Pattern | Git Tracking Status | Storage Type | Distribution Safety | Description |
| :--- | :---: | :---: | :---: | :--- |
| `ecg_lab/` | **Tracked** | Source Code | Safe (MIT) | Production modular Python package. |
| `app.py` | **Tracked** | Source Code | Safe (MIT) | Streamlit web application. |
| `tests/` | **Tracked** | Test Suite | Safe (MIT) | 101 automated unit and integration tests. |
| `hardware/` | **Tracked** | Hardware Prototype | Safe (MIT) | AD8232 + Uno firmware, diagrams, protocol & 46 tests. |
| `notebooks/` | **Tracked** | Teaching Labs | Safe (MIT) | 4 educational notebooks with embedded outputs. |
| `sample_data/` | **Tracked** | Fixtures | Safe (MIT) | Synthetic images and ground-truth CSVs (No patient data). |
| `docs/` | **Tracked** | Documentation | Safe (MIT) | Audit reports, model card, licenses, capture guides. |
| `docs/assets/` | **Tracked** | Visual Assets | Safe (MIT) | Publication-grade scientific figures. |
| `scripts/` | **Tracked** | Scripts | Safe (MIT) | Automation scripts for building notebooks and figures. |
| `artifacts/waveform_ml/` | **Tracked** | Model & Manifests | Safe | Small vetted model (10.5 KB) + SHA-256 manifests. |
| `data/ptb-xl/records*` | **Gitignored** | Raw Waveforms | Requires PhysioNet | 21,799 WFDB recordings (520 MB); downloaded via CLI. |
| `data/cache/` | **Gitignored** | Local Cache | Ephemeral | Extracted 148-feature arrays; regenerated on demand. |
| `.venv/`, `__pycache__/` | **Gitignored** | Runtime | Excluded | Virtual environment and bytecode. |
| `legacy/` | **Tracked** | Historical Archive | Quarantined | Legacy student artifacts preserved for audit comparison. |

---

## 3. Data & Model Provenance

### Waveform Machine Learning Model (`waveform_lr_model.joblib`)
- **Trained By:** `python -m ecg_lab.waveform_ml.cli train`
- **Source Data:** PTB-XL v1.0.3 (Folds 1–8: 17,418 records, 15,023 patients)
- **Validation Tuning:** Fold 9 (2,183 records, 1,942 patients) for F1-maximizing thresholds
- **Held-Out Test:** Fold 10 (2,198 records, 1,904 patients)
- **Algorithm:** One-vs-Rest Logistic Regression ($C=1.0$, balanced class weights)
- **Feature Schema:** Version 1.0.0 (148 deterministic features)
- **Artifact Size:** 10,472 bytes (10.5 KB)
- **SHA-256 Digest:** `55d364ed57cc971167f7ad813443c249f792567e406cce4c3dd84188829328d7`
- **Trust Manifest:** `waveform_lr_model.joblib.manifest.json`

### Synthetic Digitization Fixture (`synthetic_12lead.png`)
- **Generated By:** Mathematical synthesis (superimposed sinusoids & Gaussian wavelets)
- **Ground Truth:** `sample_data/synthetic_12lead_gt.csv` (255 samples per lead $\times$ 12 leads $= 3,060$ points)
- **Patient Data:** None (`SYNTHETIC DEMO — NOT PATIENT DATA`)
- **Calibration Status:** `UNRECOVERED` (Relative normalized amplitude units)
- **Benchmark Metrics:** 12/12 leads detected, Mean Pearson $r = 0.6932$, Mean MAE $= 0.0725$
