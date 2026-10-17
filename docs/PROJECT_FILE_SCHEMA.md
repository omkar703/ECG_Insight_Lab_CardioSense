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
│   └── waveform_ml/                   # Subsystem for digital waveform ML (Workflow B)
│       ├── config.py                  # PTB-XL constants, fold definitions, SHA-256 checksums
│       ├── dataset.py                 # Metadata parser, SCP-ECG mapping, patient split integrity
│       ├── features.py                # 148 deterministic lead & rhythm feature extractors
│       ├── models.py                  # OvR Logistic Regression, threshold tuner, verified loader
│       ├── train.py                   # Full-fold model training & validation threshold pipeline
│       ├── evaluate.py                # Multi-label metrics & patient-clustered bootstrap
│       └── cli.py                     # Unified CLI command suite
├── notebooks/                         # Interactive educational Jupyter notebooks
│   ├── README.md                      # Notebook guide, execution order & defense script
│   ├── 00_Project_Story_and_Architecture.ipynb # Story, Phase 0–4 history, decoupled architecture
│   ├── 01_ECG_Image_Digitization.ipynb         # Workflow A: Input validation, centroid digitizer, accuracy
│   ├── 02_PTBXL_Data_and_Features.ipynb        # Workflow B: PTB-XL composition, zero leakage, 148 features
│   └── 03_Model_Predictions_and_Evaluation.ipynb # Workflow B: Verified model inference, Fold 10 benchmark
├── tests/                             # Comprehensive automated test suite (64 tests)
│   ├── test_validation.py             # Image security limits, format verification & design tokens
│   ├── test_digitization.py           # Morphological digitizer, synthetic benchmark, CSV/JSON export
│   ├── test_waveform_dataset.py       # PTB-XL metadata, SCP-statement mapping, counts reconciliation
│   ├── test_waveform_splits.py        # Patient split integrity & zero fold leakage assertions
│   ├── test_waveform_features.py      # 148 deterministic signal features & cache invalidation
│   ├── test_waveform_isolation.py     # SHA-256 manifest checks & legacy pickle quarantine
│   ├── test_waveform_metrics.py       # Multi-label metrics & validation threshold tuning
│   ├── test_waveform_training_integrity.py # Full-fold training, patient-cluster bootstrap & delta CI
│   └── test_productization.py         # System diagnostics, in-memory privacy, safe fixtures
├── sample_data/                       # Small, licensed test assets & synthetic benchmarks
│   ├── README.md                      # Sample data documentation & synthetic labeling
│   ├── synthetic_12lead.png           # 12-lead synthetic benchmark (SYNTHETIC DEMO — NOT PATIENT DATA)
│   ├── synthetic_12lead_gt.csv        # Mathematical ground-truth waveform values
│   ├── synthetic_12lead_meta.json     # Synthetic fixture bounding boxes & layout metadata
│   └── synthetic_sample.png           # Fallback synthetic lead strip
├── docs/                              # Technical reports, audits & academic documentation
│   ├── AUDIT_REPORT.md                # Phase 0 technical audit report (monolithic legacy defects)
│   ├── PHASE2_FEASIBILITY.md          # Upstream deep learning digitizer feasibility study
│   ├── PHASE2_REPORT.md               # Phase 2 classical morphological digitizer report
│   ├── PHASE3_REPORT.md               # Phase 3 digital waveform ML initial pipeline report
│   ├── PHASE3_1_EVALUATION_AUDIT.md   # Phase 3.1 label mapping audit & Fold 10 evaluation
│   ├── PHASE3_2_FULL_FOLD_TRAINING_REPORT.md # Full-fold (17,418 records) training report
│   ├── PHASE3_2_1_PAIRED_COMPARISON_AUDIT.md # Paired patient-clustered bootstrap audit
│   ├── PHASE4_DEPLOYMENT_ASSESSMENT.md       # Feasibility, RAM footprint, privacy assessment
│   ├── PHASE4_REPORT.md               # Phase 4 productization & acceptance criteria report
│   ├── PHASE5_REPORT.md               # Phase 5 notebook verification & schema report
│   ├── MODEL_CARD.md                  # Standardized research-use model card
│   ├── DATA_SOURCES_AND_LICENSES.md   # Dataset citations, SCP taxonomy & dependency licenses
│   ├── PROJECT_FILE_SCHEMA.md         # This taxonomy and schema document
│   ├── UI_CAPTURE_GUIDE.md            # Visual walkthrough & manual screenshot guide
│   └── assets/                        # Publication-grade authentic scientific figures
│       ├── synthetic_digitization_eval.png    # 12-lead recovery vs ground truth plot
│       ├── ptbxl_class_distribution.png      # Partition sizes & superclass prevalence
│       ├── ptbxl_12lead_sample.png           # Calibrated 12-lead digital ECG voltage plot
│       ├── model_evaluation_metrics.png      # Fold 10 benchmark performance bar chart
│       └── paired_comparison_bootstrap.png   # Paired bootstrap delta distribution & CIs
├── scripts/                           # Reproducible developer & build automation scripts
│   ├── build_notebooks.py             # Programmatic notebook builder & nbclient executor
│   └── generate_scientific_assets.py  # Scientific figure generation script
├── artifacts/                         # [Tracked / Ignored split] Auditable model artifacts
│   └── waveform_ml/
│       ├── waveform_lr_model.joblib   # Tracked: 10.5 KB trained Logistic Regression model
│       ├── waveform_lr_model.joblib.manifest.json # Tracked: Cryptographic SHA-256 sidecar manifest
│       ├── benchmark_summary.md       # Tracked: Full Fold 10 evaluation benchmark markdown
│       ├── metrics.json               # Tracked: Complete evaluation metrics dictionary
│       ├── paired_model_comparison.json # Tracked: Paired bootstrap comparison results
│       ├── paired_model_comparison.md # Tracked: Paired comparison summary report
│       ├── scp_statements_audit.json  # Tracked: Audit of SCP statement mappings
│       ├── dataset_completeness_manifest.json # Tracked: 21,799 WFDB file manifest
│       └── pilot_evaluation/          # Tracked: Preserved historical Phase 3.1 pilot model
├── data/                              # [GITIGNORED] Large raw physiological data
│   ├── ptb-xl/                        # PhysioNet PTB-XL v1.0.3 (21,799 WFDB records, ~520 MB)
│   │   ├── ptbxl_database.csv         # Tracked/Cached: Dataset metadata
│   │   ├── scp_statements.csv         # Tracked/Cached: Statement annotations
│   │   ├── records100/                # Gitignored: 100 Hz binary waveforms (.dat / .hea)
│   │   └── records500/                # Gitignored: 500 Hz high-resolution waveforms
│   └── cache/                         # [GITIGNORED] Extracted feature vectors cache
└── legacy/                            # [TRACKED - QUARANTINED] Historical student artifacts
    ├── finalmodel.pkl                 # Quarantined: Untrusted pickle (loading blocked by code)
    ├── model_test.pkl                 # Quarantined: Untrusted pickle
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
| `tests/` | **Tracked** | Test Suite | Safe (MIT) | 64 automated unit and integration tests. |
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
