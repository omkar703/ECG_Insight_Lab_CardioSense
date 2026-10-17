# ECG Insight Lab — Research Platform

**ECG Image Digitization and Signal Analysis Platform**  
*An academic research prototype for paper ECG image digitization, validation, and reproducible cardiac waveform benchmarking.*

---

## 1. Project Overview & Scope

ECG Insight Lab is an open-source, software-only research platform developed to evaluate the algorithmic boundaries of extracting continuous 1D physiological signals from paper electrocardiograms and evaluating benchmark machine learning models on digital waveform data.

### Non-Clinical Research Disclaimer
> [!IMPORTANT]
> **Research Prototype Only:** This software is designed strictly for academic research and educational exploration. It is **not** a certified medical device and has not been approved for clinical diagnosis, patient triage, or healthcare decision-making. No physical hardware sensors (ESP32/AD8232) are connected.

### Key Architectural Tenets
1. **Two Decoupled Laboratories:**
   - **Workflow A (ECG Image Lab):** Ingestion, format/resolution validation, 12-lead segmentation, and continuous signal digitization with CSV/JSON export.
   - **Workflow B (Waveform ML Lab):** Reproducible diagnostic classification trained on PhysioNet's public **PTB-XL v1.0.3** dataset with verified patient-level splits.
2. **Gated Integration Bridge:** The image digitizer is **not** connected to the diagnostic classifier without independent empirical calibration and validation on paired datasets.
3. **Quiet Academic Editorial Aesthetic:** The user interface follows a restrained, publication-grade design system influenced by academic journals (Source Serif 4, IBM Plex Sans, warm ivory `#F7F6F2`, deep charcoal `#252722`, and forest green `#455D4A`).

---

## 2. Digitization Subsystem (Phase 2)

### Digitization Strategy & Feasibility
- **Active Engine:** `ClassicalBaselineDigitizer` (`ecg_lab/digitization/baseline.py`). Operates natively on CPU using NumPy, SciPy, and Pillow. It extracts 1D continuous waveforms via intensity-weighted centroid tracking across standard clinical $3 \times 4$ 12-lead bounding boxes.
- **Upstream Adapter:** `OpenECGDigitizerAdapter` (`ecg_lab/digitization/adapter.py`). Provides a safe protocol boundary for Ahus-AIM's deep learning U-Net digitizer. Automatically checks runtime dependencies and Git LFS weights, failing safely with guidance if dependencies are missing.
- **Calibration Honesty:** Amplitudes are strictly output as `AmplitudeUnit.RELATIVE` and `CalibrationStatus.UNRECOVERED`. The platform never fabricates millivolts or seconds without empirical grid-spacing calibration.

### Evaluation on Synthetic 12-Lead Benchmark
The platform includes a deterministic synthetic benchmark fixture (`sample_data/synthetic_12lead.png`) with verified ground-truth values (`sample_data/synthetic_12lead_gt.csv`):
- **12-Lead Detection Rate:** 100% (12/12 leads detected)
- **Mean Pearson Correlation ($r$):** $0.6932$ across all 12 leads
- **Mean Absolute Error (MAE):** $0.0725$ in normalized amplitude units
- **Inference Runtime:** $\approx 210\,\text{ms}$ on standard CPU

---

## 3. Repository Structure

```text
code/
├── app.py                     # Streamlit research web application
├── ecg_lab/                   # Core modular library
│   ├── __init__.py            # Package exports
│   ├── config.py              # Editorial design tokens & validation limits
│   ├── models.py              # Typed dataclasses & validation result models
│   ├── validation.py          # Robust image safety & format validation
│   ├── digitization/          # Subsystem for waveform digitization (Workflow A)
│   │   ├── __init__.py
│   │   ├── types.py           # DigitizationResult, LeadSignal, Enums
│   │   ├── adapter.py         # ECGDigitizer protocol & Open-ECG adapter
│   │   ├── baseline.py        # Classical morphological centroid digitizer
│   │   └── service.py         # High-level dispatch & CSV/JSON export
│   └── waveform_ml/           # Subsystem for digital waveform ML (Workflow B)
│       ├── __init__.py
│       ├── config.py          # Constants, official checksums, fold splits
│       ├── dataset.py         # Metadata, SCP mapping, patient split integrity
│       ├── features.py        # 148 deterministic signal features & cache
│       ├── models.py          # Prevalence baseline, One-vs-Rest LR, verified joblib
│       ├── train.py           # Training pipeline & validation threshold tuning
│       ├── evaluate.py        # Multi-label metrics (Precision, Recall, F1, PR-AUC)
│       └── cli.py             # Unified CLI commands
├── tests/                     # Automated test suite (47 unit & integration tests)
│   ├── __init__.py
│   ├── test_validation.py         # Phase 1: Input validation & security safeguards
│   ├── test_digitization.py       # Phase 2: Morphological digitizer & synthetic benchmark
│   ├── test_waveform_dataset.py   # Phase 3: Metadata loading & SCP-statement mapping
│   ├── test_waveform_splits.py    # Phase 3: Patient split integrity & leakage assertions
│   ├── test_waveform_features.py  # Phase 3: Deterministic 148 features & cache invalidation
│   ├── test_waveform_metrics.py   # Phase 3: Multi-label evaluation & threshold tuning
│   └── test_waveform_isolation.py # Phase 3: Cryptographic manifests & legacy quarantine
├── sample_data/               # Small, licensed test assets & synthetic benchmarks
├── legacy/                    # Quarantined historical artifacts (March 2025)
├── docs/                      # Technical reports & audit documentation
│   ├── AUDIT_REPORT.md        # Comprehensive Phase 0 technical audit report
│   ├── PHASE2_FEASIBILITY.md  # Feasibility assessment for upstream digitizer
│   ├── PHASE2_REPORT.md       # Comprehensive Phase 2 implementation report
│   └── PHASE3_REPORT.md       # Comprehensive Phase 3 implementation report
├── requirements.txt           # Pinned, reproducible Python dependencies
└── README.md                  # This documentation
```

---

## 4. Getting Started

### Prerequisites
- Linux / macOS / Windows with **Python 3.10+** (Python 3.12 or 3.14 supported).
- Dedicated virtual environment.

### Setup Instructions

1. **Activate the Virtual Environment:**
   ```bash
   # From code/
   source .venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute Automated Test Suite:**
   ```bash
   pytest -v
   ```
   *Expected output: 47 passed tests in ~3.0s (100% pass rate, zero skips).*

4. **Digital Waveform ML Pipeline (CLI):**
   ```bash
   # Validate dataset metadata and verify zero patient leakage
   python -m ecg_lab.waveform_ml.cli validate --verify-checksums

   # Extract 148 features and build cache
   python -m ecg_lab.waveform_ml.cli extract-features

   # Train naive baseline and lightweight model (thresholds tuned on fold 9 only)
   python -m ecg_lab.waveform_ml.cli train
   ```

5. **Launch Application:**
   ```bash
   streamlit run app.py
   ```
   *Access the web dashboard locally at `http://localhost:8501`.*

---

## 5. Licensing & Citations

- **Open ECG Digitizer Literature:**  
  Ahus-AIM / Electrocardiogram-Digitization. Evaluated for feasibility under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- **PTB-XL ECG Dataset:**  
  Wagner, P., Strodthoff, N., Bousseljot, R. D., Samek, W., & Schaeffter, T. (2022). *PTB-XL, a large publicly available electrocardiography dataset* (version 1.0.3). PhysioNet. [https://doi.org/10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45). Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Application Codebase:**  
  Licensed under the MIT License.
