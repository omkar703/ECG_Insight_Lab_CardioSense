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
   - **Workflow A (ECG Image Lab):** Ingestion, format/resolution validation, perspective-aware lead segmentation, and continuous signal digitization with CSV export.
   - **Workflow B (Waveform ML Lab):** Reproducible diagnostic classification trained on PhysioNet's public **PTB-XL v1.0.3** dataset with verified patient-level splits.
2. **Gated Integration Bridge:** The image digitizer is **not** blindly connected to the diagnostic classifier without independent empirical calibration and validation on paired datasets.
3. **Quiet Academic Editorial Aesthetic:** The user interface follows a restrained, publication-grade design system influenced by academic journals (Source Serif 4, IBM Plex Sans, warm ivory `#F7F6F2`, deep charcoal `#252722`, and forest green `#455D4A`).

---

## 2. Repository Structure

```text
code/
├── app.py                     # Main Streamlit web application
├── ecg_lab/                   # Core modular library
│   ├── __init__.py            # Package exports
│   ├── config.py              # Editorial design tokens & validation limits
│   ├── models.py              # Typed dataclasses & validation result models
│   └── validation.py          # Robust image safety & format validation
├── tests/                     # Automated test suite
│   ├── __init__.py
│   └── test_validation.py     # Unit tests for input validation & safeguards
├── sample_data/               # Small, licensed test assets & synthetic benchmarks
│   ├── README.md              # Sample data manifest & privacy policies
│   └── synthetic_sample.png   # Synthetic 1200x800 test ECG grid pattern
├── legacy/                    # Quarantined historical artifacts (March 2025)
│   ├── README.md              # Historical context & security notices
│   ├── test.py                # Legacy fixed-crop inference script
│   ├── test1.ipynb            # Legacy notebook with recorded PCA failure traces
│   └── *.pkl                  # Historical PCA and ensemble classifier artifacts
├── docs/                      # Technical reports & audit documentation
│   └── AUDIT_REPORT.md        # Comprehensive Phase 0 technical audit report
├── requirements.txt           # Pinned, reproducible Python dependencies
└── README.md                  # This documentation
```

---

## 3. Getting Started

### Prerequisites
- Linux / macOS / Windows with **Python 3.10+** (Python 3.12 or 3.14 supported).
- Dedicated virtual environment.

### Setup Instructions

1. **Activate the Virtual Environment:**
   ```bash
   # From the project root (code/)
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
   *Expected output: 12 passed tests in ~0.10s.*

4. **Launch Application:**
   ```bash
   streamlit run app.py
   ```
   *Access the web dashboard locally at `http://localhost:8501`.*

---

## 4. Quarantined Legacy Artifacts

Historical files from the original prototype have been quarantined in [`legacy/`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/legacy) to preserve academic provenance while ensuring clean, reproducible forward development:
- **`finalmodel.pkl`:** An `sklearn.decomposition.PCA` transformer ($3,060 \to 400$).
- **`Heart_Disease_Prediction_using_ECG (4).pkl`:** An ensemble `VotingClassifier` requiring 400 features.
- *Notice:* Refer to [`docs/AUDIT_REPORT.md`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/docs/AUDIT_REPORT.md) for the complete root-cause analysis of the 3,060-to-400 feature mismatch identified in the legacy implementation.

---

## 5. Licensing & Attribution

- **PTB-XL ECG Dataset:** Wagner et al., PhysioNet, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Open ECG Digitizer:** Ahus-AIM, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Kept strictly behind an external adapter interface.
- **Application Codebase:** Licensed under the MIT License.
