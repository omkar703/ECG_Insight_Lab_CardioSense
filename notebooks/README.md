# Educational & Demonstration Notebooks Guide

**ECG Insight Lab / CardioSense — Phase 5 Portfolio Deliverable**  
*Interactive teaching labs designed for professor evaluations, code walkthroughs, and pedagogical demonstrations.*

---

## 1. Overview & Demonstration Sequence

The four Jupyter notebooks in this directory provide a complete, interactive walkthrough of the ECG Insight Lab research platform. Each notebook functions as an educational lab with learning objectives, step-by-step code execution, publication-grade figures, and academic defense discussions.

```text
Sequence of Demonstration:
00_Project_Story_and_Architecture.ipynb  --> Historical origins, audit findings, architectural decoupling
01_ECG_Image_Digitization.ipynb          --> Optical image recovery, classical centroid baseline, relative units
02_PTBXL_Data_and_Features.ipynb         --> PTB-XL dataset, patient stratification, 148 deterministic features
03_Model_Predictions_and_Evaluation.ipynb --> Verified ML model inference, Fold 10 benchmark, paired bootstrap
```

### Notebook Inventory

| Notebook | Focus | Primary Outputs & Figures | Expected Runtime |
| :--- | :--- | :--- | :---: |
| **`00_Project_Story_and_Architecture.ipynb`** | Historical evolution, Phase 0–4 audit trail, decoupled architecture | System diagnostics status table, architectural decoupling block diagram | $\approx 2\,\text{s}$ |
| **`01_ECG_Image_Digitization.ipynb`** | Workflow A: Input validation, morphological centroid digitizer, benchmark accuracy | Input image viewer, 12-lead recovery plots, Pearson $r$ ($0.6932$) & MAE ($0.0725$), in-memory CSV/JSON | $\approx 3\,\text{s}$ |
| **`02_PTBXL_Data_and_Features.ipynb`** | Workflow B: PTB-XL v1.0.3 composition, patient leakage verification, feature extraction | Fold sizes table ($0$ leakage), superclass prevalence chart, calibrated 12-lead mV plot, 148 feature vector | $\approx 4\,\text{s}$ |
| **`03_Model_Predictions_and_Evaluation.ipynb`** | Workflow B: Verified model loading, Fold 10 inference, paired bootstrap comparison | SHA-256 integrity check, multi-label probability decisions, benchmark table (F1: 0.6093), bootstrap density plot | $\approx 3\,\text{s}$ |

---

## 2. Environment Setup & Kernel Registration

To run the notebooks with full interactivity in JupyterLab or VS Code:

### Step 1: Install Notebook Dependencies
```bash
# From the code/ directory with active virtual environment:
source .venv/bin/activate
pip install -r requirements-notebooks.txt
```

### Step 2: Register the Dedicated Jupyter Kernel
```bash
python -m ipykernel install --user --name ecg-insight-lab --display-name "Python (ECG Insight Lab)"
```

### Step 3: Launch Jupyter Interface
```bash
jupyter lab notebooks/
# Or: jupyter notebook notebooks/
```
In Jupyter, select the **Python (ECG Insight Lab)** kernel from the top-right kernel menu.

---

## 3. Dataset & Model Prerequisites

The notebooks are pre-executed with all figures and outputs embedded so that evaluators can inspect results without re-running. If re-executing cells live:

1. **Synthetic Image Benchmark:** Pre-packaged in `sample_data/` (`synthetic_12lead.png`, `synthetic_12lead_gt.csv`). No download required.
2. **PTB-XL Dataset:** Requires metadata (`data/ptb-xl/ptbxl_database.csv`) and 100 Hz waveform files (`data/ptb-xl/records100/`). If absent, run:
   ```bash
   python -m ecg_lab.waveform_ml.cli download-missing
   ```
3. **Active Model Artifact:** Pre-packaged in `artifacts/waveform_ml/waveform_lr_model.joblib` with its sidecar SHA-256 manifest.

---

## 4. Live Professor Walkthrough & Defense Script

When presenting this work to an academic professor or technical interviewer:

### 1. The Opening Pitch (Notebook 00)
- Highlight that the project rebuilt a flawed legacy student codebase into a rigorous, publication-grade research platform.
- Emphasize the **strict decoupling**: Workflow A (paper ECG image processing) and Workflow B (calibrated digital waveform ML) are intentionally separate.
- Explain that claiming an optical-image-to-classifier pipeline without empirical calibration and domain adaptation would be scientifically dishonest.

### 2. The Optical Recovery Demonstration (Notebook 01)
- Show that the digitizer runs natively on CPU using classical morphological operations in $\approx 210\,\text{ms}$.
- Point to the benchmark accuracy ($r = 0.6932$, $\text{MAE} = 0.0725$) on the deterministic synthetic fixture.
- Explain the `UNRECOVERED` calibration status: pixel displacement is not converted to millivolts unless grid scaling is proven.
- Emphasize the zero-persistence privacy guarantee: uploads exist only in RAM.

### 3. The Data Science Foundation (Notebook 02)
- Point to the patient-level fold stratifications: **17,418 training records**, **2,183 validation records**, and **2,198 test records**.
- Prove that patient leakage is zero ($0$ overlapping patient IDs).
- Walk through the 148 deterministic physiological features (mean, std, peak-to-peak, spectral power, QRS duration) and explain why a transparent linear model was selected over an opaque deep network.

### 4. The Evaluation Integrity (Notebook 03)
- Highlight the cryptographic trust boundary: the model is loaded only after SHA-256 verification against its manifest.
- Present the Fold 10 benchmark results: **Macro F1 = 0.6093**, **Micro F1 = 0.6392**, **Macro PR-AUC = 0.6292**.
- Explain the paired model comparison: Full-Fold Model outperforms the Pilot Model by **+0.1510 Macro F1** ($95\%$ patient-clustered bootstrap CI $[0.1343, 0.1681]$, $B=2,000$).
- Address the two mandatory caveats:
  1. This is a comparison between *complete pipelines* (differing in training set size and validation sample sizes for threshold tuning: 30 vs. 2,183 records).
  2. 0/2,000 non-positive replicates is an empirical observation from finite resampling, describing test-cohort sampling uncertainty conditional on fixed fitted models.

---

## 5. Troubleshooting & FAQ

- **Kernel not found in Jupyter:** Run `python -m ipykernel install --user --name ecg-insight-lab --display-name "Python (ECG Insight Lab)"`.
- **Missing Module `ecg_lab`:** Launch Jupyter from the `code/` directory, or set `export PYTHONPATH=.` before launching.
- **Decompression bomb warning on images:** Handled automatically by `ecg_lab/validation.py` (maximum resolution capped at $6000 \times 6000$).
