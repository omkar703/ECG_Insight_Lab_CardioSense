# CardioSense Waveform ML Model Card (Research Release v1.0.0)

**Date:** 2026-10-17  
**Model Name:** `waveform_lr_model.joblib`  
**Model Architecture:** Standardized Features + One-vs-Rest Balanced Logistic Regression  
**Feature Schema Version:** 1.0.0 (148 deterministic signal descriptors)  
**Trained Artifact SHA-256:** `55d364ed57cc971167f7ad813443c249f792567e406cce4c3dd84188829328d7`  
**License:** MIT License (Code) / [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (PTB-XL Data Attribution)  

---

## 1. Model Overview

The **CardioSense Waveform ML Model** is a lightweight, fully reproducible multi-label classification pipeline trained on digital voltage waveforms from the PhysioNet **PTB-XL v1.0.3** benchmark dataset. It maps 10-second 12-lead digital electrocardiograms (sampled at 100 Hz) to five diagnostic superclasses:
1. **NORM:** Normal ECG
2. **MI:** Myocardial Infarction
3. **STTC:** ST/T Changes
4. **CD:** Conduction Disturbance
5. **HYP:** Hypertrophy

The model is designed as an interpretable research baseline to establish transparent, deterministic performance metrics against naive prevalence heuristics and earlier pilot pipelines.

---

## 2. Intended & Prohibited Uses

### 2.1 Intended Academic Uses
- **Algorithmic Benchmarking:** Providing a reproducible, computationally lightweight linear baseline for physiological signal classification.
- **Methodological Auditing:** Serving as a reference for statistical evaluation protocols, patient-clustered bootstrapping, and threshold optimization.
- **Educational Demonstrations:** Demonstrating reproducible machine learning pipelines, split integrity verification, and cryptographic artifact management.

### 2.2 Strictly Prohibited Uses
- **Clinical Decision Support:** This model is **NOT** a certified medical device and has never been validated in a prospective clinical trial. It must **NOT** be used for patient diagnosis, acute coronary syndrome triage, arrhythmia detection, or therapy selection.
- **Paper ECG Inference:** This model operates exclusively on calibrated digital voltage signals ($\text{mV}$). Passing uncalibrated morphological traces from paper ECG scans is strictly prohibited due to severe domain shift and unverified physical calibration.
- **Commercial Diagnostic Marketing:** The model must not be deployed or marketed as a healthcare product.

---

## 3. Training & Evaluation Data

### 3.1 Dataset Attribution & Provenance
- **Dataset:** PTB-XL v1.0.3 (Wagner et al., 2020/2022).
- **Signal Modality:** 12-lead digital voltage recordings, downsampled to 100 Hz (`records100/`), 1,000 samples per lead over 10 seconds.
- **Label Standard:** Mapped from official clinical statements (`scp_statements.csv`) following the SCP-ECG European standard.

### 3.2 Partition Splits & Patient Disjointness
Training, validation, and testing follow the official PTB-XL `strat_fold` assignments to enforce strict patient separation:

| Partition | Folds | Records ($N$) | Unique Patients | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Training** | Folds 1–8 | 17,418 | 15,023 | Feature scaling, baseline base rates, and model weight estimation |
| **Validation** | Fold 9 | 2,183 | 1,942 | Per-class decision threshold tuning ($\tau$) |
| **Held-Out Test** | Fold 10 | 2,198 | 1,904 | Final offline evaluation with patient-clustered bootstrap |

*Mathematical Split Integrity:* $\text{Patients}(\text{Train}) \cap \text{Patients}(\text{Val}) \cap \text{Patients}(\text{Test}) = \emptyset$ (Verified 0 patient leakage).

---

## 4. Feature Extraction Schema (Version 1.0.0)

The pipeline extracts **148 deterministic numerical features** sequentially per record:
1. **Per-Lead Descriptors (144 features):** 12 statistical metrics calculated independently across each of the 12 leads (`I`, `II`, `III`, `aVR`, `aVL`, `aVF`, `V1`, `V2`, `V3`, `V4`, `V5`, `V6`):
   - Mean, Standard Deviation, Median, Minimum, Maximum, Peak-to-Peak Range
   - Skewness, Kurtosis, Root-Mean-Square (RMS)
   - Zero-Crossing Rate, Signal Energy, Mean Absolute Derivative
2. **Cross-Lead Descriptors (4 features):**
   - Global Limb Lead RMS (Leads I, II, III, aVR, aVL, aVF)
   - Global Precordial Lead RMS (Leads V1 through V6)
   - Maximum Absolute Global Amplitude
   - Limb-to-Precordial Energy Ratio

---

## 5. Model Architecture & Decision Thresholds

- **Preprocessing:** `sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)`
- **Classifier:** `sklearn.multiclass.OneVsRestClassifier(LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs'))`
- **Validation-Tuned Decision Thresholds (Fold 9 Only):**
  - **NORM:** $\tau = 0.50$
  - **MI:** $\tau = 0.50$
  - **STTC:** $\tau = 0.55$
  - **CD:** $\tau = 0.55$
  - **HYP:** $\tau = 0.65$

---

## 6. Quantitative Performance Metrics (Fold 10, N=2,198)

### 6.1 Benchmark Metrics Summary

| Superclass | Support ($N$) | Threshold ($\tau$) | Precision | Recall (Sens.) | Specificity | F1 Score | PR-AUC | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NORM** | 963 | 0.50 | 0.6995 | 0.8484 | 0.7158 | **0.7668** | 0.8128 | 0.8647 |
| **MI** | 550 | 0.50 | 0.4708 | 0.6891 | 0.7415 | **0.5594** | 0.5468 | 0.7857 |
| **STTC** | 521 | 0.55 | 0.5173 | 0.7159 | 0.7925 | **0.6006** | 0.5856 | 0.8258 |
| **CD** | 496 | 0.55 | 0.5581 | 0.6492 | 0.8502 | **0.6002** | 0.6671 | 0.8351 |
| **HYP** | 262 | 0.65 | 0.4520 | 0.6107 | 0.8998 | **0.5195** | 0.5338 | 0.8277 |
| **Macro Average** | **2,198** | — | **0.5395** | **0.7027** | **0.7999** | **0.6093** | **0.6292** | **0.8278** |
| **Micro Average** | **2,198** | — | **0.5759** | **0.7180** | — | **0.6392** | — | — |

### 6.2 Patient-Clustered Bootstrap Uncertainty (B=1,000, 1,904 Patient Clusters)
- **Macro F1 (95% CI):** **[0.5910, 0.6269]** (Bootstrap Mean: 0.6091, SE: 0.0092)

### 6.3 Paired Pipeline Comparison vs. Phase 3.1 Pilot Model (B=2,000, Seed: 42)
- **Point Delta ($\Delta = \text{Macro F1}_{\text{full}} - \text{Macro F1}_{\text{pilot}}$):** **+0.1510**
- **Bootstrap Standard Error ($SE_{\Delta}$):** **0.0087**
- **95% Paired Confidence Interval:** **[0.1343, 0.1681]**
- **Empirical Non-Positive Rate:** 0 / 2,000 replicates non-positive ($P \le 0$).
- *Methodological Caveat:* This comparison measures the difference between two complete pipelines (including training data scale and validation threshold-tuning sample sizes). The interval describes test-cohort sampling uncertainty conditional on fixed fitted models, not all retraining variability.

---

## 7. Technical Limitations & Bias Considerations

1. **Demographic & Geographic Specificity:** PTB-XL was collected in a single European healthcare system between 1987 and 2001. Performance may degrade on pediatric cohorts, ambulatory wearable sensors, or ethnically diverse patient groups.
2. **Frequency Band Limitation:** Evaluated exclusively at 100 Hz downsampled resolution. High-frequency notch filtering (e.g. pacemaker spikes, late potentials) cannot be reliably detected.
3. **Class Confounding:** Multi-label co-occurrence (e.g., MI with concurrent STTC or CD) is modeled through One-vs-Rest independent sigmoids, which do not model joint label covariance.
4. **Offline Evaluation Only:** Evaluation reflects retrospective dataset metrics, which systematically overestimate performance compared to live, real-world deployment.

---

## 8. Cryptographic Provenance & Delivery

- **Primary Model File:** `artifacts/waveform_ml/waveform_lr_model.joblib`
- **Manifest:** `artifacts/waveform_ml/waveform_lr_model.joblib.manifest.json`
- **Verification Rule:** The model must only be loaded via `ecg_lab.waveform_ml.models.load_verified_model()`. Deserialization is automatically rejected if the SHA-256 digest does not match the manifest.
