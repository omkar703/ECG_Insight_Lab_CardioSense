# Verified Scientific Results & Clinical Limitations

**Project:** CardioSense — ECG Insight Lab  
**Subtitle:** Reproducible Machine Learning and Signal Analysis for ECG Data  
**Status:** Educational & Academic Research Prototype  
**Date:** October 2026  

---

## 1. Executive Summary of Verified Results

All metrics documented below were generated from reproducible local runs on held-out test cohorts and audited against cryptographic model manifests:

| Subsystem / Benchmark | Dataset Cohort | Primary Metric | Baseline Metric | Status |
| :--- | :--- | :---: | :---: | :---: |
| **12-Lead Pathology ML (Workflow B)** | PTB-XL Fold 10 ($N=2,198$) | **Macro F1: 0.6093** | Pilot Model F1: 0.4583 ($\Delta = +0.1510$) | **VERIFIED** |
| **12-Lead PR-AUC (Workflow B)** | PTB-XL Fold 10 ($N=2,198$) | **Macro PR-AUC: 0.6292** | Naive Prevalence: 0.2540 | **VERIFIED** |
| **Single-Lead Beat ML — Oracle (Workflow C)** | MIT-BIH Test ($N=47,561$) | **Macro F1: 0.5396** | Majority Class: 0.2350 | **VERIFIED** |
| **Single-Lead Beat ML — Matched (Workflow C)** | MIT-BIH Test ($N=47,034$) | **Macro F1: 0.5274** | Weighted F1: 0.9011 | **VERIFIED** |
| **Autonomous QRS Detector (Workflow C)** | MIT-BIH Test ($N=47,576$ beats) | **Detector F1: 0.9932** | Sens: 98.89%, PPV: 99.76% | **VERIFIED** |
| **Class Imbalance Experiment (Workflow C)** | MIT-BIH Test ($N=47,561$) | **Candidate 3 F1: 0.5592** | Class S Recall: +6.68% | **VERIFIED** |
| **AD8232 Offline Telemetry (Hardware)** | Simulated ADC Transport | **Reconstruction MAE: 0.0000 mV** | Perfect round-trip framing | **VERIFIED** |
| **Optical Signal Recovery (Archived)** | Synthetic 12-Lead Fixture | **Pearson $r$: 0.6932** | $\text{MAE} = 0.0725$ (relative units) | **ARCHIVED** |

---

## 2. Workflow B: 12-Lead Diagnostic Classification (PTB-XL)

### 2.1 Benchmark Results on Held-Out Test Fold 10 ($N=2,198$ Records, $1,904$ Patients)

Evaluated across all five SCP-ECG diagnostic superclasses with validation-tuned decision thresholds:

| Diagnostic Superclass | Test Support | Decision Threshold | Precision | Recall (Sensitivity) | F1-Score | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NORM** (Normal ECG) | 963 | 0.50 | 0.6995 | 0.8484 | **0.7668** | 0.8128 |
| **MI** (Myocardial Infarction) | 550 | 0.50 | 0.4708 | 0.6891 | **0.5594** | 0.5468 |
| **STTC** (ST/T Changes) | 521 | 0.55 | 0.5173 | 0.7159 | **0.6006** | 0.5856 |
| **CD** (Conduction Disturbance) | 496 | 0.55 | 0.5581 | 0.6492 | **0.6002** | 0.6671 |
| **HYP** (Hypertrophy) | 262 | 0.65 | 0.4520 | 0.6107 | **0.5195** | 0.5338 |
| **Macro Average** | 2,198 | — | 0.5633 | 0.6667 | **0.6093** | 0.6292 |
| **Micro Average** | 2,198 | — | 0.5828 | 0.7077 | **0.6392** | — |

*Model Artifact:* `artifacts/waveform_ml/waveform_lr_model.joblib`  
*SHA-256 Digest:* `eaec703ce5317781a9eb5332f939e6a25e24bcf8f6d7aeb150c226fc97931c81`

### 2.2 Paired Patient-Clustered Bootstrap Comparison
A paired bootstrap analysis ($B=2,000$ replicates, random seed 42) evaluated difference in Macro F1 between the Phase 3.1 Pilot Model ($N_{\text{train}}=240$) and the Phase 3.2 Full-Fold Model ($N_{\text{train}}=17,418$):
- **Pilot Macro F1:** $0.4583$ vs. **Full-Fold Macro F1:** $0.6093$
- **Point Estimate Difference ($\Delta$):** $+0.1510$
- **95% Clustered Bootstrap Confidence Interval:** $[+0.1343, +0.1681]$ ($\text{SE} = 0.0087$)
- **Statistical Significance:** $p < 0.0001$ (strictly superior with zero CI overlap with 0).

---

## 3. Workflow C: Single-Lead Arrhythmia Beat Analysis (MIT-BIH)

### 3.1 Dual Benchmark Evaluation Framework
Single-lead classifier performance is reported under two distinct protocols to ensure transparency:

#### A. Oracle / Reference-Beat Benchmark ($N=47,561$ beats across 21 held-out test patients)
Evaluates classifier performance given ground-truth R-peak locations:

| Class | Support | Precision | Recall | F1-Score | Confusion Matrix (Row: True, Col: Pred) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **N** (Normal / BBB) | 42,184 | 0.9595 | 0.9247 | **0.9418** | `[39006, 492, 535, 2151]` |
| **S** (Supraventricular Ectopic) | 1,782 | 0.4035 | 0.2043 | **0.2712** | `[1380, 364, 38, 0]` |
| **V** (Premature Ventricular) | 3,018 | 0.7912 | 0.9222 | **0.8517** | `[203, 45, 2952, 1]` |
| **Q** (Fusion / Other) | 394 | 0.0549 | 0.3173 | **0.0936** | `[62, 1, 206, 125]` |
| **Macro Average** | 47,561 | 0.5523 | 0.5921 | **0.5396** | — |
| **Weighted Average** | 47,561 | 0.9161 | 0.9036 | **0.9036** | — |

*Model Artifact:* `artifacts/single_lead_ml/single_lead_rf_model.joblib`  
*SHA-256 Digest:* `3889a5f5ab97803da339710d4ae9fde8ea50f8e5a5ff76ed2015a0fb0b6ec8e9`

#### B. Matched-Detection Conditional Benchmark ($N=47,034$ beats across 21 test records)
Evaluates classifier performance on peaks proposed autonomously by the Pan-Tompkins detector and matched within $\pm 50\,\text{ms}$ ($\pm 18$ samples @ 360 Hz):

| Class | Matched Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **N** (Normal / BBB) | 41,883 | 0.9592 | 0.9218 | **0.9401** |
| **S** (Supraventricular Ectopic) | 1,757 | 0.3555 | 0.1793 | **0.2384** |
| **V** (Premature Ventricular) | 3,018 | 0.7781 | 0.9298 | **0.8472** |
| **Q** (Fusion / Other) | 376 | 0.0489 | 0.2979 | **0.0840** |
| **Macro Average** | 47,034 | 0.5354 | 0.5822 | **0.5274** |
| **Weighted Average** | 47,034 | 0.9161 | 0.8988 | **0.9011** |

#### C. Autonomous Pan-Tompkins QRS Detector Accounting
Evaluated on continuous raw signals across all 21 test records without ground-truth annotations:
- **Total Reference Beats:** $47,576$
- **Total Detected Peaks:** $47,163$
- **True Positives (matched within $\pm 50\,\text{ms}$):** $47,048$
- **False Positives (spurious peak detections):** $115$
- **False Negatives (missed reference beats):** $528$
- **Boundary Dropped Peaks ($<90$ or $>L-110$ samples):** $14$ ($47,048 - 14 = 47,034$)
- **Sensitivity:** **98.89%** | **Positive Predictive Value (PPV):** **99.76%** | **Detector F1:** **99.32%**

---

## 4. Controlled Research Experiment: Class Imbalance

### 4.1 Evaluation Summary Across Architectures

| Architecture | Validation Macro F1 | Held-Out Test Macro F1 | Test Weighted F1 | Class N F1 | Class S F1 | Class V F1 | Class Q F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Balanced RF)** | **0.4469** *(Selected)* | **0.5396** | 0.9036 | 0.9418 | 0.2712 | 0.8517 | 0.0936 |
| **Candidate 1 (Balanced Subsample RF)** | 0.4355 | 0.5066 | 0.9160 | 0.9608 | 0.1234 | 0.8707 | 0.0716 |
| **Candidate 2 (Balanced Logistic Regression)** | 0.4468 | 0.5036 | 0.7969 | 0.8214 | 0.2458 | 0.8700 | 0.0772 |
| **Candidate 3 (Minority-Boosted RF)** | 0.4151 | **0.5592** | 0.9037 | 0.9369 | **0.3245** | **0.8889** | 0.0867 |

### 4.2 Scientific Analysis & Trade-Offs
1. **Minority Recall vs. Precision Trade-Off:** Candidate 3 (cost-sensitive boosting) increased Class S recall from $20.43\%$ to $27.10\%$ ($+6.68\%$) and Class S F1 from $0.2712$ to $0.3245$ ($+0.0533$). However, this shift increased false positives on normal sinus beats, lowering Class N precision slightly.
2. **Strict Leakage Prevention:** Because Candidate 3 achieved a lower Macro F1 on the validation set ($0.4151$ vs. $0.4469$), the baseline model was selected under strict validation protocol without peeking at the test set.
3. **Repository Policy:** The production model was not overwritten and remains frozen at SHA-256 `3889a5f5...`.

---

## 5. Critical Clinical & Engineering Limitations

### 5.1 Non-Diagnostic Academic Prototype
- CardioSense is an educational and research prototype developed to study biomedical signal processing and machine learning methodologies.
- It is **not** a certified medical device and has **not** been approved or cleared by the FDA (United States), CE (European Union), or CDSCO (India) for diagnostic use, patient monitoring, triage, or clinical management.
- Algorithmic outputs are statistical pattern estimates and must never be used as a substitute for professional clinical judgment by board-certified cardiologists.

### 5.2 Decoupled Subsystem Air-Gaps
- **Optical Trace Air-Gap:** Extracted traces from paper ECG scans exist in uncalibrated relative units and must **never** be routed into digital ML classifiers. Doing so introduces severe domain shift that invalidates prediction guarantees.
- **Single-Lead Diagnostic Scope:** Single-lead Holter signals lack spatial orthogonal vectors and **cannot** be used to diagnose acute myocardial infarction (MI), bundle branch blocks, or ventricular hypertrophy. Acute ischemia requires a full 12-lead surface electrocardiogram.

### 5.3 Hardware Simulation Status
- All hardware biopotential acquisition screens in CardioSense represent **offline software simulations**.
- Physical hardware acquisition with live electrodes has **not** been tested.
- Standard microcontrollers connected via USB lack galvanic patient isolation (IEC 60601-1). Connecting uncertified electrode circuits to human subjects carries serious shock risk and is strictly prohibited.
