# Single-Lead Beat ML Model Card (Phase 6 Research Release v1.0.0)

**Date:** 2026-10-18  
**Model Name:** `single_lead_rf_model.joblib`  
**Model Architecture:** Standardized Features + Balanced Random Forest Classifier (100 estimators, max depth 12)  
**Feature Schema Version:** 1.0.0 (47 deterministic timing and morphological descriptors)  
**Trained Artifact SHA-256:** `3889a5f5ab97803da339710d4ae9fde8ea50f8e5a5ff76ed2015a0fb0b6ec8e9`  
**Dataset Provenance:** MIT-BIH Arrhythmia Database v1.0.0 (PhysioNet)  
**Dataset DOI:** [10.13026/C2F305](https://doi.org/10.13026/C2F305)  
**Citation:** Moody G, Mark R. MIT-BIH Arrhythmia Database, version 1.0.0. PhysioNet, 2005.  
**Dataset License:** Open Data Commons Attribution License v1.0 (ODC-By v1.0)  
**Code License:** MIT License  

---

## 1. Model Overview

The **Single-Lead Beat ML Model** is an academic research classifier developed under Phase 6 of the CardioSense / ECG Insight Lab project. It performs automated single-channel electrocardiogram (ECG) beat-type classification.

> [!NOTE]
> **Taxonomy Disclosure:** Project-defined 4-class beat grouping: N, S, V, Q, with ventricular fusion beats (F) merged into Q. This is not a standard five-class AAMI EC57 implementation.  
> In evaluated non-paced records, **98.2% of Class Q beats are ventricular fusion beats (`F`)** rather than unclassifiable beats.

The four project-defined beat classes are:
1. **N (Normal / Bundle Branch Block):** Normal sinus beats, left/right bundle branch block, nodal/atrial escape beats (`N`, `L`, `R`, `e`, `j`).
2. **S (Supraventricular Ectopic Beat - SVEB):** Premature atrial contractions (PAC), aberrated atrial beats, nodal premature beats (`A`, `a`, `J`, `S`).
3. **V (Ventricular Ectopic Beat - VEB):** Premature ventricular contractions (PVC), ventricular escape beats (`V`, `E`).
4. **Q (Fusion & Other):** Ventricular fusion beats (`F`), paced beats (`/`), paced fusion beats (`f`), and unclassifiable waveforms (`Q`). In evaluated non-paced records, 98.2% of Class Q consists of ventricular fusion beats (`F`).

The pipeline incorporates a decoupled, signal-based R-peak detector (Pan-Tompkins filter) to locate beats from continuous raw voltage traces without consuming ground-truth annotation indices at inference.

---

## 2. Intended & Prohibited Uses

### 2.1 Intended Academic Uses
- **Algorithmic Benchmarking:** Demonstrating single-lead arrhythmia beat detection and classification on standardized Holter records.
- **Signal Processing Research:** Evaluating the performance of classical Pan-Tompkins QRS detection and deterministic time/morphology feature extraction.
- **Offline Recorded Playback Simulation:** Providing interactive educational replay in Streamlit to explore ECG morphology and beat labeling.

### 2.2 Strictly Prohibited Uses
- **Clinical Decision Support:** This model is **NOT** a certified medical device and has never undergone clinical trial validation. It must **NOT** be used for patient diagnosis, arrhythmia therapy, or patient monitoring.
- **Acute Coronary Syndrome / Heart Attack Diagnosis:** Single-lead Holter signals cannot reliably diagnose myocardial infarction (MI), ST/T changes (STTC), conduction disturbances (CD), or hypertrophy (HYP). Claiming heart attack detection from a single channel is scientifically invalid and strictly forbidden.
- **Hardware Equivalence Claims:** MIT-BIH recordings were collected from reel-to-reel Holter cassette tape recorders on hospitalized patients. They do **NOT** represent data acquired from consumer AD8232 or ESP32 front-ends. Never claim physical hardware compatibility or live sensor operation.
- **Image-to-Classifier Routing:** Routing digitized paper ECG images into this single-lead classifier is strictly prohibited.

---

## 3. Training, Validation & Held-Out Test Partitions

### 3.1 Patient-Disjoint Partition Strategy
There are 48 half-hour two-channel recordings from 47 unique subjects in MIT-BIH. Records **201** and **202** originate from the same patient. To prevent patient data leakage, both records are forced into the Training partition under subject identifier `sub_201_202`. Four records with predominantly paced rhythms (`102`, `104`, `107`, `217`) are quarantined per AAMI EC57 guidelines.

| Partition | Records ($N$) | Subjects ($N$) | Beats Evaluated | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Training** | 16 (`101`, `108`, `109`, `112`, `114`, `115`, `116`, `119`, `122`, `124`, `201`, `202`, `203`, `208`, `209`, `215`) | 15 | 37,419 | Feature scaling, baseline base rates, Random Forest training |
| **Validation** | 7 (`106`, `118`, `205`, `207`, `220`, `223`, `230`) | 7 | 15,725 | Hyperparameter verification and threshold checking |
| **Held-Out Test** | 21 (`100`, `103`, `105`, `111`, `113`, `117`, `121`, `123`, `200`, `210`, `212`, `213`, `214`, `219`, `221`, `222`, `228`, `231`, `232`, `233`, `234`) | 21 | 47,561 | Strict final frozen benchmark evaluation |
| **Quarantined** | 4 (`102`, `104`, `107`, `217`) | 4 | Quarantined | Excluded paced recordings per AAMI EC57 |

*Subject Disjointness:* $\text{Subjects}(\text{Train}) \cap \text{Subjects}(\text{Val}) \cap \text{Subjects}(\text{Test}) = \emptyset$ (Zero patient overlap verified).

---

## 4. Signal Processing & Feature Extraction Schema (47 Features)

Each beat is windowed across 200 samples at 360 Hz ($-90$ to $+110$ samples around R-peak, approx. $-250\,\text{ms}$ to $+305\,\text{ms}$):
1. **Timing & Rhythm Descriptors (5 features):**
   - `pre_rr_sec`: Preceding RR interval in seconds.
   - `post_rr_sec`: Following RR interval in seconds.
   - `local_rr_sec`: Running average RR across up to 10 surrounding beats.
   - `rr_ratio_pre`: Ratio of pre-RR to local-RR (identifies premature contractions).
   - `rr_ratio_post`: Ratio of post-RR to local-RR (identifies compensatory pauses).
2. **Global Beat Morphology (10 features):**
   - Amplitude mean, standard deviation, minimum, maximum, peak-to-peak amplitude.
   - Skewness, kurtosis, root-mean-square (RMS), signal energy, zero-crossing rate.
3. **Subsegment Descriptors (12 features):**
   - QRS core ($-30$ to $+30$ samples): Mean, RMS, peak-to-peak, energy, QRS-to-total energy ratio.
   - Pre-QRS / P-wave region ($-90$ to $-30$ samples): Mean, RMS, energy.
   - Post-QRS / ST-T region ($+30$ to $+110$ samples): Mean, RMS, energy, peak-to-peak.
4. **Subsampled Morphological Template (20 features):**
   - 20 evenly-spaced amplitude samples across the 200-sample window capturing normalized wave contour.

---

## 5. Quantitative Evaluation Results (Held-Out Test Set: 21 Patients)

### 5.1 Signal-Based R-Peak Detector Performance (Pan-Tompkins)
Evaluated across all 21 held-out test recordings with a tolerance of $\pm 50\,\text{ms}$ ($\pm 18$ samples @ 360 Hz) without consuming annotations at inference:
- **Total Reference Beats:** 47,576
- **Total Detected Peaks:** 47,163
- **True Positives (TP):** 47,048
- **False Positives (FP):** 115 (unmatched artifacts)
- **False Negatives (FN):** 528 (missed reference beats)
- **Sensitivity:** **98.89%** ($0.9889$)
- **Positive Predictive Value (PPV):** **99.76%** ($0.9976$)
- **F1 Score:** **99.32%** ($0.9932$)

### 5.2 Primary Benchmark: Matched-Detection Conditional Beat-Classification Metrics (N = 47,034)
Evaluates beat classification on detected R-peaks proposed by the automated Pan-Tompkins detector, matched to reference annotations within $\pm 50\,\text{ms}$ tolerance, and retained after feature-window boundary checks.

- **Evaluated Detections (Feature-Valid):** **47,034 beats** (47,048 matched detections minus 14 boundary-truncated beats)
- **Macro F1-Score:** **0.5274**
- **Weighted F1-Score:** **0.9011**

| Beat Class Grouping | True Support | Precision | Recall | F1 Score | Grouping Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **N** | 41,883 | 0.9592 | 0.9218 | **0.9401** | Normal / Bundle Branch Block Beats (`N`, `L`, `R`, `e`, `j`) |
| **S** | 1,757 | 0.3555 | 0.1793 | **0.2384** | Supraventricular Ectopic Beats (`A`, `a`, `J`, `S`) |
| **V** | 3,018 | 0.7781 | 0.9298 | **0.8472** | Ventricular Ectopic Beats (`V`, `E`) |
| **Q** | 376 | 0.0489 | 0.2979 | **0.0840** | Ventricular Fusion (`F`) & Unclassifiable (`Q`) |
| **Macro Average** | **47,034** | **0.5354** | **0.5822** | **0.5274** | Unweighted mean across all 4 classes |
| **Weighted Average** | **47,034** | **0.9178** | **0.8896** | **0.9011** | Prevalence-weighted mean |

> [!NOTE]
> **Benchmark Interpretation Note:** Matched-detection conditional beat-classification metrics — classification F1 is calculated only for detector peaks matched to a reference annotation and retained after feature-window boundary checks. Detector false positives (115 peaks) and missed reference beats (542 beats, including boundary-dropped matched beats) are reported separately in detector/count-reconciliation metrics and are not included as errors in this classifier confusion matrix.

### 5.3 Secondary Benchmark: Annotation-Centred (Oracle) Classifier Reference
Diagnostic reference isolating classifier feature discrimination assuming ideal QRS localization. Beat windows are centred on expert ground-truth annotation indices (47,561 valid windows after dropping 15 boundary reference annotations); **this is not an autonomous end-to-end result**:

| Model | Macro F1 | Weighted F1 | Class N F1 | Class S F1 | Class V F1 | Class Q F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Prevalence Baseline** | 0.2350 | 0.8338 | 0.9401 | 0.0000 | 0.0000 | 0.0000 |
| **Balanced Random Forest** | **0.5396** | **0.9036** | **0.9418** | **0.2712** | **0.8517** | **0.0936** |

#### Detailed Per-Class Breakdown (Oracle Reference on 47,561 Valid Windows)

| Class | Description | Test Support | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **N** | Normal / BBB Beat | 42,184 | 0.9595 | 0.9247 | **0.9418** |
| **S** | Supraventricular Ectopic (SVEB/PAC) | 1,782 | 0.4035 | 0.2043 | **0.2712** |
| **V** | Ventricular Ectopic (VEB/PVC) | 3,201 | 0.7912 | 0.9222 | **0.8517** |
| **Q** | Fusion (`F`) & Other (`Q`) | 394 | 0.0549 | 0.3173 | **0.0936** |
| **Macro Average** | **Unweighted Average** | **47,561** | **0.5523** | **0.5921** | **0.5396** |
| **Weighted Average** | **Prevalence Average** | **47,561** | **0.9192** | **0.9002** | **0.9036** |

#### Oracle Confusion Matrix

$$\begin{pmatrix} & \text{Pred N} & \text{Pred S} & \text{Pred V} & \text{Pred Q} \\ \text{True N} & 39,006 & 492 & 535 & 2,151 \\ \text{True S} & 1,380 & 364 & 38 & 0 \\ \text{True V} & 203 & 45 & 2,952 & 1 \\ \text{True Q} & 62 & 1 & 206 & 125 \end{pmatrix}$$

### 5.4 Beat Count Reconciliation & Key Limitations
- **Beat Count Reconciliation:** Total reference annotations across 21 test records = $47,576$.
  - *Oracle Benchmark ($N=47,561$):* 15 reference annotations occur too close to recording edges ($<90$ samples at start or $>649,890$ samples at end: 14 Class N, 1 Class V in Record 233) to extract a complete 200-sample window ($47,576 - 15 = 47,561$).
  - *Detector Benchmark ($N=47,034$):* Pan-Tompkins detects 47,163 peaks; 47,048 match reference annotations within $\pm 50\,\text{ms}$. Of these, 14 fall too close to recording boundaries to extract features ($47,048 - 14 = 47,034$).
  - *Decomposition of the 542 Unrepresented Reference Beats:* 527 non-boundary detector FNs + 1 boundary detector FN (Record 213, sample 649,992, Class N) + 14 boundary-dropped matched detections (13 Class N, 1 Class V) = **542 unrepresented reference beats**.
    - Class N: 301 non-boundary FNs + 1 boundary FN + 13 boundary-dropped detections = **315** unrepresented beats ($42,198 - 315 = 41,883$ true support).
    - Class S: 25 non-boundary FNs = **25** unrepresented beats ($1,782 - 25 = 1,757$ true support).
    - Class V: 183 non-boundary FNs + 1 boundary-dropped detection = **184** unrepresented beats ($3,202 - 184 = 3,018$ true support).
    - Class Q: 18 non-boundary FNs = **18** unrepresented beats ($394 - 18 = 376$ true support).
    Non-boundary FNs total $301 + 25 + 183 + 18 = 527$.
- **Severe Class Imbalance:** Class N represents ~89% of beats, which elevates Weighted F1 (~0.90) relative to Macro F1 (~0.53).
- **Minority Class Performance:** Supraventricular ectopic beats (S: F1 = 0.2384) and Fusion/Other beats (Q: F1 = 0.0840) exhibit poor sensitivity and high false-alarm rates.
- **Academic Replay Boundary:** Workflow C evaluates historic Holter data (1975–1979) and has not been validated on physical AD8232/ESP32 hardware or for clinical monitoring.

---

## 6. Trust Boundary & Cryptographic Verification

The model is serialized to `artifacts/single_lead_ml/single_lead_rf_model.joblib` and bound to `single_lead_rf_model.joblib.manifest.json`.
- The repository loader `load_verified_single_lead_model()` strictly verifies that the bit-for-bit SHA-256 digest on disk matches `3889a5f5ab97803da339710d4ae9fde8ea50f8e5a5ff76ed2015a0fb0b6ec8e9`.
- Quarantined legacy pickles (e.g. `finalmodel.pkl`) and tampered files are rejected with security errors.
