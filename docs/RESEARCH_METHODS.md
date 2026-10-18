# Research Methods & Methodological Protocol

**Project:** CardioSense — ECG Insight Lab  
**Subtitle:** Reproducible Machine Learning and Signal Analysis for ECG Data  
**Scope:** Educational & Academic Research Prototype  
**Date:** October 2026  

---

## 1. Overview & Research Objectives

CardioSense (ECG Insight Lab) investigates reproducible machine learning, biomedical signal conditioning, and telemetry emulation for electrocardiogram (ECG) data. The project is organized around three strictly decoupled active workflows:

1. **Workflow B — 12-Lead Diagnostic Classification (PTB-XL):** Multi-label prediction across five clinical superclasses using deterministic waveform feature engineering and patient-clustered evaluation.
2. **Workflow C — Single-Lead Arrhythmia Beat Analysis (MIT-BIH):** Autonomous QRS detection and beat-by-beat classification using timing dynamics, windowed morphology, and patient-disjoint partitions.
3. **Hardware Prototype — Offline AD8232 Telemetry Simulation:** Forward 10-bit ADC modeling, line-delimited serial protocol transport, and calibration inversion under simulated sensor noise.
4. **Archived Research Module — Optical Paper Digitizer:** Classical morphological signal recovery, retained in an isolated research section for reference but removed from primary navigation due to unproven generalization on real-world paper scans.

---

## 2. Workflow B: 12-Lead ECG Classification (PTB-XL)

### 2.1 Cohort Partitioning & Leakage Elimination
- **Dataset:** PTB-XL v1.0.3 ($N=21,799$ records, $18,869$ patients, 12 leads, 100 Hz calibrated WFDB signals).
- **Stratified Partitioning:** Standard 10-fold patient-clustered split:
  - **Training Cohort:** Folds 1–8 ($17,418$ records, $15,023$ patients).
  - **Validation Cohort:** Fold 9 ($2,183$ records, $1,942$ patients).
  - **Held-Out Test Cohort:** Fold 10 ($2,198$ records, $1,904$ patients).
- **Zero-Overlap Invariant:** $\text{Patients}(\text{Train}) \cap \text{Patients}(\text{Val}) \cap \text{Patients}(\text{Test}) = \emptyset$. All records belonging to the same patient are co-located in the same partition.

### 2.2 Signal Preprocessing
- **Bi-directional Butterworth Filter:** Forward-backward recursive filtering (`scipy.signal.filtfilt`) using a 2nd-order bandpass filter ($0.05\,\text{Hz} - 40\,\text{Hz}$).
- **Zero-Phase Distortion:** Bidirectional filtering ensures zero phase shift ($\theta = 0^\circ$), strictly preventing artificial ST-segment elevation or depression artifacts.

### 2.3 Feature Engineering Schema (148 Features)
Features are computed deterministically per recording without learned parameters:
- **Per-Lead Descriptors ($12 \times 12 = 144$ features):** Mean, standard deviation, median, interquartile range (IQR), minimum, maximum, peak-to-peak amplitude, skewness, kurtosis, root-mean-square (RMS), zero-crossing count, and derivative RMS.
- **Cross-Lead Global Descriptors ($4$ features):** Global mean energy, global maximum amplitude, limb-lead RMS energy ratio, and precordial-lead RMS energy ratio.
- **Total:** Exactly **148 deterministic features**.

### 2.4 Classifier & Decision Calibration
- **Model:** One-vs-Rest Logistic Regression with balanced class weighting ($C=1.0$).
- **Threshold Calibration:** Decision cutoffs were selected on **Validation Fold 9** to maximize macro F1 under class imbalance:
  - `NORM`: $0.50$
  - `MI`: $0.50$
  - `STTC`: $0.55$
  - `CD`: $0.55$
  - `HYP`: $0.65$
- **Held-Out Test Evaluation:** Evaluated once on **Fold 10** ($N=2,198$) with frozen thresholds and model weights.

---

## 3. Workflow C: Single-Lead Beat Analysis (MIT-BIH)

### 3.1 Patient-Disjoint Partitioning
- **Dataset:** PhysioNet MIT-BIH Arrhythmia Database v1.0.0 ($48$ half-hour ambulatory records, $47$ patients, 360 Hz).
- **Split Definitions:**
  - **Train (16 records, 15 patients, 37,419 beats):** 101, 108, 109, 112, 114, 115, 116, 119, 122, 124, 201, 202, 203, 208, 209, 215. (Records 201 and 202 belong to the same patient `sub_201_202` and are strictly co-located).
  - **Validation (7 records, 7 patients, 15,725 beats):** 106, 118, 205, 207, 220, 223, 230.
  - **Held-Out Test (21 records, 21 patients, 47,561 beats):** 100, 103, 105, 111, 113, 117, 121, 123, 200, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234.
  - **Quarantined Paced Records (4 records):** 102, 104, 107, 217 (paced rhythms excluded from general beat evaluation).
- **Zero-Overlap Invariant:** Zero patient overlap between train, validation, and test sets.

### 3.2 Signal-Based QRS Detection (Pan-Tompkins)
- Operates strictly on raw single-lead ECG signal at 360 Hz.
- **Inference Principle:** Ground-truth annotations are **never** provided to the detector or feature extraction engine at inference time.
- **Detector Processing Cascade:**
  1. Low-pass filter ($f_c \approx 11\,\text{Hz}$, 2nd order)
  2. High-pass filter ($f_c \approx 5\,\text{Hz}$, 2nd order)
  3. Five-point first-derivative approximation
  4. Non-linear squaring operator
  5. Moving-window integrator ($150\,\text{ms}$, 54 samples)
  6. Adaptive dual-threshold state machine with $200\,\text{ms}$ refractory lockout ($72$ samples)

### 3.3 Feature Extraction Schema (47 Features)
Each beat is windowed across 200 samples ($-90$ to $+110$ samples around the R-peak, approximately $-250\,\text{ms}$ to $+305\,\text{ms}$):
1. **Timing & RR Rhythm Dynamics ($5$ features):** `pre_rr_sec`, `post_rr_sec`, `local_rr_sec`, `rr_ratio_pre`, `rr_ratio_post`.
2. **Global Beat Morphology ($10$ features):** Amplitude mean, std, min, max, peak-to-peak, skewness, kurtosis, RMS, signal energy, zero-crossing count.
3. **Subsegment Descriptors ($12$ features):**
   - QRS core ($-30$ to $+30$ samples): Mean, RMS, peak-to-peak, energy, QRS-to-total energy ratio ($5$ features).
   - Pre-QRS / P-wave region ($-90$ to $-30$ samples): Mean, RMS, energy ($3$ features).
   - Post-QRS / ST-T region ($+30$ to $+110$ samples): Mean, RMS, energy, peak-to-peak ($4$ features).
4. **Subsampled Morphological Contour ($20$ features):** 20 evenly-spaced amplitude points across the 200-sample window.
- **Total:** Exactly **47 deterministic features** ($5 + 10 + 12 + 20 = 47$).

### 3.4 Project-Defined 4-Class Mapping
The project maps MIT-BIH symbols to four target classes:
- **`N` (Normal):** Normal sinus beats, left/right bundle branch blocks (`N`, `L`, `R`, `e`, `j`).
- **`S` (Supraventricular Ectopic):** Atrial premature, aberrated, nodal premature, and supraventricular beats (`A`, `a`, `J`, `S`).
- **`V` (Ventricular Ectopic):** Premature ventricular contractions, ventricular escape beats (`V`, `E`).
- **`Q` (Fusion & Other):** Fusion beats, paced beats, and unclassifiable beats (`F`, `/`, `f`, `Q`).
- *Important Disclosure:* Merging ventricular fusion beats (`F`) into `Q` is a project-defined research grouping and does not constitute an unqualified standard five-class AAMI EC57 implementation.

---

## 4. Controlled Research Experiment: Class Imbalance

### 4.1 Objective & Research Question
> **Can class-imbalance handling improve minority-class beat classification on unseen subjects without materially degrading performance on common classes?**

### 4.2 Candidate Architectures
1. **Baseline (Balanced RF):** 100 trees, max depth 12, global `class_weight='balanced'`.
2. **Candidate 1 (Balanced Subsample RF):** 100 trees, max depth 12, `class_weight='balanced_subsample'` (weights dynamically calculated per bootstrap sample).
3. **Candidate 2 (Balanced Logistic Regression):** Linear classification baseline with global balanced weighting and L2 regularization.
4. **Candidate 3 (Minority-Boosted Cost-Sensitive RF):** 100 trees, max depth 12, with focal weight multipliers ($2\times$ on S, $3\times$ on Q).

### 4.3 Protocol & Leakage Prevention
- Feature scalers and classifiers were fit **strictly on the Train partition** ($N=37,419$).
- Model selection was executed on the **Validation partition** ($N=15,725$) without peeking at the test set.
- Final evaluation was performed **once on the untouched held-out Test partition** ($N=47,561$).

---

## 5. Hardware Prototype: Offline AD8232 Simulation

### 5.1 Telemetry Emulation Model
- **Forward Simulation:** Continuous analog ECG signals ($x(t)$ in mV) are mapped to 10-bit integer ADC counts:
  $$\text{ADC}[n] = \text{clip}\left(\text{round}\left(V_{\text{baseline}} + G \cdot x[n] + \epsilon[n]\right), 0, 1023\right)$$
  where baseline count $V_{\text{baseline}} = 512$, gain $G = 150\,\text{counts/mV}$, and $\epsilon[n]$ models additive Gaussian sensor noise, baseline drift ($0.25\,\text{Hz}$ wander), and mains hum ($50\,\text{Hz}$).
- **UART Protocol Framing:** Line-oriented serial frames (`VAL=<count>,LO+=<0/1>,LO-=<0/1>\n`) at 115,200 baud with microsecond timing.
- **Inverse Calibration:** Reconstructs estimated voltage:
  $$\hat{x}[n] = \frac{\text{ADC}[n] - V_{\text{baseline}}}{G}$$
- **Safety Policy:** Physical hardware acquisition remains untested. Connecting non-isolated USB circuits or electrodes to human subjects is strictly prohibited (IEC 60601-1).

---

## 6. Archived Module: Optical Paper Digitizer

### 6.1 Status & Archival Rationale
The Optical Paper Digitizer uses classical computer vision (morphological thresholding, bounding-box lead segmentation, and vertical centroid tracing) to extract 1D waveforms from scanned 12-lead paper charts.
- **Benchmark Performance:** Evaluated on synthetic fixture (`sample_data/synthetic_12lead.png`), achieving $r=0.6932$ and $\text{MAE}=0.0725$ relative units across all 12 leads.
- **Reason for Archival:** Synthetic benchmark evaluation does not establish reliable signal recovery on real-world paper printouts exhibiting non-linear paper folds, perspective skew, low contrast, or grid degradation.
- **Archival Implementation:** Relocated from default navigation into a collapsible research section. The module remains strictly air-gapped from downstream ML classifiers to prevent uncalibrated optical noise from corrupting diagnostic inferences.
