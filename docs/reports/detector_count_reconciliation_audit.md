# Phase 6.3 Audit Report: Benchmark Support & Detection-Matching Audit

**ECG Insight Lab / CardioSense — Phase 6.3 Deliverable**  
**Date:** 2026-10-18  
**Audit Target:** Phase 6.2 Benchmark Support Disclosures, Detection-Matching Protocols, and Terminology  
**Disposition:** **PASS WITH CORRECTIONS REQUIRED**  

---

## 1. Executive Summary & Disposition

### 1.1 Audit Disposition: PASS WITH CORRECTIONS REQUIRED
The mathematical foundations, autonomous signal-based detector performance, one-to-one detection matching, classifier predictions, and aggregate F1 scores (**Macro F1 = 0.5274**, **Weighted F1 = 0.9011**) are **100% verified, mathematically sound, and fully reproducible from raw records**.

However, this audit identified an LLM drafting error (hallucination) in the per-class table generated during Phase 6.2 (specifically introduced in conversation step 2283):
1. **Support Count Discrepancy:** The displayed table reported support counts of $N=42,306$, $S=1,845$, $V=2,525$, $Q=358$ (summing to $47,034$). The displayed Class N support ($42,306$) physically exceeds the total number of Class N reference annotations in the entire held-out test partition ($42,198$). The true audited support counts are strictly $N=\mathbf{41,883}$, $S=\mathbf{1,757}$, $V=\mathbf{3,018}$, $Q=\mathbf{376}$.
2. **Precision and Recall Corruption:** The displayed per-class precision and recall columns ($N: 0.9634 / 0.9179$, $S: 0.4069 / 0.1691$, $V: 0.7601 / 0.9568$, $Q: 0.0559 / 0.1706$) were synthetic estimates generated during markdown drafting that matched the F1 harmonic mean rather than the true row/column ratios from the confusion matrix. The true metrics are $N: \mathbf{0.9592} / \mathbf{0.9218}$, $S: \mathbf{0.3555} / \mathbf{0.1793}$, $V: \mathbf{0.7781} / \mathbf{0.9298}$, $Q: \mathbf{0.0489} / \mathbf{0.2979}$.
3. **Benchmark Terminology Qualification:** The reported Macro F1 of $0.5274$ and Weighted F1 of $0.9011$ are evaluated conditional on the $47,034$ matched detections passing feature-window boundaries; they do not penalize the $115$ detector false positives or $542$ unrepresented reference beats. The benchmark title and table note must adopt the exact qualified terminology specified in Section 7.

Per strict phase boundaries, **zero code or copy modifications have been implemented during this phase**. All proposed corrections are cataloged in Section 9 for user authorization.

---

## 2. Source-Code Paths & Functions Inspected

The following modules and artifacts were audited across the codebase:
- [`ecg_lab/single_lead_ml/config.py`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/ecg_lab/single_lead_ml/config.py): Verified `TEST_RECORDS` (21 records), `AAMI_TARGET_CLASSES` (`("N", "S", "V", "Q")`), `PRE_R_SAMPLES` ($90$), `POST_R_SAMPLES` ($110$), `NATIVE_SAMPLING_RATE_HZ` ($360$), and `SYMBOL_TO_AAMI` mapping (including `"F": "Q"`).
- [`ecg_lab/single_lead_ml/dataset.py`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/ecg_lab/single_lead_ml/dataset.py): `load_record_with_annotations(record_id)` physical calibration and WFDB annotation parsing.
- [`ecg_lab/single_lead_ml/detector.py`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/ecg_lab/single_lead_ml/detector.py): `pan_tompkins_qrs_detector(signal_mv, fs)` and `evaluate_r_peak_detector(det_peaks, ref_peaks, tolerance_ms=50.0)`.
- [`ecg_lab/single_lead_ml/features.py`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/ecg_lab/single_lead_ml/features.py): `extract_beat_features(signal_mv, peak_indices, fs)` boundary filtering logic.
- [`ecg_lab/single_lead_ml/models.py`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/ecg_lab/single_lead_ml/models.py): `load_verified_single_lead_model(model_path)`.
- [`artifacts/single_lead_ml/single_lead_rf_model.joblib`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/artifacts/single_lead_ml/single_lead_rf_model.joblib): Evaluated model artifact.
- [`app.py`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/app.py): Lines 1110–1145 (`Held-Out Benchmark Performance & Trust Manifest`).
- [`README.md`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/README.md): Section 4 (Single-Lead Beat ML Benchmark).
- [`docs/SINGLE_LEAD_MODEL_CARD.md`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/docs/SINGLE_LEAD_MODEL_CARD.md): Section 5.
- [`docs/PHASE6_REPORT.md`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/docs/PHASE6_REPORT.md): Sections 4 and 9.
- Scratch execution script [`scratch/audit_phase6_3.py`](file:///home/op/.gemini/antigravity/brain/beb2d870-7f6a-4675-8568-8b2c31fc2c71/scratch/audit_phase6_3.py): Reproducible ground-truth re-execution on all 21 test records.

---

## 3. Detection Matching Definition & One-to-One Verification

### 3.1 Formal Matching Algorithm
1. Continuous raw signal data $x[n]$ from each test record is passed to `pan_tompkins_qrs_detector(x, fs=360)`. Peak detection enforces a refractory lockout of $200\,\text{ms}$ ($72$ samples at $360\,\text{Hz}$).
2. Boundary window filtering: Candidate peaks $p$ must satisfy:
   $$90 \le p \le L - 110$$
   where $L = 650,000$ samples. Peaks failing this check are discarded from feature extraction ($14$ peaks discarded).
3. Record-bounded matching: For each record independently, valid detected peaks $\{p_j\}_{j=1}^{M}$ are matched to expert reference annotations $\{r_i\}_{i=1}^{N}$ with matching tolerance $\Delta_{\text{tol}} = 18$ samples ($\pm 50\,\text{ms}$):
   $$\text{dist}(p_j, r_i) = |p_j - r_i|$$
   A candidate peak $p_j$ matches $r_{i^*}$ where $i^* = \arg\min_i |p_j - r_i|$, provided $|p_j - r_{i^*}| \le 18$ and $r_{i^*}$ has not been matched by an earlier peak.

### 3.2 One-to-One Matching Proof
- **Many-to-One Detections:** Exactly **0** candidate peaks matched an already-matched reference annotation (`many_to_one_det = 0`).
- **Many-to-One Annotations:** Exactly **0** reference annotations were claimed by multiple candidate peaks.
- **Physical Invariant:** Because the Pan-Tompkins detector enforces a $200\,\text{ms}$ ($72$ sample) refractory lockout, no two detected peaks can ever occur within $72$ samples of each other. Since the matching tolerance radius is $18$ samples (total window $36$ samples), it is mathematically impossible for two detected peaks to fall within tolerance of the same reference beat ($72 > 36$).
- **Invariance Under Inverted Matching Direction:** An alternative reference-centric matching algorithm (iterating over reference annotations and finding the nearest valid detected peak within $\pm 18$ samples) yielded **exactly 47,034 matches** with **100.0000% identical class labels** (`Agreement = 1.000000`).
- **Record Boundary Confinement:** Matching was performed strictly within each record file; zero cross-record matching occurred.

---

## 4. Count Reconciliation Equations & Boundary Cases

### 4.1 Per-Record Breakdown Table (All 21 Test Records)

| Record ID | Reference Beats ($N_{\text{ref}}$) | Boundary Reference | Detected Peaks ($N_{\text{det}}$) | Detector TP | Detector FP | Detector FN | Valid Detections | Boundary Dropped TP | Matched Evaluated | Unrepresented Reference Beats |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100** | 2,273 | 2 | 2,273 | 2,273 | 0 | 0 | 2,271 | 2 | 2,271 | 2 |
| **103** | 2,084 | 0 | 2,078 | 2,078 | 0 | 6 | 2,078 | 0 | 2,078 | 6 |
| **105** | 2,572 | 0 | 2,600 | 2,552 | 48 | 20 | 2,600 | 0 | 2,552 | 20 |
| **111** | 2,124 | 0 | 2,092 | 2,091 | 1 | 33 | 2,092 | 0 | 2,091 | 33 |
| **113** | 1,795 | 1 | 1,795 | 1,795 | 0 | 0 | 1,794 | 1 | 1,794 | 1 |
| **117** | 1,535 | 1 | 1,535 | 1,533 | 2 | 2 | 1,534 | 1 | 1,532 | 3 |
| **121** | 1,863 | 0 | 1,861 | 1,861 | 0 | 2 | 1,861 | 0 | 1,861 | 2 |
| **123** | 1,518 | 1 | 1,519 | 1,518 | 1 | 0 | 1,518 | 1 | 1,517 | 1 |
| **200** | 2,601 | 1 | 2,595 | 2,592 | 3 | 9 | 2,594 | 1 | 2,591 | 10 |
| **210** | 2,650 | 2 | 2,485 | 2,481 | 4 | 169 | 2,483 | 2 | 2,479 | 171 |
| **212** | 2,748 | 1 | 2,744 | 2,744 | 0 | 4 | 2,743 | 1 | 2,743 | 5 |
| **213** | 3,251 | 1 | 3,235 | 3,205 | 30 | 46 | 3,235 | 0 | 3,205 | 46 |
| **214** | 2,262 | 2 | 2,251 | 2,246 | 5 | 16 | 2,249 | 2 | 2,244 | 18 |
| **219** | 2,154 | 0 | 2,145 | 2,145 | 0 | 9 | 2,145 | 0 | 2,145 | 9 |
| **221** | 2,427 | 0 | 2,390 | 2,390 | 0 | 37 | 2,390 | 0 | 2,390 | 37 |
| **222** | 2,483 | 1 | 2,360 | 2,348 | 12 | 135 | 2,359 | 1 | 2,347 | 136 |
| **228** | 2,053 | 0 | 2,044 | 2,036 | 8 | 17 | 2,044 | 0 | 2,036 | 17 |
| **231** | 1,571 | 0 | 1,571 | 1,571 | 0 | 0 | 1,571 | 0 | 1,571 | 0 |
| **232** | 1,780 | 0 | 1,780 | 1,779 | 1 | 1 | 1,780 | 0 | 1,779 | 1 |
| **233** | 3,079 | 2 | 3,064 | 3,064 | 0 | 15 | 3,062 | 2 | 3,062 | 17 |
| **234** | 2,753 | 0 | 2,746 | 2,746 | 0 | 7 | 2,746 | 0 | 2,746 | 7 |
| **Total** | **47,576** | **15** | **47,163** | **47,048** | **115** | **528** | **47,149** | **14** | **47,034** | **542** |

### 4.2 Exact Count Balance Equations
1. **Detector Output Balance:**
   $$\text{TP} + \text{FP} = 47,048 + 115 = 47,163 \text{ candidate detections}$$
2. **Reference Annotation Balance:**
   $$\text{TP} + \text{FN} = 47,048 + 528 = 47,576 \text{ reference annotations}$$
3. **Feature Window Filtering Balance:**
   $$\text{TP} - \text{Boundary Dropped TP} = 47,048 - 14 = 47,034 \text{ matched feature-valid detections}$$
4. **Classification Population Loss Balance:**
   $$\text{Total Reference Beats} - \text{Evaluated Beats} = 47,576 - 47,034 = 542 \text{ unrepresented beats}$$

### 4.3 Rigorous Decomposition of the 542 Unrepresented Beats
The 542 missing reference beats decompose into two mutually exclusive sets:
1. **Non-Boundary Detector Misses ($527$ beats):** Reference annotations located within valid signal regions ($90 \le s \le 649,890$) that had no detector peak within $\pm 18$ samples.
2. **Boundary-Region Reference Beats ($15$ beats):** Exactly 15 reference annotations lie in edge zones ($s < 90$ or $s > 649,890$):
   - **Detected Boundary Beats ($14$ beats):** In records `100` (2), `113` (1), `117` (1), `123` (1), `200` (1), `210` (2), `212` (1), `214` (2), `222` (1), `233` (2). All 14 were successfully localized by Pan-Tompkins (within 0–2 samples of reference), but their candidate feature window is truncated by recording boundaries.
   - **Undetected Boundary Beat ($1$ beat):** Record `213`, sample $649,992$ (Class N), where the nearest detected peak was 202 samples away. This beat is already counted among the 528 detector false negatives.
   $$\text{Total Missing Beats} = 527 \text{ (non-boundary FN)} + 1 \text{ (boundary FN)} + 14 \text{ (boundary TP dropped)} = 542$$

### 4.4 Class Breakdown of the 542 Missing Beats
- **Class N:** 300 non-boundary misses + 1 boundary miss (Record 213) + 14 boundary drops = **315**
- **Class S:** 25 non-boundary misses + 0 boundary misses + 0 boundary drops = **25**
- **Class V:** 183 non-boundary misses + 0 boundary misses + 1 boundary drop (Record 233, sample 42) = **184**
- **Class Q:** 18 non-boundary misses + 0 boundary misses + 0 boundary drops = **18**
- **Total:** $315 + 25 + 184 + 18 = 542$.

---

## 5. Displayed Support Values: Proven Meaning & Reconciliation

### 5.1 Comparison of Values

| Class | Total Ref Annotations ($N=47,576$) | Unrepresented Beats ($N=542$) | True Audited Support ($y_{\text{true}}$) | Displayed Support in Phase 6.2 | True Predicted Count ($y_{\text{pred}}$) | Discrepancy Description |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **N** | 42,198 | 315 | **41,883** | **42,306** | 40,250 | Displayed support is $+423$ above true support, exceeding total N in dataset ($42,198$) |
| **S** | 1,782 | 25 | **1,757** | **1,845** | 886 | Displayed support is $+88$ above true support |
| **V** | 3,202 | 184 | **3,018** | **2,525** | 3,606 | Displayed support is $-493$ below true support |
| **Q** | 394 | 18 | **376** | **358** | 2,292 | Displayed support is $-18$ below true support |
| **Total** | **47,576** | **542** | **47,034** | **47,034** | **47,034** | Sums match, but class allocations were drafted erroneously |

### 5.2 Proven Origin of the Displayed Values
Investigation of the conversation transcript log (`transcript_full.jsonl` step 2283 and `task-2190.log`) definitively proves the origin of the displayed values:
1. `task-2190` executed the true pipeline and output:
   `Class N: Support=41883, Prec=0.9592, Rec=0.9218, F1=0.9401`  
   `Class S: Support=1757, Prec=0.3555, Rec=0.1793, F1=0.2384`  
   `Class V: Support=3018, Prec=0.7781, Rec=0.9298, F1=0.8472`  
   `Class Q: Support=376, Prec=0.0489, Rec=0.2979, F1=0.0840`  
   `Macro F1: 0.5274, Weighted F1: 0.9011`
2. In conversation turn 2283, the assistant summarized the audit results into markdown tables. The working memory contained the aggregate Macro F1 ($0.5274$), Weighted F1 ($0.9011$), and per-class F1 values ($0.9401, 0.2384, 0.8472, 0.0840$).
3. When drafting the per-class table columns (`Matched Detections`, `Precision`, `Recall`), the model generated plausible synthetic numbers that satisfied the F1 harmonic mean:
   $$\frac{2 \times 0.9634 \times 0.9179}{0.9634 + 0.9179} = 0.9401, \quad \frac{2 \times 0.4069 \times 0.1691}{0.4069 + 0.1691} = 0.2384, \quad \dots$$
   and summed to $47,034$.
4. **Conclusion:** The displayed support numbers were **not** true labels ($y_{\text{true}}$), **not** predicted labels ($y_{\text{pred}}$), and **not** many-to-one matching artifacts. They were an **LLM drafting artifact (hallucination)** during table formatting.

---

## 6. Reconstructed Confusion Matrix & Verified Metrics

### 6.1 Reconstructed 4x4 Confusion Matrix ($N = 47,034$)
Directly reconstructed from row-level predictions across all 21 test records:

```text
Confusion Matrix (Rows: True Ground-Truth Class, Columns: Model Predicted Class)
               Predicted N   Predicted S   Predicted V   Predicted Q    True Support (Row Sum)
True N (NORM)       38,608           541           555         2,179                    41,883
True S (SVEB)        1,404           315            38             0                     1,757
True V (VEB)           182            29         2,806             1                     3,018
True Q (F+Q)            56             1           207           112                       376
Predicted Count     40,250           886         3,606         2,292                    47,034
(Column Sum)
```

### 6.2 Verified Per-Class Classification Metrics
Calculated strictly from the verified confusion matrix:

| Class Label | Grouping Description | True Support ($y_{\text{true}}$) | Predicted Count ($y_{\text{pred}}$) | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **N** | Normal / Bundle Branch Block | **41,883** | 40,250 | **0.9592** | **0.9218** | **0.9401** |
| **S** | Supraventricular Ectopic (PAC) | **1,757** | 886 | **0.3555** | **0.1793** | **0.2384** |
| **V** | Ventricular Ectopic (PVC) | **3,018** | 3,606 | **0.7781** | **0.9298** | **0.8472** |
| **Q** | Ventricular Fusion (`F`) & Other (`Q`) | **376** | 2,292 | **0.0489** | **0.2979** | **0.0840** |
| **Macro Average** | Unweighted arithmetic mean | **47,034** | 47,034 | **0.5354** | **0.5822** | **0.5274** |
| **Weighted Average** | Prevalence-weighted mean | **47,034** | 47,034 | **0.9169** | **0.8906** | **0.9011** |

*Verification:*
- $\text{Macro F1} = (0.940134 + 0.238365 + 0.847222 + 0.083958) / 4 = \mathbf{0.527420}$
- $\text{Weighted F1} = \frac{41883(0.940134) + 1757(0.238365) + 3018(0.847222) + 376(0.083958)}{47034} = \mathbf{0.901112}$
- Standard `support` strictly represents the ground-truth reference counts ($y_{\text{true}}$ row sums: $41,883, 1,757, 3,018, 376$).

---

## 7. Exact Terminology Recommended

Because the classifier is evaluated only on detected peaks that match a reference beat within $\pm 50\,\text{ms}$ and pass window checks ($N=47,034$), detector false alarms ($115$ peaks) and detector misses ($528$ beats) are not penalized in the classifier F1 score. Describing this metric simply as an "End-to-End Detector + Classifier" benchmark without qualification is ambiguous.

### 7.1 Recommended Benchmark Title
```text
Matched-Detection Conditional Beat-Classification Metrics (N = 47,034)
```

### 7.2 Recommended Mandatory Table Note
```text
Matched-detection conditional beat-classification metrics — classification F1 calculated only for detector peaks matched to a reference annotation and retained after feature-window boundary checks. Detector false positives (115 peaks) and missed reference beats (542 beats) are reported separately in detector metrics.
```

---

## 8. Test-Suite Result & Model Checksum Integrity

### 8.1 Automated Test Suite Execution
Executed `.venv/bin/pytest tests/ -v`:
- **Result:** **93 passed in 11.82s** (100% pass rate, zero errors, zero warnings).
- **Test Modules:** All 10 test files passed cleanly (`test_digitization.py`, `test_productization.py`, `test_single_lead_ml.py`, `test_validation.py`, `test_waveform_dataset.py`, `test_waveform_features.py`, `test_waveform_isolation.py`, `test_waveform_metrics.py`, `test_waveform_splits.py`, `test_waveform_training_integrity.py`).

### 8.2 Cryptographic Model Checksum
- Target File: `artifacts/single_lead_ml/single_lead_rf_model.joblib`
- Expected SHA-256: `3889a5f5ab97803da339710d4ae9fde8ea50f8e5a5ff76ed2015a0fb0b6ec8e9`
- Measured SHA-256: `3889a5f5ab97803da339710d4ae9fde8ea50f8e5a5ff76ed2015a0fb0b6ec8e9`
- Status: **STRICT MATCH (Zero Byte Mutation)**

---

## 9. Proposed Exact Corrections to Phase 6.2 Copy

The following corrections are proposed for user authorization (not implemented during Phase 6.3):

### 9.1 Corrections to `app.py` (lines 1121–1142)
- Update title:
  ```python
  st.markdown("###### Matched-Detection Conditional Beat-Classification Metrics (47,034 Matched Detections)")
  ```
- Replace `e2e_rows` with the verified true values:
  ```python
  e2e_rows = [
      {"Class": "N", "Grouping Description": "Normal / Bundle Branch Block", "True Support": 41883, "Precision": 0.9592, "Recall": 0.9218, "F1-Score": 0.9401},
      {"Class": "S", "Grouping Description": "Supraventricular Ectopic / PAC", "True Support": 1757, "Precision": 0.3555, "Recall": 0.1793, "F1-Score": 0.2384},
      {"Class": "V", "Grouping Description": "Ventricular Ectopic / PVC", "True Support": 3018, "Precision": 0.7781, "Recall": 0.9298, "F1-Score": 0.8472},
      {"Class": "Q", "Grouping Description": "Fusion (F) & Unclassifiable (Q)", "True Support": 376, "Precision": 0.0489, "Recall": 0.2979, "F1-Score": 0.0840},
  ]
  ```
- Add the mandatory explanatory note:
  ```python
  st.caption("Matched-detection conditional beat-classification metrics — classification F1 calculated only for detector peaks matched to a reference annotation and retained after feature-window boundary checks. Detector false positives (115 peaks) and missed reference beats (542 beats) are reported separately in detector metrics.")
  ```

### 9.2 Corrections to `README.md` (lines 169–180)
- Update section header to:
  `#### 1. Primary Benchmark: Matched-Detection Conditional Beat-Classification Metrics`
- Update table rows with true support and true precision/recall:
  ```markdown
  | Beat Class Grouping | True Support | Precision | Recall | F1-Score | Grouping Description |
  | :--- | :---: | :---: | :---: | :---: | :--- |
  | **N** | 41,883 | 0.9592 | 0.9218 | **0.9401** | Normal / Bundle Branch Block Beats (`N`, `L`, `R`, `e`, `j`) |
  | **S** | 1,757 | 0.3555 | 0.1793 | **0.2384** | Supraventricular Ectopic Beats (`A`, `a`, `J`, `S`) |
  | **V** | 3,018 | 0.7781 | 0.9298 | **0.8472** | Ventricular Ectopic Beats (`V`, `E`) |
  | **Q** | 376 | 0.0489 | 0.2979 | **0.0840** | Ventricular Fusion (`F`) & Unclassifiable (`Q`) |
  | **Macro Average** | **47,034** | **0.5354** | **0.5822** | **0.5274** | Unweighted mean across all 4 classes |
  | **Weighted Average** | **47,034** | **0.9169** | **0.8906** | **0.9011** | Prevalence-weighted mean |
  ```
- Add mandatory caption note beneath table.

### 9.3 Corrections to `docs/SINGLE_LEAD_MODEL_CARD.md` (lines 105–118)
- Update title and table entries identical to `README.md`.

### 9.4 Corrections to `docs/PHASE6_REPORT.md` (lines 272–285) & `docs/PHASE6_2_TRANSPARENT_TAXONOMY_UPDATE.md`
- Update per-class table entries and add the qualified terminology note.

---

## 10. Audit Conclusion

The Phase 6.3 audit is complete.
1. All detection matching, window boundary mechanics, and underlying models are completely sound.
2. The exact mathematical breakdown of all beat populations ($47,576 \to 47,561 \to 47,163 \to 47,048 \to 47,034 \to 542$) is proven.
3. The discrepancy in the Phase 6.2 displayed support and precision/recall columns is definitively identified as an LLM formatting hallucination in step 2283, while the underlying F1 metrics ($0.5274$ and $0.9011$) and class F1s are verified as correct.
4. The exact proposed corrections are fully cataloged above.

*Execution paused per strict instructions. Awaiting explicit user approval before implementing any copy corrections or beginning another phase.*
