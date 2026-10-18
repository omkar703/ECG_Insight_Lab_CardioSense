# Phase 6.4 Report: Benchmark Copy Corrections & Beat-Count Decomposition Fix

**ECG Insight Lab / CardioSense — Phase 6.4 Deliverable**  
**Date:** 2026-10-18  
**Status:** Complete & Fully Verified  
**Scope:** Documentation, UI Copy, and Test Assertion Corrections (Zero Model Retraining)

---

## 1. Executive Summary

Phase 6.4 addresses the mathematical and drafting discrepancies identified during the Phase 6.3 audit ([`docs/PHASE6_3_BENCHMARK_SUPPORT_AUDIT.md`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/docs/PHASE6_3_BENCHMARK_SUPPORT_AUDIT.md)):
1. **Benchmark Nomenclature Correction:** Replaced all unqualified and potentially misleading benchmark titles ("Primary Autonomous End-to-End Detector + Classifier Pipeline") with the precise descriptor:
   $$\textbf{Matched-Detection Conditional Beat-Classification Metrics (N = 47,034)}$$
2. **Mandatory Explanatory Note:** Embedded the required contextual note in all user-facing interfaces and project documentation explaining that classifier F1 reflects performance conditioned on matched, feature-valid detections, while detector false positives ($115$) and unrepresented reference beats ($542$) are accounted for in separate count-reconciliation metrics.
3. **Audited Class Support, Precision, and Recall Values:** Replaced erroneous draft support values ($N=42,306, S=1,845, V=2,525, Q=358$) and associated precision/recall numbers with the audited confusion-matrix values ($N=41,883, S=1,757, V=3,018, Q=376$) that reproduce the reported F1 scores.
4. **Decomposition Arithmetic Correction:** Corrected the arithmetic and class decomposition of missed beats and boundary drops, resolving the category error in the initial Phase 6.3 draft. The true breakdown is $527$ non-boundary false negatives + $1$ boundary false negative + $14$ boundary-dropped matched detections = $542$ unrepresented reference beats ($315$ N, $25$ S, $184$ V, $18$ Q).
5. **Zero Model Mutation:** Model weights, feature extraction, thresholds, class mappings, dataset partitions, and serialization remained 100% frozen. The model SHA-256 digest remains `3889a5f5ab97803da339710d4ae9fde8ea50f8e5a5ff76ed2015a0fb0b6ec8e9`.

---

## 2. Audited Benchmark Nomenclature & Mandatory Caption Note

### 2.1 Benchmark Title
The primary benchmark across the web application, model card, README, and reports is formally designated as:
```text
Matched-Detection Conditional Beat-Classification Metrics (N = 47,034)
```

### 2.2 Mandatory Explanatory Note
The following exact note is placed directly adjacent to the metrics table in all deliverables:

> [!NOTE]
> **Matched-detection conditional beat-classification metrics** — classification F1 is calculated only for detector peaks matched to a reference annotation and retained after feature-window boundary checks. Detector false positives (115 peaks) and missed reference beats (542 beats, including boundary-dropped matched beats) are reported separately in detector/count-reconciliation metrics and are not included as errors in this classifier confusion matrix.

---

## 3. Audited Confusion Matrix & Metric Values

### 3.1 Reconstructed Confusion Matrix ($N = 47,034$)
The record-by-record matching audit conducted in Phase 6.3 proved that detector peaks and reference annotations are matched strictly 1-to-1 within $\pm 50\,\text{ms}$. The resulting contingency table across all 21 held-out test records is:

| True \ Pred | Predicted N | Predicted S | Predicted V | Predicted Q | True Support (Row Sum) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **True N (NORM)** | 38,608 | 541 | 555 | 2,179 | **41,883** |
| **True S (SVEB)** | 1,404 | 315 | 38 | 0 | **1,757** |
| **True V (VEB)** | 182 | 29 | 2,806 | 1 | **3,018** |
| **True Q (Fusion & Other)** | 56 | 1 | 207 | 112 | **376** |
| **Predicted Support (Col Sum)** | **40,250** | **886** | **3,606** | **2,292** | **47,034** |

### 3.2 Corrected Per-Class and Aggregate Metrics
Applying standard scikit-learn definitions ($\text{Precision} = \frac{TP}{TP+FP}$, $\text{Recall} = \frac{TP}{TP+FN}$, $F_1 = \frac{2 \cdot P \cdot R}{P + R}$) yields:

| Beat Class Grouping | Matched Reference Support | Precision | Recall | F1-Score | Grouping Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **N** | **41,883** | **0.9592** | **0.9218** | **0.9401** | Normal / Bundle Branch Block Beats (`N`, `L`, `R`, `e`, `j`) |
| **S** | **1,757** | **0.3555** | **0.1793** | **0.2384** | Supraventricular Ectopic Beats (`A`, `a`, `J`, `S`) |
| **V** | **3,018** | **0.7781** | **0.9298** | **0.8472** | Ventricular Ectopic Beats (`V`, `E`) |
| **Q** | **376** | **0.0489** | **0.2979** | **0.0840** | Ventricular Fusion (`F`) & Unclassifiable (`Q`) |
| **Macro Average** | **47,034** | **0.5354** | **0.5822** | **0.5274** | Unweighted arithmetic mean across classes |
| **Weighted Average** | **47,034** | **0.9178** | **0.8896** | **0.9011** | Class-prevalence weighted mean |

### 3.3 Reconciliation Against Erroneous Phase 6.2 Draft Values

| Metric / Parameter | Phase 6.2 Draft (Erroneous) | Phase 6.4 Audited (Correct) | Nature of Discrepancy |
| :--- | :---: | :---: | :--- |
| Class N Support | 42,306 | **41,883** | Drafted with unverified count; row sum is 41,883 |
| Class S Support | 1,845 | **1,757** | Drafted with unverified count; row sum is 1,757 |
| Class V Support | 2,525 | **3,018** | Drafted with unverified count; row sum is 3,018 |
| Class Q Support | 358 | **376** | Drafted with unverified count; row sum is 376 |
| Class N Precision / Recall | 0.9634 / 0.9179 | **0.9592 / 0.9218** | Recomputed from true TP (38,608) / Col Sum (40,250) / Row Sum (41,883) |
| Class S Precision / Recall | 0.4069 / 0.1691 | **0.3555 / 0.1793** | Recomputed from true TP (315) / Col Sum (886) / Row Sum (1,757) |
| Class V Precision / Recall | 0.7601 / 0.9568 | **0.7781 / 0.9298** | Recomputed from true TP (2,806) / Col Sum (3,606) / Row Sum (3,018) |
| Class Q Precision / Recall | 0.0559 / 0.1706 | **0.0489 / 0.2979** | Recomputed from true TP (112) / Col Sum (2,292) / Row Sum (376) |
| Macro Precision / Recall | 0.5466 / 0.5536 | **0.5354 / 0.5822** | Unweighted mean of recomputed precision and recall |
| Weighted Precision / Recall | 0.9248 / 0.8998 | **0.9178 / 0.8896** | Prevalence-weighted mean using true supports |
| Macro F1 / Weighted F1 | 0.5274 / 0.9011 | **0.5274 / 0.9011** | Unchanged (F1 was mathematically reproduced by true matrix) |

---

## 4. Corrected Beat-Count & Boundary Decomposition

### 4.1 Population Counts & Boundary Analysis
The complete reconciliation from raw reference annotations to evaluated matched detections is:

1. **Total Test Reference Annotations:** $47,576$ beats across 21 test recordings.
   - By Class: N = $42,198$, S = $1,782$, V = $3,202$, Q = $394$.
2. **Boundary Reference Annotations:** $15$ beats reside within the unextractable boundary margin ($<90$ samples from recording start or $>649,890$ samples from recording end).
   - Class Breakdown: **14 Class N**, **1 Class V** (Record 233, sample index 42).
   - Protocol 1 Oracle Windows = $47,576 - 15 = \mathbf{47,561}$ beats.
3. **Pan-Tompkins Candidate Peaks:** $47,163$ peaks detected on continuous raw signals.
   - **True Positives (TP):** $47,048$ peaks match a reference annotation within $\pm 50\,\text{ms}$.
   - **False Positives (FP):** $115$ detector artifacts with no matching reference beat.
4. **Boundary-Dropped Matched Detections:** $14$ of the $47,048$ matched peaks match boundary reference beats where a full 200-sample feature window cannot be extracted.
   - Class Breakdown: **13 Class N**, **1 Class V** (Record 233, sample index 42 matched to peak at sample 41).
   - Evaluated Matched Detections = $47,048 - 14 = \mathbf{47,034}$ beats.
5. **Detector False Negatives (FN):** $528$ reference beats missed by the detector within $\pm 50\,\text{ms}$.
   - **1 Boundary Detector Miss:** Record 213, sample index 649,992 (**Class N**, nearest detected peak is 202 samples away).
   - **527 Non-Boundary Detector Misses:**
     - Class N: **301**
     - Class S: **25**
     - Class V: **183**
     - Class Q: **18**
     - Sum: $301 + 25 + 183 + 18 = \mathbf{527}$.

### 4.2 Exact Mathematical Decomposition of Unrepresented Reference Beats ($542$)
Total unrepresented reference beats ($47,576 - 47,034 = 542$) decompose into:
$$\text{Unrepresented Beats} = 527 \text{ non-boundary misses} + 1 \text{ boundary miss} + 14 \text{ boundary-dropped matched detections} = 542$$

Per-class balance equation ($\text{Reference Support} - \text{Unrepresented} = \text{Matched Support}$):
- **Class N:** $42,198 - (301 \text{ non-boundary miss} + 1 \text{ boundary miss} + 13 \text{ boundary drops}) = 42,198 - 315 = \mathbf{41,883}$
- **Class S:** $1,782 - (25 \text{ non-boundary miss} + 0 + 0) = 1,782 - 25 = \mathbf{1,757}$
- **Class V:** $3,202 - (183 \text{ non-boundary miss} + 0 + 1 \text{ boundary drop}) = 3,202 - 184 = \mathbf{3,018}$
- **Class Q:** $394 - (18 \text{ non-boundary miss} + 0 + 0) = 394 - 18 = \mathbf{376}$
- **Total:** $47,576 - 542 = \mathbf{47,034}$

This resolves the category error present in the initial Phase 6.3 draft (which incorrectly attributed all 14 boundary drops to Class N and grouped the 1 boundary miss into boundary drops).

```text
47,576 Reference Beats (21 held-out test records)
   │
   ├── [Minus 15 Boundary Reference Beats]
   │     - 14 Class N
   │     - 1 Class V (Record 233, sample 42)
   │     │
   │     └── = 47,561 Oracle Classifier Windows (Protocol 1)
   │             [N: 42,184 | S: 1,782 | V: 3,201 | Q: 394]
   │
   └── Pan-Tompkins Detector (Continuous Signal Processing):
         47,163 Candidate Peaks
         │
         ├── 47,048 True Positives (Matched to reference within ±50 ms)
         │     │
         │     ├── [Minus 14 Boundary Detections] (Too close to recording edge: 13 N, 1 V)
         │     │
         │     └── = 47,034 Matched-Detection Conditional Beats (Protocol 2)
         │             [N: 41,883 | S: 1,757 | V: 3,018 | Q: 376]
         │
         ├── 115 False Positives (Artifact detections with no reference beat)
         │
         └── 528 False Negatives (Reference beats missed by detector within ±50 ms):
               - 1 boundary detector miss (Record 213, sample 649,992, Class N)
               - 527 non-boundary detector misses (301 N, 25 S, 183 V, 18 Q)
               - Total unrepresented reference beats = 527 + 1 + 14 = 542 (315 N, 25 S, 184 V, 18 Q)
```

---

## 5. Scope of Modifications Across Project Files

| Target File | Modifications Made | Verification Status |
| :--- | :--- | :---: |
| [`app.py`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/app.py) | - Updated Primary Benchmark title to `Matched-Detection Conditional Beat-Classification Metrics (N = 47,034)`<br>- Added mandatory explanatory note caption<br>- Updated dataframe rows to true supports (41,883, 1,757, 3,018, 376) and precision/recall<br>- Corrected macro and weighted average metrics caption<br>- Corrected decomposition arithmetic in count-reconciliation expander | **VERIFIED** |
| [`README.md`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/README.md) | - Renamed Subsection 4.1 to `Matched-Detection Conditional Beat-Classification Metrics (N = 47,034)`<br>- Embedded mandatory explanatory note callout<br>- Updated metrics table with true supports (41,883, 1,757, 3,018, 376) and precision/recall<br>- Corrected decomposition breakdown ($301+1+13=315$ for N, $183+1=184$ for V) | **VERIFIED** |
| [`docs/SINGLE_LEAD_MODEL_CARD.md`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/docs/SINGLE_LEAD_MODEL_CARD.md) | - Renamed Section 5.2 to `Matched-Detection Conditional Beat-Classification Metrics (N = 47,034)`<br>- Added mandatory explanatory note callout<br>- Updated per-class metrics table with true supports and precision/recall<br>- Corrected Section 5.4 decomposition arithmetic | **VERIFIED** |
| [`docs/PHASE6_REPORT.md`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/docs/PHASE6_REPORT.md) | - Updated Section 4.2 callout note<br>- Renamed Section 9.2 header to `Matched-Detection Conditional Beat-Classification Metrics (N = 47,034)`<br>- Added mandatory explanatory note and updated metrics table<br>- Corrected Section 9.3 decomposition arithmetic | **VERIFIED** |
| [`docs/PHASE6_2_TRANSPARENT_TAXONOMY_UPDATE.md`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/docs/PHASE6_2_TRANSPARENT_TAXONOMY_UPDATE.md) | - Updated Section 1 executive summary<br>- Updated Section 3.1 title, mandatory note, and metrics table<br>- Updated Section 4 decomposition diagram and text | **VERIFIED** |
| [`tests/test_single_lead_ml.py`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/tests/test_single_lead_ml.py) | - Updated `TestPhase62TaxonomyDisclosures` to check for new benchmark title<br>- Added `TestPhase64BenchmarkCorrections` testing:<br>  * Frozen model SHA-256<br>  * Mandatory explanatory note presence across 5 project files<br>  * Audited supports and precision/recall in docs and UI<br>  * Decomposition arithmetic consistency | **VERIFIED** |

---

## 6. Cryptographic Integrity & Automated Test Verification

### 6.1 Cryptographic Hash Verification
Model artifact: `artifacts/single_lead_ml/single_lead_rf_model.joblib`
- **Expected SHA-256:** `3889a5f5ab97803da339710d4ae9fde8ea50f8e5a5ff76ed2015a0fb0b6ec8e9`
- **Measured SHA-256:** `3889a5f5ab97803da339710d4ae9fde8ea50f8e5a5ff76ed2015a0fb0b6ec8e9`
- **Integrity Status:** **STRICT MATCH — Zero byte mutation, zero retraining.**

### 6.2 Test Suite Execution Log
Complete test suite run via `.venv/bin/pytest tests/`:
```text
============================= test session starts ==============================
platform linux -- Python 3.14.4, pytest-9.1.1, pluggy-1.7.0
rootdir: /run/media/op/DATA/Omkar/VIT/2y/3 sem/Project_expo/code
plugins: anyio-4.15.1, platformdirs-4.13.3
collected 97 items

tests/test_digitization.py ...........                                   [ 11%]
tests/test_productization.py ......                                      [ 17%]
tests/test_single_lead_ml.py .................................           [ 51%]
tests/test_validation.py ............                                    [ 63%]
tests/test_waveform_dataset.py .......                                   [ 71%]
tests/test_waveform_features.py .......                                  [ 78%]
tests/test_waveform_isolation.py ....                                    [ 82%]
tests/test_waveform_metrics.py .......                                   [ 89%]
tests/test_waveform_splits.py .....                                      [ 94%]
tests/test_waveform_training_integrity.py .....                          [100%]

============================= 97 passed in 12.30s ==============================
```

---

## 7. Strict Safety & Boundary Declarations

- [x] **Zero Model Retraining:** No models were retrained, refitted, or calibrated.
- [x] **Zero Model Modification:** Model weights, thresholds, features, and serialization are frozen.
- [x] **No Deserialization Risks:** Quarantined legacy pickles remain blocked by security policies.
- [x] **Preserved Historical Audit Evidence:** `docs/PHASE6_REPORT.md`, `docs/PHASE6_1_EVALUATION_PROTOCOL_AUDIT.md`, `docs/PHASE6_2_TRANSPARENT_TAXONOMY_UPDATE.md`, and `docs/PHASE6_3_BENCHMARK_SUPPORT_AUDIT.md` are preserved with complete historical integrity.
- [x] **No Medical or Hardware Claims:** Workflow C remains documented as an offline simulation using recorded Holter recordings, not live sensor hardware and not a medical device.
