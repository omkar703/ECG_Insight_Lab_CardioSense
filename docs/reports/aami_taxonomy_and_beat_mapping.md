# Phase 6.2 Report: Transparent Taxonomy & Evaluation Disclosure

**ECG Insight Lab / CardioSense — Phase 6.2 Deliverable**  
**Date:** 2026-10-18  
**Status:** Complete & Fully Verified  
**Decision Followed:** Option 1 (Retain verified frozen model; update documentation, disclosures, and UI copy)  

---

## 1. Executive Summary

Phase 6.2 implements **Option 1** as selected following the comprehensive methodological audit recorded in [`docs/PHASE6_1_EVALUATION_PROTOCOL_AUDIT.md`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/docs/PHASE6_1_EVALUATION_PROTOCOL_AUDIT.md).

Rather than mutating the trained model or silently remapping labels, this phase ensures **complete scientific transparency and academic honesty** by:
1. Formally disclosing that the single-lead beat classifier implements a **project-defined 4-class beat grouping (N, S, V, Q, with ventricular fusion beats F merged into Q)**, rather than a standard ANSI/AAMI EC57 five-class implementation.
2. Presenting the **Matched-Detection Conditional Beat-Classification Metrics** ($N=47,034$, Macro F1 = **0.5274**, Weighted F1 = **0.9011**) as the primary benchmark in all interfaces and documentation.
3. Repositioning the **Annotation-Centred Classifier Reference** ($N=47,561$, Macro F1 = **0.5396**, Weighted F1 = **0.9036**) as a secondary diagnostic reference that isolates feature discrimination when beat localization is supplied.
4. Fully explaining the exact mathematical derivation of all beat population counts ($47,576$ reference annotations, $47,561$ oracle reference beats, $47,163$ detected peaks, $47,048$ detector true positives, $47,034$ matched feature-valid detections, and $542$ unrepresented reference beats).
5. Enforcing that the model artifact remained **100% frozen** with zero retraining and zero changes to model weights, feature schemas, or decision thresholds.

---

## 2. Mandatory Taxonomy Disclosure

Across all user-facing interfaces, model cards, and reports, the following exact disclosure is enforced:

> **Taxonomy Disclosure:** Project-defined 4-class beat grouping: N, S, V, Q, with ventricular fusion beats (F) merged into Q. This is not a standard five-class AAMI EC57 implementation.

### Empirical Beat Distribution of Class Q
In the evaluated non-paced MIT-BIH recordings (21 held-out test patients), **98.2% of the beats assigned to Class Q are ventricular fusion beats (`F`)** rather than unclassifiable beats:
- Total Reference Beats in Test Partition: $47,576$
- Normal group (`N`, `L`, `R`, `j`): $42,198$ ($88.69\%$)
- Supraventricular Ectopic group (`A`, `J`, `a`): $1,782$ ($3.75\%$)
- Ventricular Ectopic group (`V`, `E`): $3,202$ ($6.73\%$)
- Ventricular Fusion group (`F`): $387$ ($0.81\%$)
- Unclassifiable group (`Q`): $7$ ($0.01\%$)
- Implemented Class Q in Test = $387 (\text{F}) + 7 (\text{Q}) = 394$ beats ($98.2\% \text{ F}$).
- Paced rhythm beats (`/` and `f`) reside almost exclusively in the four quarantined records (`102`, `104`, `107`, `217`) and do not appear in the evaluated test set.

---

## 3. Separated Benchmark Protocols & Quantitative Findings

### 3.1 Primary Benchmark: Matched-Detection Conditional Beat-Classification Metrics (N = 47,034)
Evaluates continuous raw ECG signals without ground-truth annotations. The Pan-Tompkins algorithm identifies R-peak candidates without consuming ground-truth annotations; detected peaks are matched against expert annotations within $\pm 50\,\text{ms}$ tolerance and classified by the Balanced Random Forest model:

> [!NOTE]
> **Matched-detection conditional beat-classification metrics** — classification F1 is calculated only for detector peaks matched to a reference annotation and retained after feature-window boundary checks. Detector false positives (115 peaks) and missed reference beats (542 beats, including boundary-dropped matched beats) are reported separately in detector/count-reconciliation metrics and are not included as errors in this classifier confusion matrix.

- **Evaluated Detections (Feature-Valid):** **47,034 beats**
- **Macro F1-Score:** **0.5274**
- **Weighted F1-Score:** **0.9011**

| Beat Class Grouping | Matched Reference Support | Precision | Recall | F1-Score | Grouping Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **N** | 41,883 | 0.9592 | 0.9218 | **0.9401** | Normal / Bundle Branch Block Beats (`N`, `L`, `R`, `e`, `j`) |
| **S** | 1,757 | 0.3555 | 0.1793 | **0.2384** | Supraventricular Ectopic Beats (`A`, `a`, `J`, `S`) |
| **V** | 3,018 | 0.7781 | 0.9298 | **0.8472** | Ventricular Ectopic Beats (`V`, `E`) |
| **Q** | 376 | 0.0489 | 0.2979 | **0.0840** | Ventricular Fusion (`F`) & Unclassifiable (`Q`) |
| **Macro Average** | **47,034** | **0.5354** | **0.5822** | **0.5274** | Unweighted arithmetic mean across classes |
| **Weighted Average** | **47,034** | **0.9178** | **0.8896** | **0.9011** | Class-prevalence weighted mean |

**Detector Operating Characteristics:**
- Total Reference Beats: $47,576$
- Total Detected Peaks: $47,163$
- True Positives (TP): $47,048$ ($\pm 50\,\text{ms}$ match)
- False Positives (FP): $115$ (unmatched artifact detections)
- False Negatives (FN): $528$ (missed reference beats)
- Sensitivity: **98.89%** | PPV: **99.76%** | F1-Score: **99.32%**
- Pipeline Loss Accounting: $542$ reference beats unrepresented ($315$ N, $25$ S, $184$ V, $18$ Q) due to $527$ non-boundary detector misses + $1$ boundary detector miss + $14$ boundary-dropped matched detections.

### 3.2 Secondary Benchmark: Annotation-Centred (Oracle) Classifier Reference
Diagnostic reference isolating classifier feature discrimination assuming ideal QRS localization. Beat windows are centred on expert ground-truth annotation indices (47,561 valid windows after dropping 15 boundary reference annotations); **this is not an autonomous end-to-end result**:

| Beat Class Grouping | Reference Support | Precision | Recall | F1-Score | Baseline F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **N** (Normal / BBB Beat) | 42,184 | 0.9595 | 0.9247 | **0.9418** | 0.9401 |
| **S** (Supraventricular Ectopic / PAC) | 1,782 | 0.4035 | 0.2043 | **0.2712** | 0.0000 |
| **V** (Ventricular Ectopic / PVC) | 3,201 | 0.7912 | 0.9222 | **0.8517** | 0.0000 |
| **Q** (Ventricular Fusion & Other) | 394 | 0.0549 | 0.3173 | **0.0936** | 0.0000 |
| **Macro Average** | **47,561** | **0.5523** | **0.5921** | **0.5396** | 0.2350 |
| **Weighted Average** | **47,561** | **0.9192** | **0.9002** | **0.9036** | 0.8338 |

---

## 4. Complete Beat Count Reconciliation

The discrepancy between the various reported beat counts is reconciled below:

```text
47,576 Reference Beats (Total expert annotations in 21 test records)
   │
   ├── [Minus 15 Boundary Drops] (<90 samples from file start or >649,890 samples from file end)
   │     - 14 Class N beats
   │     - 1 Class V beat (Record 233, sample 42)
   │
   └── = 47,561 Evaluated Oracle Beats (Protocol 1 Confusion Matrix Support)
           [N: 42,184 | S: 1,782 | V: 3,201 | Q: 394]

Autonomous Detector Processing Pipeline:
   47,163 Candidate Peaks Detected by Pan-Tompkins Algorithm
   │
   ├── 47,048 True Positives (Matched to reference within ±50 ms tolerance)
   │     │
   │     ├── [Minus 14 Boundary Detections] (Too close to recording edges for 200-sample window: 13 N, 1 V)
   │     │
   │     └── = 47,034 Evaluated Matched Detections (Protocol 2 Evaluated Beats)
   │             [N: 41,883 | S: 1,757 | V: 3,018 | Q: 376]
   │
   ├── 115 False Positives (Detector artifacts with no matching reference beat)
   │
   └── 528 False Negatives (Reference beats missed by detector within ±50 ms)
         - 1 boundary reference beat is already in this count (Record 213, sample 649,992, Class N)
         - 527 non-boundary reference beats missed (301 N, 25 S, 183 V, 18 Q)
         - 527 non-boundary misses + 1 boundary miss + 14 boundary-dropped matched detections (13 N, 1 V) = 542 Unrepresented Reference Beats (315 N, 25 S, 184 V, 18 Q)
```

---

## 5. Scope of Changes Across Project Files

| File | Nature of Changes | Verification Status |
| :--- | :--- | :---: |
| [`app.py`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/app.py) | Added top taxonomy disclosure banner in Single-Lead Lab; updated legend badge to `Class Q (Fusion & Other)` with description `Ventricular Fusion (F) & Unclassifiable (Q)`; replaced benchmark expander with separated Primary End-to-End and Secondary Oracle tables + count reconciliation; added Single-Lead taxonomy tab in Methods & Benchmarks. | **VERIFIED** |
| [`README.md`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/README.md) | Updated Workflow C description under Architectural Tenets; overhauled Section 4 (Single-Lead Beat ML Benchmark) to prominently display Primary End-to-End metrics, Secondary Oracle reference, exact count reconciliation, and updated test suite count. | **VERIFIED** |
| [`docs/SINGLE_LEAD_MODEL_CARD.md`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/docs/SINGLE_LEAD_MODEL_CARD.md) | Updated Section 1 (Model Overview) to disclose project-defined 4-class grouping with F merged into Q; restructured Section 5 into Primary End-to-End ($N=47,034$), Secondary Oracle ($N=47,561$), and Beat Count Reconciliation. | **VERIFIED** |
| [`docs/PHASE6_REPORT.md`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/docs/PHASE6_REPORT.md) | Added oracle clarification callout in Section 4.2; appended **Section 9: Phase 6.1 Audit Resolution & Phase 6.2 Transparent Taxonomy Update** detailing both protocols, counts, and zero-retraining confirmation. | **VERIFIED** |
| [`tests/test_single_lead_ml.py`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/tests/test_single_lead_ml.py) | Added `TestPhase62TaxonomyDisclosures` with 4 automated regression test cases asserting presence of required taxonomy disclosure strings and metrics in `app.py`, `README.md`, `SINGLE_LEAD_MODEL_CARD.md`, and `PHASE6_REPORT.md`. | **VERIFIED** |

---

## 6. Verification & Quality Assurance Evidence

### 6.1 Automated Test Execution Log
```text
============================= test session starts ==============================
platform linux -- Python 3.14.4, pytest-9.1.1, pluggy-1.7.0
rootdir: /run/media/op/DATA/Omkar/VIT/2y/3 sem/Project_expo/code
plugins: anyio-4.15.1, platformdirs-4.13.3
collected 93 items

tests/test_digitization.py ...........                                   [ 11%]
tests/test_productization.py ......                                      [ 18%]
tests/test_single_lead_ml.py .............................               [ 49%]
tests/test_validation.py ............                                    [ 62%]
tests/test_waveform_dataset.py .......                                   [ 70%]
tests/test_waveform_features.py .......                                  [ 77%]
tests/test_waveform_isolation.py ....                                    [ 81%]
tests/test_waveform_metrics.py .......                                   [ 89%]
tests/test_waveform_splits.py .....                                      [ 94%]
tests/test_waveform_training_integrity.py .....                          [100%]

============================= 93 passed in 13.88s ==============================
```

### 6.2 Cryptographic Model Hash Confirmation
- **Target File:** `artifacts/single_lead_ml/single_lead_rf_model.joblib`
- **Expected SHA-256:** `3889a5f5ab97803da339710d4ae9fde8ea50f8e5a5ff76ed2015a0fb0b6ec8e9`
- **Measured SHA-256:** `3889a5f5ab97803da339710d4ae9fde8ea50f8e5a5ff76ed2015a0fb0b6ec8e9`
- **Integrity Status:** **MATCH (Zero Retraining, Zero Byte Mutation)**

---

## 7. Strict Safety Boundaries Checklist

- [x] **Zero Model Retraining:** No model weights, decision thresholds, or hyperparameters were adjusted.
- [x] **Zero Code Policy Breach:** No changes to `SYMBOL_TO_AAMI` mapping, feature extraction code, or dataset partitions.
- [x] **Preserved Audit Reports:** `docs/PHASE6_1_EVALUATION_PROTOCOL_AUDIT.md` and historical reports were preserved untouched.
- [x] **No Untrusted Deserialization:** Quarantined legacy pickles remain blocked by policy.
- [x] **No Clinical or Hardware Claims:** Workflow C is explicitly documented as an offline recorded Holter playback simulation, not live hardware and not an acute coronary diagnostic tool.
- [x] **Non-Conformant Taxonomy Disclosed:** Documented across all user-facing copy that the model uses a project-defined 4-class grouping and is not a 5-class AAMI EC57 conformant classifier.
