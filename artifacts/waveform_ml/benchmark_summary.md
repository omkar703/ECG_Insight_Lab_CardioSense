# CardioSense Phase 3.2 Waveform ML Full-Fold Benchmark Report

**Generated:** 2026-10-17 17:17:49 UTC

**Dataset:** PTB-XL v1.0.3 (100 Hz WFDB)

**Model:** Logistic Regression (One-vs-Rest, Balanced Class Weights, C=1.0)

**Features:** 148 deterministic lead & cross-lead statistics (Version 1.0.0)

**Split Statistics:** Train: 17,418 records (15,023 patients, Folds 1–8) | Validation: 2,183 records (1,942 patients, Fold 9) | Test: 2,198 records (1,904 patients, Fold 10, strictly held out)

**Peak Memory Footprint:** 515.25 MiB RSS

### Training Scale Comparison: Phase 3.1 Pilot Model vs. Phase 3.2 Full-Fold Model (Fold 10)

> [!NOTE]
> **Methodological Disclosure:** The Phase 3.1 pilot model was trained on 240 records (30 per fold, Folds 1–8) and thresholded on 30 records (Fold 9). The Phase 3.2 model is trained on the full permitted training partition and thresholded on the full validation partition. Both models are evaluated on the exact same Fold 10 test cohort.

| Superclass | Test Support | Pilot Model F1 (N_tr=240) | Full-Fold Model F1 | Delta ($\Delta\text{ F1}$) | Full Precision | Full Recall | Full PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NORM** | 963 | 0.5882 | **0.7668** | +0.1785 | 0.6995 | 0.8484 | 0.8128 |
| **MI** | 550 | 0.3883 | **0.5594** | +0.1711 | 0.4708 | 0.6891 | 0.5468 |
| **STTC** | 521 | 0.4909 | **0.6006** | +0.1097 | 0.5173 | 0.7159 | 0.5856 |
| **CD** | 496 | 0.4933 | **0.6002** | +0.1068 | 0.5581 | 0.6492 | 0.6671 |
| **HYP** | 262 | 0.3305 | **0.5195** | +0.1890 | 0.4520 | 0.6107 | 0.5338 |
| **Macro Aggregate** | 2198 | 0.4583 | **0.6093** | +0.1510 | — | — | 0.6292 |

### Benchmark Comparison: Naive Baseline vs. Lightweight Model

| Superclass | Support | Baseline F1 | Model F1 | Delta ($\Delta$) | Model Precision | Model Recall | Model PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NORM** | 963 | 0.0000 | **0.7668** | +0.7668 | 0.6995 | 0.8484 | 0.8128 |
| **MI** | 550 | 0.0000 | **0.5594** | +0.5594 | 0.4708 | 0.6891 | 0.5468 |
| **STTC** | 521 | 0.0000 | **0.6006** | +0.6006 | 0.5173 | 0.7159 | 0.5856 |
| **CD** | 496 | 0.0000 | **0.6002** | +0.6002 | 0.5581 | 0.6492 | 0.6671 |
| **HYP** | 262 | 0.0000 | **0.5195** | +0.5195 | 0.4520 | 0.6107 | 0.5338 |
| **Macro Aggregate** | 2198 | 0.0000 | **0.6093** | +0.6093 | — | — | 0.6292 |

### Validation Fold 9 Evaluation (Full Cohort, N=2,183, 1,942 Patients)
**Sample Count ($N$):** 2183 | **Macro F1:** 0.6352 | **Micro F1:** 0.6639 | **Macro PR-AUC:** 0.6334

| Superclass | Support | Threshold | TP / FP / FN / TN | Precision | Recall (Sens.) | F1 Score | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NORM** | 955 | 0.50 | 833 / 317 / 122 / 911 | 0.7243 | 0.8723 | **0.7914** | 0.8098 |
| **MI** | 540 | 0.50 | 395 / 413 / 145 / 1230 | 0.4889 | 0.7315 | **0.5861** | 0.5630 |
| **STTC** | 528 | 0.55 | 390 / 334 / 138 / 1321 | 0.5387 | 0.7386 | **0.6230** | 0.5874 |
| **CD** | 495 | 0.55 | 333 / 238 / 162 / 1450 | 0.5832 | 0.6727 | **0.6248** | 0.6582 |
| **HYP** | 268 | 0.65 | 173 / 187 / 95 / 1728 | 0.4806 | 0.6455 | **0.5510** | 0.5485 |

### Held-Out Test Fold 10 Re-Evaluation (Full Cohort, N=2,198, 1,904 Patients)
**Sample Count ($N$):** 2198 | **Macro F1:** 0.6093 | **Micro F1:** 0.6392 | **Macro PR-AUC:** 0.6292

| Superclass | Support | Threshold | TP / FP / FN / TN | Precision | Recall (Sens.) | F1 Score | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NORM** | 963 | 0.50 | 817 / 351 / 146 / 884 | 0.6995 | 0.8484 | **0.7668** | 0.8128 |
| **MI** | 550 | 0.50 | 379 / 426 / 171 / 1222 | 0.4708 | 0.6891 | **0.5594** | 0.5468 |
| **STTC** | 521 | 0.55 | 373 / 348 / 148 / 1329 | 0.5173 | 0.7159 | **0.6006** | 0.5856 |
| **CD** | 496 | 0.55 | 322 / 255 / 174 / 1447 | 0.5581 | 0.6492 | **0.6002** | 0.6671 |
| **HYP** | 262 | 0.65 | 160 / 194 / 102 / 1742 | 0.4520 | 0.6107 | **0.5195** | 0.5338 |

### 95% Patient-Clustered Bootstrap Confidence Intervals (Fold 10, N=2,198, 1,904 Patients)
**Bootstrap Resamples:** 1000 (patient_cluster, 1904 patients) | **Macro F1 (95% CI):** 0.6091 [0.5910, 0.6269]

| Superclass | Bootstrap Mean F1 | 95% Confidence Interval | CI Width |
| :--- | :--- | :--- | :--- |
| **NORM** | 0.7668 | [0.7451, 0.7872] | 0.0421 |
| **MI** | 0.5599 | [0.5255, 0.5951] | 0.0695 |
| **STTC** | 0.5997 | [0.5673, 0.6315] | 0.0643 |
| **CD** | 0.5995 | [0.5617, 0.6394] | 0.0777 |
| **HYP** | 0.5193 | [0.4671, 0.5662] | 0.0991 |