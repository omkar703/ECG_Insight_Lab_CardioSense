# CardioSense Phase 3 Waveform ML Benchmark Report

**Generated:** 2026-10-17 16:17:01 UTC

**Dataset:** PTB-XL v1.0.3 (100 Hz WFDB)

**Model:** Logistic Regression (One-vs-Rest, Balanced Class Weights, C=1.0)

**Features:** 148 deterministic lead & cross-lead statistics (Version 1.0.0)

**Split Statistics:** Train: 240 records (15,023 patients, Folds 1–8) | Validation: 30 records (1,942 patients, Fold 9) | Test: 2,198 records (1,904 patients, Fold 10, strictly held out)

### Benchmark Comparison: Naive Baseline vs. Lightweight Model

| Superclass | Support | Baseline F1 | Model F1 | Delta ($\Delta$) | Model Precision | Model Recall | Model PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NORM** | 963 | 0.6093 | **0.5882** | -0.0211 | 0.7070 | 0.5036 | 0.6872 |
| **MI** | 550 | 0.0000 | **0.3883** | +0.3883 | 0.3196 | 0.4945 | 0.3378 |
| **STTC** | 521 | 0.0000 | **0.4909** | +0.4909 | 0.3639 | 0.7543 | 0.4503 |
| **CD** | 496 | 0.0000 | **0.4933** | +0.4933 | 0.4406 | 0.5605 | 0.4729 |
| **HYP** | 262 | 0.0000 | **0.3305** | +0.3305 | 0.2961 | 0.3740 | 0.2756 |
| **Macro Aggregate** | 2198 | 0.1219 | **0.4583** | +0.3364 | — | — | 0.4448 |

### Validation Fold 9 Evaluation
**Sample Count ($N$):** 30 | **Macro F1:** 0.5232 | **Micro F1:** 0.5538 | **Macro PR-AUC:** 0.4189

| Superclass | Support | Threshold | TP / FP / FN / TN | Precision | Recall (Sens.) | F1 Score | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NORM** | 10 | 0.80 | 8 / 3 / 2 / 17 | 0.7273 | 0.8000 | **0.7619** | 0.7306 |
| **MI** | 5 | 0.30 | 3 / 5 / 2 / 20 | 0.3750 | 0.6000 | **0.4615** | 0.2722 |
| **STTC** | 8 | 0.10 | 4 / 7 / 4 / 15 | 0.3636 | 0.5000 | **0.4211** | 0.3917 |
| **CD** | 2 | 0.55 | 2 / 3 / 0 / 25 | 0.4000 | 1.0000 | **0.5714** | 0.4500 |
| **HYP** | 1 | 0.55 | 1 / 3 / 0 / 26 | 0.2500 | 1.0000 | **0.4000** | 0.2500 |

### Held-Out Test Fold 10 Evaluation (Full Cohort, N=2,198)
**Sample Count ($N$):** 2198 | **Macro F1:** 0.4583 | **Micro F1:** 0.4790 | **Macro PR-AUC:** 0.4448

| Superclass | Support | Threshold | TP / FP / FN / TN | Precision | Recall (Sens.) | F1 Score | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NORM** | 963 | 0.80 | 485 / 201 / 478 / 1034 | 0.7070 | 0.5036 | **0.5882** | 0.6872 |
| **MI** | 550 | 0.30 | 272 / 579 / 278 / 1069 | 0.3196 | 0.4945 | **0.3883** | 0.3378 |
| **STTC** | 521 | 0.10 | 393 / 687 / 128 / 990 | 0.3639 | 0.7543 | **0.4909** | 0.4503 |
| **CD** | 496 | 0.55 | 278 / 353 / 218 / 1349 | 0.4406 | 0.5605 | **0.4933** | 0.4729 |
| **HYP** | 262 | 0.55 | 98 / 233 / 164 / 1703 | 0.2961 | 0.3740 | **0.3305** | 0.2756 |

### 95% Bootstrap Confidence Intervals (Fold 10, N=2198)
**Bootstrap Resamples:** 1000 | **Macro F1 (95% CI):** 0.4578 [0.4411, 0.4748]

| Superclass | Bootstrap Mean F1 | 95% Confidence Interval | CI Width |
| :--- | :--- | :--- | :--- |
| **NORM** | 0.5874 | [0.5576, 0.6158] | 0.0583 |
| **MI** | 0.3881 | [0.3558, 0.4215] | 0.0657 |
| **STTC** | 0.4901 | [0.4600, 0.5194] | 0.0593 |
| **CD** | 0.4932 | [0.4575, 0.5305] | 0.0730 |
| **HYP** | 0.3301 | [0.2833, 0.3784] | 0.0951 |