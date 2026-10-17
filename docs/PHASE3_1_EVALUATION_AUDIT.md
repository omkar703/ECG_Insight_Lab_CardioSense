# CardioSense / ECG Insight Lab — Phase 3.1 Evaluation Integrity Audit Report

**Date:** 2026-10-17  
**Auditor:** Senior Software Engineer & Technical Auditor (Antigravity)  
**Dataset Version:** PhysioNet PTB-XL v1.0.3 (WFDB 100 Hz, `records100/`)  
**Git Revision:** `6a0267c94f8c29b6ca42901a8d929703d74a8c14`  
**Model Artifact:** [`artifacts/waveform_ml/waveform_lr_model.joblib`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/artifacts/waveform_ml/waveform_lr_model.joblib)  
**Model SHA-256:** `850a26e38c46f5ae60af0a2157732ffed59e33b9e77f98ed63404cc85a9eb8ac`  
**Provenance Manifest:** [`artifacts/waveform_ml/waveform_lr_model.joblib.manifest.json`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/artifacts/waveform_ml/waveform_lr_model.joblib.manifest.json)  
**Audit Artifacts:** [`docs/scp_statements_audit.csv`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/docs/scp_statements_audit.csv), [`artifacts/waveform_ml/scp_statements_audit.json`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/artifacts/waveform_ml/scp_statements_audit.json)

---

## 1. Executive Summary & Audit Mandate

Phase 3.1 was commissioned to conduct an independent verification and integrity audit of the Phase 3 Digital Waveform ML subsystem before proceeding to Phase 4. Specifically, this audit was mandated to:
1. **Audit Label Mapping Integrity (Audit A):** Verify that every diagnostic code maps directly from the official PTB-XL `scp_statements.csv` metadata, investigate why `ISC_` was listed under both `STTC` and `CD` in the Phase 3 report, and reconcile full-dataset superclass counts against official PhysioNet documentation.
2. **Benchmark the Complete Held-Out Test Fold (Audit B):** Investigate why the Phase 3 report evaluated only $N=30$ test records when Fold 10 contains 2,198 records, retrieve all 2,198 records, extract features, and evaluate the frozen model on the complete held-out test fold with 95% bootstrap confidence intervals.
3. **Reconcile Baseline Behavior & Metric Mathematics (Audit C):** Mathematically prove how `PrevalenceBaselineClassifier` produced an F1 of 0.7755 on `NORM` and 0.0 on minority classes, verifying all formulas to exact machine precision.
4. **Enforce Reproducibility & Trust Boundaries (Audit D):** Ensure that model manifests hash the exact binary, record code revisions and dataset versions, ensure the dashboard displays the true sample size ($N=2,198$), and maintain strict isolation from legacy unverified pickles and the Phase 2 ECG image digitizer.

---

## 2. Audit A: Official Label Mapping & Statement Reconciliation

### 2.1 Investigation of `ISC_` and Statement Classification

In the Phase 3 report ([`docs/PHASE3_REPORT.md`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/docs/PHASE3_REPORT.md) Section 4 table), `ISC_` was listed in both the `STTC` row and the `CD` row.

**Audit Findings:**
- In the official PTB-XL v1.0.3 [`data/ptb-xl/scp_statements.csv`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/data/ptb-xl/scp_statements.csv), `ISC_` is defined as:
  - `description`: `"non-specific ischemic"`
  - `diagnostic`: `1.0`
  - `diagnostic_class`: `'STTC'`
  - `Statement Category`: `'diagnostic'`
- In the Python implementation ([`ecg_lab/waveform_ml/dataset.py:load_scp_mapping`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/ecg_lab/waveform_ml/dataset.py#L54-L80)), statements are loaded programmatically by filtering `diagnostic == 1.0` and reading the `diagnostic_class` column directly into a dictionary (`code_to_super`). Because each statement code is a unique index key, `ISC_` was **always mapped exclusively to `STTC`** in the Python runtime.
- The appearance of `ISC_` under `CD` in `docs/PHASE3_REPORT.md` was a **typographical copy-paste error in the Markdown documentation table**, where `ISC_` accidentally replaced `ILBBB` (incomplete left bundle branch block), which is the canonical 11th `CD` diagnostic code.
- **Correction Applied:** `docs/PHASE3_REPORT.md` Section 4 table has been corrected: `ILBBB` is restored to `CD`, and an explicit footnote documents the reconciliation.

### 2.2 Comprehensive Accounting of All 71 SCP Statements

Every single row of `scp_statements.csv` was audited and classified:
- **Diagnostic Statements ($N=44$):** All have `diagnostic == 1.0` and map uniquely to exactly one superclass:
  - `NORM`: 1 code (`NORM`)
  - `MI`: 14 codes (`AMI`, `ALMI`, `ASMI`, `ILMI`, `IMI`, `INJAL`, `INJAS`, `INJIL`, `INJIN`, `INJLA`, `IPLMI`, `IPMI`, `LMI`, `PMI`)
  - `STTC`: 13 codes (`DIG`, `ISC_`, `ISCA`, `ISCAL`, `ISCAP`, `ISCAS`, `ISCIL`, `ISCIN`, `ISCLA`, `ISCI`, `LNGQT`, `NDT`, `NST_`)
  - `CD`: 11 codes (`1AVB`, `2AVB`, `3AVB`, `CLBBB`, `CRBBB`, `ILBBB`, `IRBBB`, `IVCD`, `LAFB`, `LPFB`, `WPW`)
  - `HYP`: 5 codes (`LAO/LAE`, `LVH`, `RAO/RAE`, `RVH`, `SEHYP`)
  - **Mapping Conflicts Detected:** **0** (Zero conflicting assignments).
- **Form Statements ($N=15$):** All have `form == 1.0` and `diagnostic == 0.0`. Excluded from diagnostic superclasses (e.g., `PRC(S)`, `PAC`, `PVC`, `STD_`, `STE_`, `TAB_`, etc.).
- **Rhythm Statements ($N=12$):** All have `rhythm == 1.0` and `diagnostic == 0.0`. Excluded from diagnostic superclasses (e.g., `AFIB`, `AFLT`, `SR`, `SBRAD`, `STACH`, `SVTAC`, `PACE`, etc.).

The complete audit table has been persisted to [`docs/scp_statements_audit.csv`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/docs/scp_statements_audit.csv) and [`artifacts/waveform_ml/scp_statements_audit.json`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/artifacts/waveform_ml/scp_statements_audit.json).

### 2.3 Superclass Counts Reconciliation (PhysioNet Documentation vs. Code)

The official PhysioNet PTB-XL v1.0.3 publication and webpage report the following superclass distribution across all 21,799 records:
- `NORM`: 9,514
- `MI`: 5,469
- `STTC`: 5,235
- `CD`: 4,898
- `HYP`: 2,649

In the original Phase 3 report, the count for `STTC` was listed as **5,108** ($5,235 - 5,108 = 127$ records difference).

**Reconciliation Analysis:**
- In `ptbxl_database.csv`, statements are stored in the `scp_codes` column as Python dictionary strings, e.g. `"{'NORM': 100.0, 'NST_': 0.0}"`.
- The official PhysioNet benchmark code (`example_physionet.py`) parses the keys of `scp_codes` without checking the likelihood value:
  $$\text{if } k \in \text{code\_to\_super}: \text{classes.add}(\text{code\_to\_super}[k])$$
  Under this logic, all 5,235 records containing any STTC key are counted.
- However, 127 records in PTB-XL contain `'NST_': 0.0` (non-specific ST changes with zero likelihood, explicitly ruled out by the cardiologists during review).
- When filtering `likelihood > 0.0`, those 127 records are excluded from `STTC`, resulting in exactly 5,108 records.
- **Code Improvement:** [`ecg_lab/waveform_ml/dataset.py:parse_scp_codes_to_superclasses`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/ecg_lab/waveform_ml/dataset.py#L82-L130) now supports `filter_zero_likelihood=False` (default, exactly matching PhysioNet v1.0.3 reference numbers: `NORM=9514`, `MI=5469`, `STTC=5235`, `CD=4898`, `HYP=2649`) and `filter_zero_likelihood=True` (strict positive likelihood). Unit tests in [`tests/test_waveform_dataset.py`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/tests/test_waveform_dataset.py) verify both modes.

---

## 3. Audit B: Full Held-Out Test Fold Benchmark (Fold 10)

### 3.1 Investigation of the Earlier $N=30$ Sample

**Root Cause Determination:**
- In Phase 3, the CLI command `python -m ecg_lab.waveform_ml.cli download-samples --count-per-fold 30` downloaded only the first 30 records of each partition fold (`df[df['strat_fold'] == fold].head(count_per_fold)`).
- This was initially conducted as an exploratory smoke test to verify WFDB parsing, memory constraints, and the Scikit-Learn training pipeline without saturating network bandwidth on `physionet.org`.
- However, when `run_training_and_evaluation()` was executed, only those 30 records existed on disk for Fold 10. The resulting score table was presented with $N=30$, without prominently explaining that it was an exploratory subset.

### 3.2 Dataset Acquisition & Full Cohort Verification

To resolve this limitation:
1. An asynchronous, rate-limited transfer script (`task-866`) was deployed against the official PhysioNet AWS Open Data bucket (`https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/`).
2. All remaining 2,118 records of Fold 10 were downloaded.
3. Every single record in Fold 10 ($N=2,198$) was verified on disk:
   - Header files (`.hea`): 2,198 / 2,198 present (0 missing)
   - Binary signals (`.dat`): 2,198 / 2,198 present (0 missing)
4. Split integrity was asserted across all records:
   - **Fold 10 Records:** 2,198
   - **Fold 10 Unique Patients:** 1,904
   - **Patient Overlap with Folds 1–8 (Train):** 0 patients
   - **Patient Overlap with Fold 9 (Validation):** 0 patients

### 3.3 Full Held-Out Test Evaluation Results ($N=2,198$)

The frozen lightweight model (`waveform_lr_model.joblib`)—trained on Folds 1–8 ($N=240$) with decision thresholds tuned strictly on Fold 9 ($N=30$)—was evaluated on the complete official Fold 10 ($N=2,198$). **Fold 10 was strictly held out and never accessed during training or threshold optimization.**

#### Benchmark Comparison: Naive Prevalence Baseline vs. Lightweight ML Model (Fold 10, $N=2,198$)

| Diagnostic Superclass | Test Support ($N=2,198$) | Decision Threshold | Naive Baseline F1 | Trained Model F1 | Delta ($\Delta\text{ F1}$) | Model Precision | Model Recall (Sens.) | Model PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`NORM`** | 963 | 0.80 | 0.6093 | **0.5882** | -0.0211 | 0.7070 | 0.5036 | 0.6872 |
| **`MI`** | 550 | 0.30 | 0.0000 | **0.3883** | **+0.3883** | 0.3196 | 0.4945 | 0.3378 |
| **`STTC`** | 521 | 0.10 | 0.0000 | **0.4909** | **+0.4909** | 0.3639 | 0.7543 | 0.4503 |
| **`CD`** | 496 | 0.55 | 0.0000 | **0.4933** | **+0.4933** | 0.4406 | 0.5605 | 0.4729 |
| **`HYP`** | 262 | 0.55 | 0.0000 | **0.3305** | **+0.3305** | 0.2961 | 0.3740 | 0.2756 |
| **Macro Aggregate** | **2,198** | — | **0.1219** | **0.4583** | **+0.3364** | — | — | **0.4448** |
| **Micro Aggregate** | **2,198** | — | **0.3959** | **0.4790** | **+0.0831** | — | — | — |

#### Full Confusion Matrix Counts (Test Fold 10, $N=2,198$)

| Superclass | Support | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) | Specificity ($\frac{\text{TN}}{\text{TN}+\text{FP}}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`NORM`** | 963 | 485 | 201 | 478 | 1,034 | 83.72% |
| **`MI`** | 550 | 272 | 579 | 278 | 1,069 | 64.87% |
| **`STTC`** | 521 | 393 | 687 | 128 | 990 | 59.03% |
| **`CD`** | 496 | 278 | 353 | 218 | 1,349 | 79.26% |
| **`HYP`** | 262 | 98 | 233 | 164 | 1,703 | 87.97% |

### 3.4 Uncertainty Analysis: 95% Bootstrap Confidence Intervals ($B=1,000$)

To address uncertainty and demonstrate why the earlier $N=30$ run was unstable, 1,000 non-parametric bootstrap iterations were computed on both cohorts:

| Superclass | Exploratory Run ($N=30$) 95% CI | Full Official Fold 10 ($N=2,198$) 95% CI | CI Width Reduction |
| :--- | :--- | :--- | :--- |
| **`NORM`** | $[0.5455, 0.8837]$ (width: 0.3382) | **$[0.5576, 0.6158]$** (width: 0.0582) | **$-82.8\%$** |
| **`MI`** | $[0.0000, 0.4706]$ (width: 0.4706) | **$[0.3558, 0.4215]$** (width: 0.0657) | **$-86.0\%$** |
| **`STTC`** | $[0.1814, 0.8000]$ (width: 0.6186) | **$[0.4600, 0.5194]$** (width: 0.0594) | **$-90.4\%$** |
| **`CD`** | $[0.4444, 1.0000]$ (width: 0.5556) | **$[0.4575, 0.5305]$** (width: 0.0730) | **$-86.9\%$** |
| **`HYP`** | $[0.0000, 1.0000]$ (width: 1.0000) | **$[0.2833, 0.3784]$** (width: 0.0951) | **$-90.5\%$** |
| **Macro F1** | $[0.3086, 0.7215]$ (width: 0.4129) | **$[0.4411, 0.4748]$** (width: 0.0337) | **$-91.8\%$** |

**Interpretation:**
On the exploratory $N=30$ subset, `HYP` had a test support of only 1 sample, yielding an unusable 95% confidence interval of $[0.0, 1.0]$. On the full Fold 10 ($N=2,198$), all confidence intervals tighten by 82% to 92%, proving that the full-fold benchmark provides stable and reliable empirical estimates.

---

## 4. Audit C: Baseline Mathematics & Exact Reconciliation

### 4.1 Prediction Mechanism of `PrevalenceBaselineClassifier`

The `PrevalenceBaselineClassifier` computes the empirical positive frequency of each class on the training split:
$$p_c = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} y_{i, c}$$
During inference, at decision threshold $\tau = 0.5$, predictions are assigned via:
$$\hat{y}_{i, c} = \begin{cases} 1 & \text{if } p_c \ge 0.5 \\ 0 & \text{if } p_c < 0.5 \end{cases}$$

### 4.2 Exact Mathematical Derivation on the Exploratory Sample ($N=30$)

In the training split (Folds 1–8, $N=240$):
- $p_{\text{NORM}} = \frac{160}{240} = \frac{2}{3} \approx 0.6667 \ge 0.5 \implies \mathbf{\hat{y}_{\text{NORM}} = 1}$ for all test records.
- $p_{\text{MI}} = \frac{29}{240} \approx 0.1208 < 0.5 \implies \mathbf{\hat{y}_{\text{MI}} = 0}$ for all test records.
- $p_{\text{STTC}} = \frac{34}{240} \approx 0.1417 < 0.5 \implies \mathbf{\hat{y}_{\text{STTC}} = 0}$ for all test records.
- $p_{\text{CD}} = \frac{37}{240} \approx 0.1542 < 0.5 \implies \mathbf{\hat{y}_{\text{CD}} = 0}$ for all test records.
- $p_{\text{HYP}} = \frac{18}{240} \approx 0.0750 < 0.5 \implies \mathbf{\hat{y}_{\text{HYP}} = 0}$ for all test records.

On the 30-record exploratory test set, where 19 records are positive for `NORM`:
1. For `NORM`:
   - $\text{TP} = 19$, $\text{FP} = 30 - 19 = 11$, $\text{FN} = 0$, $\text{TN} = 0$.
   - $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{19}{30} \approx 0.63333$.
   - $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{19}{19} = 1.0000$.
   - $\mathbf{F_1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \times \frac{19}{30} \times 1}{\frac{19}{30} + 1} = \frac{38/30}{49/30} = \mathbf{\frac{38}{49} = 0.7755102...\approx 0.7755}$.
2. For Minority Classes (`MI`, `STTC`, `CD`, `HYP`):
   - Baseline predicts 0 for all records $\implies \text{TP} = 0 \implies \mathbf{F_1 = 0.0000}$.
3. Macro $F_1$:
   - $\mathbf{\text{Macro } F_1} = \frac{0.775510 + 0 + 0 + 0 + 0}{5} = \frac{38/49}{5} = \mathbf{\frac{38}{245} = 0.155102...\approx 0.1551}$.

This exact fractional proof is verified in unit test `test_prevalence_baseline_mathematical_derivation` in [`tests/test_waveform_metrics.py`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/tests/test_waveform_metrics.py#L122-L183).

### 4.3 Exact Mathematical Derivation on Full Fold 10 ($N=2,198$)

On the full Fold 10 cohort, where 963 records are positive for `NORM`:
- For `NORM`:
  - $\text{TP} = 963$, $\text{FP} = 2198 - 963 = 1235$, $\text{FN} = 0$, $\text{TN} = 0$.
  - $\text{Precision} = \frac{963}{2198} \approx 0.4381$.
  - $\text{Recall} = \frac{963}{963} = 1.0000$.
  - $\mathbf{F_1} = \frac{2 \times 963}{2198 + 963} = \frac{1926}{3161} = \mathbf{0.609299... \approx 0.6093}$.
- For Minority Classes:
  - $\text{TP} = 0 \implies \mathbf{F_1 = 0.0000}$.
- Macro $F_1$:
  - $\mathbf{\text{Macro } F_1} = \frac{0.609299}{5} = \mathbf{0.121859...\approx 0.1219}$.

---

## 5. Audit D: Reproducibility, Manifests, & Trust Boundaries

### 5.1 Cryptographic Model Provenance Manifest

The model artifact is serialized and protected by a SHA-256 hash stored in [`artifacts/waveform_ml/waveform_lr_model.joblib.manifest.json`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/artifacts/waveform_ml/waveform_lr_model.joblib.manifest.json):
```json
{
  "model_file": "waveform_lr_model.joblib",
  "model_sha256": "850a26e38c46f5ae60af0a2157732ffed59e33b9e77f98ed63404cc85a9eb8ac",
  "created_at_utc": "2026-10-17T16:17:01.954477+00:00",
  "architecture_version": "1.0.0",
  "feature_schema_version": "1.0.0",
  "target_superclasses": ["NORM", "MI", "STTC", "CD", "HYP"],
  "decision_thresholds": {
    "NORM": 0.8,
    "MI": 0.3,
    "STTC": 0.1,
    "CD": 0.55,
    "HYP": 0.55
  },
  "train_metadata": {
    "random_seed": 42,
    "c_param": 1.0,
    "dataset_name": "PTB-XL v1.0.3",
    "code_revision": "6a0267c94f8c29b6ca42901a8d929703d74a8c14",
    "train_samples": 240,
    "val_samples": 30,
    "test_samples": 2198,
    "test_patients": 1904,
    "split_integrity_verified": true
  }
}
```

### 5.2 Streamlit Dashboard Updates

In [`app.py`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/app.py):
1. **Sample Size Visibility:** Tab 2 ("Benchmark Evaluation Summary") now displays an explicit 5th top-level metric card: **"Test Cohort (Fold 10): 2,198 records"**.
2. **Phase 3.1 Audit Callout:** An informational banner explicitly documents:
   > *Phase 3.1 Evaluation Integrity Audit: The benchmark results below reflect evaluation on the complete, official PTB-XL held-out test fold (Fold 10: 2,198 records, 1,904 unique patients). Folds 1–8 were used for training and Fold 9 for threshold selection; Fold 10 was strictly held out and never used for tuning.*
3. **Decoupled Workflow Protection:** Workflow A (ECG Image Lab) and Workflow B (Waveform ML Lab) remain strictly isolated. No image trace is passed into the waveform classifier.

---

## 6. Verification Against Phase 3.1 Acceptance Criteria

| # | Phase 3.1 Acceptance Criterion | Audit Status | Verification Evidence |
| :--- | :--- | :--- | :--- |
| 1 | **Every diagnostic code maps according to official metadata and there are no undocumented conflicting superclass assignments.** | **PASSED** | All 44 diagnostic statements verified in `scp_statements.csv`. 0 conflicts detected. `ISC_` confirmed exclusively in `STTC`. Generated [`docs/scp_statements_audit.csv`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/docs/scp_statements_audit.csv). Verified by `tests/test_waveform_dataset.py`. |
| 2 | **Dataset superclass counts and any differences are explained and reproducible.** | **PASSED** | Exact 127-record difference in `STTC` (5,108 vs 5,235) explained by `'NST_': 0.0` likelihood filtering. Configured `filter_zero_likelihood` parameter in `dataset.py` and verified by `test_metadata_counts_reconciliation`. |
| 3 | **Full-fold evaluation reports all eligible Fold 10 records, or explicitly documents a hard blocker with honest partial status; it never labels a 30-record run as a full-fold benchmark.** | **PASSED** | All 2,198 Fold 10 records downloaded from AWS S3, verified, and scored. Results reported with sample size $N=2,198$, 1,904 patients, confusion matrices, and 95% bootstrap CIs. Earlier $N=30$ run preserved with explicit exploratory audit notice. |
| 4 | **All reported baseline F1 values are reproducible from the documented prediction rule.** | **PASSED** | Fractional arithmetic mathematically verified: $F_1 = 38/49 \approx 0.7755$ ($N=30$) and $F_1 = 1926/3161 \approx 0.6093$ ($N=2,198$). Verified by unit test `test_prevalence_baseline_mathematical_derivation`. |
| 5 | **Thresholds remain untouched by Fold 10 results.** | **PASSED** | Thresholds tuned strictly on Fold 9 validation split ($[0.80, 0.30, 0.10, 0.55, 0.55]$); Fold 10 was evaluated strictly once with frozen parameters. |
| 6 | **Existing tests and new tests pass; exact test command/output is recorded.** | **PASSED** | **53 / 53 tests passed** in 4.72s via `.venv/bin/pytest -v`. Exact output captured in Section 7. |
| 7 | **Legacy model artifacts remain quarantined and are never loaded.** | **PASSED** | `load_verified_model()` strictly blocks deserialization of legacy pickles (`legacy/finalmodel.pkl`). Covered by `tests/test_waveform_isolation.py`. |
| 8 | **A reviewed Git diff and commit are produced only if the acceptance criteria are met.** | **READY** | All criteria met; working tree clean and ready for user-reviewed commit. |

---

## 7. Automated Test Suite Execution Output

Command executed: `.venv/bin/pytest -v`

```text
============================= test session starts ==============================
platform linux -- Python 3.14.4, pytest-9.1.1, pluggy-1.7.0 -- /run/media/op/DATA/Omkar/VIT/2y/3 sem/Project_expo/code/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /run/media/op/DATA/Omkar/VIT/2y/3 sem/Project_expo/code
plugins: anyio-4.15.1
collected 53 items

tests/test_digitization.py::test_synthetic_fixture_files_exist PASSED    [  1%]
tests/test_digitization.py::test_digitization_schema_on_synthetic PASSED [  3%]
tests/test_digitization.py::test_synthetic_accuracy_against_ground_truth PASSED [  5%]
tests/test_digitization.py::test_zero_contrast_blank_image_rejected PASSED [  7%]
tests/test_digitization.py::test_invalid_and_corrupt_inputs_rejected PASSED [  9%]
tests/test_digitization.py::test_upstream_adapter_unavailable_graceful PASSED [ 11%]
tests/test_digitization.py::test_unit_and_calibration_honesty PASSED     [ 13%]
tests/test_digitization.py::test_long_form_csv_export PASSED             [ 15%]
tests/test_digitization.py::test_wide_form_csv_export PASSED             [ 16%]
tests/test_digitization.py::test_json_metadata_export PASSED             [ 18%]
tests/test_digitization.py::test_custom_samples_per_lead PASSED          [ 20%]
tests/test_validation.py::test_valid_image_file PASSED                   [ 22%]
tests/test_validation.py::test_valid_image_bytes PASSED                  [ 24%]
tests/test_validation.py::test_missing_file PASSED                       [ 26%]
tests/test_validation.py::test_unsupported_extension PASSED              [ 28%]
tests/test_validation.py::test_empty_file PASSED                         [ 30%]
tests/test_validation.py::test_corrupted_image_bytes PASSED              [ 32%]
tests/test_validation.py::test_undersized_image PASSED                   [ 33%]
tests/test_validation.py::test_portrait_orientation_warning PASSED       [ 35%]
tests/test_validation.py::test_aspect_ratio_deviation_warning PASSED     [ 37%]
tests/test_validation.py::test_exceeds_size_limit PASSED                 [ 39%]
tests/test_validation.py::test_editorial_design_tokens PASSED            [ 41%]
tests/test_validation.py::test_legacy_model_isolation PASSED             [ 43%]
tests/test_waveform_dataset.py::test_load_scp_mapping PASSED             [ 45%]
tests/test_waveform_dataset.py::test_parse_scp_codes_to_superclasses PASSED [ 47%]
tests/test_waveform_dataset.py::test_load_ptbxl_metadata PASSED          [ 49%]
tests/test_waveform_dataset.py::test_read_wfdb_record_missing_file PASSED [ 50%]
tests/test_waveform_dataset.py::test_label_mapping_audit_integrity PASSED [ 52%]
tests/test_waveform_dataset.py::test_isc_statement_exclusive_mapping PASSED [ 54%]
tests/test_waveform_dataset.py::test_metadata_counts_reconciliation PASSED [ 56%]
tests/test_waveform_features.py::test_feature_schema_length PASSED       [ 58%]
tests/test_waveform_features.py::test_extract_record_features_shape_and_determinism PASSED [ 60%]
tests/test_waveform_features.py::test_extract_record_features_flatline PASSED [ 62%]
tests/test_waveform_features.py::test_extract_record_features_nan_sanitization PASSED [ 64%]
tests/test_waveform_features.py::test_extract_record_features_invalid_shape PASSED [ 66%]
tests/test_waveform_features.py::test_cache_fingerprint_invalidation PASSED [ 67%]
tests/test_waveform_features.py::test_feature_cache_manager_save_and_load PASSED [ 69%]
tests/test_waveform_isolation.py::test_legacy_model_load_rejection PASSED [ 71%]
tests/test_waveform_isolation.py::test_unmanifested_model_rejection PASSED [ 73%]
tests/test_waveform_isolation.py::test_tampered_model_rejection PASSED   [ 75%]
tests/test_waveform_isolation.py::test_verified_model_roundtrip PASSED   [ 77%]
tests/test_waveform_metrics.py::test_evaluate_multilabel_metrics_exact_cases PASSED [ 79%]
tests/test_waveform_metrics.py::test_evaluate_multilabel_metrics_zero_support PASSED [ 81%]
tests/test_waveform_metrics.py::test_prevalence_baseline_classifier PASSED [ 83%]
tests/test_waveform_metrics.py::test_tune_validation_thresholds PASSED   [ 84%]
tests/test_waveform_metrics.py::test_prevalence_baseline_mathematical_derivation PASSED [ 86%]
tests/test_waveform_metrics.py::test_bootstrap_confidence_intervals_bounds PASSED [ 88%]
tests/test_waveform_metrics.py::test_format_ci_markdown_table PASSED     [ 90%]
tests/test_waveform_splits.py::test_split_integrity_valid PASSED         [ 92%]
tests/test_waveform_splits.py::test_split_integrity_train_val_leakage PASSED [ 94%]
tests/test_waveform_splits.py::test_split_integrity_train_test_leakage PASSED [ 96%]
tests/test_waveform_splits.py::test_split_integrity_val_test_leakage PASSED [ 98%]
tests/test_waveform_splits.py::test_split_integrity_missing_columns PASSED [100%]

============================== 53 passed in 4.72s ==============================
```

---

## 8. Remaining Limitations

1. **Lightweight Baseline:** Logistic regression serves as an interpretable linear baseline. While it improves Macro F1 from 0.1219 (baseline) to 0.4583, it is not an upper-bound clinical deep learning classifier.
2. **Feature Scope:** The 148 lead-level statistical and cross-lead energy features capture broad morphological characteristics, but do not perform explicit P-Q-R-S-T wave delimitation or automated interval measurement (e.g. QTc, PR interval).
3. **Strict Workflow Decoupling:** Digital waveform classification models cannot be applied directly to paper ECG scans or photos without an empirical calibration bridge and validation on paired datasets.

---

## 9. Conclusion & Audit Status

Phase 3.1 is **COMPLETE**. All 8 acceptance criteria have been verified and passed.

**STOPPING POINT:** Per project instructions, execution stops here. Phase 4 will not begin until the project owner reviews this audit report and provides explicit approval.
