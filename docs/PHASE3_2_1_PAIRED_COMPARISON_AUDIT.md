# CardioSense — Phase 3.2.1 Paired Model-Comparison Integrity Audit

**Date:** 2026-10-17  
**Status:** Completed & Statistically Verified  
**Scope:** Metrics-Only Paired Model-Comparison Audit on PTB-XL Fold 10  
**Authors:** Senior Software Engineer & Technical Auditor (Antigravity)  
**Evaluated Artifacts:**
- Phase 3.1 Pilot Model: `artifacts/waveform_ml/pilot_evaluation/waveform_lr_pilot_model.joblib` (SHA-256: `850a26e38c46f5ae60af0a2157732ffed59e33b9e77f98ed63404cc85a9eb8ac`)
- Phase 3.2 Full-Fold Model: `artifacts/waveform_ml/waveform_lr_model.joblib` (SHA-256: `55d364ed57cc971167f7ad813443c249f792567e406cce4c3dd84188829328d7`)

---

## Executive Summary

In Phase 3.2, the full-fold training pipeline scaled up model training from a 240-record pilot subset to all 17,418 records of PTB-XL v1.0.3 Folds 1–8, with decision thresholds tuned on all 2,183 records of Fold 9. When re-evaluating on the official held-out test partition (Fold 10: 2,198 records, 1,904 patients), the full-fold model achieved a Macro F1 score of **0.6093**, with a patient-clustered 95% bootstrap confidence interval of **[0.5910, 0.6269]**. 

The Phase 3.2 report originally stated:
> *"The entire 95% confidence interval for the full-fold Macro F1 ([0.5910, 0.6269]) lies completely above the Phase 3.1 pilot point estimate (0.4583), confirming that the +0.1510 gain is statistically significant and not an artifact of random sampling."*

**Audit Purpose:** As identified in the project audit brief, this statement was too strong by itself. A confidence interval for one model's score compared with another model's point estimate does **not** constitute a confidence interval for the paired difference. Because both models were evaluated on the exact same Fold 10 patient cohort, their performance must be compared using a **paired patient-clustered bootstrap** that samples matched patient clusters simultaneously.

**Key Audit Findings:**
1. **Verified Comparison Inputs:** Both the Phase 3.1 pilot model and Phase 3.2 full-fold model were loaded strictly through the verified model loader with cryptographic SHA-256 validation. Their predictions on the shared Fold 10 cohort (2,198 records across 1,904 unique patients, joined explicitly by `ecg_id`) reproduced all previously reported headline point estimates bit-for-bit.
2. **Paired Patient-Cluster Bootstrap ($B=2,000$, Seed: 42):**
   - Point Estimate of Paired Difference: **$\Delta = +0.1510$** ($\text{Macro F1}_{\text{full}} = 0.6093$ vs. $\text{Macro F1}_{\text{pilot}} = 0.4583$)
   - Bootstrap Standard Error ($SE_{\Delta}$): **0.0087**
   - **95% Paired Confidence Interval:** **[0.1343, 0.1681]**
   - Empirical Non-Positive Replicate Rate: $P(\Delta \le 0) = 0.000$ (0 / 2,000 replicates $\le 0$)
   - Per-Class Paired CIs: All five superclasses demonstrated paired 95% CIs strictly above zero (NORM: $[0.1503, 0.2071]$, MI: $[0.1356, 0.2070]$, STTC: $[0.0789, 0.1398]$, CD: $[0.0729, 0.1406]$, HYP: $[0.1346, 0.2393]$).
3. **Corrected Statistical Interpretation:** The paired 95% CI for the difference lies entirely above zero, providing rigorous evidence of a positive performance difference under this specified bootstrap comparison. However, this is disclosed as a preplanned comparison on a previously examined cohort (Fold 10), and does **not** imply clinical diagnostic efficacy, medical device safety, or cross-population generalization.

---

## 1. Task A — Verification of Comparison Inputs

### 1.1 Model Artifacts and Provenance
Both models were loaded and audited using `load_verified_model()`, enforcing SHA-256 integrity and schema checks:

| Attribute | Phase 3.1 Pilot Model | Phase 3.2 Full-Fold Model | Alignment Status |
| :--- | :--- | :--- | :--- |
| **Artifact Path** | `artifacts/waveform_ml/pilot_evaluation/waveform_lr_pilot_model.joblib` | `artifacts/waveform_ml/waveform_lr_model.joblib` | Verified Present |
| **Manifest Path** | `artifacts/waveform_ml/pilot_evaluation/waveform_lr_pilot_model.joblib.manifest.json` | `artifacts/waveform_ml/waveform_lr_model.joblib.manifest.json` | Verified Present |
| **SHA-256 Digest** | `850a26e38c46f5ae60af0a2157732ffed59e33b9e77f98ed63404cc85a9eb8ac` | `55d364ed57cc971167f7ad813443c249f792567e406cce4c3dd84188829328d7` | Exact Match |
| **Training Split** | Folds 1–8 Pilot Subset ($N_{\text{tr}}=240$, 30 per fold) | Folds 1–8 Complete ($N_{\text{tr}}=17,418$, 15,023 patients) | As Documented |
| **Validation Split** | Fold 9 Pilot Subset ($N_{\text{val}}=30$) | Fold 9 Complete ($N_{\text{val}}=2,183$, 1,942 patients) | As Documented |
| **Feature Schema** | 148 deterministic lead & cross-lead features (v1.0.0) | 148 deterministic lead & cross-lead features (v1.0.0) | Identical Schema |
| **Label Order** | `['NORM', 'MI', 'STTC', 'CD', 'HYP']` | `['NORM', 'MI', 'STTC', 'CD', 'HYP']` | Identical Order |
| **Decision Thresholds** | NORM: 0.80, MI: 0.30, STTC: 0.10, CD: 0.55, HYP: 0.55 | NORM: 0.50, MI: 0.50, STTC: 0.55, CD: 0.55, HYP: 0.65 | Frozen in Manifest |

### 1.2 Evaluation Cohort Verification
- **Cohort Source:** PhysioNet PTB-XL v1.0.3, `strat_fold == 10`.
- **Sample Count ($N$):** Exactly 2,198 records.
- **Unique Patients:** Exactly 1,904 unique `patient_id` values.
- **Record Join Integrity:** Predictions from both models were joined strictly by `ecg_id`. No positional row ordering was used.
- **Truth Labels:** Ground-truth multi-label binary vectors derived from audited SCP statement mapping.

### 1.3 Exact Point Estimate Reproduction
Running inference for both models across the verified Fold 10 cohort confirmed exact reproduction of all reported metrics:

| Metric | Pilot Model Reported (Phase 3.1) | Pilot Model Reproduced | Full-Fold Model Reported (Phase 3.2) | Full-Fold Model Reproduced | Match? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Macro F1** | **0.4583** | **0.4583** | **0.6093** | **0.6093** | **Exact** |
| **Micro F1** | **0.4790** | **0.4790** | **0.6392** | **0.6392** | **Exact** |
| NORM F1 | 0.5882 | 0.5882 | 0.7668 | 0.7668 | Exact |
| MI F1 | 0.3883 | 0.3883 | 0.5594 | 0.5594 | Exact |
| STTC F1 | 0.4909 | 0.4909 | 0.6006 | 0.6006 | Exact |
| CD F1 | 0.4933 | 0.4933 | 0.6002 | 0.6002 | Exact |
| HYP F1 | 0.3305 | 0.3305 | 0.5195 | 0.5195 | Exact |

---

## 2. Task B — Paired Patient-Cluster Bootstrap Methodology

### 2.1 Resampling Design
1. **Cluster Unit:** Resampling operates on the 1,904 unique `patient_id` clusters. When a patient is drawn in a bootstrap replication, **all** ECG recordings from that patient are retained in that replicate.
2. **Paired Evaluation:** In every bootstrap replicate $b \in \{1, \dots, B\}$, the resampled record subset is passed to both the Full-Fold model and the Pilot model simultaneously.
3. **Paired Delta Calculation:**
   $$\Delta_{\text{macro}, b} = \text{Macro F1}_{\text{full}, b} - \text{Macro F1}_{\text{pilot}, b}$$
   $$\Delta_{c, b} = F1_{\text{full}, c, b} - F1_{\text{pilot}, c, b} \quad \text{for } c \in \{\text{NORM, MI, STTC, CD, HYP}\}$$
4. **Replication Parameters:** $B = 2,000$ resamples, deterministic random seed `42`, 95% percentile confidence bounds ($\alpha = 0.025$).

### 2.2 Paired Bootstrap Results ($B=2,000$)

| Superclass / Metric | Pilot Model F1 ($N_{\text{tr}}=240$) | Full-Fold Model F1 ($N_{\text{tr}}=17,418$) | Paired Delta ($\Delta$) | Bootstrap SE ($SE_{\Delta}$) | 95% Paired CI | Empirical $P(\Delta \le 0)$ | Statistically Significant Difference? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NORM** | 0.5882 | **0.7668** | **+0.1785** | 0.0142 | **[0.1503, 0.2071]** | 0.0000 | **Yes (Positive)** |
| **MI** | 0.3883 | **0.5594** | **+0.1711** | 0.0186 | **[0.1356, 0.2070]** | 0.0000 | **Yes (Positive)** |
| **STTC** | 0.4909 | **0.6006** | **+0.1097** | 0.0156 | **[0.0789, 0.1398]** | 0.0000 | **Yes (Positive)** |
| **CD** | 0.4933 | **0.6002** | **+0.1068** | 0.0176 | **[0.0729, 0.1406]** | 0.0000 | **Yes (Positive)** |
| **HYP** | 0.3305 | **0.5195** | **+0.1890** | 0.0266 | **[0.1346, 0.2393]** | 0.0000 | **Yes (Positive)** |
| **Macro Aggregate** | **0.4583** | **0.6093** | **+0.1510** | **0.0087** | **[0.1343, 0.1681]** | **0.0000** | **Yes (Positive)** |

---

## 3. Task C — Corrected Statistical Interpretation & Boundaries

### 3.1 Statistical Distinction: Score Interval vs. Difference Interval
- **Full-Fold Model Score Interval:** **[0.5910, 0.6269]**. This represents sampling uncertainty in the full-fold model's Macro F1 score on the Fold 10 population. It must **not** be used to claim statistical significance relative to another model's point estimate.
- **Paired Difference Interval:** **[0.1343, 0.1681]**. This represents the sampling uncertainty of the paired performance difference between the two models on matched patient clusters.
- **Audit Conclusion on Significance:** Because the 95% paired confidence interval for the difference (**[0.1343, 0.1681]**) is strictly positive and bounded away from zero by more than 13 percentage points (and the bootstrap SE is 0.0087), there is rigorous evidence of a positive performance difference between the full-fold model and the pilot model on this evaluation cohort.

### 3.2 Methodological Caveats
1. **Previously Examined Cohort:** Fold 10 was previously evaluated in Phase 3.1. Therefore, this evaluation is not a "first-time" holdout test; it is an offline, preplanned comparative audit on a previously examined benchmark split.
2. **Clinical Boundary:** Statistical superiority under this paired bootstrap comparison on digital voltage recordings from PTB-XL does **not** establish clinical diagnostic accuracy, safety for patient management, or generalizability to distinct patient populations, paper ECG scans, or alternate electrocardiographs.
3. **Decoupled System:** The waveform ML pipeline remains strictly decoupled from the Phase 2 ECG image digitization subsystem.

---

## 4. Updates Applied Across Project Deliverables

1. **Phase 3.2 Full-Fold Training Report (`docs/PHASE3_2_FULL_FOLD_TRAINING_REPORT.md`):**
   - Replaced Section 5 overclaim sentence with Section 5.1 detailing the paired patient-clustered bootstrap results ($B=2,000$, $\Delta = +0.1510$, 95% paired CI: $[0.1343, 0.1681]$, SE: $0.0087$).
   - Explicitly clarified the mathematical distinction between score CIs and paired difference CIs.
2. **Streamlit Application (`app.py`):**
   - Updated Tab 2 informational banner to reflect the Phase 3.2.1 paired bootstrap audit.
   - Added automated loading and rendering of `artifacts/waveform_ml/paired_model_comparison.md`.
3. **Machine-Readable Artifacts:**
   - Generated `artifacts/waveform_ml/paired_model_comparison.json`.
   - Generated `artifacts/waveform_ml/paired_model_comparison.md`.
4. **Automated Test Suite:**
   - Added unit test `test_paired_patient_cluster_bootstrap_ci()` in `tests/test_waveform_training_integrity.py`.
   - All 58 unit and integration tests passing.

---

## 5. Phase 3.2.1 Acceptance Criteria Evaluation

| Criteria | Target Requirement | Status | Verification Detail |
| :--- | :--- | :--- | :--- |
| **AC-3.2.1.1: Input Verification** | Predictions correspond to exact Fold 10 `ecg_id` records and patient IDs. | **MET** | Verified 2,198 Fold 10 records and 1,904 patient IDs aligned explicitly by `ecg_id`. |
| **AC-3.2.1.2: Pilot Prediction Audit** | Pilot predictions verified or absence documented. | **MET** | Verified pilot model loaded via `load_verified_model` (SHA-256 matched); exact metrics reproduced bit-for-bit. |
| **AC-3.2.1.3: Paired Cluster Bootstrap** | Estimate CI of the Macro F1 difference on patient clusters. | **MET** | Executed $B=2,000$ patient-cluster bootstrap iterations; point delta $+0.1510$, 95% paired CI: $[0.1343, 0.1681]$. |
| **AC-3.2.1.4: Correct Interpretation** | Significance wording matches paired interval; no overclaim from single-model CI. | **MET** | Corrected in `PHASE3_2_FULL_FOLD_TRAINING_REPORT.md`, `app.py`, and this audit report. |
| **AC-3.2.1.5: No Retraining / Tuning** | No model refitting, threshold tuning, or feature schema changes. | **MET** | Models and thresholds remained frozen throughout the entire audit. |
| **AC-3.2.1.6: Automated Tests** | All unit and integration tests pass. | **MET** | 58 / 58 pytest tests passing. |

---

## 6. Audit Trail & Execution Command

To deterministically re-run this paired model comparison audit:
```bash
.venv/bin/python3 -m ecg_lab.waveform_ml.cli audit-paired-comparison --n-bootstraps 2000 --random-seed 42
```

---
*Phase 3.2.1 audit is complete and verified. Antigravity awaits explicit user review before Phase 4.*
