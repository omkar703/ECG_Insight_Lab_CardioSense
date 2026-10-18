# Data Sources, Licensing, and Provenance

**ECG Insight Lab / CardioSense — Phase 4 Documentation**  
*Comprehensive inventory of physiological datasets, standard ontologies, synthetic assets, third-party libraries, and license obligations.*

---

## 1. Primary Dataset: PTB-XL v1.0.3

The digital waveform machine learning laboratory (Workflow B) utilizes the **PTB-XL** dataset, a large, publicly accessible clinical 12-lead electrocardiography dataset released via PhysioNet.

### Citation & Academic Provenance
- **Dataset Title:** PTB-XL, a large publicly available electrocardiography dataset
- **Version:** 1.0.3 (Published 2022; initial release 2020)
- **Authors:** Patrick Wagner, Nils Strodthoff, Ralf-Dieter Bousseljot, Wojciech Samek, Tobias Schaeffter
- **Persistent DOI:** [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45)
- **PhysioNet URL:** `https://physionet.org/content/ptb-xl/1.0.3/`
- **Primary Literature Reference:**  
  Wagner, P., Strodthoff, N., Bousseljot, R. D., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T. (2020). *PTB-XL, a large publicly available electrocardiography dataset*. Scientific Data, 7(1), 154. [https://doi.org/10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6)

### Licensing & Use Terms
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Permitted Uses:** Academic research, commercial evaluation, modification, and redistribution with appropriate attribution.
- **Attribution Requirement:** Explicit credit given to the authors and PhysioNet repository, including citation, DOI, and license link.
- **Patient Privacy & De-identification:** Data is fully de-identified in compliance with HIPAA Safe Harbor and GDPR standards. Record timestamps and patient identifiers are anonymized.

### Dataset Composition in ECG Insight Lab
- **Total Records:** 21,799 12-lead records (each 10 seconds in duration at 100 Hz / 500 Hz).
- **Format:** PhysioNet / WFDB format (`.dat` binary waveforms, `.hea` headers) paired with `ptbxl_database.csv` metadata.
- **Partitioning:** Official patient-separated recommended stratifications (Folds 1–8: Training [17,418 records]; Fold 9: Validation [2,183 records]; Fold 10: Held-Out Test [2,198 records]).
- **Zero Patient Overlap:** Evaluated and confirmed in `tests/test_waveform_splits.py` (0 overlapping patient IDs across folds).

---

## 2. Single-Lead Arrhythmia Dataset: MIT-BIH Arrhythmia Database v1.0.0

The single-lead beat classification laboratory (Workflow C) utilizes the **MIT-BIH Arrhythmia Database**, a standard reference dataset of ambulatory electrocardiograms released via PhysioNet.

### Citation & Academic Provenance
- **Dataset Title:** MIT-BIH Arrhythmia Database
- **Version:** 1.0.0 (Published 2005)
- **Authors:** George B. Moody, Roger G. Mark
- **Persistent DOI:** [10.13026/C2F305](https://doi.org/10.13026/C2F305)
- **PhysioNet URL:** `https://physionet.org/content/mitdb/1.0.0/`
- **Primary Literature Reference:**
  Moody, G. B., & Mark, R. G. (2001). *The impact of the MIT-BIH Arrhythmia Database*. IEEE Engineering in Medicine and Biology Magazine, 20(3), 45-50. [https://doi.org/10.1109/51.932724](https://doi.org/10.1109/51.932724)

### Licensing & Use Terms
- **License:** [Open Data Commons Attribution License v1.0 (ODC-By v1.0)](https://opendatacommons.org/licenses/by/1-0/)
- **Permitted Uses:** Academic research, modification, and redistribution with explicit attribution.
- **Attribution Requirement:** Explicit credit given to the authors, MIT-BIH database project, and PhysioNet repository.
- **Patient Privacy:** Ambulatory Holter recordings are anonymized; record numbers represent study IDs rather than patient health identifiers.

### Dataset Composition in ECG Insight Lab
- **Total Records:** 48 half-hour two-channel ambulatory ECG recordings from 47 unique subjects.
- **Sampling Rate & Quantization:** 360 Hz per channel; 11-bit digitization over a 10 mV dynamic range ($(ADC - 1024) / 200\,\text{mV}$).
- **Annotations:** Approximately 110,000 expert-reviewed beat and non-beat annotations.
- **Channel Policy:** Primary preference: `MLII` (modified limb lead II, available in 46/48 records); fallback: `V5` (available in records 102 and 104).
- **Patient Separation & Split Strategy:**
  - Records 201 and 202 originate from the same subject (`sub_201_202`) and are strictly co-located in the Training partition.
  - Four paced records (`102`, `104`, `107`, `217`) are quarantined from natural arrhythmia evaluation per AAMI EC57.
  - Partitions: Training (16 records / 15 subjects; 37,419 beats); Validation (7 records / 7 subjects; 15,725 beats); Held-Out Test (21 records / 21 subjects; 47,561 beats).
  - Subject Overlap: Evaluated and confirmed in `tests/test_single_lead_ml.py` (0 overlapping subjects across partitions).

---

## 3. Standard Diagnostic Taxonomy: SCP-ECG & ANSI/AAMI EC57

The diagnostic classification ontology implemented in Workflow B aligns with the international Standard Communications Protocol for Computer-Assisted Electrocardiography (**SCP-ECG**, ISO 11073-91064 / ANSI/AAMI EC71).

### Schema & Mapping Provenance
- **Annotation Source:** `scp_statements.csv` supplied alongside PTB-XL v1.0.3.
- **Statement Count:** 71 clinical diagnostic, form, and rhythm statements mapped to 5 core diagnostic superclasses:
  1. `NORM`: Normal ECG (30 statements, 9,514 occurrences across dataset)
  2. `MI`: Myocardial Infarction (19 statements, 5,469 occurrences)
  3. `STTC`: ST/T changes (11 statements, 5,235 occurrences)
  4. `CD`: Conduction Disturbance (9 statements, 4,898 occurrences)
  5. `HYP`: Hypertrophy (2 statements, 2,649 occurrences)
- **Licensing:** Part of the PTB-XL v1.0.3 dataset release under CC BY 4.0.
- **Mapping Implementation:** Validated in `ecg_lab/waveform_ml/dataset.py` with zero ambiguous assignments across all evaluated folds.

### Single-Lead Beat Grouping Taxonomy (MIT-BIH / Workflow C)
- **Taxonomy Disclosure:** Project-defined 4-class beat grouping: N, S, V, Q, with ventricular fusion beats (F) merged into Q. This is not a standard five-class AAMI EC57 implementation.
- **Classes:**
  - `N`: Normal and bundle branch block beats (`N`, `L`, `R`, `e`, `j`)
  - `S`: Supraventricular ectopic beats (`A`, `a`, `J`, `S`)
  - `V`: Ventricular ectopic beats (`V`, `E`)
  - `Q`: Fusion and other beats (`F`, `/`, `f`, `Q`). In evaluated non-paced records, 98.2% of Class Q beats are ventricular fusion beats (`F`).

---

## 3. Digitization Literature & Upstream Implementations

The image digitization laboratory (Workflow A) draws upon published methodologies for optical electrocardiogram signal recovery.

### Ahus-AIM / Open-ECG Literature
- **Project:** Electrocardiogram-Digitization (Open-ECG)
- **Authors:** Akershus University Hospital AI in Medicine (Ahus-AIM) group
- **Repository:** `https://github.com/Ahus-AIM/Electrocardiogram-Digitization`
- **License:** [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)
- **Repository Integration Status:**
  - Evaluated in Phase 2 feasibility study (`docs/PHASE2_FEASIBILITY.md`).
  - Integrated via a strict adapter interface (`OpenECGDigitizerAdapter` in `ecg_lab/digitization/adapter.py`).
  - Active execution defaults to native CPU classical morphological extraction (`ClassicalBaselineDigitizer`) to avoid unverified binary weights and excessive memory overhead.
  - No legacy or untrusted weights are loaded without explicit cryptographic verification.

---

## 4. Synthetic Demo Fixtures

To ensure safe, reproducible demonstrations and headless CI test runs without handling real patient health records or requiring large downloads, the repository provides synthetic benchmark fixtures.

### Assets Located in `sample_data/`
- **`sample_data/synthetic_12lead.png`:** Synthesized 12-lead ECG grid generated algorithmically using mathematical sinusoids and parameterized Gaussian wavelets.
- **`sample_data/synthetic_12lead_gt.csv`:** Exact ground-truth numerical values corresponding to the synthesized lead waveforms.
- **Labeling Policy:** Explicitly labeled as `SYNTHETIC DEMO — NOT PATIENT DATA` across all UI screens and code outputs.
- **Licensing:** Created specifically for this project; licensed under the project's root MIT License.
- **Medical Disclaimer:** Synthetic assets do not represent true human physiological phenomena and are not used to evaluate clinical model inference.

---

## 5. Software Stack & Open Source Licenses

The platform is constructed entirely using permissively licensed open-source libraries:

| Library | Version | License | Primary Purpose |
| :--- | :--- | :--- | :--- |
| **Python** | 3.10 – 3.14 | PSF License | Runtime language environment |
| **Streamlit** | $\ge 1.30.0$ | Apache 2.0 | Research dashboard and interactive UI |
| **NumPy** | $\ge 1.24.0$ | BSD-3-Clause | Numerical vector operations and matrix math |
| **SciPy** | $\ge 1.10.0$ | BSD-3-Clause | Signal processing (Butterworth filters, peaks) |
| **scikit-learn** | $\ge 1.3.0$ | BSD-3-Clause | One-vs-Rest Logistic Regression & evaluation metrics |
| **Pandas** | $\ge 2.0.0$ | BSD-3-Clause | Tabular metadata parsing and alignment |
| **wfdb** | $\ge 4.1.0$ | MIT License | PhysioNet WFDB waveform and header I/O |
| **Matplotlib** | $\ge 3.8.0$ | PSF / BSD Compatible | Calibrated diagnostic waveform plotting |
| **Pillow** | $\ge 10.0.0$ | HPND | Image parsing, format checking, and bounding box crops |
| **pytest** | $\ge 8.0.0$ | MIT License | Automated unit and integration test suite |
| **joblib** | $\ge 1.3.0$ | BSD-3-Clause | Cryptographically verified model serialization |

All third-party libraries comply with academic redistribution guidelines and permit offline execution without telemetry.

---

## 6. Redistribution, Privacy, and Attribution Compliance

### Attribution Notice Requirement
Any academic publication, distribution, or fork of this repository must retain:
1. The original PTB-XL PhysioNet citation and DOI ([10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45)).
2. The project's root MIT License notice.
3. The explicit non-clinical disclaimer.

### Data Privacy & In-Memory Guarantee
- **Zero-Persistence Policy:** User-uploaded ECG images or files are processed entirely in ephemeral RAM buffers and are never saved to disk, logged, or transmitted off the host.
- **De-identification:** No patient identifiable information (PII) or protected health information (PHI) is collected, stored, or processed.
