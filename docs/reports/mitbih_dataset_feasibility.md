# Phase 6 Dataset Feasibility & Provenance Report

**MIT-BIH Arrhythmia Database v1.0.0 — Single-Lead Beat Classification Study**  
*Comprehensive audit of signal channels, expert annotations, subject-level split strategy, and hardware absence disclaimers.*

---

## 1. Executive Summary & Acquisition Status

In accordance with `CardioSense_Phase_6_MITBIH_Single_Lead_ML_Prompt.md`, the primary dataset for Workflow C is the **MIT-BIH Arrhythmia Database v1.0.0**. The dataset was verified and acquired from local storage at `/run/media/op/DATA/Omkar/VIT/2y/3 sem/Project_expo/.data/MIT-BHI` and symlinked to `data/mitdb/raw` within the project root.

### Dataset Overview
- **Dataset Title:** MIT-BIH Arrhythmia Database
- **Version:** 1.0.0
- **Publishers:** Massachusetts Institute of Technology & Beth Israel Hospital (via PhysioNet)
- **Primary Authors:** George B. Moody and Roger G. Mark
- **Persistent DOI:** [10.13026/C2F305](https://doi.org/10.13026/C2F305)
- **Official Citation:**  
  Moody, G. B., & Mark, R. G. (2001). *The impact of the MIT-BIH Arrhythmia Database*. IEEE Engineering in Medicine and Biology Magazine, 20(3), 45–50. [https://doi.org/10.1109/51.932724](https://doi.org/10.1109/51.932724)
- **Licence:** Open Data Commons Attribution License v1.0 (ODC-By v1.0). Permitted for research and derivative works with appropriate attribution.

---

## 2. Dataset Composition & Physical Characteristics

The database consists of 48 two-channel ambulatory ECG recordings of approximately 30 minutes each, collected from 47 distinct subjects:
- **Total Recordings:** 48 records.
- **Total Subjects:** 47 subjects (Record 201 and Record 202 originate from the same individual).
- **Sampling Frequency:** $360\,\text{Hz}$ ($360\,\text{samples/second/channel}$).
- **Recording Duration:** $\approx 30.09\,\text{minutes}$ ($650,000$ samples per channel).
- **ADC Resolution & Range:** 11-bit digitization over a $10\,\text{mV}$ physical range ($200\,\text{adc units / mV}$, baseline at $1024$).
- **Disk Footprint:** $470.8\,\text{MB}$ across 48 CSV data files and 48 expert annotation text files.

---

## 3. Signal Channel Configurations & Lead Selection Policy

Across the 48 recordings, lead configurations vary based on electrode placement on the patient's chest during Holter monitoring:

| Lead Pair (Channel 0, Channel 1) | Record Count | Records |
| :--- | :---: | :--- |
| **`('MLII', 'V1')`** | 40 | 101, 103, 105, 106, 107, 108, 109, 111, 112, 113, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234 |
| **`('MLII', 'V5')`** | 2 | 100, 202 |
| **`('MLII', 'V2')`** | 2 | 102, 104 |
| **`('MLII', 'V4')`** | 1 | 114 |
| **`('V5', 'MLII')`** | 1 | 114 (alternate channel order) |
| **`('V5', 'V2')`** | 2 | 102, 104 (lead reassignment) |

### Formal Channel Selection Policy
To establish a consistent single-lead input pipeline:
1. **Primary Selection:** `MLII` (Modified Lead II) is selected whenever present in the recording (46 of 48 records).
2. **Fallback Selection:** If `MLII` is not available, `V5` (modified lead V5) is selected.
3. **No Cross-Lead Contamination:** The selected channel name, ADC gain, and baseline are recorded in sample metadata for full auditability.

---

## 4. Expert Beat Annotations & AAMI EC57 Mapping

Across the 48 recordings, expert cardiologists reviewed and annotated a total of **112,647 events**, of which **109,494 are cardiac beats**:

| Original MIT-BIH Symbol | Clinical Description | AAMI Superclass | Count Across Dataset |
| :---: | :--- | :---: | :---: |
| `N` | Normal beat | **N** (Normal) | 75,052 |
| `L` | Left bundle branch block beat | **N** (Normal) | 8,075 |
| `R` | Right bundle branch block beat | **N** (Normal) | 7,259 |
| `e` | Atrial escape beat | **N** (Normal) | 16 |
| `j` | Nodal (junctional) escape beat | **N** (Normal) | 229 |
| `A` | Atrial premature beat | **S** (Supraventricular Ectopic) | 2,546 |
| `a` | Aberrated atrial premature beat | **S** (Supraventricular Ectopic) | 150 |
| `J` | Nodal (junctional) premature beat | **S** (Supraventricular Ectopic) | 83 |
| `S` | Supraventricular premature or ectopic beat | **S** (Supraventricular Ectopic) | 2 |
| `V` | Premature ventricular contraction | **V** (Ventricular Ectopic) | 7,130 |
| `E` | Ventricular escape beat | **V** (Ventricular Ectopic) | 106 |
| `F` | Fusion of ventricular and normal beat | **F** (Fusion) | 803 |
| `/` | Paced beat | **Q** (Unknown / Paced) | 7,028 |
| `f` | Fusion of paced and normal beat | **Q** (Unknown / Paced) | 982 |
| `Q` | Unclassifiable beat | **Q** (Unknown / Paced) | 33 |
| `+`, `~`, `!`, `"`, `x`, `|`, `[`, `]` | Rhythm changes, signal noise, artifacts | *Non-Beat Event* | 3,153 |

### Superclass Totals
- **N (Normal / Bundle Branch Block):** $90,631$ beats ($82.8\%$)
- **S (Supraventricular Ectopic - SVEB):** $2,781$ beats ($2.5\%$)
- **V (Ventricular Ectopic - VEB):** $7,236$ beats ($6.6\%$)
- **F (Ventricular Fusion):** $803$ beats ($0.7\%$)
- **Q (Paced / Unclassifiable):** $8,043$ beats ($7.3\%$)
- **Total Valid Beats:** **$109,494$ beats**

---

## 5. Subject-Level Partitioning & Patient Leakage Prevention

### Critical Constraint: Records 201 and 202
Records 201 and 202 were obtained from the exact same patient. In older literature (e.g., De Chazal et al., 2004), placing 201 in the training set and 202 in the testing set caused patient-level data leakage. In this project, **Records 201 and 202 are forced into the same partition**.

### Partition Stratification Design
To ensure leak-free evaluation with balanced representation:
- **Paced Record Quarantine (4 records):** Records `102`, `104`, `107`, and `217` contain predominantly artificial pacemaker spikes ($>8,000$ paced beats) and are excluded from natural arrhythmia classification in accordance with AAMI standards.
- **Training Set (16 records, 15 distinct subjects):** `101`, `108`, `109`, `112`, `114`, `115`, `116`, `119`, `122`, `124`, `201`, `202`, `203`, `208`, `209`, `215`. (Contains both 201 and 202).
- **Validation Set (7 records, 7 distinct subjects):** `106`, `118`, `205`, `207`, `220`, `223`, `230`.
- **Held-Out Test Set (21 records, 21 distinct subjects):** `100`, `103`, `105`, `111`, `113`, `117`, `121`, `123`, `200`, `210`, `212`, `213`, `214`, `219`, `221`, `222`, `228`, `231`, `232`, `233`, `234`.

### Partition Beat Distributions

| Partition | Records | Subjects | Normal (N) | SVEB (S) | VEB (V) | Fusion (F) | Paced/Other (Q) | Total Beats |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training** | 16 | 15 | 33,886 | 626 | 2,516 | 391 | 8 | **37,427** |
| **Validation** | 7 | 7 | 14,041 | 373 | 1,291 | 25 | 0 | **15,730** |
| **Held-Out Test** | 21 | 21 | 42,198 | 1,782 | 3,202 | 387 | 7 | **47,576** |
| **Paced (Quarantined)** | 4 | 4 | 506 | 0 | 227 | 0 | 8,028 | **8,761** |
| **Total Cohort** | **48** | **47** | **90,631** | **2,781** | **7,236** | **803** | **8,043** | **109,494** |

**Zero Overlap Proof:** $\text{Subjects}(\text{Train}) \cap \text{Subjects}(\text{Val}) \cap \text{Subjects}(\text{Test}) = \emptyset$.

---

## 6. Hardware Absence & Non-Clinical Disclaimers

> [!IMPORTANT]
> **RECORDED DATA REPLAY — NOT LIVE HARDWARE:**  
> The MIT-BIH Arrhythmia Database was collected between 1975 and 1979 at Beth Israel Hospital using analog Del Mar Avionics reel-to-reel Holter recorders and digitized at $360\,\text{Hz}$ with custom analog front-ends.  
> - It is **not** an ECG recording from an AD8232 analog front-end or an ESP32 microcontroller.
> - Electrode configuration, analog bandpass filtering, ADC resolution, and motion artifact characteristics differ substantially from consumer single-lead sensors.
> - This study is an **algorithmic beat-classification demonstration** and does not claim physical hardware integration or clinical diagnosis.
