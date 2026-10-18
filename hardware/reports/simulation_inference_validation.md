# Phase 7.1 Validation Report: Streamlit Single-Lead Predictions & Offline AD8232 Simulation

**CardioSense Research Platform**  
**Phase:** 7.1 — Streamlit Prediction Visualization & Offline AD8232 Ingestion Pipeline  
**Execution Timestamp:** 2026-10-18T14:40:00Z  
**Primary Artifact:** `hardware/reports/PHASE7_1_STREAMLIT_PREDICTION_AND_SIMULATION_VALIDATION.md`  
**Structured Results Artifact:** `hardware/reports/phase7_1_validation_results.json`  

---

## 1. Executive Summary & Root Cause Analysis

### 1.1 Root Cause of Missing Predictions in Streamlit
The user reported that they **cannot see AD8232 / single-lead ECG beat predictions in the Streamlit application**. Investigation of the source code (`app.py`, `ecg_lab/single_lead_ml/playback.py`, `hardware/host/`) revealed three concrete causes:

1. **Omission of Predictions Results Table in Tab 2 (Single-Lead Lab):**
   In `app.py` (lines 1744–1751), Tab 2 loaded recorded MIT-BIH slices via `get_playback_slice()` and rendered 4 aggregate count metrics (`cn.metric("Class N (Normal)", ...)`), but **completely omitted a predictions results table**. Users had no way to view individual classified beats, timestamps, detector peak indices, class probability distributions, model confidence, or reference comparison rows.
2. **Absence of AD8232 Simulation UI in Streamlit:**
   The offline AD8232 simulation tools developed in Phase 7 (`simulate_adc_stream.py`, `serial_protocol.py`, `calibration.py`) were CLI/host scripts and were **never wired into the Streamlit application**. The web app offered no interface for generating an AD8232-like 10-bit ADC stream, serializing/parsing it, performing simulator inverse conversion, and running beat predictions.
3. **Decoupled Architecture Without Common Inference Interface:**
   The codebase lacked a consolidated inference service function that could be called uniformly by recorded playback, offline simulation, and future physical acquisitions.

### 1.2 Remedy & Deliverables Implemented
1. **Consolidated Inference Service (`ecg_lab/single_lead_ml/inference.py`):**
   Implemented `run_single_lead_inference(...)` providing autonomous Pan-Tompkins QRS detection, 50-dimensional feature extraction, verified frozen model classification, and strict post-hoc decoupled annotation matching.
2. **Simulation Inversion & Presets (`hardware/host/simulate_adc_stream.py`):**
   Implemented `simulated_adc_to_voltage(...)` for exact simulator round-trip inversion with clipping and MAE metrics, alongside named presets (`Near-lossless software round trip` and `Noisy sensor-like stress test`).
3. **Dedicated Streamlit UI Section (`app.py`):**
   Added **`Hardware Prototype — Offline AD8232 Simulation`** under Practical Lab with source selector (MIT-BIH vs Synthetic), configurable presets, run button, dual-layer Plotly charts (ADC counts [0–1023] vs reconstructed simulated mV), visible prediction results table, and post-hoc evaluation matrix.
4. **Visible Results Table in Tab 2:**
   Added the beat-by-beat classification results table to Tab 2 (`Single-Lead Lab (Workflow C)`), ensuring individual predictions (`N`, `S`, `V`, `Q`) are visible across all single-lead views.
5. **Comprehensive Automated Test Suite (`tests/test_phase7_1_simulation_inference.py`):**
   Added 13 automated tests covering the full pipeline, annotation decoupling, calibration guards, preset inversion, and Streamlit AppTest integration. All 172 tests in the repository pass.

---

## 2. Architecture & Data Flow

The following data-flow diagram illustrates the complete route from raw signal source through the forward simulator, transport layer, parser, calibration inversion, model inference, and UI rendering:

```
+---------------------------------------------------------------------------------------------------+
| 1. SIGNAL INGESTION LAYER                                                                         |
|    - MIT-BIH Recorded ECG (100 Hz / 360 Hz calibrated digital millivolts)                        |
|    - Deterministic Synthetic ECG Fixture (Gaussian P-Q-R-S-T wave superposition)                  |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| 2. FORWARD SENSOR SIMULATOR (simulate_adc_stream.py)                                              |
|    - Optional Perturbations: Additive Gaussian Noise, 0.25 Hz Baseline Drift, 50 Hz Mains Hum     |
|    - Transfer Function: sensor_counts = baseline_count + gain_counts_per_mv * ecg_mV              |
|    - Quantization: 10-bit integer clamping [0, 1023]; clipping detection                          |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| 3. FIRMWARE SERIAL TRANSPORT PROTOCOL (serial_protocol.py)                                       |
|    - Format: seq,timestamp_micros,raw_adc,lo_plus,lo_minus\n                                     |
|    - Line-oriented UART stream emulation (115200 baud, 360 Hz nominal clock)                     |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| 4. HOST SERIAL STREAM PARSER (SerialStreamParser)                                                 |
|    - Validates ASCII line structure, token counts, integer parsing, and bounds [0, 1023]          |
|    - Detects sequence gaps, timer rollover, and lead-off flags (LO+, LO-)                         |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| 5. SIMULATION CALIBRATION INVERSION (simulated_adc_to_voltage)                                    |
|    - Inverse: reconstructed_mV = (raw_adc - baseline_count) / gain_counts_per_mv                 |
|    - Tagged: calibration_provenance = "SIMULATION PRESET — NOT HARDWARE CALIBRATION"              |
|    - Enforces Guard: Physical ADC counts without verified calibration are REJECTED                |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| 6. CONSOLIDATED INFERENCE SERVICE (run_single_lead_inference)                                     |
|    - Input Prerequisite Check: 1D array, finite values, units=="mV", fs==360 Hz                   |
|    - Autonomous Pan-Tompkins QRS Peak Detector (NO access to ground-truth annotations)            |
|    - Feature Extraction: 50 features (5 RR dynamics, 10 window, 12 subsegment, 20 morph points)   |
|    - Model Inference: Frozen Balanced Random Forest (100 trees, 4 classes: N, S, V, Q)            |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| 7. USER INTERFACE & AUDIT DISPLAY (app.py)                                                        |
|    - Status Panel: Source, Lead, Samples, ADC Range, Clipping %, Reconstruction MAE, Provenance   |
|    - Dual Waveform Plot: Raw ADC counts [0–1023] & Reconstructed ECG with classified peaks        |
|    - Visible Predictions Table: Beat #, Time (s), Sample Index, Class, Confidence, Probabilities  |
|    - Decoupled Post-Hoc Evaluation: Matched Sensitivity/PPV and Confusion Matrix (±50 ms window)  |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Reproduction Steps & Exact Commands

### 3.1 Reproduce the Missing Predictions in Streamlit (Prior State)
1. Launch Streamlit: `.venv/bin/streamlit run app.py`
2. Navigate to `Live Experiment` -> Tab 2 (`Single-Lead Lab`).
3. Notice that only 4 metric boxes were visible; no beat prediction table was rendered on screen.
4. Notice that no AD8232 simulation tab existed in the application.

### 3.2 Run the Streamlit Application (Fixed State)
To launch the updated application with both the Single-Lead Lab prediction table and the dedicated Hardware Prototype simulation tab:
```bash
.venv/bin/streamlit run app.py --server.port 8502 --server.headless true
```
Navigate to:
- **`Practical Lab (Live Experiment)`** -> **`Hardware Prototype — Offline AD8232 Simulation`**
  - Select signal source: `MIT-BIH recorded ECG` (Record 100) or `Deterministic synthetic ECG fixture`
  - Select preset: `Near-lossless software round trip` or `Noisy sensor-like stress test`
  - Click **`Run Offline Sensor Simulation & Predict Beats`**
  - View the status panel, dual-view waveforms, summary counts, and the interactive results table.
- **`Practical Lab (Live Experiment)`** -> **`Single-Lead Lab (Workflow C)`**
  - Select record (e.g. 100), start time, duration.
  - View the newly integrated **Beat-by-Beat Classification Results** table.

### 3.3 Standalone Evaluation Command
To execute the automated validation benchmark without user interface interaction:
```bash
.venv/bin/python hardware/run_phase7_1_validation.py
```
This generates the reproducible results artifact at `hardware/reports/phase7_1_validation_results.json`.

---

## 4. Experimental Results & Numerical Verification

The validation suite was executed against MIT-BIH Record 100 and the deterministic synthetic fixture. The numerical outputs are summarized below:

### 4.1 Numerical Fidelity & Inversion Accuracy
| Configuration Preset | Source Signal | Samples | ADC Range | Clipped Samples | Max Quantization Error | Reconstruction MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Near-lossless Round Trip** | MIT-BIH Rec 100 (10 s) | 3,600 | [415, 656] | 0 (0.00%) | 0.0033 mV | 0.0017 mV |
| **Noisy Stress Test** | MIT-BIH Rec 100 (10 s) | 3,600 | [389, 695] | 0 (0.00%) | 0.2841 mV | 0.1084 mV |
| **Synthetic Fixture** | Gaussian Model (10 s) | 3,600 | [475, 692] | 0 (0.00%) | 0.0512 mV | 0.0239 mV |

> [!NOTE]
> Under the `Near-lossless` preset (150 counts/mV), the 10-bit integer round-trip error is strictly bounded by quantization: maximum absolute error is $1 / (2 \times 150) \approx 0.0033\text{ mV}$ ($3.33\ \mu\text{V}$), with zero clipping.

### 4.2 Autonomous Detector & Classifier Metrics (MIT-BIH Record 100, 10.0 s Window)
- **Autonomous QRS Detection (±50 ms matching tolerance):**
  - Reference Beats in Window: 13
  - Detected Candidate Peaks: 13
  - True Positives (TP): 13
  - False Positives (FP): 0
  - False Negatives (FN): 0
  - Detector Sensitivity: 100.00%
  - Positive Predictive Value (PPV): 100.00%
  - Detector F1-Score: 1.0000

- **Beat Classification Performance (Matched-Detection Conditional):**
  - Evaluated Beats: 11 (2 edge beats skipped due to $[-90, +110]$ sample window boundary constraints)
  - Predictions Breakdown: Class N: 11, Class S: 0, Class V: 0, Class Q: 0
  - Matched Reference Ground Truth: Class N: 10, Class S: 1, Class V: 0, Class Q: 0
  - Window Macro F1: 0.2381 (reflects $N=10$ correctly classified, $S=1$ missed in this short 10 s slice)
  - Window Weighted F1: 0.8658

### 4.3 Classification Confusion Matrix (Record 100 Window)
```
                  Predicted Class
                N     S     V     Q    Support
True Class N   10     0     0     0      10
True Class S    1     0     0     0       1
True Class V    0     0     0     0       0
True Class Q    0     0     0     0       0
```

> [!IMPORTANT]
> The historical held-out test partition DS2 benchmark ($N=47,034$ beats) achieved Macro F1 **0.5274** and Weighted F1 **0.9011**. The short-window metrics reported above reflect the specific 10-second slice evaluated and are not a substitute for the full-cohort benchmark.

---

## 5. Security, Provenance & Guardrail Verification

### 5.1 Cryptographic Model Integrity
- **Artifact:** `artifacts/single_lead_ml/single_lead_rf_model.joblib`
- **Expected SHA-256 Digest:** `3889a5f5ab97803da339710d4ae9fde8ea50f8e5a5ff76ed2015a0fb0b6ec8e9`
- **Measured Post-Change Digest:** `3889a5f5ab97803da339710d4ae9fde8ea50f8e5a5ff76ed2015a0fb0b6ec8e9`
- **Status:** **MATCH (100% Bit-for-Bit Verified)**. No model retraining, re-fitting, or weight alterations were performed.

### 5.2 Annotation Decoupling (No Data Leakage)
A dedicated test (`test_shuffling_annotations_leaves_inference_predictions_invariant`) verified that:
1. When reference annotations are permuted, scrambled, or completely removed, the Pan-Tompkins peak detector produces bit-for-bit identical peak sample indices.
2. The extracted feature vectors, predicted classes, and class probabilities remain completely invariant.
3. Ground-truth annotations are consulted exclusively in the separate post-hoc evaluation layer after inference has completed.

### 5.3 Hardware Calibration Safety Guards
1. Passing raw ADC counts into `run_single_lead_inference` raises an explicit rejection:  
   `"Classifier requires input in calibrated millivolts (mV). Received uncalibrated unit: 'raw_adc_counts'"`
2. Marking a stream with `source_kind = "physical Arduino/AD8232 stream"` without verified calibration raises:  
   `"Inference blocked: hardware calibration unknown. Physical ADC stream requires verified calibration."`
3. Passing incompatible sampling rates (e.g. 100 Hz or 250 Hz) raises:  
   `"Sampling rate 100 Hz is incompatible with model requirements (360 Hz)."`

---

## 6. Test Suite Results

The complete automated test suite was executed across the repository:
```bash
.venv/bin/pytest tests/ hardware/tests/ -v
```

### Summary:
- **Total Tests Executed:** 172
- **Passed:** 172
- **Failed:** 0
- **Execution Time:** 23.43 seconds
- **Breakdown:**
  - `tests/test_phase7_1_simulation_inference.py`: 13 passed (Model SHA-256, Simulation pipeline, Annotation decoupling, Calibration guards, Preset inversion, Streamlit AppTest integration)
  - `hardware/tests/`: 46 passed (Calibration guards, Diagram spec, ECG filters, Firmware spec, Serial protocol, Simulated stream)
  - `tests/test_single_lead_ml.py`: 32 passed (Provenance, dataset, feature extraction, detector, model persistence, playback)
  - `tests/test_waveform_*.py`: 55 passed (PTB-XL 12-lead pipeline, splits, metrics, bootstrap CI)
  - `tests/test_dashboard_helpers.py`: 14 passed
  - `tests/test_hardware_integration.py`: 12 passed

### Compiler Check:
- `arduino-cli` / `PlatformIO`: Not installed on the execution environment. Static firmware specification and memory safety were verified via `hardware/tests/test_firmware_spec.py` (all tests passing).

---

## 7. Modified Files & Changelog

| File Path | Description of Changes |
| :--- | :--- |
| `ecg_lab/single_lead_ml/inference.py` | **NEW:** Implemented consolidated single-lead inference service `run_single_lead_inference(...)` and dataclasses `BeatPrediction` and `SingleLeadInferenceResult`. |
| `hardware/host/simulate_adc_stream.py` | Added `simulated_adc_to_voltage(...)` inverse transformation, `SIMULATION_PRESETS` dictionary, and perturbation injection parameters. |
| `app.py` | Added **`Hardware Prototype — Offline AD8232 Simulation`** tab with status panel, dual Plotly charts, visible prediction table, and post-hoc evaluation; added visible prediction results table to **`Single-Lead Lab (Workflow C)`**. |
| `hardware/run_phase7_1_validation.py` | **NEW:** Standalone CLI script executing reproducible validation across presets and generating `phase7_1_validation_results.json`. |
| `hardware/reports/phase7_1_validation_results.json` | **NEW:** Machine-readable validation output containing configuration, numerical errors, and confusion matrices. |
| `tests/test_phase7_1_simulation_inference.py` | **NEW:** Automated test suite (13 tests) verifying end-to-end simulation, AppTest UI components, calibration guards, and annotation decoupling. |
| `hardware/README.md` | Updated documentation to include Streamlit simulation instructions, validation script usage, and architecture flow. |

---

## 8. Mandatory Safety Disclosures & Known Limitations

> [!CAUTION]
> **ELECTRICAL SAFETY & PROTOTYPE BOUNDARIES:**
> 1. **Offline Sensor Simulation Only:** The software streams processed in this phase are generated from recorded MIT-BIH waveforms and mathematical synthetic fixtures. No physical sensor (AD8232 / Arduino Uno), electrode, or analog front-end has been connected, acquired, or validated live.
> 2. **Simulator Assumptions:** Simulator gain ($150\text{ counts/mV}$) and baseline offset ($512\text{ counts}$) are software approximations, not measured board calibrations.
> 3. **No Clinical Claims:** Predicted beat classes (`N`, `S`, `V`, `Q`) are experimental pattern classifications for research exploration. CardioSense is not a medical device and is not calibrated for acute coronary syndrome triage or clinical diagnosis.
> 4. **Electrical Safety Warning:** USB-connected hobby microcontrollers are not galvanically isolated medical devices. Never attach electrodes connected to non-isolated USB hardware to a living person.
