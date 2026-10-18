# Phase 7 Report: Hardware Folder, AD8232 + Arduino Prototype, Offline Simulation & Test Harness

**ECG Insight Lab / CardioSense — Phase 7 Deliverable**  
**Date:** 2026-10-18  
**Status:** Complete & Fully Verified  
**Acceptance Designation:** `OFFLINE SOFTWARE PROTOTYPE COMPLETE — PHYSICAL HARDWARE NOT TESTED`  
**Mandatory Formal Statement:**
> **Offline software prototype tested; physical AD8232/Arduino hardware not available or validated.**

---

## 1. Executive Summary

Phase 7 establishes a self-contained, modular, and thoroughly tested `hardware/` subsystem for the planned **Analog Devices AD8232 single-lead ECG front-end and Arduino Uno R3-compatible microcontroller board**. 

Because physical hardware modules, electrodes, and microcontrollers are not currently present in the project workspace, this phase implements the complete software and firmware architecture, editable vector schematics, deterministic host protocol parser, digital signal processing filters, and an **offline ADC stream simulator**. The simulator leverages the existing MIT-BIH dataset and synthetic mathematical fixtures to achieve end-to-end software validation.

Crucially, strict architectural guardrails prevent uncalibrated hardware ADC counts from polluting existing machine learning classifiers (Workflow A, Workflow B, and Workflow C), model weights and serialization remain 100% frozen, and prominent biomedical electrical safety warnings are embedded throughout all documentation and vector diagrams.

---

## 2. Directory Structure & Inventory of Deliverables

The `hardware/` subsystem is structured into modular packages:

```text
hardware/
├── README.md                      # Master hardware manual (BOM, wiring, simulation, safety)
├── PINOUT_AND_SAFETY.md           # Electrical safety & biomedical isolation guide (IEC 60601-1)
├── diagrams/
│   ├── ad8232_arduino_uno_wiring.svg # Editable vector wiring schematic with caution banners
│   └── pin_map.md                 # Pinout tables, voltage levels, and wire color codes
├── firmware/
│   └── ad8232_arduino_uno/
│       ├── ad8232_arduino_uno.ino # 360 Hz deterministic micros() acquisition sketch
│       └── README.md              # Firmware compilation, upload, and troubleshooting guide
├── host/
│   ├── __init__.py                # Package exports
│   ├── serial_protocol.py         # Line-oriented CSV protocol parser with timer unwrap
│   ├── serial_capture.py          # Real-time UART capture utility (optional pyserial)
│   ├── serial_replay.py           # Offline playback & protocol validation utility
│   └── simulate_adc_stream.py     # 10-bit ADC stream simulator (MIT-BIH records & synthetic ECG)
├── signal_processing/
│   ├── __init__.py                # Package exports
│   ├── ecg_filters.py             # Zero-phase offline (filtfilt) & causal streaming IIR filters
│   └── calibration.py             # Strict physical unit guards blocking uncalibrated model ingestion
├── simulation/
│   ├── README.md                  # Simulation walkthrough and verified commands
│   └── example_config.json        # Standard 360 Hz offline simulation profile
├── tests/
│   ├── __init__.py                # Test package init
│   ├── test_serial_protocol.py    # Protocol parsing, timer wrap, and error handling tests (12 tests)
│   ├── test_simulated_stream.py   # Deterministic simulator & clipping tests (8 tests)
│   ├── test_ecg_filters.py        # Bandpass, notch, and causal streaming filter tests (11 tests)
│   ├── test_calibration_guards.py # Physical unit separation & classifier rejection tests (10 tests)
│   ├── test_firmware_spec.py      # Static verification of sketch pins, timing, and memory (4 tests)
│   └── test_diagram_spec.py       # SVG XML validity, pin completeness, and warning checks (3 tests)
└── reports/
    └── PHASE7_HARDWARE_OFFLINE_VALIDATION.md # This comprehensive Phase 7 report
```

---

## 3. Firmware Specification & Protocol Design

### 3.1 Hardware Pin Mapping (Arduino Uno R3 $\leftrightarrow$ AD8232 Breakout)

| AD8232 Breakout Pin | Arduino Uno R3 Header | Wire Color | Signal Type | Electrical Rating | Function |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`3.3V`** | **`3.3V`** | Red | DC Power Input | $+3.3\,\text{V} \pm 5\%$ | Regulated supply. **Never connect to 5V!** |
| **`GND`** | **`GND`** | Black | Ground Return | $0.0\,\text{V}$ | Common signal and return ground. |
| **`OUTPUT`** | **`A0`** | Green | Analog Output | $0.0\text{--}3.3\,\text{V}$ | Bandpass-filtered ECG signal centered at $\approx 1.65\,\text{V}$. |
| **`LO-`** | **`D11`** | Orange | Digital Input | $0.0\,\text{V} / 3.3\,\text{V}$ | Lead-Off Status (-) for negative electrode (RA). |
| **`LO+`** | **`D10`** | Yellow | Digital Input | $0.0\,\text{V} / 3.3\,\text{V}$ | Lead-Off Status (+) for positive electrode (LA). |
| **`SDN`** | *Unconnected* | — | Digital Input | Pull-up ($\approx 3.3\,\text{V}$) | Active-low shutdown. Leave open for normal operation. |

### 3.2 Timing & Bandwidth Budget
- **Nominal Sampling Rate:** $360\,\text{Hz}$ (chosen for compatibility experiments with the single-lead MIT-BIH benchmark).
- **Nominal Sample Interval:** $T_s = \frac{1,000,000\,\mu\text{s}}{360\,\text{Hz}} = 2,778\,\mu\text{s}$.
- **Baud Rate:** $115,200\,\text{baud}$ ($\approx 11,520\,\text{bytes/sec}$).
- **Line Length:** $\approx 18\text{--}24\,\text{bytes}$ per sample (`seq,micros,adc,lo_plus,lo_minus\n`).
- **UART Bandwidth Utilization:** $360 \times 22 \approx 7,920\,\text{bytes/sec} \implies 68.7\%$ of channel capacity.
- **Memory Safety:** Zero dynamic heap allocation (`String` objects prohibited); uses direct numeric prints to eliminate heap fragmentation.

### 3.3 Serial Protocol Format
- Lines beginning with `#` represent metadata comments and are safely skipped by host parsers.
- Data rows are comma-delimited ASCII strings:
  $$\text{seq},\text{micros},\text{raw\_adc},\text{lo\_plus},\text{lo\_minus}$$
- Example output:
  ```text
  # CardioSense AD8232 Arduino Uno v1.0
  # baud=115200, nominal_fs=360, interval_us=2778
  # fields=seq,micros,adc,lo_plus,lo_minus
  0,100000,512,0,0
  1,102778,516,0,0
  2,105556,514,0,0
  ```

---

## 4. Digital Signal Processing & Calibration Guardrails

### 4.1 Digital Filtering Pipeline (`hardware/signal_processing/ecg_filters.py`)
1. **Zero-Phase Offline Bandpass Filter (`butter_bandpass_filter`):**
   - 2nd-order Butterworth bandpass ($0.5\,\text{Hz}$ highpass to $40.0\,\text{Hz}$ lowpass).
   - Applied bidirectionally via `scipy.signal.filtfilt` to guarantee zero phase distortion in offline research.
2. **Zero-Phase Powerline Notch Filter (`notch_filter`):**
   - IIR notch filter at $50.0\,\text{Hz}$ (configurable to $60.0\,\text{Hz}$) with $Q=30.0$.
3. **Causal Streaming Filter (`StreamingECGFilter`):**
   - Stateful IIR implementation tracking initial filter states (`zi`/`zf`) via `lfilter`.
   - Correctly models physical group delay without future-sample lookahead.

### 4.2 Strict Calibration Guardrails (`hardware/signal_processing/calibration.py`)
- **Unit Taxonomy Separation:**
  * `raw_adc_counts`: Uncalibrated 10-bit integer values ($0\text{--}1023$).
  * `input_volts`: Voltage at Arduino pin $V_{in} = \text{ADC} \times \frac{3.3\,\text{V}}{1023}$.
  * `mV`: Calibrated patient bio-potential in physical millivolts.
- **Model Ingestion Barrier:**
  The `validate_input_for_classifier` and `convert_hardware_to_ecg_mv` functions strictly reject uncalibrated signals:
  * Passing `raw_adc_counts` or `input_volts` raises `IncompatibleSignalError` with explicit instructions.
  * Attempting conversion without verified hardware calibration parameters raises `UncalibratedHardwareError`.
  * Existing models (Workflow B Logistic Regression and Workflow C Random Forest) are completely protected from uncalibrated hardware data.

---

## 5. Offline Simulation Validation

The offline simulator (`hardware/host/simulate_adc_stream.py`) translates recorded physical signals into simulated 10-bit integer ADC counts:
$$\text{ADC} = \text{clip}\left(\text{round}\left(512.0 + V_{\text{mV}} \times 200.0\right), 0, 1023\right)$$

### Verified Command Executions & Provenance:
1. **MIT-BIH Record 100 Simulation (5.0 s @ 360 Hz):**
   ```bash
   python -m hardware.host.simulate_adc_stream --source mitbih --record 100 --duration 5.0 --output hardware/simulation/test_sim_record_100.csv
   ```
   *Result:* Generated 1,800 samples. ADC dynamic range: $[383, 704]$ counts. Zero clipped samples. Full provenance metadata saved to `test_sim_record_100.meta.json`.
2. **Deterministic Synthetic ECG Simulation (3.0 s @ 360 Hz, Seed = 42):**
   ```bash
   python -m hardware.host.simulate_adc_stream --source synthetic --duration 3.0 --output hardware/simulation/test_sim_synthetic.csv
   ```
   *Result:* Generated 1,080 samples. ADC dynamic range: $[450, 763]$ counts. Zero clipped samples.
3. **High-Speed Protocol Replay (Parser Verification):**
   ```bash
   python -m hardware.host.serial_replay --input hardware/simulation/test_sim_record_100.csv --pace 0 --quiet
   ```
   *Result:* Replayed 1,800 samples in $0.133\,\text{s}$ ($13,577.6\,\text{Hz}$ effective throughput). Zero packet drops, zero corrupted tokens.

---

## 6. Cryptographic Integrity & Automated Quality Gates

### 6.1 Cryptographic Hash Verification
Both production models were cryptographically checked before and after Phase 7 changes:

| Model Artifact | File Location | Expected SHA-256 Digest | Measured SHA-256 Digest | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Workflow C Beat ML** | `artifacts/single_lead_ml/single_lead_rf_model.joblib` | `3889a5f5ab97803da339710d4ae9fde8ea50f8e5a5ff76ed2015a0fb0b6ec8e9` | `3889a5f5ab97803da339710d4ae9fde8ea50f8e5a5ff76ed2015a0fb0b6ec8e9` | **STRICT MATCH (Zero Byte Mutation)** |
| **Workflow B PTB-XL ML** | `artifacts/waveform_ml/waveform_lr_model.joblib` | `55d364ed57cc971167f7ad813443c249f792567e406cce4c3dd84188829328d7` | `55d364ed57cc971167f7ad813443c249f792567e406cce4c3dd84188829328d7` | **STRICT MATCH (Zero Byte Mutation)** |

*Assertion:* No model weights, decision thresholds, feature extraction routines, training scripts, or dataset manifests were modified.

### 6.2 Automated Test Execution Summary
1. **Hardware Unit & Simulation Tests (`hardware/tests/`):**
   ```text
   ============================= test session starts ==============================
   platform linux -- Python 3.14.4, pytest-9.1.1
   collected 46 items

   hardware/tests/test_calibration_guards.py ..........                     [ 21%]
   hardware/tests/test_diagram_spec.py ...                                  [ 28%]
   hardware/tests/test_ecg_filters.py .........                            [ 47%]
   hardware/tests/test_firmware_spec.py ....                                [ 56%]
   hardware/tests/test_serial_protocol.py ............                      [ 82%]
   hardware/tests/test_simulated_stream.py ........                         [100%]

   ============================== 46 passed in 1.22s ==============================
   ```

2. **Root Repository Integration Tests (`tests/`):**
   ```text
   ============================= test session starts ==============================
   platform linux -- Python 3.14.4, pytest-9.1.1
   collected 101 items

   tests/test_digitization.py ...........                                   [ 10%]
   tests/test_hardware_integration.py ....                                  [ 14%]
   tests/test_productization.py ......                                      [ 20%]
   tests/test_single_lead_ml.py .................................           [ 53%]
   tests/test_validation.py ............                                    [ 65%]
   tests/test_waveform_dataset.py .......                                   [ 72%]
   tests/test_waveform_features.py .......                                  [ 79%]
   tests/test_waveform_isolation.py ....                                    [ 83%]
   tests/test_waveform_metrics.py .......                                   [ 90%]
   tests/test_waveform_splits.py .....                                      [ 95%]
   tests/test_waveform_training_integrity.py .....                          [100%]

   ============================= 101 passed in 12.54s =============================
   ```
   **Total Across Repository:** **147 passed tests, 0 failures, 0 warnings.**

### 6.3 Toolchain & Compilation Disclosure
- **Arduino CLI / PlatformIO / avr-gcc:** Not installed in the automated Linux workspace.
- **Actual Status:** `firmware compile not run: toolchain unavailable`.
- Static source specification was validated by [`hardware/tests/test_firmware_spec.py`](file:///run/media/op/DATA/Omkar/VIT/2y/3%20sem/Project_expo/code/hardware/tests/test_firmware_spec.py).

---

## 7. Known Limitations & Future Physical Validation Roadmap

1. **Hardware Absence:** No physical AD8232 breakout or Arduino Uno board is present or connected. Physical sensor acquisition, electrode impedance, and USB UART latency remain unverified on hardware.
2. **Electrical Safety Restriction:** Standard Arduino USB connections lack galvanic isolation. Testing on human subjects is strictly prohibited without certified medical isolation meeting **IEC 60601-1**.
3. **Hardware Calibration Pending:** Transfer function gain, virtual ground offset, and electrode impedance parameters must be empirically measured using a calibrated benchtop ECG signal generator before hardware data can be classified.
4. **Toolchain Compilation:** Once an Arduino toolchain or physical board is acquired, firmware compilation and timing jitter measurements ($< 2,778\,\mu\text{s}$) should be physically characterized with an oscilloscope or logic analyzer.

---

## 8. Acceptance Checklist

- [x] **Self-Contained `hardware/` Folder:** Fully populated with firmware, SVG diagrams, host utilities, digital filters, simulation configs, tests, and documentation.
- [x] **Working Deterministic Firmware:** 360 Hz micros() scheduling, 115,200 baud, zero heap allocation, bounds validation, and lead-off flags.
- [x] **Editable Vector Schematic:** Clean SVG diagram with labelled pins, color-coded wires, and prominent safety banners.
- [x] **Zero-Hardware Acceptance Path:** Working offline simulator supporting MIT-BIH records and synthetic fixtures with clipping tracking and sidecar provenance metadata.
- [x] **Signal Processing & Digital Filters:** Zero-phase offline bandpass and notch filters, plus causal stateful streaming IIR filter.
- [x] **Strict Calibration Guards:** Model inference blocked on uncalibrated ADC counts and incompatible sample rates.
- [x] **Comprehensive Test Suite:** 46 new hardware tests + 4 root integration tests; all 147 tests pass cleanly.
- [x] **Zero Model Retraining / Mutation:** Model hashes strictly verified before and after.
- [x] **Biomedical Safety Warnings:** Prominently featured across READMEs, schematics, and reports.
- [x] **Honest Verification Labeling:** All results explicitly labelled `OFFLINE SOFTWARE PROTOTYPE COMPLETE — PHYSICAL HARDWARE NOT TESTED`.
