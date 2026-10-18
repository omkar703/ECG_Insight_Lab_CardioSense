# CardioSense Hardware Prototype Module (Phase 7)

**Hardware Target:** Analog Devices AD8232 Heart Rate Monitor Breakout $\leftrightarrow$ Arduino Uno R3 (ATmega328P)  
**Verification Level:** `OFFLINE SOFTWARE PROTOTYPE COMPLETE — PHYSICAL HARDWARE NOT TESTED`  
**Baud Rate:** 115,200 baud | **Nominal Sample Rate:** 360 Hz ($\Delta t = 2,778\,\mu\text{s}$)  
**Release Version:** 1.0.0

---

## 1. Critical Electrical Safety Notice — Mandatory

> [!CAUTION]
> ### LIFE SAFETY WARNING: DO NOT CONNECT ELECTRODES TO HUMAN SUBJECTS
> 
> 1. **Zero Galvanic Isolation:** Standard Arduino Uno boards and commercial hobbyist AD8232 breakout modules **lack medical-grade galvanic isolation**. The microcontroller ground is directly coupled through the USB ground shield to the host PC chassis and building AC mains ground.
> 2. **Electrocution Hazard:** In the event of a PC power supply fault, ground loop differential, or electrical surge, hazardous AC mains current ($120\,\text{V} / 230\,\text{V}$) could conduct directly through the electrode leads and across a human thorax.
> 3. **Medical Safety Standards (IEC 60601-1):** Clinical biomedical recording equipment mandates Type CF (Cardiac Floating) galvanic isolation rated for $\ge 4,000\,\text{V}_{\text{RMS}}$ dielectric withstand voltage, leakage currents strictly below $10\,\mu\text{A}$ RMS, and patient defibrillation protection resistors.
> 4. **Authorized Research Scope:** The firmware, wiring schematics, and host utilities in this folder are designed **exclusively for benchtop electronic simulation, educational inspection, and offline software testing**. Testing with human electrodes is strictly prohibited until a medically isolated analog front-end is acquired, verified, and approved.

---

## 2. Directory Structure

The `hardware/` tree is fully self-contained, preserving strict isolation from the primary ML pipelines:

```text
hardware/
├── README.md                      # This master hardware documentation guide
├── PINOUT_AND_SAFETY.md           # Comprehensive pinout & biomedical electrical safety guide
├── diagrams/                      # Editable schematics and pin mappings
│   ├── ad8232_arduino_uno_wiring.svg # Editable vector wiring schematic with safety banners
│   └── pin_map.md                 # Detailed pin table, electrical levels, and wire color codes
├── firmware/                      # Microcontroller source code
│   └── ad8232_arduino_uno/
│       ├── ad8232_arduino_uno.ino # 360 Hz deterministic micros() acquisition sketch
│       └── README.md              # Firmware compilation, upload, and troubleshooting manual
├── host/                          # Host workstation communication & simulation utilities
│   ├── __init__.py                # Host package exports
│   ├── serial_protocol.py         # Line-oriented CSV protocol parser with timer unwrap
│   ├── serial_capture.py          # Real-time UART capture utility (optional pyserial)
│   ├── serial_replay.py           # Offline playback & protocol validation utility
│   └── simulate_adc_stream.py     # 10-bit ADC simulator (MIT-BIH records & synthetic ECG)
├── signal_processing/             # Digital filtering & calibration safeguards
│   ├── __init__.py                # Signal processing exports
│   ├── ecg_filters.py             # Zero-phase offline (filtfilt) & causal streaming IIR filters
│   └── calibration.py             # Strict physical unit guards blocking uncalibrated model ingestion
├── simulation/                    # Offline simulation profiles & sample fixtures
│   ├── README.md                  # Step-by-step offline simulation walkthrough
│   └── example_config.json        # Standard 360 Hz offline simulation profile
├── tests/                         # 46 automated unit, static, and simulation tests
│   ├── __init__.py                # Test package init
│   ├── test_serial_protocol.py    # Protocol parsing, timer wrap, and error handling tests
│   ├── test_simulated_stream.py   # Deterministic simulator & clipping tests
│   ├── test_ecg_filters.py        # Bandpass, notch, and causal streaming filter tests
│   ├── test_calibration_guards.py # Physical unit separation & classifier rejection tests
│   ├── test_firmware_spec.py      # Static verification of sketch pins, timing, and memory safety
│   └── test_diagram_spec.py       # SVG XML validity, pin completeness, and warning checks
└── reports/                       # Formal verification reports
    ├── hardware_offline_validation.md     # Offline hardware simulation validation report
    ├── simulation_inference_validation.md # Streamlit simulation and inference benchmark report
    └── simulation_validation_results.json # Structured benchmark metrics output
```

---

## 3. Planned Bill of Materials (BOM)

*Note: The components below represent the planned hardware procurement list for future benchtop laboratory validation. No physical hardware has been purchased or verified in this phase.*

| Item # | Component | Manufacturer / Model | Quantity | Target Interface | Purpose |
| :---: | :--- | :--- | :---: | :--- | :--- |
| 1 | Single-Lead ECG Front-End | Analog Devices AD8232 Breakout (SparkFun WIG-12650 or equivalent) | 1 | 3.3V, GND, A0, D10, D11 | Bio-potential instrumentation amplifier & active filter. |
| 2 | Development Board | Arduino Uno R3 (ATmega328P @ 16 MHz) | 1 | USB Type-B to Workstation | 10-bit analog-to-digital conversion & UART transmission. |
| 3 | Biomedical Lead Cable | 3.5 mm TRRS Audio Jack to 3-Snap Leads | 1 | RA, LA, RL snap terminals | Electrode lead interface (for phantom/simulator connection). |
| 4 | USB Cable | USB 2.0 Type-A to Type-B Shielded Cable | 1 | Host PC USB Port | 115,200 baud serial communication & 5V board power. |
| 5 | Interconnect Wires | 22 AWG Dupont Jumper Wires (M-to-F) | 5 | Breakout to Uno Headers | Red (3.3V), Black (GND), Green (A0), Orange (D11), Yellow (D10). |
| 6 | ECG Phantom / Generator | Handheld ECG Patient Simulator (Planned future testing) | 1 | Snap lead inputs | Safe electronic signal generator replacing human skin contact. |

---

## 4. Hardware Pinout & Wiring Summary

Refer to [`hardware/diagrams/ad8232_arduino_uno_wiring.svg`](diagrams/ad8232_arduino_uno_wiring.svg) for the editable vector schematic and [`hardware/diagrams/pin_map.md`](diagrams/pin_map.md) for electrical ratings.

| AD8232 Breakout Pin | Arduino Uno R3 Pin | Function | Electrical Rating |
| :--- | :--- | :--- | :--- |
| **`3.3V`** | **`3.3V`** (Power) | Sensor Module Supply | $+3.3\,\text{V} \pm 5\%$ (**Never connect to 5V!**) |
| **`GND`** | **`GND`** (Power) | Common Ground | $0.0\,\text{V}$ common return |
| **`OUTPUT`** | **`A0`** (Analog) | Analog Voltage Output | $0.0\text{--}3.3\,\text{V}$ (Centered at $\approx 1.65\,\text{V}$ virtual ground) |
| **`LO-`** | **`D11`** (Digital) | Lead-Off Status (-) | Logic HIGH ($3.3\,\text{V}$) when negative electrode is detached |
| **`LO+`** | **`D10`** (Digital) | Lead-Off Status (+) | Logic HIGH ($3.3\,\text{V}$) when positive electrode is detached |
| **`SDN`** | *Unconnected* | Shutdown Control | Pulled HIGH internally; leave unconnected |

---

## 5. Offline Simulation & Verification (Executable Today Without Hardware)

The complete software stack can be executed and validated immediately using recorded MIT-BIH recordings or synthetic fixtures.

### Step 1: Generate Simulated ADC Stream from MIT-BIH Record 100
```bash
python -m hardware.host.simulate_adc_stream \
  --source mitbih \
  --record 100 \
  --duration 10.0 \
  --output hardware/simulation/simulated_stream.csv
```
*Output: Generates 3,600 samples (10.0 s @ 360 Hz) mapped to 10-bit integer ADC counts, with full metadata sidecar.*

### Step 2: Generate Synthetic ECG Stream with Lead-Off Fault Injection
```bash
python -m hardware.host.simulate_adc_stream \
  --source synthetic \
  --duration 8.0 \
  --lead-off-start 3.0 \
  --lead-off-duration 1.5 \
  --output hardware/simulation/simulated_leadoff.csv
```
*Output: Simulates electrode detachment between seconds 3.0 and 4.5, triggering digital lead-off flags.*

### Step 3: High-Speed Protocol Replay & Parser Validation
```bash
python -m hardware.host.serial_replay \
  --input hardware/simulation/simulated_stream.csv \
  --pace 0 \
  --quiet
```
*Output: Verifies that 3,600 samples are parsed with zero frame drops and zero corrupted tokens.*

### Step 4: Run Streamlit Web Application (Hardware Prototype Tab)
Launch the Streamlit research platform:
```bash
.venv/bin/streamlit run app.py --server.port 8502 --server.headless true
```
1. Open `http://localhost:8502`
2. Switch top navigation to **`Live Experiment`**
3. Select tab **`Hardware Prototype — Offline AD8232 Simulation`**
4. Configure signal source (MIT-BIH Record 100 or Synthetic Fixture) and simulation preset (`Near-lossless` or `Noisy`)
5. Click **`Run Offline Sensor Simulation & Predict Beats`** to view the live status panel, dual-view Plotly waveforms, and the visible beat predictions table.

### Step 5: Execute Automated Validation & Benchmark Script
```bash
.venv/bin/python hardware/validate_simulation_pipeline.py
```
*Output: Executes end-to-end simulation across MIT-BIH Record 100 and synthetic fixtures, performs post-hoc evaluation, verifies frozen model SHA-256 digest, and outputs `hardware/reports/simulation_validation_results.json`.*

### Step 6: Execute Automated Quality Gates
```bash
# Run simulation and Streamlit AppTest suite (13 tests)
.venv/bin/pytest tests/test_simulation_inference.py -v

# Run hardware-specific test suite (46 tests)
.venv/bin/pytest hardware/tests/ -v

# Run complete repository integration test suite (192 tests)
.venv/bin/pytest tests/ hardware/tests/ -v
```

---

## 6. Serial Protocol Specification

- **Baud Rate:** 115,200 baud, 8 data bits, no parity, 1 stop bit (8-N-1).
- **Line Framing:** ASCII newline-terminated CSV (`\n`).
- **Header Lines:** Any line starting with `#` is treated as a comment/metadata header and ignored by numerical processors.
- **Data Line Format:**
  $$\text{seq},\text{micros},\text{raw\_adc},\text{lo\_plus},\text{lo\_minus}$$
  Example:
  ```text
  0,100000,512,0,0
  1,102778,515,0,0
  2,105556,519,0,0
  ```
- **Field Definitions:**
  * `seq` (uint32): Sequential packet counter.
  * `micros` (uint32): Microcontroller `micros()` clock (unwrapped by host parser past 71.58 min rollover).
  * `raw_adc` (uint16): Uncalibrated integer ADC count $[0, 1023]$. **NEVER labelled mV.**
  * `lo_plus` (uint8): $1$ if `LO+` asserted, $0$ otherwise.
  * `lo_minus` (uint8): $1$ if `LO-` asserted, $0$ otherwise.

---

## 7. Digital Signal Processing & Calibration Safeguards

### Digital Filters (`hardware/signal_processing/ecg_filters.py`):
1. **Zero-Phase Offline Bandpass Filter (`butter_bandpass_filter`):**
   - 2nd-order Butterworth bandpass ($0.5\,\text{Hz}$ highpass to $40.0\,\text{Hz}$ lowpass).
   - Applied bidirectionally via `filtfilt` for zero phase distortion.
2. **Zero-Phase Powerline Notch Filter (`notch_filter`):**
   - IIR notch filter centered at $50.0\,\text{Hz}$ (configurable to $60.0\,\text{Hz}$) with $Q=30.0$.
3. **Causal Streaming Filter (`StreamingECGFilter`):**
   - Stateful IIR implementation for real-time sample-by-sample and block processing.
   - Accurately introduces physical group delay without future-sample leakage.

### Calibration Guardrails (`hardware/signal_processing/calibration.py`):
- **Raw ADC Count $\ne$ Pin Voltage $\ne$ Patient ECG mV:**
  $$\text{Pin Voltage } V_{in} = \text{ADC} \times \frac{3.3\,\text{V}}{1023}$$
  $$\text{ECG } (\text{mV}) = \frac{V_{in} - V_{\text{virtual\_gnd}}}{\text{Analog Gain}} \times 1000$$
- **Model Ingestion Barrier:**
  The single-lead ML classifier was trained strictly on calibrated physical millivolts ($200\,\text{ADC/mV}$). Passing raw hardware ADC counts or uncalibrated analog volts triggers an immediate `IncompatibleSignalError` or `UncalibratedHardwareError`, preventing uncalibrated inference by default.

---

## 8. Summary of Known Limitations & Future Validation Tasks

1. **Hardware Absence:** No physical AD8232 or Arduino board is present in the workspace. All verification is based on offline simulation and static analysis.
2. **Compilation Status:** Arduino toolchains (`arduino-cli`, `platformio`, `avr-gcc`) are not installed in the automated environment; firmware compilation was not executed.
3. **Hardware Calibration Pending:** Transfer function gain, DC bias, and analog filter tolerance must be empirically measured with precision bench instruments before hardware data can be classified.
4. **Clinical Safety Boundary:** The prototype is an academic engineering exercise and has not undergone medical device regulatory review (FDA, CE-MDR, or CDSCO).
