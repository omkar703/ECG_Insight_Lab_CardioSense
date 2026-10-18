# Hardware Pinout & Electrical Safety Guide

**Project:** CardioSense / ECG Insight Lab  
**Module:** Phase 7 Hardware Prototype (`hardware/`)  
**Target Hardware:** AD8232 Heart Rate Monitor Breakout + Arduino Uno R3  
**Status:** `OFFLINE SOFTWARE PROTOTYPE — PHYSICAL HARDWARE NOT TESTED`

---

## 1. Mandatory Electrical Safety Notice

> [!CAUTION]
> ### CRITICAL SAFETY WARNING: DO NOT CONNECT ELECTRODES TO HUMAN SUBJECTS
> 
> 1. **Lack of Galvanic Isolation:** Standard Arduino Uno boards and commercial hobbyist AD8232 breakout modules **lack medical-grade galvanic isolation**. The Arduino ground is directly tied through the USB cable shield to the host computer chassis and, ultimately, to building electrical mains ground.
> 2. **Electrocution Hazard:** In the event of an electrical fault, transient surge, faulty PC power supply, or ground potential differential, lethal AC mains voltage ($120\,\text{V} / 230\,\text{V}$) can travel directly through the USB ground into the electrode leads and across the subject's thorax.
> 3. **Medical Standards (IEC 60601-1):** Clinical biomedical recording equipment requires certified Type CF (Cardiac Floating) galvanic isolation rated for $\ge 4,000\,\text{V}_{\text{RMS}}$ dielectric breakdown, microampere-level patient leakage current limits ($\le 10\,\mu\text{A}$ normal, $\le 50\,\mu\text{A}$ single-fault), and defibrillation protection resistors.
> 4. **Permitted Testing Scope:** The hardware designs and firmware sketches in this repository are designed **exclusively for benchtop research, educational inspection, and offline software simulation**. Physical testing is authorized only with certified electronic ECG patient simulators (phantom signal generators) or battery-powered, optoisolated test fixtures.

---

## 2. Complete Breakout Pinout & Safeguards

```text
       AD8232 Breakout                  Arduino Uno R3
    ┌────────────────────┐          ┌────────────────────┐
    │  [3.5mm JACK]      │          │                    │
    │  (Bio-potential)   │          │  [USB UART]        │
    │                    │          │                    │
    │  3.3V  ────────────┼──────────┼── 3.3V (Power Rail)│
    │  GND   ────────────┼──────────┼── GND  (Ground)    │
    │  OUTPUT ───────────┼──────────┼── A0   (Analog ADC)│
    │  LO-   ────────────┼──────────┼── D11  (Digital In)│
    │  LO+   ────────────┼──────────┼── D10  (Digital In)│
    │  SDN   (No Connect)│          │                    │
    └────────────────────┘          └────────────────────┘
```

### Safety Rules During Wiring:
1. **Never apply 5V to the AD8232 3.3V pin:** The AD8232 IC is fabricated on a low-voltage submicron process. Connecting the Arduino 5V header pin will permanently destroy the front-end amplifier.
2. **Do not short analog output:** The `OUTPUT` pin drives the unbuffered output of the operational amplifier filter stage. Do not connect it to external power rails.
3. **Shutdown Pin (`SDN`):** The `SDN` pin has an internal pull-up resistor. Leaving it unconnected keeps the module active. Do not pull it to 5V.

---

## 3. Signal Paths & Data Integrity

### Analog Signal Path (Pin `A0`):
- The AD8232 incorporates an on-chip instrumentation amplifier (fixed gain $\approx 100\,\text{V/V}$) and a two-pole high-pass filter ($0.5\,\text{Hz}$) to reject large half-cell electrode DC potentials.
- The output signal is amplified through a second active low-pass Sallen-Key filter ($40\,\text{Hz}$) and referenced to an internal virtual reference voltage ($V_R \approx 1.65\,\text{V}$).
- The Arduino Uno 10-bit analog-to-digital converter (ADC) quantizes this signal against its reference voltage into raw integers:
  $$\text{Raw ADC Count} \in [0, 1023]$$

### Digital Status Signals (Pins `D10` & `D11`):
- The AD8232 features leads-off detection circuitry that injects small high-frequency or DC currents into the electrode paths to monitor contact impedance.
- When an electrode falls off or exhibits excessive skin impedance:
  - `LO+` (Pin `D10`) asserts HIGH ($3.3\,\text{V}$) if the positive electrode is disconnected.
  - `LO-` (Pin `D11`) asserts HIGH ($3.3\,\text{V}$) if the negative electrode is disconnected.
- The host software treats asserted lead-off status as **unusable/corrupted data** and flags it in downstream processing rather than attempting classification.

---

## 4. Software Safety Guardrails (Zero-Hardware Simulation Path)

To enable thorough testing without requiring physical boards, the project provides:
1. **Synthetic & MIT-BIH ADC Simulator (`hardware/host/simulate_adc_stream.py`):** Generates identical UART data frames without physical hardware.
2. **Model Input Isolation (`hardware/signal_processing/calibration.py`):** Strictly prevents uncalibrated ADC counts from entering the single-lead Random Forest classifier.
3. **Immutability of Historical ML Workflows:** Preserves Workflow A (image digitizer), Workflow B (12-lead PTB-XL), and Workflow C (single-lead MIT-BIH) without modification.
