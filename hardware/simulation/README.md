# Hardware Offline Simulation Guide

**Module:** CardioSense Hardware Simulation (`hardware/simulation/`)  
**Purpose:** Enable 100% offline development, verification, and automated testing of the serial protocol, parser, digital filters, and replay utilities without physical microcontrollers or sensors.

---

## 1. Simulation Architecture

Because physical AD8232 and Arduino boards are not currently available, the simulation pipeline translates real digital ECG signals into **simulated 10-bit integer ADC counts** using the exact transmission framing expected from the Arduino Uno firmware:

```text
┌───────────────────────────────┐
│ MIT-BIH Arrhythmia Database   │
│ (e.g. Record 100, MLII Lead)  │
│ or Synthetic ECG Generator    │
└──────────────┬────────────────┘
               │ Calibrated Physical Signal (mV)
               ▼
┌───────────────────────────────┐
│ Quantization & Offset Model   │
│ ADC = 512 + (mV * 200)        │
│ Range Clamp: [0, 1023]        │
└──────────────┬────────────────┘
               │ Simulated 10-bit Raw ADC Counts
               ▼
┌───────────────────────────────┐
│ Protocol Framing Generator    │
│ "seq,micros,adc,lo+,lo-\n"    │
└──────────────┬────────────────┘
               ▼
    [CSV File / In-Memory Stream / Host Parser]
```

> [!IMPORTANT]
> **SIMULATION SCOPE DISCLAIMER:**  
> The simulator parameters (offset: $512$, scale: $200\,\text{counts/mV}$) are software test vectors to emulate ADC integer transportation. They do **not** reflect the physical analog transfer function, component tolerances, or electrode polarization of an actual physical AD8232 breakout.

---

## 2. Running Simulations (Verified Commands)

### Command A: Simulate from MIT-BIH Record 100 (5 Seconds @ 360 Hz)
```bash
python -m hardware.host.simulate_adc_stream \
  --source mitbih \
  --record 100 \
  --duration 5.0 \
  --output hardware/simulation/simulated_stream.csv
```
**Output:** Generates `simulated_stream.csv` (1,800 samples) and `simulated_stream.meta.json` with provenance details.

### Command B: Simulate from Synthetic Waveform Generator (Standalone Mode)
```bash
python -m hardware.host.simulate_adc_stream \
  --source synthetic \
  --duration 10.0 \
  --seed 42 \
  --output hardware/simulation/simulated_synthetic.csv
```

### Command C: Simulate with Lead-Off Fault Injection
```bash
python -m hardware.host.simulate_adc_stream \
  --source mitbih \
  --record 100 \
  --duration 8.0 \
  --lead-off-start 3.0 \
  --lead-off-duration 1.5 \
  --output hardware/simulation/simulated_leadoff.csv
```
*Between seconds 3.0 and 4.5, `lo_plus` and `lo_minus` will assert `1`, and ADC count will saturate to 1023, verifying fault handling in host software.*

### Command D: Replay Simulated Data (High Speed Protocol Test)
```bash
python -m hardware.host.serial_replay \
  --input hardware/simulation/simulated_stream.csv \
  --pace 0 \
  --quiet
```
*Replays all samples through the protocol parser at full CPU throughput.*
