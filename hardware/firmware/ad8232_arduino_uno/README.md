# AD8232 + Arduino Uno Firmware Guide

**Firmware Target:** Arduino Uno R3 (ATmega328P, 16 MHz)  
**Sensor Front-End:** Analog Devices AD8232 Heart Rate Monitor Breakout  
**Acquisition Frequency:** 360 Hz nominal ($\Delta t = 2,778\,\mu\text{s}$)  
**Communication:** UART Serial over USB @ 115,200 baud  
**Protocol Version:** 1.0.0 (Line-oriented CSV)

---

## 1. Electrical Safety Warning — Mandatory

> [!CAUTION]
> **ELECTRICAL SAFETY & RESEARCH-USE ONLY WARNING:**  
> This firmware is intended solely for bench-top prototyping, educational exploration, and testing with electronic signal generators or hardware-in-the-loop simulators.  
> **DO NOT CONNECT ELECTRODES TO HUMAN SUBJECTS.**  
> A standard hobby Arduino Uno connected via USB to a PC is **not galvanically isolated** from mains electricity. In the event of a ground fault or surge, lethal current could pass through electrode leads into a human subject. Clinical ECG equipment mandates strict galvanic isolation meeting **IEC 60601-1** medical safety standards.

---

## 2. Hardware Pin Connections

Assuming an AD8232 breakout (SparkFun or pin-compatible module):

| AD8232 Breakout Pin | Arduino Uno R3 Pin | Function / Direction | Electrical Notes |
| :--- | :--- | :--- | :--- |
| **`3.3V`** | **`3.3V`** | Power Supply (Input to AD8232) | AD8232 rated for 2.0V to 3.5V operation. Do **NOT** connect to 5V! |
| **`GND`** | **`GND`** | Common Ground | Common return path. |
| **`OUTPUT`** | **`A0`** | Analog Voltage Output | Analog signal centered around virtual ground (~1.65V). 0–3.3V dynamic range. |
| **`LO+`** | **`D10`** | Digital Lead-Off Status (+) | High when positive electrode is detached. |
| **`LO-`** | **`D11`** | Digital Lead-Off Status (-) | High when negative electrode is detached. |
| **`SDN`** | *Unconnected* | Shutdown Control | Pulled high on breakout board; leave disconnected for normal operation. |

---

## 3. Firmware Design & Timing Integrity

1. **Deterministic 360 Hz Scheduling:**
   - Sample interval: $T_s = \frac{1,000,000\,\mu\text{s}}{360\,\text{Hz}} = 2,777.78\,\mu\text{s} \approx 2,778\,\mu\text{s}$.
   - Scheduling is governed by `micros()` tracking without blocking `delay()` calls.
   - Timing drift is tracked; severe overruns ($> 2 \times T_s$) increment an internal overrun counter.

2. **Serial Bandwidth Budget:**
   - Baud rate: **115,200 baud** ($\approx 11,520\,\text{bytes/second}$).
   - Sample line length: $\approx 18\text{--}24\,\text{bytes}$ (`seq,micros,adc,lo_plus,lo_minus\n`).
   - Throughput demand: $360\,\text{Hz} \times 22\,\text{bytes} = 7,920\,\text{bytes/second}$.
   - Bus utilization: $\approx 68.7\%$, comfortably within UART capacity.

3. **Memory Safety:**
   - Zero dynamic heap allocation (`String` objects are strictly prohibited).
   - Direct numeric stream writes prevent heap fragmentation in long-running experiments.

---

## 4. Expected Serial Protocol Stream

Upon startup, the firmware transmits metadata header lines (prefixed by `#`), followed by continuous line-oriented CSV rows:

```text
# CardioSense AD8232 Arduino Uno v1.0
# baud=115200, nominal_fs=360, interval_us=2778
# fields=seq,micros,adc,lo_plus,lo_minus
# NOTICE: RAW 10-BIT ADC COUNTS (0-1023). NOT CALIBRATED ECG MV. PROTOTYPE ONLY.
0,2778,512,0,0
1,5556,514,0,0
2,8334,510,0,0
3,11112,508,0,0
```

### Field Definitions:
- `seq` (uint32): Monotonically increasing sequence number (wraps at $2^{32}-1$).
- `micros` (uint32): Arduino hardware timer timestamp in microseconds (wraps every ~71.58 minutes).
- `adc` (uint16): Raw 10-bit analog conversion value ($0 \le \text{adc} \le 1023$). **Not calibrated millivolts.**
- `lo_plus` (uint8): Digital state of `LO+` pin (1 = lead detached, 0 = lead attached).
- `lo_minus` (uint8): Digital state of `LO-` pin (1 = lead detached, 0 = lead attached).

---

## 5. Build & Upload Instructions

### Method A: Arduino IDE
1. Open the Arduino IDE (v1.8+ or v2.0+).
2. Connect the Arduino Uno to your workstation via USB.
3. Open `hardware/firmware/ad8232_arduino_uno/ad8232_arduino_uno.ino`.
4. Select **Tools > Board > Arduino AVR Boards > Arduino Uno**.
5. Select the appropriate serial port under **Tools > Port** (e.g. `/dev/ttyACM0` on Linux, `COM3` on Windows).
6. Click **Upload** (Ctrl+U or Cmd+U).
7. Open **Tools > Serial Monitor** and set the baud rate to **115200**.

### Method B: Arduino CLI
```bash
# Compile the sketch for Arduino Uno
arduino-cli compile --fqbn arduino:avr:uno hardware/firmware/ad8232_arduino_uno/

# Upload to the detected port
arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:avr:uno hardware/firmware/ad8232_arduino_uno/
```

*Note: In the current automated CI/research environment, `arduino-cli` is not installed on this system. Compilation has not been executed here, and physical upload has not been performed.*

---

## 6. Troubleshooting & Diagnostics

| Symptom | Probable Cause | Corrective Action |
| :--- | :--- | :--- |
| Garbled text in Serial Monitor | Baud rate mismatch | Ensure host monitor or capture script is set to **115200 baud**. |
| `adc` stays flat at 0 or 1023 | Wiring disconnect or missing power | Verify 3.3V supply and GND pins. Check connection to pin `A0`. |
| `lo_plus` and `lo_minus` are always 1 | Leads detached or leads-off circuitry triggered | Ensure electrodes/simulation signals are connected to input jack. |
| Excessive 50 Hz/60 Hz hum | Mains interference or floating inputs | Use shielded cables; apply the software digital filters in `hardware/signal_processing/`. |
