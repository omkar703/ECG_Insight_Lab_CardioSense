# AD8232 & Arduino Uno R3 Pin Mapping

**Architecture Target:** Analog Devices AD8232 Heart Rate Monitor Breakout $\leftrightarrow$ Arduino Uno R3 (ATmega328P)  
**Document Status:** Benchmark Prototype Specification  
**Verification Level:** `SIMULATION / SPECIFICATION ONLY — PHYSICAL HARDWARE NOT TESTED`

---

## 1. Pin Interconnection Table

| AD8232 Pin Label | Arduino Uno R3 Header | Wire Color | Signal Type | Voltage Level | Description & Operational Function |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`3.3V`** | **`3.3V`** (Power) | Red | DC Power Input | $+3.3\,\text{V} \pm 5\%$ | Regulated 3.3V power supply rail for the AD8232 front-end. **CAUTION: Never connect to the 5V pin**, as 5V exceeds the AD8232 absolute maximum rating (3.6V). |
| **`GND`** | **`GND`** (Power) | Black | Ground Reference | $0.0\,\text{V}$ | Common signal and chassis reference ground return. |
| **`OUTPUT`** | **`A0`** (Analog) | Green | Analog Output | $0.0\,\text{V} \text{ to } 3.3\,\text{V}$ | Amplified and bandpass-filtered analog ECG signal centered on an internal virtual ground ($\approx 1.65\,\text{V}$). Sampled by the 10-bit ATmega328P ADC ($0\text{--}1023$ counts). |
| **`LO-`** | **`D11`** (Digital) | Orange | Digital Output | $0.0\,\text{V} \text{ or } 3.3\,\text{V}$ | **Lead-Off Status (-)**: Logic HIGH when the negative electrode (Right Arm / RA) is detached or has poor skin contact. |
| **`LO+`** | **`D10`** (Digital) | Yellow | Digital Output | $0.0\,\text{V} \text{ or } 3.3\,\text{V}$ | **Lead-Off Status (+)**: Logic HIGH when the positive electrode (Left Arm / LA) is detached or has poor skin contact. |
| **`SDN`** | *Unconnected* | — | Digital Input | Pull-up ($\approx 3.3\,\text{V}$) | **Shutdown Control**: Active-LOW shutdown pin. Pulled HIGH internally on standard breakout boards. Leave unconnected for normal continuous operation. |

---

## 2. Bio-Potential Electrode Connections (Reference Only)

> [!CAUTION]
> **ELECTRICAL SAFETY RESTRICTION:**  
> The electrode connections below describe standard Lead I / Lead II configurations for testing with an **isolated patient simulator or benchtop ECG signal generator**. Connecting electrodes to human skin while powered via non-isolated USB is strictly prohibited.

| Electrode Lead Name | Typical Cable Marking | Placement (Standard 3-Lead) | Function |
| :--- | :---: | :--- | :--- |
| **`RA`** | Red or Black | Right Arm / Right Subclavicular | Negative differential input ($V_-$). Connected to differential preamplifier. |
| **`LA`** | Yellow or White | Left Arm / Left Subclavicular | Positive differential input ($V_+$). Connected to differential preamplifier. |
| **`RL`** | Green | Right Leg / Lower Abdomen | Right Leg Drive (RLD) / Driven reference ground for common-mode noise cancellation. |

*Note: Cable color conventions differ across manufacturers (AAMI vs IEC standards). Always follow the physical breakout silkscreen and labelled jacks rather than cable color alone.*

---

## 3. Electrical Characteristics & Operating Envelope

- **Operating Supply Voltage ($V_{DD}$):** $2.0\,\text{V}$ to $3.5\,\text{V}$ (Nominal: $3.3\,\text{V}$).
- **Quiescent Supply Current:** $\approx 170\,\mu\text{A}$.
- **Analog Output Range:** Rail-to-rail, typical linear dynamic range $0.2\,\text{V}$ to $3.1\,\text{V}$ on $3.3\,\text{V}$ supply.
- **On-Board Instrumentation Amplifier Gain:** Typical default factory configuration provides an overall instrumentation amplifier + second stage filter gain of $\approx 100\,\text{V/V}$ to $1100\,\text{V/V}$ depending on breakout component values.
- **Arduino ADC Resolution:** 10 bits ($2^{10} = 1024$ quantization levels: $0$ to $1023$).
- **Nominal Sampling Rate:** $360\,\text{Hz}$ ($\Delta t = 2,778\,\mu\text{s}$).
