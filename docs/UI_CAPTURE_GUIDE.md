# User Interface Walkthrough & Visual Capture Guide

**ECG Insight Lab / CardioSense — Phase 4 Portfolio Deliverable**  
*Comprehensive visual walkthrough, UI component hierarchy, and step-by-step instructions for capturing publication-grade screenshots across all 5 application journeys.*

---

## 1. Overview & Capture Environment

The ECG Insight Lab web application adheres to the **Quiet Academic Editorial** design system:
- **Typography:** Serif headings (`Source Serif 4` / `Merriweather`), clean sans-serif data labels (`IBM Plex Sans` / `Inter`), and tabular monospaced numbers.
- **Palette:** Warm ivory background (`#F7F6F2`), deep charcoal typography (`#252722`), forest green accents (`#455D4A`), subdued borders (`#E5E3DC`), and muted amber warnings (`#D9822B`).
- **Optimal Capture Viewport:** $1280 \times 800\,\text{px}$ (Desktop landscape) or $1920 \times 1080\,\text{px}$ at $100\%$ scale.

### Launching the Application for Visual Capture
```bash
# 1. Activate the dedicated virtual environment
source .venv/bin/activate

# 2. Launch Streamlit locally
streamlit run app.py --server.port 8501

# 3. Access in modern browser (Chrome, Firefox, Safari, Edge)
# URL: http://localhost:8501
```

---

## 2. Visual Walkthrough by User Journey

```
+--------------------------------------------------------------------------------------------------+
|  ECG INSIGHT LAB                                  [Quiet Academic Editorial UI]                  |
|  Research Platform for ECG Image Digitization & Calibrated Waveform ML                           |
+--------------------------------------------------------------------------------------------------+
|  [Tabs]: (1) Overview  | (2) ECG Image Lab  | (3) Waveform ML Lab  | (4) Methods  | (5) Diagnostics |
+--------------------------------------------------------------------------------------------------+
```

---

### Journey 1: Overview & System Status
**Target View:** Initial landing screen displaying project motivation, architectural separation, and system status cards.

#### Key Visual Elements to Capture
1. **Header & Non-Clinical Disclaimer:** Bold research prototype notice emphasizing that this is an academic exploration tool, not a medical device, and live sensors/ESP32 are not connected.
2. **Architectural Decoupling Diagram:** Clean conceptual cards contrasting **Workflow A** (Paper Image Digitization) and **Workflow B** (Digital Waveform ML) with an explicit notice that no unvalidated paper-to-model bridge exists.
3. **Headline Benchmark Metric Cards:**
   - Active Model Macro F1: `0.6090` (Held-Out Fold 10, $N=2,198$)
   - Digitizer Lead Detection: `100%` (12/12 leads on synthetic benchmark)
   - Verified Zero-Leakage Folds: `10` official stratifications
   - Test Cohort: `2,198` records across `1,883` distinct patients
4. **Live System Readiness Status:** Green status badge confirming whether model weights and demo fixtures are operational.

---

### Journey 2: ECG Image Lab (Workflow A)
**Target View:** Interactive paper ECG image digitization and waveform recovery workspace.

#### Step-by-Step Capture Procedure
1. Click the **ECG Image Lab** tab.
2. Observe the **Privacy & Data Safety Banner**:
   > *Zero-Persistence Guarantee: Uploaded images are processed entirely in ephemeral RAM and never saved to disk, logged, or retained.*
3. In the sidebar or input container, select **Use Synthetic Benchmark (sample_data/synthetic_12lead.png)** or upload an image file.
4. Note the prominent indicator: `SYNTHETIC DEMO — NOT PATIENT DATA`.
5. Click **Run Digitization Analysis**.
6. **Capture Section:**
   - Image validation metrics (Resolution, Aspect Ratio, Color Depth, Validation status: `PASSED`).
   - 12 individual lead waveform recovery plots arranged in standard clinical $3 \times 4$ layout (Leads I, II, III, aVR, aVL, aVF, V1–V6).
   - Calibration Status Notice: Displaying `RELATIVE` amplitude units and `UNRECOVERED` calibration status.
   - Data Export Section: Interactive `Export Waveforms as CSV` and `Export Metadata as JSON` buttons.
   - Strict Boundary Banner: Emphasizing that recovered 1D signals are not piped into the diagnostic classifier.

---

### Journey 3: Waveform ML Lab (Workflow B)
**Target View:** Calibrated PhysioNet PTB-XL digital waveform inspection, multi-label model inference, and held-out benchmark evaluation.

#### Step-by-Step Capture Procedure
1. Click the **Waveform ML Lab** tab.
2. Choose a record from the dropdown (e.g., `Record #1 (ID: 00001_hr)`).
3. **Capture Section A — Calibrated Waveform Plot:**
   - 12-lead voltage plot rendered with accurate physical axes: Time in seconds ($0.0 - 10.0\,\text{s}$) and Voltage in millivolts ($-1.5 - +2.5\,\text{mV}$).
   - High-contrast clinical grid lines ($0.2\,\text{s}$ major grid, $0.04\,\text{s}$ minor grid).
4. **Capture Section B — Model Inference & Diagnostic Probabilities:**
   - 5 diagnostic superclasses (`NORM`, `MI`, `STTC`, `CD`, `HYP`).
   - Probability progress bars and positive decision badges evaluated against validation-tuned thresholds.
   - Ground truth annotation comparison badges (`Verified Research Annotation`).
5. **Capture Section C — Benchmark Evaluation & Paired Comparison Tab:**
   - Full held-out benchmark summary table across 2,198 records in Fold 10.
   - **Paired Model Comparison View:** Expanding the comparison between the Phase 3.1 Pilot Model and the Phase 3.2 Full-Fold Model.
   - **Mandatory Caveats Displayed:**
     - *Pipeline Comparison:* Improvement (+0.1510 Macro F1) represents the difference between complete pipelines (training subsets AND validation sample sizes for threshold tuning: 30 records vs 2,193 records).
     - *Empirical Bootstrap Replicates:* `0 of 2,000` paired bootstrap replicates had a non-positive difference; confidence interval $[0.1343, 0.1681]$ describes test-cohort sampling uncertainty conditional on the fixed fitted models.

---

### Journey 4: Methods & Benchmarks
**Target View:** Transparent academic documentation of data splits, taxonomy, 148 feature definitions, and model trust boundaries.

#### Key Visual Elements to Capture
1. **Partition Splits & Patient Isolation:** Table documenting Folds 1–8 (17,418 records, 14,996 patients), Fold 9 (2,183 records, 1,883 patients), and Fold 10 (2,198 records, 1,883 patients) with verified $0$ patient overlap.
2. **Diagnostic Superclasses Taxonomy:** SCP-ECG 71 statement ontology mapped to the 5 superclasses.
3. **Signal Feature Schema:** 148 deterministic features across 12 leads (RR intervals, heart rate, QRS durations, mean, std, skewness, kurtosis, frequency-domain spectral energy).
4. **Cryptographic Model Integrity Boundary:** Displaying SHA-256 hash match (`55d364ed...`), manifest creation timestamp, and strict rejection policy for untrusted legacy pickles.

---

### Journey 5: Diagnostics & Privacy Safeguards
**Target View:** Live self-test suite and local deployment configuration.

#### Key Visual Elements to Capture
1. **Live Diagnostics Health Checks:** 7 green verification checkmarks covering dataset completeness, model SHA-256 match, synthetic fixtures, and in-memory upload safeguards.
2. **Privacy & Ephemeral Lifecycle Guarantee:** Detailed technical breakdown of RAM-only processing and zero telemetry.
3. **Reproducible Offline Run Guide:** Direct copyable commands for running tests, CLI pipelines, and launching the application.

---

## 3. Recommended Tools for Manual Screenshot Capture

- **Browser Built-in Capture:** In Chrome/Edge, open Developer Tools (`F12`), press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS), type `Capture full size screenshot`, and press Enter.
- **OS Native Tools:**
  - Linux: `gnome-screenshot -a` or `flameshot gui`
  - macOS: `Cmd + Shift + 4`
  - Windows: `Win + Shift + S`
- **Output Storage:** Save captured visual assets in `docs/screenshots/` (e.g., `overview.png`, `image_lab.png`, `waveform_ml.png`, `methods.png`, `diagnostics.png`).
