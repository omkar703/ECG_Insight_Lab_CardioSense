# Changelog & Repository Evolution

All notable changes to the **CardioSense — ECG Insight Lab** platform are documented in this file.

---

## [2.0.0] - 2026-10-18 — Admissions Portfolio Refactoring & Academic Rigor

### Major Product & Architectural Realignment
- **Primary Platform Purpose:** Standardized application identity as **CardioSense — ECG Insight Lab: Reproducible Machine Learning and Signal Analysis for ECG Data**. Positioned strictly as an educational and research prototype evaluating public benchmarks.
- **Active Workflow Normalization:** Streamlined primary product navigation to exactly three verified, active workflows:
  1. `12-Lead ECG Classification (PTB-XL)`
  2. `Single-Lead Beat Analysis (MIT-BIH)`
  3. `Hardware Prototype — Offline AD8232 Simulation`
- **Optical Paper Digitizer Deprecation & Archival:**
  - Removed the digitizer from primary tabs, homepage cards, and guided demonstration steps.
  - Relocated module into a designated collapsible archival section (`Archived Research Module: Optical Paper Digitizer`) preserved below the primary experimental labs.
  - Added transparent disclosure: archived due to unproven generalization on real-world hospital paper scans ($r=0.6932$ benchmark was established on synthetic mathematical fixtures only).
  - Maintained complete source code, adapter protocols, and 11 unit tests (`tests/test_digitization.py`) without broken imports or dead links.

### Scientific Research & Metric Integrity
- **Controlled Class-Imbalance Research Experiment:**
  - Formulated research question: *Can class-imbalance handling improve minority-class beat classification on unseen subjects without materially degrading performance on common classes?*
  - Implemented `scripts/run_imbalance_experiment.py` evaluating 4 candidate architectures: Baseline Balanced RF, Balanced Subsample RF, Balanced Logistic Regression, and Minority-Boosted Cost-Sensitive RF.
  - Enforced strict leakage-free protocol: models fit exclusively on Train ($N=37,419$), selected on Validation ($N=15,725$), and evaluated once on held-out Test ($N=47,561$).
  - Finding: Minority boosting increased Class S recall from $20.43\%$ to $27.10\%$ ($+6.68\%$) and Class S F1 from $0.2712$ to $0.3245$, with only a minor $-0.0049$ drop in majority Class N F1.
  - Documented complete experiment in `docs/reports/single_lead_imbalance_experiment.md` and `artifacts/single_lead_ml/experiments/class_imbalance/`.
  - Preserved frozen production model (`single_lead_rf_model.joblib`) unchanged at SHA-256 `3889a5f5...`.
- **Feature Count Reconciliation & Schema Integrity:**
  - Audited and verified feature vectors: 148 features for Workflow B (12 leads $\times$ 12 per-lead metrics + 4 global metrics = 148); 47 features for Workflow C (5 timing/RR + 10 global morphology + 12 subsegments + 20 contour points = 47).
  - Corrected historical typographical error in `hardware/reports/simulation_inference_validation.md` from "50 features" to "47 features".
  - Added automated test `test_feature_schema_breakdown_reconciliation` in `tests/test_single_lead_ml.py` verifying exact mathematical breakdown.
- **Dual Single-Lead Benchmark Disclosures:**
  - Transparently distinguished Oracle/Reference-Beat metrics ($N=47,561$, Macro F1 = 0.5396) from Matched-Detection Conditional metrics ($N=47,034$, Macro F1 = 0.5274).
  - Separately reported Pan-Tompkins detector false positives (115 peaks) and missed reference beats (528 beats) from classifier confusion matrices.
  - Disclosed project-defined 4-class mapping merging ventricular fusion beats (`F`) into `Q` (not an unqualified 5-class AAMI EC57 implementation).

### Hardware Simulation Hardening
- **Labeling & UI Disclosures:** Renamed Tab 3 to `Hardware Prototype — Offline AD8232 Simulation`.
- **Persistent Notices:** Added persistent warning: *Physical hardware has not been tested. The ECG, ADC readings, and reconstructed voltage on this screen are simulated.*
- **Biomedical Electrical Safety:** Added explicit safety warning noting lack of galvanic isolation on standard USB/microcontroller circuits (IEC 60601-1) and strictly prohibiting human electrode connections.
- **Unit Qualification:** Explicitly distinguished raw integer ADC counts from reconstructed voltage, labeling voltage as `mV-equivalent (simulated)`.

### Security, Secrets & Explanations
- **Groq Model Standardization:** Updated default model ID to officially supported `llama-3.3-70b-versatile` with deterministic offline fallback when API key is unconfigured.
- **Secret Scrubbing:** Regex error sanitizer prevents API keys (`gsk_***`) or Bearer tokens from appearing in exception traces.
- **Data Privacy:** Guaranteed zero transmission of raw patient records, waveforms, or document scans.

### Continuous Integration & Testing
- **GitHub Actions CI:** Created `.github/workflows/ci.yml` running automated pytest suite on pushes and pull requests.
- **Test Suite Expansion:** Expanded test suite to **196 passing automated tests** (including UI AppTest verification, feature count breakdown assertions, and imbalance experiment artifact validation).
- **Graceful Dataset Decoupling:** Added `@mitdb_required` marker to allow offline unit tests to run fast without requiring large multi-gigabyte dataset downloads on CI runners.
