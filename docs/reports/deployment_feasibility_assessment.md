# CardioSense — Phase 4 Deployment Feasibility & Architecture Assessment

**Date:** 2026-10-17  
**Platform Version:** 1.0.0 (Research Release)  
**Target Environment:** Local-First Academic Workstation / Headless Demo Container  
**Runtime:** Python 3.14.4 virtual environment  
**Author:** Senior Software Engineer & Technical Auditor (Antigravity)  

---

## Executive Summary

This deployment feasibility assessment evaluates the operational, computational, licensing, and privacy boundaries for deploying the **CardioSense / ECG Insight Lab** platform. 

The primary recommendation of this assessment is an **architecturally enforced Local-First deployment model**. For hosted demonstration environments, the application is strictly restricted to **synthetic demo fixtures** and preloaded public reference data. In accordance with clinical research ethics, no public cloud endpoint accepting real patient ECGs may be deployed without an independent clinical institutional review, encryption at rest, and HIPAA/GDPR compliance certification.

---

## 1. Runtime Environment & Dependency Compatibility

### 1.1 Python Version Support
- **Active Environment:** Python 3.14.4 on Linux x86_64.
- **Compatibility Matrix:**
  - Standard scientific stack (`numpy>=2.0.0`, `scipy>=1.14.0`, `scikit-learn>=1.5.0`, `pandas>=2.2.0`, `pillow>=10.4.0`) is fully functional.
  - UI framework (`streamlit>=1.38.0`, `plotly>=5.24.0`) operates cleanly with zero deprecated API warnings.
  - WFDB parsing (`wfdb>=4.1.2`) operates natively on binary `.hea` and `.dat` signals.
- **Packaging:** Dependencies are pinned in `requirements.txt`. A fresh checkout can construct the operational virtual environment via `pip install -r requirements.txt`.

### 1.2 Resource Footprint & System Sizing

| Metric | Measured Value | Hosted Container Recommendation |
| :--- | :--- | :--- |
| **Peak Memory (Feature Extraction & Training)** | **515.25 MiB RSS** | 1.0 GiB RAM minimum |
| **Active Streamlit UI Idle Memory** | **~85.0 MiB RSS** | 512 MiB RAM minimum |
| **Waveform Digitization Latency (CPU)** | **~210 ms** per 12-lead scan | 1 vCPU adequate |
| **Waveform ML Inference Latency (CPU)** | **~1.2 ms** per 100 Hz record | 1 vCPU adequate |
| **Full Dataset On-Disk Footprint (21,799 records)** | **~520 MiB** (raw signals) | 2.0 GiB disk storage |
| **Feature Cache Compressed (`.npz` + `.json`)** | **~26.0 MiB** | 100 MiB disk storage |
| **Model Artifact + Manifest** | **~12.0 KiB** | Negligible |

*Conclusion:* The platform is exceptionally lightweight and can easily execute on standard personal workstations, entry-level cloud VMs (e.g., AWS t3.small, GCP e2-small), or containerized serverless runtimes.

---

## 2. Privacy Safeguards & Data Lifecycle

### 2.1 In-Memory Upload Lifecycle
To eliminate accidental patient data leakage or unintended persistence:
1. **Zero Disk Persistence:** User uploads to Workflow A (paper ECG scans) are received into volatile memory buffers (`BytesIO`). They are validated, processed by the morphological digitizer, visualized in Plotly, and released. At no point is an uploaded file written to disk or stored in temporary directories.
2. **Zero Logging of Physiological Data:** Log files and stdout streams log only operational metadata (dimensions, channel count, lead recovery counts, execution runtime in milliseconds). Raw image bytes, pixel arrays, and signal floats are excluded from system logs.
3. **Session Containment:** Uploaded data exists solely within the user's active browser session and is purged upon page refresh or session termination.

### 2.2 Network Boundary & External Telemetry
- **Zero Telemetry:** The platform contains zero third-party telemetry beacons, analytics trackers, or external API calls.
- **Local Execution:** Model inference and morphological digitization execute 100% locally on the host CPU. No waveform or scan data leaves the host perimeter.

---

## 3. Model Artifact Delivery & Cryptographic Integrity

### 3.1 Trust Model & Delivery Architecture
Because Python pickle/joblib deserialization carries security risks if artifacts are untrusted or tampered with:
1. **Legacy Quarantine:** Historical models (`legacy/finalmodel.pkl`) remain strictly quarantined. The platform's loader rejects them with `PermissionError`.
2. **Verified Manifest Anchor:** The active model artifact (`waveform_lr_model.joblib`) is accompanied by `waveform_lr_model.joblib.manifest.json`.
3. **Cryptographic Checksum:**
   - Active Model SHA-256: `55d364ed57cc971167f7ad813443c249f792567e406cce4c3dd84188829328d7`
   - Pilot Model SHA-256: `850a26e38c46f5ae60af0a2157732ffed59e33b9e77f98ed63404cc85a9eb8ac`
4. **Delivery Recommendation:** Due to its minimal footprint (10.5 KB), the verified joblib model can be distributed directly as a release attachment or generated deterministically using `python -m ecg_lab.waveform_ml.cli train`. If absent, the application displays an informative "Model Unavailable" diagnostic state rather than fabricating predictions.

---

## 4. Licensing, Data Provenance & Attribution Obligations

| Component | Source / Citation | License | Permitted Commercial / Academic Use | Attribution Requirement |
| :--- | :--- | :--- | :--- | :--- |
| **PTB-XL v1.0.3 Dataset** | Wagner et al., PhysioNet (2022) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Academic and commercial use permitted | Must cite Wagner et al. and link to CC BY 4.0 |
| **Open-ECG Digitizer Adapter** | Ahus-AIM (2023) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | ShareAlike required if adapter code is distributed | Documented behind isolated protocol adapter |
| **Synthetic Test Assets** | CardioSense Internal | MIT License | Permitted | Documented as synthetic fixtures |
| **Platform Codebase** | CardioSense Repository | MIT License | Permitted | Preserved in repository |

---

## 5. Deployment Options & Execution Checklist

### Option A: Local-First Research Installation (Recommended)
```bash
# 1. Clone repository
git clone <repo-url> && cd code

# 2. Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Verify system health & run tests
pytest tests/ -v
python -m ecg_lab.waveform_ml.cli diagnostics

# 4. Launch research interface
streamlit run app.py
```

### Option B: Hosted Demonstration Mode
If deployed to a cloud platform (e.g., Streamlit Community Cloud, Hugging Face Spaces, GCP Cloud Run):
- **Safety Rule:** Public deployments must default to `sample_data/synthetic_12lead.png` or preloaded PTB-XL public reference records.
- **Disabled Features:** File uploads from unknown public users must display the non-clinical disclaimer and require explicit user acknowledgment that uploaded documents contain no protected health information.
- **Resource Limits:** Configure container memory ceiling to $\ge 1.0\,\text{GiB}$ and timeout to $60\,\text{s}$.

---
*Assessment approved for Phase 4 release.*
