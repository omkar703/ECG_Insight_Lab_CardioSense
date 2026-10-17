# Phase 2 Feasibility Assessment: ECG Image Digitizer

**Date of Assessment:** October 17, 2026  
**Auditor / Engineer:** Senior Technical Auditor & Product Engineer (Antigravity)  
**Project Workspace:** `Project_expo/code/`  

---

## 1. Context & Objectives

The purpose of Phase 2 is to deliver the first functional end-to-end user workflow:
$$\text{ECG Image Ingestion} \longrightarrow \text{Validation} \longrightarrow \text{Trace Digitization} \longrightarrow \text{Waveform Visualization} \longrightarrow \text{Data Export}$$

Before implementing digitization logic, this feasibility gate evaluates the preferred upstream candidate ([`Ahus-AIM/Open-ECG-Digitizer`](https://github.com/Ahus-AIM/Open-ECG-Digitizer)) against local machine constraints, licensing obligations, and runtime requirements.

---

## 2. Technical Evaluation of Open-ECG-Digitizer

| Evaluation Criteria | Upstream Specification | Local Environment Status | Feasibility Verdict |
| :--- | :--- | :--- | :--- |
| **Python Runtime** | Requires Python 3.12+ | Python 3.14.4 installed (`/usr/bin/python3.14`) | **INCOMPATIBLE / HIGH RISK:** PyTorch 2.x and Ray wheels are not published for Python 3.14 on Linux x86_64. |
| **Model Weights** | Pretrained U-Net weights stored via Git LFS (32.3 MB total) | `git lfs` is **not installed** (`git: 'lfs' is not a git command`) | **BLOCKED:** Cloning or fetching weights yields 133-byte LFS pointer files rather than PyTorch `.pt` model weights. |
| **Heavy Dependencies** | `torch`, `torchvision`, `torchaudio`, `ray[data,train,tune]`, `tensorboard`, `yacs` | Light virtualenv with `numpy`, `scipy`, `pandas`, `pillow`, `scikit-learn`, `streamlit`, `plotly` | **BLOCKED:** Ray cluster orchestration and PyTorch are omitted from the project to avoid dependency bloat and installation failure. |
| **Available RAM** | Recommended $\ge 8$ GB system RAM for Ray + PyTorch segmentation | 7.0 GiB RAM total, ~1.2 GiB available (Swap: 2.2 GiB used) | **HIGH RISK:** Running PyTorch deep learning alongside Streamlit and Ray in 1.2 GiB free RAM triggers OOM kills. |
| **GPU / VRAM** | Tested on CUDA with Linux | NVIDIA GTX 1650 Ti Mobile (4,096 MiB VRAM), CUDA 13.2 | **MARGINAL:** 4 GB VRAM is shared with desktop UI (`gnome-shell`); marginal for Ray distributed workers. |
| **Licensing Terms** | Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) | Project codebase is MIT-compatible open research | **COPYLEFT RESTRICTION:** Cannot be vendored into the project tree; must be executed strictly as an isolated subprocess or external service. |

---

## 3. Decision & Architectural Route

### Primary Decision: Gated Adapter + Classical Baseline

Attempting to install PyTorch, Ray, and Git LFS on the bleeding-edge Python 3.14 host environment will fail due to missing binary wheels and host memory exhaustion.

Therefore, following the directives in `phase_2.md`, the project adopts the **Gated Adapter + Classical Morphological Baseline** route:

```mermaid
flowchart TD
    ImgIn["Input Image Scan / Upload"] --> Val["Validation Module (`ecg_lab.validation`)"]
    Val -->|Invalid| Err["Structured Validation Error & Disclosures"]
    Val -->|Valid| Router{"Digitizer Router (`ecg_lab.digitization.service`)"}
    
    Router -->|Preferred Route (If Configured)| Upstream["`OpenECGDigitizerAdapter`<br/>(Checks environment, binaries & weights)"]
    Upstream -->|Weights/Deps Missing| Graceful["Graceful Status: UNAVAILABLE<br/>Emits instructions to install Git LFS & Python 3.12"]
    
    Router -->|Active Native Route| Baseline["`ClassicalBaselineDigitizer`<br/>(Native NumPy/SciPy/Pillow Pipeline)"]
    Baseline --> Out["`DigitizationResult`<br/>(Waveforms, Explicit Units, Calibration Metadata)"]
```

### Strategic Safeguards & Contract Guarantees
1. **Stable Protocol Interface (`ECGDigitizer`):**  
   Both the `OpenECGDigitizerAdapter` and the `ClassicalBaselineDigitizer` implement the exact same protocol:
   `digitize(image: Image.Image, metadata: ImageMetadata, options: Optional[dict]) -> DigitizationResult`
2. **Honest Calibration Metadata:**  
   Amplitudes from the classical baseline are tagged as `AmplitudeUnit.RELATIVE` and `CalibrationStatus.UNRECOVERED`. The system **never** fabricates millivolts or seconds without physical grid calibration.
3. **No Blind Contour Assumption:**  
   The classical baseline avoids the legacy flaw of selecting `sorted(contours)[0]`. It employs morphological grid suppression, lead window localization, and column-wise intensity tracking to extract continuous 1D signals while rejecting blank or noisy leads.
4. **Zero Diagnostic Coupling:**  
   No classifier is connected. Digitized signals are displayed interactively and exported as CSV/JSON, with no automated diagnostic claims.
