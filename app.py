"""ECG Insight Lab — Research Platform Web Application.

Editorial academic interface designed according to the Quiet Academic Editorial
aesthetic specification (restrained typography, warm ivory palette, research focus).
"""

import streamlit as st
from PIL import Image

from ecg_lab.config import (
    ColorPalette,
    TypographySystem,
    ValidationLimits,
    DISCLAIMER_RESEARCH_ONLY,
    DISCLAIMER_HARDWARE_ABSENT,
)
from ecg_lab.validation import validate_image_bytes, validate_image_file


# ==============================================================================
# Page Configuration & Editorial CSS Injection
# ==============================================================================

st.set_page_config(
    page_title="ECG Insight Lab",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

def inject_editorial_theme():
    """Inject typography and color tokens adhering to Quiet Academic Editorial design."""
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap');

    /* Global Typography & Palette */
    html, body, [class*="st-"] {{
        font-family: {TypographySystem.FONT_BODY};
        color: {ColorPalette.TEXT_PRIMARY};
    }}

    .stApp {{
        background-color: {ColorPalette.BG_PRIMARY};
    }}

    /* Editorial Headings */
    h1, h2, h3, h4 {{
        font-family: {TypographySystem.FONT_HEADINGS} !important;
        color: {ColorPalette.TEXT_PRIMARY} !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
        line-height: {TypographySystem.LINE_HEIGHT_HEADINGS};
    }}

    h1 {{
        font-size: 2.25rem !important;
        margin-bottom: 0.5rem !important;
    }}

    h2 {{
        font-size: 1.65rem !important;
        border-bottom: 1px solid {ColorPalette.BORDER_DIVIDER};
        padding-bottom: 0.4rem;
        margin-top: 1.5rem !important;
    }}

    h3 {{
        font-size: 1.25rem !important;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {ColorPalette.SURFACE_INSET};
        border-right: 1px solid {ColorPalette.BORDER_DIVIDER};
    }}

    /* Editorial Containers & Cards */
    div[data-testid="stMetric"], .academic-card {{
        background-color: #FFFFFF;
        border: 1px solid {ColorPalette.BORDER_DIVIDER};
        border-radius: 4px;
        padding: 1.2rem;
        box-shadow: none !important;
    }}

    /* Code & Technical Data */
    code, pre, .technical-mono {{
        font-family: {TypographySystem.FONT_MONO} !important;
        background-color: {ColorPalette.SURFACE_INSET} !important;
        color: {ColorPalette.TEXT_PRIMARY} !important;
        border-radius: 3px;
    }}

    /* Editorial Badges & Metadata */
    .meta-label {{
        font-family: {TypographySystem.FONT_BODY};
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {ColorPalette.TEXT_SECONDARY};
    }}

    .disclaimer-banner {{
        background-color: {ColorPalette.SURFACE_INSET};
        border-left: 3px solid {ColorPalette.ACCENT_PRIMARY};
        padding: 0.8rem 1.2rem;
        margin: 1.0rem 0;
        font-size: 0.9rem;
        color: {ColorPalette.TEXT_SECONDARY};
    }}

    .watermark-demo {{
        display: inline-block;
        border: 1px solid {ColorPalette.ACCENT_PRIMARY};
        color: {ColorPalette.ACCENT_PRIMARY};
        font-size: 0.75rem;
        padding: 0.15rem 0.5rem;
        border-radius: 3px;
        font-weight: 500;
        letter-spacing: 0.04em;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_editorial_theme()


# ==============================================================================
# Sidebar Navigation
# ==============================================================================

with st.sidebar:
    st.markdown("### ECG Insight Lab")
    st.markdown("<span class='meta-label'>Research Prototype · v1.0.0</span>", unsafe_allow_html=True)
    st.markdown("---")

    nav_choice = st.radio(
        "Navigation",
        [
            "Research Overview",
            "ECG Image Lab",
            "Waveform ML Lab",
            "Sample Benchmark Data",
            "Governance & Ethics",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        f"<div class='disclaimer-banner'><strong>Notice:</strong> {DISCLAIMER_RESEARCH_ONLY}</div>",
        unsafe_allow_html=True,
    )


# ==============================================================================
# Page: Research Overview
# ==============================================================================

if nav_choice == "Research Overview":
    st.markdown("# ECG Image Digitization & Waveform Analysis Platform")
    st.markdown(
        "<p style='font-size: 1.15rem; color: #6F716B; line-height: 1.6;'>"
        "An open-access academic framework evaluating the feasibility, computational boundaries, "
        "and validation requirements of digitizing paper electrocardiograms and benchmark cardiac machine learning."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='disclaimer-banner'><strong>System Scope:</strong> {DISCLAIMER_HARDWARE_ABSENT}</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Image Extraction Architecture", "12 Leads", "Morphological & U-Net")
    with col2:
        st.metric("Benchmark Dataset", "PTB-XL v1.0.3", "21,801 Records")
    with col3:
        st.metric("Evaluation Protocol", "Stratified Patient Folds", "Zero Leakage")

    st.markdown("## Architectural Principles")
    st.markdown(
        """
        1. **Strict Subsystem Decoupling:** Workflow A (Paper ECG Image Digitization) and Workflow B 
           (Digital Waveform ML) operate as independent research labs. Paper image signals are not passed to 
           classifiers without empirical calibration validation.
        2. **Reproducibility Over Decoration:** All training splits, feature configurations, and validation 
           metrics are strictly derived from documented benchmarks without fabricated performance claims.
        3. **Honest Limitations:** Physical hardware claims are explicitly disclaimed. Prerecorded benchmark 
           waveforms are visibly watermarked as demonstration data.
        """
    )


# ==============================================================================
# Page: ECG Image Lab
# ==============================================================================

elif nav_choice == "ECG Image Lab":
    st.markdown("# Workflow A: ECG Image Ingestion & Validation")
    st.markdown(
        "Upload scanned or photographed 12-lead electrocardiograms to inspect structural fidelity, "
        "dimensions, color channels, and lead integrity."
    )

    uploaded_file = st.file_uploader(
        "Select ECG Scan (JPEG, PNG, TIFF)",
        type=list(ValidationLimits.ALLOWED_EXTENSIONS),
        help="Maximum file size: 25 MB. Minimum resolution: 400x300.",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        validation = validate_image_bytes(file_bytes, filename=uploaded_file.name)

        if validation.is_valid:
            st.success(f"✓ {validation.message}")
            meta = validation.metadata

            # Metadata Display
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Dimensions", f"{meta.width} × {meta.height}")
            c2.metric("Color Mode", f"{meta.mode} ({meta.channels} ch)")
            c3.metric("Aspect Ratio", f"{meta.aspect_ratio}")
            c4.metric("File Size", f"{round(meta.size_bytes / 1024, 1)} KB")

            # Warnings
            if validation.warnings:
                for w in validation.warnings:
                    st.warning(f"Advisory: {w}")

            # Image Preview
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded ECG Scan: {uploaded_file.name}", use_container_width=True)
        else:
            st.error(f"Validation Failed [{validation.status.value}]: {validation.message}")

    else:
        st.info("No file uploaded. Select an image file or navigate to 'Sample Benchmark Data' to inspect sample inputs.")


# ==============================================================================
# Page: Waveform ML Lab
# ==============================================================================

elif nav_choice == "Waveform ML Lab":
    st.markdown("# Workflow B: Digital Waveform ML Lab")
    st.markdown(
        "Independent machine learning benchmark on PhysioNet's **PTB-XL v1.0.3** dataset "
        "(10-second 12-lead digital electrocardiograms from 18,869 patients)."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Benchmark Protocol")
        st.markdown(
            """
            - **Target Taxonomy:** Multilabel classification of 5 diagnostic superclasses:
              - `NORM`: Normal ECG
              - `MI`: Myocardial Infarction
              - `STTC`: ST/T Change
              - `CD`: Conduction Disturbance
              - `HYP`: Hypertrophy
            - **Data Representation:** 100 Hz downsampled WFDB recordings (`records100/`).
            - **Split Integrity:** Official `strat_fold` column (folds 1–8 train, fold 9 val, fold 10 test) 
              ensuring zero patient overlap across splits.
            """
        )
    with col2:
        st.markdown("### Scope Boundary")
        st.markdown(
            """
            > [!IMPORTANT]
            > **Digital Signals vs. Paper Scans:** PTB-XL provides native digital voltages ($\mu\text{V}$), 
            > not scanned paper photographs. This research lab does not claim that a model trained on PTB-XL 
            > can accurately diagnose raw camera photos without a validated end-to-end calibration bridge.
            """
        )


# ==============================================================================
# Page: Sample Benchmark Data
# ==============================================================================

elif nav_choice == "Sample Benchmark Data":
    st.markdown("# Sample & Integration Test Assets")
    st.markdown(
        "Deterministic benchmark assets used for automated unit testing and offline demonstrations. "
        "<span class='watermark-demo'>DEMO / NON-CLINICAL</span>",
        unsafe_allow_html=True,
    )

    sample_path = "sample_data/synthetic_sample.png"
    result = validate_image_file(sample_path)

    if result.is_valid:
        img = Image.open(sample_path)
        st.image(img, caption="Synthetic 1200x800 Test ECG Pattern (sample_data/synthetic_sample.png)", use_container_width=True)
        st.json(result.to_dict())
    else:
        st.error(f"Sample asset verification failed: {result.message}")


# ==============================================================================
# Page: Governance & Ethics
# ==============================================================================

elif nav_choice == "Governance & Ethics":
    st.markdown("# Governance, Licensing & Attribution")
    st.markdown(
        """
        ### Data & Code Licenses
        - **PhysioNet PTB-XL v1.0.3:** Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://physionet.org/content/ptb-xl/1.0.3/).
        - **Open ECG Digitizer Candidate:** Licensed under [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://github.com/Ahus-AIM/Open-ECG-Digitizer). Maintained strictly behind an external adapter interface.
        - **Scikit-Learn Model Persistence:** Legacy models are quarantined under `legacy/` following [official Scikit-Learn security guidance](https://scikit-learn.org/stable/model_persistence.html).

        ### Research Integrity
        - **No Physical Sensors:** No live microcontroller telemetry is connected.
        - **Privacy Assurance:** User uploads are validated in memory and are not permanently recorded or transmitted to third-party APIs.
        """
    )
