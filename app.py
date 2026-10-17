"""ECG Insight Lab — Research Platform Web Application.

Editorial academic interface designed according to the Quiet Academic Editorial
aesthetic specification (restrained typography, warm ivory palette, research focus).
"""

from io import BytesIO
from pathlib import Path
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import pandas as pd

from ecg_lab.config import (
    ColorPalette,
    TypographySystem,
    ValidationLimits,
    DISCLAIMER_RESEARCH_ONLY,
    DISCLAIMER_HARDWARE_ABSENT,
)
from ecg_lab.validation import validate_image_bytes, validate_image_file
from ecg_lab.digitization import (
    digitize_ecg_image,
    DigitizationStatus,
    CalibrationStatus,
    AmplitudeUnit,
    LeadSignal,
)


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
        font-size: 2.2rem !important;
        margin-bottom: 0.5rem !important;
    }}

    h2 {{
        font-size: 1.55rem !important;
        border-bottom: 1px solid {ColorPalette.BORDER_DIVIDER};
        padding-bottom: 0.4rem;
        margin-top: 1.5rem !important;
    }}

    h3 {{
        font-size: 1.2rem !important;
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
        padding: 1.0rem;
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


def create_lead_chart(lead_name: str, lead_sig: LeadSignal) -> go.Figure:
    """Create a crisp Plotly chart for a single digitized lead waveform."""
    fig = go.Figure()

    x_vals = lead_sig.sample_indices
    y_vals = lead_sig.samples

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            line=dict(color="#1B2022", width=1.75),
            name=f"Lead {lead_name}",
            hovertemplate="Index: %{x}<br>Relative Amp: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Lead {lead_name}</b> (Confidence: {lead_sig.confidence:.2f})",
            font=dict(family="IBM Plex Sans", size=13, color="#252722"),
            x=0.03,
            y=0.92,
        ),
        margin=dict(l=35, r=15, t=32, b=28),
        height=190,
        plot_bgcolor="#FAF9F5",
        paper_bgcolor="#FFFFFF",
        xaxis=dict(
            title=dict(text="Sample Index", font=dict(size=10, color="#6F716B")),
            showgrid=True,
            gridcolor="#EBEAE4",
            zeroline=False,
            tickfont=dict(size=9, color="#6F716B"),
        ),
        yaxis=dict(
            title=dict(text="Rel. Amp", font=dict(size=10, color="#6F716B")),
            showgrid=True,
            gridcolor="#EBEAE4",
            zeroline=True,
            zerolinecolor="#DCDDD5",
            range=[-1.15, 1.15],
            tickfont=dict(size=9, color="#6F716B"),
        ),
    )
    return fig


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
        st.metric("Image Extraction Architecture", "12 Leads", "Morphological Baseline")
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
    st.markdown("# Workflow A: ECG Image Ingestion & Digitization")
    st.markdown(
        "Upload scanned or photographed 12-lead electrocardiograms to inspect structural fidelity, "
        "extract continuous 1D waveforms across all 12 leads, and export validated data."
    )

    input_mode = st.radio(
        "Input Selection Mode",
        ["Upload Local ECG Scan", "Preloaded Deterministic Benchmark", "Historical Reference Scan"],
        horizontal=True,
    )

    active_bytes = None
    active_filename = None

    if input_mode == "Upload Local ECG Scan":
        uploaded_file = st.file_uploader(
            "Select ECG Scan (JPEG, PNG, TIFF)",
            type=list(ValidationLimits.ALLOWED_EXTENSIONS),
            help="Maximum file size: 25 MB. Minimum resolution: 400x300.",
        )
        if uploaded_file is not None:
            active_bytes = uploaded_file.read()
            active_filename = uploaded_file.name

    elif input_mode == "Preloaded Deterministic Benchmark":
        bench_path = Path("sample_data/synthetic_12lead.png")
        if bench_path.is_file():
            active_bytes = bench_path.read_bytes()
            active_filename = bench_path.name
            st.info("Loaded deterministic synthetic 12-lead benchmark pattern (`synthetic_12lead.png`).")

    else:
        hist_path = Path("legacy/test_image.jpg")
        if hist_path.is_file():
            active_bytes = hist_path.read_bytes()
            active_filename = hist_path.name
            st.info("Loaded historical clinical scan template (`legacy/test_image.jpg`).")

    if active_bytes is not None:
        validation = validate_image_bytes(active_bytes, filename=active_filename)

        if validation.is_valid:
            meta = validation.metadata

            # Metadata Display
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Dimensions", f"{meta.width} × {meta.height}")
            c2.metric("Color Mode", f"{meta.mode} ({meta.channels} ch)")
            c3.metric("Aspect Ratio", f"{meta.aspect_ratio}")
            c4.metric("File Size", f"{round(meta.size_bytes / 1024, 1)} KB")

            if validation.warnings:
                for w in validation.warnings:
                    st.warning(f"Advisory: {w}")

            with st.expander("Inspect Source Image", expanded=False):
                img = Image.open(BytesIO(active_bytes))
                st.image(img, caption=f"Source Document: {active_filename}", use_container_width=True)

            # Digitization Action Trigger
            st.markdown("### Waveform Extraction Engine")
            col_engine, col_btn = st.columns([2, 1])
            with col_engine:
                engine_choice = st.selectbox(
                    "Digitization Method",
                    ["Classical Morphological Baseline", "Open-ECG-Digitizer (Upstream Adapter)"],
                    help="Classical baseline uses native CPU centroid tracking. Open-ECG adapter checks upstream runtime availability.",
                )
            with col_btn:
                st.write("")
                st.write("")
                run_btn = st.button("Digitize 12-Lead Waveform", type="primary")

            if run_btn:
                with st.spinner("Processing lead regions and tracing continuous signals..."):
                    engine_key = "open-ecg" if "Upstream" in engine_choice else "baseline"
                    result = digitize_ecg_image(active_bytes, engine=engine_key)

                st.markdown("---")

                if result.status in (DigitizationStatus.SUCCESS, DigitizationStatus.PARTIAL):
                    st.success(
                        f"✓ Digitization Complete [{result.status.value}] — "
                        f"Recovered {result.lead_count} leads in {result.runtime_ms:.1f} ms."
                    )

                    # Diagnostic Calibration Notice
                    st.markdown(
                        f"<div class='disclaimer-banner'>"
                        f"<strong>Calibration Status:</strong> {result.calibration_status.value} · "
                        f"<strong>Amplitude Units:</strong> {result.amplitude_unit.value}<br>"
                        f"<em>Notice: Signals are in relative normalized coordinates. "
                        f"Physical millivolt and millisecond calibration is not claimed without automated grid line detection.</em>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    if result.warnings:
                        for rw in result.warnings:
                            st.warning(rw)

                    # Plotly Waveform Visualization Grid (3 columns x 4 rows)
                    st.markdown("### Reconstructed 12-Lead Waveforms")
                    lead_order = ["I", "aVR", "V1", "V4", "II", "aVL", "V2", "V5", "III", "aVF", "V3", "V6"]

                    for r in range(0, len(lead_order), 3):
                        cols = st.columns(3)
                        for c_idx in range(3):
                            l_idx = r + c_idx
                            if l_idx < len(lead_order):
                                lname = lead_order[l_idx]
                                with cols[c_idx]:
                                    if lname in result.leads and result.leads[lname].is_valid:
                                        fig = create_lead_chart(lname, result.leads[lname])
                                        st.plotly_chart(fig, use_container_width=True, key=f"plot_{lname}")
                                    else:
                                        st.error(f"Lead {lname}: Missing or unrecovered.")

                    # Data Export Area
                    st.markdown("### Export Digitized Waveforms")
                    dcol1, dcol2, dcol3 = st.columns(3)

                    # Long-form CSV
                    long_csv = result.to_long_dataframe().to_csv(index=False)
                    dcol1.download_button(
                        label="Download Long-Form CSV",
                        data=long_csv,
                        file_name=f"{Path(active_filename).stem}_long.csv",
                        mime="text/csv",
                    )

                    # Wide-form CSV
                    wide_csv = result.to_wide_dataframe().to_csv(index=False)
                    dcol2.download_button(
                        label="Download Wide-Form CSV",
                        data=wide_csv,
                        file_name=f"{Path(active_filename).stem}_wide.csv",
                        mime="text/csv",
                    )

                    # JSON Metadata Sidecar
                    json_meta = result.to_json(indent=2)
                    dcol3.download_button(
                        label="Download JSON Metadata",
                        data=json_meta,
                        file_name=f"{Path(active_filename).stem}_meta.json",
                        mime="application/json",
                    )

                elif result.status == DigitizationStatus.UNAVAILABLE:
                    st.warning(f"⚠️ Engine Unavailable: {result.error_message}")
                    for uw in result.warnings:
                        st.info(f"Guidance: {uw}")
                else:
                    st.error(f"Digitization Failed [{result.status.value}]: {result.error_message}")

        else:
            st.error(f"Validation Failed [{validation.status.value}]: {validation.message}")
    else:
        st.info("Select an input scan or choose a preloaded sample to begin digitization.")


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

    tab1, tab2 = st.tabs(["Synthetic 12-Lead Pattern", "Basic Test Grid"])

    with tab1:
        syn_path = "sample_data/synthetic_12lead.png"
        res_syn = validate_image_file(syn_path)
        if res_syn.is_valid:
            st.image(Image.open(syn_path), caption="Deterministic 12-Lead Fixture (sample_data/synthetic_12lead.png)", use_container_width=True)
            st.json(res_syn.to_dict())
            gt_df = pd.read_csv("sample_data/synthetic_12lead_gt.csv")
            st.markdown(f"**Ground Truth Reference Samples:** {len(gt_df)} rows across 12 leads.")
            st.dataframe(gt_df.head(10), use_container_width=True)

    with tab2:
        sample_path = "sample_data/synthetic_sample.png"
        result = validate_image_file(sample_path)
        if result.is_valid:
            img = Image.open(sample_path)
            st.image(img, caption="Synthetic 1200x800 Test ECG Pattern (sample_data/synthetic_sample.png)", use_container_width=True)
            st.json(result.to_dict())


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
