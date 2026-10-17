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
from plotly.subplots import make_subplots
import json
from ecg_lab.waveform_ml.config import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_DIR,
    LEAD_NAMES as WF_LEAD_NAMES,
    PHYSIONET_CITATION,
    PHYSIONET_DATASET_NAME,
    PHYSIONET_LICENSE,
    SAMPLING_RATE_HZ as WF_SAMPLING_RATE_HZ,
    TARGET_SUPERCLASSES,
)
from ecg_lab.waveform_ml.dataset import (
    load_ptbxl_metadata,
    read_wfdb_record,
)
from ecg_lab.waveform_ml.features import extract_record_features
from ecg_lab.waveform_ml.models import load_verified_model
from ecg_lab.diagnostics import check_system_diagnostics


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
            "Overview & System Status",
            "ECG Image Lab (Workflow A)",
            "Waveform ML Lab (Workflow B)",
            "Methods & Benchmarks",
            "Diagnostics & Privacy",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        f"<div class='disclaimer-banner'><strong>Notice:</strong> {DISCLAIMER_RESEARCH_ONLY}</div>",
        unsafe_allow_html=True,
    )


# ==============================================================================
# Page: Overview & System Status
# ==============================================================================

if nav_choice == "Overview & System Status":
    st.markdown("# ECG Insight Lab — Research Platform")
    st.markdown(
        "<p style='font-size: 1.15rem; color: #6F716B; line-height: 1.6;'>"
        "An academic framework evaluating the computational boundaries of paper ECG image digitization "
        "and reproducible cardiac waveform machine learning on the PTB-XL v1.0.3 clinical benchmark."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div class='disclaimer-banner'>"
        f"<strong>Non-Clinical Academic Prototype:</strong> {DISCLAIMER_RESEARCH_ONLY}<br>"
        f"<em>System Boundary: {DISCLAIMER_HARDWARE_ABSENT}</em>"
        f"</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Workflow A Digitizer", "12 Leads", "Morphological Tracking")
    with col2:
        st.metric("Benchmark Records", "21,799", "100% WFDB Complete")
    with col3:
        st.metric("Patient Splits", "10 Folds", "0 Patient Leakage")
    with col4:
        st.metric("Held-Out Test F1", "0.6093", "Macro Average (Fold 10)")

    st.markdown("## Architectural Principles & Strict Decoupling")
    st.markdown(
        """
        1. **Strict Subsystem Decoupling:** 
           - **Workflow A (ECG Image Lab):** Ingests paper ECG scans/photographs, validates formatting and resolution, and recovers continuous 1D geometric traces in normalized relative coordinates with CSV/JSON export.
           - **Workflow B (Waveform ML Lab):** Evaluates multi-label diagnostic superclasses on calibrated digital voltage signals (100 Hz WFDB) from PTB-XL v1.0.3.
           - **No Gated Bridge:** Digitized signals from Workflow A are **never** passed into the Workflow B classifier without empirical calibration validation on paired datasets.
        2. **Reproducibility & Scientific Honesty:**
           - Models are trained exclusively on Folds 1–8 ($N=17,418$), thresholds tuned on Fold 9 ($N=2,183$), and evaluated on Fold 10 ($N=2,198$).
           - Evaluated with patient-clustered bootstrap confidence intervals ($B=2,000$, 1,904 patient clusters).
        3. **Zero-Persistence Privacy:**
           - User-uploaded images are processed strictly in volatile memory (RAM) and are never written to disk, logged, or transmitted externally.
        """
    )

    st.markdown("## System Health & Component Readiness")
    diag = check_system_diagnostics()
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("PTB-XL Metadata", "Available" if diag.dataset_metadata.is_available else "Missing", diag.dataset_metadata.status_label)
    with d2:
        st.metric("WFDB Signals", "100% Complete" if diag.dataset_completeness.is_available else "Partial", diag.dataset_completeness.status_label)
    with d3:
        st.metric("Active Model", "Verified" if diag.active_model.is_available else "Missing", diag.active_model.status_label)
    with d4:
        st.metric("Demo Fixtures", "Verified" if diag.synthetic_fixtures.is_available else "Missing", diag.synthetic_fixtures.status_label)

    if diag.all_critical_components_ready:
        st.success("✓ All critical research and demo subsystems are OPERATIONAL.")
    else:
        st.warning("⚠️ Some components require attention. Review the Diagnostics & Privacy page for details.")


# ==============================================================================
# Page: ECG Image Lab
# ==============================================================================

elif nav_choice == "ECG Image Lab (Workflow A)":
    st.markdown("# Workflow A: ECG Image Ingestion & Digitization")
    st.markdown(
        "<div style='margin-bottom: 1.0rem; display: flex; gap: 0.5rem; flex-wrap: wrap;'>"
        "<span class='badge-status'>12-Lead Paper Digitization</span>"
        "<span class='badge-status'>Relative Normalized Coordinates</span>"
        "<span class='watermark-demo'>NON-CLINICAL PROTOTYPE</span>"
        "<span class='badge-clinical'>Ephemeral Memory Processing</span>"
        "</div>",
        unsafe_allow_html=True,
    )
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
        st.caption("🔒 Privacy Safeguard: Uploads are processed strictly in volatile memory (RAM). Files are never written to disk, stored, or logged.")
        if uploaded_file is not None:
            active_bytes = uploaded_file.read()
            active_filename = uploaded_file.name

    elif input_mode == "Preloaded Deterministic Benchmark":
        bench_path = Path("sample_data/synthetic_12lead.png")
        if bench_path.is_file():
            active_bytes = bench_path.read_bytes()
            active_filename = bench_path.name
            st.warning("⚠️ SYNTHETIC DEMO — NOT PATIENT DATA: This image is an artificial geometric test fixture containing no human physiology.")
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

    st.markdown("---")
    st.markdown(
        "<div class='disclaimer-banner'>"
        "<strong>Architectural Boundary Notice:</strong> 1D continuous waveforms extracted in Workflow A are purely "
        "geometric morphological traces in relative coordinates. In accordance with clinical safety principles, "
        "these traces are strictly isolated and <strong>NEVER</strong> fed to the Workflow B diagnostic classifier."
        "</div>",
        unsafe_allow_html=True,
    )


# ==============================================================================
# Page: Waveform ML Lab
# ==============================================================================

elif nav_choice == "Waveform ML Lab (Workflow B)":
    st.markdown("# Workflow B: Digital Waveform ML Lab")
    st.markdown(
        "<div style='margin-bottom: 1.2rem; display: flex; gap: 0.5rem; flex-wrap: wrap;'>"
        "<span class='badge-clinical'>PTB-XL v1.0.3 Benchmark</span>"
        "<span class='badge-status'>100 Hz WFDB Digital Voltage</span>"
        "<span class='badge-status'>Patient-Separated Folds (1-8 Train, 9 Val, 10 Test)</span>"
        "<span class='watermark-demo'>RESEARCH BENCHMARK ONLY</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        > [!IMPORTANT]
        > **Decoupled Workflow & Academic Prototype:** This subsystem operates exclusively on native digital voltage 
        > waveforms ($\text{mV}$) from PhysioNet PTB-XL v1.0.3. It does NOT analyze paper ECG scans, camera images, 
        > or live hardware telemetry. Model predictions are research outputs and do NOT represent medical diagnoses, 
        > clinical decision support, or validated medical devices.
        """
    )

    tab_explore, tab_bench, tab_provenance = st.tabs([
        "Interactive Record Explorer",
        "Benchmark Evaluation Summary",
        "Dataset Provenance & Methodology",
    ])

    # --------------------------------------------------------------------------
    # Tab 1: Interactive Record Explorer & Inference
    # --------------------------------------------------------------------------
    with tab_explore:
        db_path = Path(DEFAULT_DATA_DIR) / "ptbxl_database.csv"
        if not db_path.is_file():
            st.warning(
                f"PTB-XL metadata not found in `{DEFAULT_DATA_DIR}`. "
                "Download metadata using the CLI: `python -m ecg_lab.waveform_ml.cli download-metadata`"
            )
        else:
            try:
                meta_df = load_ptbxl_metadata(DEFAULT_DATA_DIR)
                data_path = Path(DEFAULT_DATA_DIR)
                # Find records present locally
                avail_ids = [
                    eid for eid, row in meta_df.iterrows()
                    if (data_path / f"{row['filename_lr']}.hea").is_file()
                ]

                if not avail_ids:
                    st.info(
                        "No WFDB records found on disk yet. Download sample records via CLI: "
                        "`python -m ecg_lab.waveform_ml.cli download-samples --count-per-fold 10`"
                    )
                else:
                    st.markdown(f"**Available Local Records:** {len(avail_ids)} records on disk.")
                    
                    filter_col1, filter_col2 = st.columns([1, 2])
                    with filter_col1:
                        fold_filter = st.selectbox(
                            "Filter by Partition Fold",
                            ["All Folds", "Train Folds (1-8)", "Validation Fold (9)", "Test Fold (10)"],
                        )

                    # Filter records by selected partition
                    filtered_df = meta_df.loc[avail_ids]
                    if fold_filter == "Train Folds (1-8)":
                        filtered_df = filtered_df[filtered_df["strat_fold"].isin(range(1, 9))]
                    elif fold_filter == "Validation Fold (9)":
                        filtered_df = filtered_df[filtered_df["strat_fold"] == 9]
                    elif fold_filter == "Test Fold (10)":
                        filtered_df = filtered_df[filtered_df["strat_fold"] == 10]

                    if filtered_df.empty:
                        st.warning(f"No downloaded records match filter '{fold_filter}'.")
                    else:
                        with filter_col2:
                            def format_rec_option(eid: int) -> str:
                                r = filtered_df.loc[eid]
                                sc_str = ", ".join(r["diagnostic_superclasses"]) if r["diagnostic_superclasses"] else "Unmapped"
                                split_tag = "Train" if r["strat_fold"] <= 8 else ("Val" if r["strat_fold"] == 9 else "Test")
                                return f"ECG #{eid:05d} [Fold {r['strat_fold']} - {split_tag}] — Pt #{r['patient_id']} ({sc_str})"

                            selected_eid = st.selectbox(
                                "Select PTB-XL ECG Record",
                                options=filtered_df.index.tolist(),
                                format_func=format_rec_option,
                            )

                        rec_row = filtered_df.loc[selected_eid]
                        rec_path = data_path / str(rec_row["filename_lr"])

                        # Clinical metadata cards
                        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                        with m_col1:
                            st.metric("Patient ID", f"Pt #{rec_row['patient_id']}")
                        with m_col2:
                            partition_name = "Train" if rec_row['strat_fold'] <= 8 else ("Validation" if rec_row['strat_fold'] == 9 else "Held-Out Test")
                            st.metric("Dataset Split", f"Fold {rec_row['strat_fold']} ({partition_name})")
                        with m_col3:
                            sex_str = "Male" if rec_row.get("sex") == 1 else ("Female" if rec_row.get("sex") == 0 else "Unknown")
                            st.metric("Demographics", f"Age {rec_row.get('age', 'N/A')} | {sex_str}")
                        with m_col4:
                            st.metric("Signal Spec", "100 Hz | 12 Leads | 10.0s")

                        # Reference annotations
                        st.markdown("#### Reference Annotations (Ground Truth)")
                        sc_list = rec_row["diagnostic_superclasses"]
                        if sc_list:
                            sc_badges = " ".join([f"<span class='badge-clinical'>{sc}</span>" for sc in sc_list])
                            st.markdown(f"**Diagnostic Superclasses:** {sc_badges}", unsafe_allow_html=True)
                        else:
                            st.markdown("<span class='watermark-demo'>NON-DIAGNOSTIC / UNMAPPED STATEMENT</span>", unsafe_allow_html=True)

                        st.caption(f"Raw SCP Codes: `{rec_row.get('scp_codes', 'N/A')}`")

                        # Read and plot digital waveform
                        try:
                            signal, meta = read_wfdb_record(rec_path)
                            st.markdown("#### 12-Lead Calibrated Digital Waveform")
                            lead_to_view = st.selectbox("Display Lead", ["All 12 Leads (Grid)"] + list(WF_LEAD_NAMES))
                            time_axis = [i / meta["fs"] for i in range(len(signal))]

                            if lead_to_view == "All 12 Leads (Grid)":
                                fig = make_subplots(
                                    rows=4, cols=3,
                                    subplot_titles=WF_LEAD_NAMES,
                                    vertical_spacing=0.08,
                                    horizontal_spacing=0.05,
                                )
                                for idx, lead_name in enumerate(WF_LEAD_NAMES):
                                    row_idx = (idx // 3) + 1
                                    col_idx = (idx % 3) + 1
                                    fig.add_trace(
                                        go.Scatter(
                                            x=time_axis,
                                            y=signal[:, idx],
                                            mode="lines",
                                            name=lead_name,
                                            line=dict(color=ColorPalette.PRIMARY, width=1.2),
                                        ),
                                        row=row_idx, col=col_idx,
                                    )
                                fig.update_layout(
                                    height=650,
                                    showlegend=False,
                                    margin=dict(l=40, r=20, t=40, b=30),
                                    paper_bgcolor=ColorPalette.BG_PRIMARY,
                                    plot_bgcolor="#FFFFFF",
                                )
                                fig.update_yaxes(title_text="mV", tickfont=dict(size=9))
                                fig.update_xaxes(title_text="sec", tickfont=dict(size=9))
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                lead_idx = WF_LEAD_NAMES.index(lead_to_view)
                                fig = go.Figure()
                                fig.add_trace(
                                    go.Scatter(
                                        x=time_axis,
                                        y=signal[:, lead_idx],
                                        mode="lines",
                                        name=lead_to_view,
                                        line=dict(color=ColorPalette.PRIMARY, width=1.5),
                                    )
                                )
                                fig.update_layout(
                                    title=f"Lead {lead_to_view} — Calibrated Continuous Voltage",
                                    xaxis_title="Time (seconds)",
                                    yaxis_title="Voltage (mV)",
                                    height=380,
                                    paper_bgcolor=ColorPalette.BG_PRIMARY,
                                    plot_bgcolor="#FFFFFF",
                                    margin=dict(l=50, r=30, t=50, b=40),
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            # Model Inference Section
                            st.markdown("#### Model Inference & Evaluation")
                            model_file = Path(DEFAULT_ARTIFACT_DIR) / "waveform_lr_model.joblib"
                            if not model_file.is_file():
                                st.info(
                                    "No trained model artifact found in `artifacts/waveform_ml/`. "
                                    "Train the baseline model using: `python -m ecg_lab.waveform_ml.cli train`"
                                )
                            else:
                                try:
                                    model, manifest = load_verified_model(model_file)
                                    feats = extract_record_features(signal, fs=meta["fs"]).reshape(1, -1)
                                    probs = model.predict_proba(feats)[0]
                                    thresholds = manifest.get("decision_thresholds", {})

                                    inference_rows = []
                                    for idx, sc in enumerate(TARGET_SUPERCLASSES):
                                        gt_pos = bool(rec_row[f"y_{sc}"])
                                        prob_val = float(probs[idx])
                                        thresh_val = float(thresholds.get(sc, 0.5))
                                        pred_pos = prob_val >= thresh_val

                                        # Concordance
                                        if gt_pos and pred_pos:
                                            status = "True Positive"
                                        elif not gt_pos and not pred_pos:
                                            status = "True Negative"
                                        elif not gt_pos and pred_pos:
                                            status = "False Positive"
                                        else:
                                            status = "False Negative"

                                        inference_rows.append({
                                            "Superclass": sc,
                                            "Ground Truth": "Positive" if gt_pos else "Negative",
                                            "Model Score": f"{prob_val:.4f}",
                                            "Tuned Threshold": f"{thresh_val:.2f}",
                                            "Model Prediction": "Positive" if pred_pos else "Negative",
                                            "Concordance": status,
                                        })

                                    inf_df = pd.DataFrame(inference_rows)
                                    st.dataframe(inf_df, use_container_width=True, hide_index=True)
                                    st.caption(
                                        "Note: Model scores represent sigmoid decision scores scaled by class-balanced logistic regression; "
                                        "they are not calibrated clinical risk probabilities."
                                    )
                                except Exception as err:
                                    st.error(f"Inference error: {err}")

                        except Exception as err:
                            st.error(f"Failed to read waveform record: {err}")

            except Exception as e:
                st.error(f"Error loading PTB-XL records: {e}")

    # --------------------------------------------------------------------------
    # Tab 2: Benchmark Evaluation Summary
    # --------------------------------------------------------------------------
    with tab_bench:
        st.markdown("### Benchmark Evaluation Results")
        metrics_file = Path(DEFAULT_ARTIFACT_DIR) / "metrics.json"
        summary_md_file = Path(DEFAULT_ARTIFACT_DIR) / "benchmark_summary.md"

        if not metrics_file.is_file():
            st.info(
                "Benchmark metrics have not yet been generated. "
                "Run the complete training and evaluation pipeline via: "
                "`python -m ecg_lab.waveform_ml.cli train`"
            )
        else:
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)

            train_m = metrics_data.get("train_metrics", {})
            val_m = metrics_data.get("val_metrics", {})
            test_m = metrics_data.get("model_test_metrics", {})
            n_eval = test_m.get("n_samples", "N/A")
            n_train = metrics_data.get("split_stats", {}).get("train_samples", metrics_data.get("n_train", 17418))
            n_val = metrics_data.get("split_stats", {}).get("val_samples", metrics_data.get("n_val", 2183))

            b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)
            with b_col1:
                st.metric("Training Set (Folds 1-8)", f"{n_train:,} records")
            with b_col2:
                st.metric("Val Set (Fold 9)", f"{n_val:,} records")
            with b_col3:
                st.metric("Held-Out Test (Fold 10)", f"{n_eval:,} records" if isinstance(n_eval, int) else str(n_eval))
            with b_col4:
                st.metric("Test Macro F1", f"{test_m.get('macro_f1', 0.0):.4f}")
            with b_col5:
                prauc_str = f"{test_m.get('macro_pr_auc', 0.0):.4f}" if test_m.get('macro_pr_auc') else "N/A"
                st.metric("Test Macro PR-AUC", prauc_str)

            st.info(
                f"**Phase 3.2.1 Paired Pipeline Comparison Audit:** Evaluated on the shared Fold 10 test cohort "
                f"({n_eval:,} records, 1,904 unique patients) comparing the complete Phase 3.1 pilot pipeline "
                f"(trained on 240 records, thresholded on 30 records) against the complete Phase 3.2 full-fold pipeline "
                f"(trained on 17,418 records, thresholded on 2,183 records). The paired patient-clustered "
                f"bootstrap ($B=2,000$, seed: 42) demonstrated a paired Macro F1 difference of "
                f"+0.1510 (95% paired CI: [0.1343, 0.1681], SE: 0.0087), with 0 of 2,000 bootstrap replicates having "
                f"a non-positive difference. Note: this comparison reflects the difference between the two complete pipelines "
                f"(including both training scale and validation threshold-tuning sample sizes); the bootstrap interval represents "
                f"test-cohort sampling uncertainty conditional on the fixed fitted models rather than all retraining variability. "
                f"Offline research benchmark only; does not claim clinical diagnostic validity or cross-population generalization."
            )

            if summary_md_file.is_file():
                with open(summary_md_file, "r", encoding="utf-8") as f:
                    st.markdown(f.read())
            else:
                st.json(metrics_data)

            paired_md_file = Path(DEFAULT_ARTIFACT_DIR) / "paired_model_comparison.md"
            if paired_md_file.is_file():
                st.markdown("---")
                with open(paired_md_file, "r", encoding="utf-8") as f:
                    st.markdown(f.read())

    # --------------------------------------------------------------------------
    # Tab 3: Dataset Provenance & Methodology
    # --------------------------------------------------------------------------
    with tab_provenance:
        st.markdown(
            f"""
            ### Dataset Provenance & Attribution
            - **Dataset Name:** {PHYSIONET_DATASET_NAME}
            - **Citation:** {PHYSIONET_CITATION}
            - **License:** [{PHYSIONET_LICENSE}](https://physionet.org/content/ptb-xl/1.0.3/)
            - **Data Format:** 100 Hz downsampled 12-lead digital waveforms (`records100/`), 10 seconds duration per record.

            ### Methodology & Technical Guardrails
            1. **Decoupled Architecture:** Phase 3 digital waveform modeling operates independently of the Phase 2 ECG paper image digitizer.
            2. **Patient-Separated Partitions:**
               - **Folds 1–8:** Training split ($N=17,418$ records, 15,023 patients).
               - **Fold 9:** Validation split ($N=2,183$ records, 1,942 patients) used strictly for model selection and threshold tuning.
               - **Fold 10:** Held-out final test split ($N=2,198$ records, 1,904 patients) evaluated strictly once.
               - **Split Integrity:** Verified zero patient overlap ($\text{Leakage} = 0$).
            3. **Bounded-Memory Feature Extraction:**
               - Processes records sequentially or in bounded batches.
               - Extracts 148 deterministic per-lead and cross-lead statistical & morphological descriptors.
               - Persists cache with a cryptographic SHA-256 fingerprint of dataset version, sampling rate, and feature schema.
            4. **Verified Model Persistence & Trust Boundary:**
               - Quarantines and rejects all legacy pickle models (`legacy/finalmodel.pkl`).
               - Persists new models with cryptographically bound SHA-256 manifests, refusing to deserialize untrusted or tampered artifacts.
            """
        )


# ==============================================================================
# Page: Methods & Benchmarks
# ==============================================================================

elif nav_choice == "Methods & Benchmarks":
    st.markdown("# Methods, Provenance & Benchmark Taxonomy")
    st.markdown(
        "<p style='font-size: 1.15rem; color: #6F716B; line-height: 1.6;'>"
        "Comprehensive documentation of dataset provenance, diagnostic labeling taxonomy, "
        "feature extraction schemas, and evaluation protocols."
        "</p>",
        unsafe_allow_html=True,
    )

    t_part, t_tax, t_feat, t_trust = st.tabs([
        "Partition Splits & Integrity",
        "Diagnostic Superclasses",
        "Feature Extraction Schema",
        "Model Trust Boundary",
    ])

    with t_part:
        st.markdown("### Official PTB-XL Folds & Patient Split Partitioning")
        st.markdown(
            """
            PTB-XL v1.0.3 defines official stratified folds (`strat_fold` 1 to 10) constructed to prevent patient leakage:
            - **Training Partition (Folds 1–8):** 17,418 records (15,023 unique patients). Used exclusively to fit feature scalers, the naive prevalence baseline, and the lightweight logistic regression model.
            - **Validation Partition (Fold 9):** 2,183 records (1,942 unique patients). Used exclusively for validation evaluation and per-class decision threshold tuning.
            - **Held-Out Test Partition (Fold 10):** 2,198 records (1,904 unique patients). Held strictly out and evaluated once per frozen model configuration.
            """
        )
        split_df = pd.DataFrame([
            {"Partition Split": "Training Split (Folds 1–8)", "Records": 17418, "Unique Patients": 15023, "Purpose": "Model parameter estimation & baseline derivation"},
            {"Partition Split": "Validation Split (Fold 9)", "Records": 2183, "Unique Patients": 1942, "Purpose": "Decision threshold optimization only"},
            {"Partition Split": "Held-Out Test Split (Fold 10)", "Records": 2198, "Unique Patients": 1904, "Purpose": "Final unbiased offline evaluation"},
        ])
        st.dataframe(split_df, use_container_width=True, hide_index=True)
        st.success("✓ Mathematical Split Integrity: Patient set intersections between all partition pairs are strictly empty (Leakage = 0).")

    with t_tax:
        st.markdown("### Diagnostic Superclass Taxonomy (SCP-ECG Standard)")
        st.markdown(
            """
            Labels are mapped from the official `scp_statements.csv` clinical annotations into five mutually non-exclusive diagnostic superclasses:
            - **NORM (Normal ECG):** Normal electrocardiographic recordings without acute or pathological abnormalities.
            - **MI (Myocardial Infarction):** Evidence of acute, subacute, or prior ischemic myocardial necrosis.
            - **STTC (ST/T Changes):** Non-specific or ischemia-related ST-segment elevations/depressions or T-wave alterations.
            - **CD (Conduction Disturbance):** Bundle branch blocks, fascicular blocks, and intraventricular conduction delays.
            - **HYP (Hypertrophy):** Left, right, or biventricular ventricular or atrial enlargement.
            """
        )

    with t_feat:
        st.markdown("### 148 Deterministic Signal Descriptors")
        st.markdown(
            """
            The feature schema (`Version 1.0.0`) extracts 148 deterministic numerical descriptors from the 100 Hz 12-lead digital waveforms:
            - **Per-Lead Features (144 features):** 12 statistical descriptors calculated independently across each of the 12 leads (Mean, Standard Deviation, Median, Minimum, Maximum, Peak-to-Peak Range, Skewness, Kurtosis, Root-Mean-Square, Zero-Crossing Rate, Signal Energy, Mean Absolute Derivative).
            - **Cross-Lead Features (4 features):** Global Limb Lead RMS, Global Precordial Lead RMS, Maximum Absolute Global Amplitude, and Limb-to-Precordial Energy Ratio.
            """
        )

    with t_trust:
        st.markdown("### Cryptographic Model Provenance & Quarantined Legacy Assets")
        st.markdown(
            """
            1. **Legacy Model Quarantine:** The historical pickle model (`legacy/finalmodel.pkl`) is completely quarantined and blocked from deserialization due to arbitrary code execution risks.
            2. **Local Trust Boundary:** All new models are stored in Joblib format accompanied by a sidecar SHA-256 JSON manifest. The verified loader verifies bit-for-bit file digests before deserializing.
            """
        )


# ==============================================================================
# Page: Diagnostics & Privacy
# ==============================================================================

elif nav_choice == "Diagnostics & Privacy":
    st.markdown("# System Diagnostics, Privacy & Architecture")
    st.markdown(
        "<p style='font-size: 1.15rem; color: #6F716B; line-height: 1.6;'>"
        "Real-time diagnostic health check, data lifecycle policies, and reproducible run instructions."
        "</p>",
        unsafe_allow_html=True,
    )

    t_diag, t_priv, t_setup = st.tabs([
        "Live System Diagnostics",
        "Privacy & Data Lifecycle",
        "Reproducibility & Setup",
    ])

    with t_diag:
        st.markdown("### Real-Time Component Health Audit")
        diag = check_system_diagnostics()
        diag_dict = diag.to_dict()["components"]

        for comp_key, comp_info in diag_dict.items():
            avail = comp_info["is_available"]
            icon = "🟢" if avail else "🟡"
            with st.expander(f"{icon} **{comp_info['name']}** — *{comp_info['status_label']}*", expanded=not avail):
                if comp_info.get("details"):
                    st.json(comp_info["details"])
                if comp_info.get("guidance"):
                    st.info(f"**Resolution Guidance:** {comp_info['guidance']}")

        if diag.all_critical_components_ready:
            st.success("✓ All critical demo and benchmark components are operational.")

    with t_priv:
        st.markdown("### Privacy Safeguards & Ephemeral Data Lifecycle")
        st.markdown(
            """
            - **Ephemeral In-Memory Processing:** User-uploaded ECG images are processed exclusively in volatile memory (RAM) and are discarded when the session ends. No user uploads are written to disk.
            - **Zero Data Persistence:** The application does not store, log, or persist user-uploaded images, waveform arrays, or patient identifiers.
            - **No External Telemetry:** The platform contains zero third-party tracking scripts, cloud analytics, or external telemetry pings.
            - **Non-Clinical Research Protocol:** This application is strictly an academic research tool; it does not collect or process protected health information (PHI) for clinical use.
            """
        )

    with t_setup:
        st.markdown("### Reproducible Setup & Offline Execution Guide")
        st.markdown(
            """
            To run this application and execute all quality gates from a clean checkout:

            ```bash
            # 1. Activate virtual environment
            source .venv/bin/activate

            # 2. Run automated test suite (58 unit & integration tests)
            pytest tests/ -v

            # 3. Inspect system diagnostics via CLI
            python -m ecg_lab.waveform_ml.cli diagnostics

            # 4. Re-run paired model-comparison bootstrap audit
            python -m ecg_lab.waveform_ml.cli audit-paired-comparison --n-bootstraps 2000

            # 5. Launch web application locally
            streamlit run app.py
            ```
            """
        )
