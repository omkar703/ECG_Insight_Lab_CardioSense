"""CardioSense / ECG Insight Lab — Minimalist Research Platform.

Theme and layout inspired by minimalist scientific web design:
A clean top bar with mode toggle (Guided Demo | Live Experiment), centered hero graphic,
distraction-free step-by-step walkthrough, and direct backend-connected practical lab.
"""

import base64
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

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
from ecg_lab.waveform_ml.config import (
    DEFAULT_ARTIFACT_DIR as WF_ARTIFACT_DIR,
    DEFAULT_DATA_DIR as WF_DATA_DIR,
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
from ecg_lab.single_lead_ml.config import (
    AAMI_CLASS_DESCRIPTIONS,
    AAMI_TARGET_CLASSES,
    DEFAULT_ARTIFACT_DIR as SL_ARTIFACT_DIR,
    DISCLAIMER_RECORDED_REPLAY,
    DISCLAIMER_SINGLE_LEAD_RESEARCH,
    MODEL_NAME as SL_MODEL_NAME,
    NATIVE_SAMPLING_RATE_HZ as SL_FS,
)
from ecg_lab.single_lead_ml.dataset import (
    BeatAnnotation,
    get_available_records,
    load_record_with_annotations,
    load_single_lead_signal,
)
from ecg_lab.single_lead_ml.models import load_verified_single_lead_model
from ecg_lab.single_lead_ml.playback import get_playback_slice
from ecg_lab.single_lead_ml.inference import (
    BeatPrediction,
    SingleLeadInferenceResult,
    run_single_lead_inference,
)
from hardware.host.simulate_adc_stream import (
    DEFAULT_ADC_BASELINE,
    DEFAULT_ADC_GAIN_COUNTS_PER_MV,
    SIMULATION_PRESETS,
    generate_synthetic_ecg,
    simulate_adc_stream,
    simulated_adc_to_voltage,
)
from hardware.host.serial_protocol import (
    SerialStreamParser,
    format_serial_line,
)
from ecg_lab.explanations.ui import (
    render_explain_results_button,
    render_explanation_panel,
)
from ecg_lab.dashboard_helpers import (
    compute_filter_bode_data,
    apply_bandpass_notch_filter,
    compute_pan_tompkins_cascade,
    create_bode_plot,
    create_pan_tompkins_cascade_figure,
    create_before_after_filter_figure,
    create_confusion_matrix_figure,
    create_12lead_matrix_heatmap,
    create_lead_correlation_heatmap,
    create_multilabel_contingency_figure,
    extract_image_preprocessing_stages,
    create_cardiac_anatomy_figure,
    create_dataset_comparison_figure,
    create_threshold_calibration_figure,
    apply_clean_scientific_theme,
    load_demo_ptbxl_records,
    load_demo_mitbih_slice,
    get_demo_playback_slice,
)


# ==============================================================================
# Page Configuration & Minimalist Editorial Theme
# ==============================================================================

st.set_page_config(
    page_title="CardioSense — ECG Research",
    page_icon="docs/assets/favicon.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_minimalist_theme():
    """Inject clean, modern minimalist CSS matching the reference interface."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

    /* Global Typography & Light Neutral Background */
    html, body, .stApp, p, h1, h2, h3, h4, h5, h6, label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: #1F1F1F;
    }

    .stApp {
        background-color: #FAFAFA;
    }

    /* Hide standard header bar and sidebar */
    header[data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stSidebar"] {
        display: none;
    }

    /* Main container max-width & padding */
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 4rem;
        margin: 0 auto;
    }

    /* Top navigation header */
    .top-header-title {
        font-size: 0.88rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #4A4A4A;
        padding-top: 0.35rem;
    }

    .top-header-divider {
        border-bottom: 1px solid #ECECEC;
        margin-top: 0.4rem;
        margin-bottom: 2.2rem;
    }

    /* Hero section styling */
    .hero-container {
        max-width: 680px;
        margin: 1.5rem auto 3rem auto;
        text-align: center;
    }

    .hero-subhead {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #222222;
        margin-top: 1.5rem;
        margin-bottom: 0.6rem;
    }

    .hero-title {
        font-size: 1.1rem;
        font-weight: 400;
        color: #333333;
        line-height: 1.5;
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        font-size: 0.92rem;
        color: #777777;
        font-weight: 400;
        line-height: 1.5;
        margin-bottom: 2rem;
    }

    /* Pill-shaped buttons: Pure white background, solid black inner text, and black border */
    div.stButton > button,
    div.stButton > button:focus,
    div.stButton > button:focus-visible,
    div.stButton > button:active,
    button[kind="secondary"],
    button[kind="primary"],
    button[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-primary"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #000000 !important;
        border-radius: 9999px !important;
        padding: 0.60rem 1.60rem !important;
        font-size: 0.90rem !important;
        font-weight: 600 !important;
        border: 1.5px solid #000000 !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05) !important;
        cursor: pointer !important;
        outline: none !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        width: auto !important;
        min-width: fit-content !important;
        max-width: fit-content !important;
        height: 44px !important;
        line-height: 1 !important;
        white-space: nowrap !important;
        box-sizing: border-box !important;
    }

    /* Enforce solid black text across all child elements inside buttons */
    div.stButton > button *,
    div.stButton > button p,
    div.stButton > button span,
    div.stButton > button div,
    button[data-testid="baseButton-secondary"] *,
    button[data-testid="baseButton-primary"] * {
        color: #000000 !important;
        font-weight: 600 !important;
    }

    /* Luminous animated glow on hover — white background, black text, glowing shadow */
    div.stButton > button:hover,
    button[data-testid="baseButton-secondary"]:hover,
    button[data-testid="baseButton-primary"]:hover {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #000000 !important;
        border-color: #000000 !important;
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 0 16px rgba(0, 0, 0, 0.32), 0 0 32px rgba(0, 0, 0, 0.16), 0 4px 10px rgba(0, 0, 0, 0.08) !important;
        animation: buttonPulseGlow 1.8s infinite alternate ease-in-out !important;
    }

    div.stButton > button:hover *,
    button[data-testid="baseButton-secondary"]:hover *,
    button[data-testid="baseButton-primary"]:hover * {
        color: #000000 !important;
    }

    div.stButton > button:active,
    button[data-testid="baseButton-secondary"]:active,
    button[data-testid="baseButton-primary"]:active {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #000000 !important;
        border-color: #000000 !important;
        transform: translateY(0px) scale(0.99);
        box-shadow: 0 0 8px rgba(0, 0, 0, 0.2) !important;
    }

    @keyframes buttonPulseGlow {
        0% {
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.18), 0 0 20px rgba(0, 0, 0, 0.10);
        }
        100% {
            box-shadow: 0 0 22px rgba(0, 0, 0, 0.40), 0 0 42px rgba(0, 0, 0, 0.20);
        }
    }

    /* Hidden Alignment Markers & Zero-Gap Container Suppression */
    .align-right-marker,
    .align-left-marker {
        display: none !important;
    }
    div[data-testid="stElementContainer"]:has(.align-right-marker),
    div[data-testid="stElementContainer"]:has(.align-left-marker) {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Navigation Footer Columns: Enforce 190px Minimum Width for Symmetric Oval Look */
    div[data-testid="column"]:has(.align-left-marker) button,
    div[data-testid="column"]:has(.align-right-marker) button,
    div[data-testid="stColumn"]:has(.align-left-marker) button,
    div[data-testid="stColumn"]:has(.align-right-marker) button {
        min-width: 190px !important;
    }

    /* Left Navigation Button Alignment */
    div[data-testid="column"]:has(.align-left-marker) div.stButton,
    div[data-testid="stColumn"]:has(.align-left-marker) div.stButton,
    div.stColumn:has(.align-left-marker) div.stButton {
        display: flex !important;
        justify-content: flex-start !important;
        align-items: center !important;
        width: 100% !important;
    }
    div[data-testid="column"]:has(.align-left-marker) button,
    div[data-testid="stColumn"]:has(.align-left-marker) button,
    div.stColumn:has(.align-left-marker) button {
        margin-right: auto !important;
        margin-left: 0 !important;
        display: inline-flex !important;
    }

    /* Forward Navigation Button Right Alignment */
    div[data-testid="column"]:has(.align-right-marker) div.stButton,
    div[data-testid="stColumn"]:has(.align-right-marker) div.stButton,
    div.stColumn:has(.align-right-marker) div.stButton {
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
        width: 100% !important;
        text-align: right !important;
    }
    div[data-testid="column"]:has(.align-right-marker) button,
    div[data-testid="stColumn"]:has(.align-right-marker) button,
    div.stColumn:has(.align-right-marker) button {
        margin-left: auto !important;
        margin-right: 0 !important;
        display: inline-flex !important;
    }

    /* Clean Card & Metric Container */
    div[data-testid="stMetric"], .clean-card {
        background-color: #FFFFFF;
        border: 1px solid #EBEBEB;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        box-shadow: none !important;
    }

    /* Inline code & code blocks: Crisp light background, solid black text, clean border */
    code,
    kbd,
    samp,
    pre,
    [data-testid="stMarkdownContainer"] code,
    [data-testid="stMarkdownContainer"] pre,
    div[data-testid="stMarkdownContainer"] p code,
    div[data-testid="stMarkdownContainer"] span code {
        background-color: #F1F3F5 !important;
        background: #F1F3F5 !important;
        color: #000000 !important;
        border: 1px solid #D0D4DC !important;
        padding: 0.15rem 0.45rem !important;
        border-radius: 4px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.88em !important;
        font-weight: 600 !important;
    }

    /* Clean, high-contrast Selectboxes & Dropdowns — Pure white background, solid black text & border */
    div[data-testid="stSelectbox"],
    div.stSelectbox,
    div[data-testid="stSelectbox"] > div,
    div.stSelectbox > div,
    div[data-testid="stSelectbox"] [role="combobox"],
    div[data-testid="stSelectbox"] [role="group"],
    div[data-testid="stSelectbox"] .react-aria-Group,
    div[data-testid="stSelectbox"] [class*="e1oobldh0"],
    div[data-baseweb="select"],
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #000000 !important;
        border-color: #000000 !important;
    }

    div[data-testid="stSelectbox"] [role="group"],
    div[data-testid="stSelectbox"] .react-aria-Group,
    div[data-testid="stSelectbox"] [class*="e1oobldh0"],
    div[data-baseweb="select"] > div {
        border: 1.5px solid #000000 !important;
        border-radius: 8px !important;
        box-shadow: none !important;
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
    }

    div[data-testid="stSelectbox"]:hover [role="group"],
    div[data-testid="stSelectbox"] [role="group"]:focus-within,
    div[data-baseweb="select"]:hover > div,
    div[data-baseweb="select"]:focus-within > div {
        border-color: #000000 !important;
        box-shadow: 0 0 8px rgba(0, 0, 0, 0.15) !important;
    }

    div[data-testid="stSelectbox"] input,
    div[data-testid="stSelectbox"] [class*="e1oobldh1"],
    div.stSelectbox input {
        background-color: #FFFFFF !important;
        background: transparent !important;
        color: #000000 !important;
        font-weight: 600 !important;
    }

    /* Force all text, spans, divs, and labels inside selectboxes to be solid black */
    div[data-testid="stSelectbox"] *,
    div.stSelectbox *,
    div[data-baseweb="select"] *,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] p,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSelectbox"] label p {
        color: #000000 !important;
        font-weight: 500 !important;
    }

    div[data-testid="stSelectbox"] label p {
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    /* Dropdown arrow SVG icon */
    div[data-testid="stSelectbox"] svg,
    div.stSelectbox svg,
    div[data-baseweb="select"] svg,
    div[data-testid="stSelectbox"] [class*="e1oobldh2"] {
        fill: #000000 !important;
        color: #000000 !important;
    }

    /* Dropdown popup menu */
    div[data-testid="stSelectboxVirtualDropdown"],
    div[data-testid="stSelectboxVirtualDropdown"] *,
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    ul[data-baseweb="menu"],
    ul[role="listbox"],
    div[role="listbox"],
    div[data-testid="stSelectboxVirtualDropdown"] [role="listbox"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #000000 !important;
    }

    div[data-testid="stSelectboxVirtualDropdown"],
    div[data-baseweb="popover"] > div {
        border: 1.5px solid #000000 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12) !important;
    }

    div[data-testid="stSelectboxVirtualDropdown"] [role="option"],
    li[data-baseweb="menu-item"],
    li[role="option"],
    div[role="option"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #000000 !important;
    }

    div[data-testid="stSelectboxVirtualDropdown"] [role="option"] *,
    li[data-baseweb="menu-item"] *,
    li[role="option"] *,
    div[role="option"] * {
        color: #000000 !important;
    }

    div[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
    li[data-baseweb="menu-item"]:hover,
    li[role="option"]:hover,
    div[role="option"]:hover,
    div[data-testid="stSelectboxVirtualDropdown"] [aria-selected="true"],
    li[aria-selected="true"],
    div[aria-selected="true"] {
        background-color: #F2F2F2 !important;
        background: #F2F2F2 !important;
        color: #000000 !important;
    }

    /* Radio buttons & text inputs */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label span {
        color: #000000 !important;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stNumberInput"] [role="group"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #000000 !important;
        border: 1.5px solid #000000 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stNumberInput"] button {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-color: #000000 !important;
    }
    div[data-testid="stNumberInput"] button svg {
        fill: #000000 !important;
    }

    div[data-testid="stTextInput"] label p,
    div[data-testid="stNumberInput"] label p,
    div[data-testid="stSlider"] label p {
        color: #000000 !important;
        font-weight: 600 !important;
    }

    /* Expander container and header: Pure white background, solid black text & border */
    div[data-testid="stExpander"],
    div.stExpander,
    div[data-testid="stExpander"] details,
    div.stExpander details {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border-radius: 8px !important;
        border: 1.5px solid #000000 !important;
        overflow: hidden !important;
        box-shadow: none !important;
    }

    div[data-testid="stExpander"] summary,
    div.stExpander summary,
    div[data-testid="stExpander"] details summary,
    div.stExpander details summary,
    div[data-testid="stExpander"] [class*="e1u8u9eb10"] {
        display: flex !important;
        align-items: center !important;
        gap: 0.75rem !important;
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #000000 !important;
        border-radius: 0px !important;
        border: none !important;
        padding: 0.65rem 1rem !important;
        cursor: pointer !important;
    }

    div[data-testid="stExpander"] summary:hover,
    div.stExpander summary:hover,
    div[data-testid="stExpander"] details summary:hover,
    div.stExpander details summary:hover {
        background-color: #F8F9FA !important;
        background: #F8F9FA !important;
    }

    /* Preserve Material Symbols icon font inside expander chevron and globally */
    [data-testid="stIconMaterial"],
    .material-symbols-rounded,
    [class*="material-symbols"],
    span[data-testid="stExpanderStepIcon"],
    span[data-testid="stExpanderStepChevron"],
    div[data-testid="stExpander"] summary [data-testid="stIconMaterial"],
    div.stExpander summary [data-testid="stIconMaterial"],
    div[data-testid="stExpander"] summary span[class*="e1vmumty0"],
    span.e1vmumty0 {
        font-family: "Material Symbols Rounded" !important;
        font-weight: normal !important;
        font-style: normal !important;
        font-size: 20px !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-smoothing: antialiased !important;
        text-rendering: optimizeLegibility !important;
        -moz-osx-font-smoothing: grayscale !important;
        font-feature-settings: 'liga' 1 !important;
        color: #000000 !important;
        fill: #000000 !important;
        margin-right: 0.35rem !important;
        flex-shrink: 0 !important;
    }

    div[data-testid="stExpander"] summary p,
    div.stExpander summary p,
    div[data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        margin: 0 !important;
        line-height: 1.4 !important;
    }

    div[data-testid="stExpander"] [data-testid="stExpanderDetails"],
    div.stExpander [data-testid="stExpanderDetails"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border-top: 1px solid #ECECEC !important;
        padding: 1rem !important;
    }

    /* Governance & Safety Framework Cards */
    .gov-card {
        background-color: #FFFFFF;
        border: 1.5px solid #000000;
        border-radius: 10px;
        padding: 1.35rem 1.25rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .gov-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.09);
    }
    .gov-badge {
        display: inline-block;
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 3px 8px;
        border-radius: 4px;
        background: #F0F0F0;
        color: #111111;
        margin-bottom: 0.65rem;
        border: 1px solid #E0E0E0;
    }
    .gov-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #000000;
        margin-bottom: 0.75rem;
        line-height: 1.3;
    }
    .gov-list {
        margin: 0;
        padding: 0;
        list-style: none;
    }
    .gov-item {
        font-size: 0.83rem;
        line-height: 1.45;
        color: #444444;
        margin-bottom: 0.65rem;
        position: relative;
        padding-left: 1.1rem;
    }
    .gov-item::before {
        content: "•";
        position: absolute;
        left: 0;
        top: 0;
        color: #000000;
        font-weight: bold;
        font-size: 1rem;
    }
    .gov-item strong {
        color: #000000;
        font-weight: 600;
    }
    .gov-footer {
        margin-top: 1rem;
        padding-top: 0.75rem;
        border-top: 1px solid #ECECEC;
        font-size: 0.76rem;
        font-weight: 600;
        color: #111111;
        letter-spacing: 0.04em;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }

    /* File uploader */
    div[data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border: 1.5px dashed #000000 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stFileUploader"] section * {
        color: #000000 !important;
    }

    /* Horizontal Radio Toggle on Right */
    div[data-testid="stRadio"] > div {
        display: flex;
        justify-content: flex-end;
        gap: 1.8rem;
    }
    div[data-testid="stRadio"] label {
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: #444444 !important;
        cursor: pointer;
    }

    /* Oval/Pill button-style tabs with light green active state (Streamlit React-Aria & BaseWeb compatible) */
    div[data-testid="stTabs"] [role="tablist"]::after,
    .stTabs [role="tablist"]::after,
    div[data-testid="stTabs"] .react-aria-SelectionIndicator,
    .stTabs .react-aria-SelectionIndicator,
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
        opacity: 0 !important;
        visibility: hidden !important;
        height: 0 !important;
        border: none !important;
        background: transparent !important;
    }

    div[data-testid="stTabs"] [role="tablist"],
    .stTabs [role="tablist"],
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.85rem !important;
        border: none !important;
        border-bottom: none !important;
        padding-bottom: 0.85rem !important;
        padding-top: 0.35rem !important;
        background: transparent !important;
        display: flex !important;
        flex-wrap: wrap !important;
        align-items: center !important;
    }

    div[data-testid="stTabs"] [data-testid="stTab"],
    div[data-testid="stTabs"] [role="tab"],
    .stTabs [data-testid="stTab"],
    .stTabs [role="tab"],
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        border: 1.5px solid #000000 !important;
        border-radius: 9999px !important;
        font-size: 0.90rem !important;
        font-weight: 600 !important;
        color: #111111 !important;
        padding: 0.65rem 1.6rem !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        height: auto !important;
        min-height: 2.4rem !important;
        box-sizing: border-box !important;
        white-space: nowrap !important;
        outline: none !important;
        cursor: pointer !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 2px 0 !important;
    }

    div[data-testid="stTabs"] [data-testid="stTab"] *,
    div[data-testid="stTabs"] [role="tab"] *,
    .stTabs [data-testid="stTab"] *,
    .stTabs [role="tab"] *,
    .stTabs [data-baseweb="tab"] * {
        color: #111111 !important;
        font-weight: 600 !important;
        font-size: 0.90rem !important;
    }

    div[data-testid="stTabs"] [data-testid="stTab"] p,
    div[data-testid="stTabs"] [role="tab"] p {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.2 !important;
    }

    div[data-testid="stTabs"] [data-testid="stTab"]:hover,
    div[data-testid="stTabs"] [role="tab"]:hover,
    div[data-testid="stTabs"] [data-testid="stTab"][data-hovered],
    .stTabs [data-testid="stTab"]:hover,
    .stTabs [role="tab"]:hover,
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #F8F9FA !important;
        background: #F8F9FA !important;
        border-color: #000000 !important;
        border-radius: 9999px !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.10) !important;
    }

    div[data-testid="stTabs"] [data-testid="stTab"]:focus,
    div[data-testid="stTabs"] [role="tab"]:focus,
    div[data-testid="stTabs"] [data-testid="stTab"]:active,
    div[data-testid="stTabs"] [role="tab"]:active,
    .stTabs [data-baseweb="tab"]:focus,
    .stTabs [data-baseweb="tab"]:active {
        outline: none !important;
        border-radius: 9999px !important;
    }

    div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"],
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"],
    div[data-testid="stTabs"] [data-testid="stTab"][data-selected],
    div[data-testid="stTabs"] [role="tab"][data-selected],
    .stTabs [data-testid="stTab"][aria-selected="true"],
    .stTabs [role="tab"][aria-selected="true"],
    .stTabs [data-testid="stTab"][data-selected],
    .stTabs [role="tab"][data-selected],
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #E8F5E9 !important;
        background: #E8F5E9 !important;
        border: 2px solid #2E7D32 !important;
        border-radius: 9999px !important;
        box-shadow: 0 0 14px rgba(46, 125, 50, 0.35), 0 2px 6px rgba(0, 0, 0, 0.06) !important;
        transform: translateY(-1px) !important;
    }

    div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] *,
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] *,
    div[data-testid="stTabs"] [data-testid="stTab"][data-selected] *,
    div[data-testid="stTabs"] [role="tab"][data-selected] *,
    .stTabs [data-testid="stTab"][aria-selected="true"] *,
    .stTabs [role="tab"][aria-selected="true"] *,
    .stTabs [data-baseweb="tab"][aria-selected="true"] * {
        color: #1B5E20 !important;
        font-weight: 700 !important;
    }

    /* Step Badge — crisp white background, solid black text & border */
    .step-badge {
        display: inline-block;
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 14px;
        border-radius: 9999px;
        border: 1.5px solid #000000 !important;
        margin-bottom: 0.5rem;
        letter-spacing: 0.06em;
    }

    /* Step transition animation */
    .step-content-container {
        animation: fadeInUp 0.45s cubic-bezier(0.16, 1, 0.3, 1) both;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Pulsing flowchart dot */
    .pulse-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #000000;
        margin-right: 6px;
        box-shadow: 0 0 0 0 rgba(0, 0, 0, 0.7);
        animation: pulseDot 2s infinite;
    }

    @keyframes pulseDot {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(0, 0, 0, 0.4);
        }
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 6px rgba(0, 0, 0, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(0, 0, 0, 0);
        }
    }

    /* Scientific 5-Stage Pipeline Track */
    .pipeline-track {
        display: flex;
        align-items: stretch;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1.5px solid #000000;
        border-radius: 10px;
        padding: 1.15rem 1.4rem;
        margin: 1.2rem 0 1.6rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        gap: 0.8rem;
    }
    .pipeline-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        flex: 1;
    }
    .pipeline-dot {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background-color: #000000;
        color: #FFFFFF;
        font-size: 0.74rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.45rem;
        letter-spacing: 0.04em;
    }
    .pipeline-label {
        font-size: 0.88rem;
        font-weight: 700;
        color: #000000;
        margin-bottom: 0.2rem;
    }
    .pipeline-sub {
        font-size: 0.75rem;
        color: #555555;
        line-height: 1.35;
        max-width: 145px;
    }
    .pipeline-connector {
        flex: 0 0 28px;
        height: 1.5px;
        background-color: #000000;
        align-self: center;
        margin-bottom: 2rem;
    }

    /* Concept Badge */
    .concept-badge {
        display: inline-block;
        background: #F0F0F0;
        color: #222222;
        font-size: 0.74rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


inject_minimalist_theme()


# ==============================================================================
# State Management
# ==============================================================================

if "nav_mode" not in st.session_state:
    st.session_state["nav_mode"] = "Guided Demo"
if "demo_step" not in st.session_state:
    st.session_state["demo_step"] = 0
if "lab_active_tab" not in st.session_state:
    st.session_state["lab_active_tab"] = "12-Lead Diagnostic Classification (PTB-XL)"

# Sync query params with demo_step if present
if "step" in st.query_params:
    try:
        q_step = int(st.query_params["step"])
        if 0 <= q_step <= 8:
            st.session_state["demo_step"] = q_step
            if q_step > 0:
                st.session_state["nav_mode"] = "Guided Demo"
    except (ValueError, TypeError):
        pass


# ==============================================================================
# Top Navigation Header (Matching Reference Image)
# ==============================================================================

# Flush any pending nav mode change BEFORE the radio widget is instantiated.
# Buttons cannot write to a widget key after it has been rendered, so we use
# a staging key (_pending_nav_mode) that is applied here on the next rerun.
if "_pending_nav_mode" in st.session_state:
    _pnm = st.session_state.pop("_pending_nav_mode")
    st.session_state["nav_mode"] = _pnm
    st.session_state["top_nav_mode_radio"] = _pnm

# Load app icon for header
header_icon_b64 = ""
fav_icon_path = Path("docs/assets/favicon.png")
if fav_icon_path.is_file():
    header_icon_b64 = base64.b64encode(fav_icon_path.read_bytes()).decode("utf-8")

header_col_left, header_col_right = st.columns([3, 2])

with header_col_left:
    icon_tag = (
        f"<img src='data:image/png;base64,{header_icon_b64}' width='24' height='24' "
        f"style='border-radius: 5px; margin-right: 9px; vertical-align: -3px; display: inline-block; box-shadow: 0 1px 3px rgba(0,0,0,0.18);' "
        f"alt='CardioSense ECG Icon' />"
        if header_icon_b64 else ""
    )
    st.markdown(
        f"<div class='top-header-title' style='display: flex; align-items: center;'>"
        f"{icon_tag}<span>CARDIOSENSE · ECG RESEARCH</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

with header_col_right:
    nav_mode = st.radio(
        "Navigation",
        options=["Guided Demo", "Live Experiment"],
        index=0 if st.session_state["nav_mode"] == "Guided Demo" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="top_nav_mode_radio",
    )
    if nav_mode != st.session_state["nav_mode"]:
        st.session_state["nav_mode"] = nav_mode
        st.rerun()

st.markdown("<div class='top-header-divider'></div>", unsafe_allow_html=True)


# ==============================================================================
# VIEW 1: GUIDED DEMO (Research Walkthrough)
# ==============================================================================

def render_step_footer(
    prev_txt: str = "",
    prev_key: str = "",
    prev_target_step: Optional[int] = None,
    prev_target_mode: Optional[str] = None,
    next_txt: str = "",
    next_key: str = "",
    next_target_step: Optional[int] = None,
    next_target_mode: Optional[str] = None,
    show_divider: bool = True,
    margins: Optional[List[float]] = None,
):
    if show_divider:
        st.markdown("<hr style='border: none; border-bottom: 1px solid #ECECEC; margin: 2rem 0 1rem 0;'>", unsafe_allow_html=True)

    if margins:
        m1, m2, m3 = st.columns(margins)
        host = m2
    else:
        host = st.container()

    with host:
        c_b1, c_b2 = st.columns([1, 1])
        with c_b1:
            st.markdown("<span class='align-left-marker'></span>", unsafe_allow_html=True)
            if prev_txt and st.button(prev_txt, key=prev_key):
                if prev_target_mode:
                    st.session_state["nav_mode"] = prev_target_mode
                    # Stage the radio update — applied before radio is instantiated on next rerun
                    st.session_state["_pending_nav_mode"] = prev_target_mode
                if prev_target_step is not None:
                    st.session_state["demo_step"] = prev_target_step
                st.rerun()
        with c_b2:
            st.markdown("<span class='align-right-marker'></span>", unsafe_allow_html=True)
            if next_txt and st.button(next_txt, key=next_key):
                if next_target_mode:
                    st.session_state["nav_mode"] = next_target_mode
                    st.session_state["_pending_nav_mode"] = next_target_mode
                if next_target_step is not None:
                    st.session_state["demo_step"] = next_target_step
                st.rerun()


if st.session_state["nav_mode"] == "Guided Demo":

    # --- LANDING SCREEN (Step 0) ---
    if st.session_state["demo_step"] == 0:
        heart_path = Path("docs/heart-anatomy.gif")
        if not heart_path.is_file():
            heart_path = Path("docs/assets/heart_anatomy_white_bg.gif")
        if not heart_path.is_file():
            heart_path = Path("docs/assets/heart.gif")

        heart_img_html = ""
        if heart_path.is_file():
            heart_b64 = base64.b64encode(heart_path.read_bytes()).decode("utf-8")
            heart_img_html = (
                f"<div style='display: flex; justify-content: center; margin-bottom: 1.5rem; background-color: #FAFAFA;'>"
                f"<img src='data:image/gif;base64,{heart_b64}' width='340' "
                f"style='display: block; margin: 0 auto; max-width: 100%; border-radius: 8px; background-color: #FAFAFA;' "
                f"alt='Cardiac Electrophysiology Anatomy' />"
                f"</div>"
            )

        hero_html = f"""
        <div style='text-align: center; max-width: 920px; margin: 0 auto; padding: 1rem 1rem 1.25rem 1rem; background-color: #FAFAFA;'>
            {heart_img_html}
            <div style='font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.14em; color: #444444; margin-bottom: 0.5rem;'>
                CARDIOVASCULAR SIGNAL INTELLIGENCE & RESEARCH PLATFORM
            </div>
            <div style='font-size: 1.85rem; font-weight: 800; color: #111111; line-height: 1.25; margin-bottom: 0.75rem; letter-spacing: -0.02em;'>
                12-Lead & Single-Lead ECG Analysis, Archival Digitization & Hardware Emulation
            </div>
            <div style='font-size: 1.05rem; font-weight: 400; color: #444444; line-height: 1.6; margin-bottom: 1.75rem; max-width: 820px; margin-left: auto; margin-right: auto;'>
                Detect cardiac pathology from 12-lead calibrated voltage signals, isolate ambulatory arrhythmias with Pan-Tompkins QRS tracking, extract continuous waveforms from paper records, and simulate real-world analog sensor telemetry.
            </div>
        </div>
        """
        st.markdown(hero_html, unsafe_allow_html=True)

        st.markdown("<div style='height: 0.25rem;'></div>", unsafe_allow_html=True)
        render_step_footer(
            prev_txt="Open Practical Labs",
            prev_key="btn_open_labs_hero",
            prev_target_step=0,
            prev_target_mode="Live Experiment",
            next_txt="Begin Guided Demo →",
            next_key="btn_start_demo",
            next_target_step=1,
            show_divider=False,
            margins=[0.05, 0.90, 0.05],
        )

        st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

        det_c1, det_c2, det_c3 = st.columns([0.05, 0.90, 0.05])
        with det_c2:
            with st.expander("Technical Architecture & Workflow Details", expanded=False):
                st.markdown(
                    """
                    ### 12-Lead Diagnostic Classification (PTB-XL)
                    Continuous 12-channel surface electrocardiograms (Leads I, II, III, aVR, aVL, aVF, and V1 through V6) are ingested from the gold-standard PTB-XL cohort at 100 Hz calibrated voltage. All multi-channel signals undergo zero-phase bi-directional Butterworth bandpass conditioning (0.05–40 Hz) using forward-backward recursive filtering (`filtfilt`). This removes baseline wander and mains powerline noise while guaranteeing zero ST-segment phase distortion (θ = 0°), ensuring critical elevation and depression metrics reflect true physiological cardiac activity.

                    The diagnostic engine predicts across five primary clinical superclasses: Myocardial Infarction (`MI`), ST/T-segment Changes (`STTC`), Conduction Disturbances (`CD`), Ventricular Hypertrophy (`HYP`), and Normal ECG (`NORM`). Rather than relying on arbitrary default thresholds, decision cutoffs are individually calibrated against independent validation splits to optimize clinical sensitivity and specificity across imbalanced pathology presentations.

                    ---

                    ### Single-Lead Arrhythmia Classification (MIT-BIH)
                    Ambulatory rhythm analysis processes continuous single-channel recordings from the PhysioNet MIT-BIH Arrhythmia Database sampled at 360 Hz. Rather than relying on rigid interval assumptions, an autonomous Pan-Tompkins QRS locator identifies ventricular depolarizations through cascaded low-pass, high-pass, derivative, non-linear squaring, and moving-window integration operations.

                    Each isolated cardiac cycle is windowed from 200 ms preceding the detected R-peak to 400 ms following it. A 100-estimator Balanced Random Forest analyzes localized morphological features alongside pre- and post-RR interval dynamics to classify heartbeats into standard AAMI diagnostic families: Normal sinus beats (`N`), Supraventricular ectopics (`S`), Ventricular ectopics or premature ventricular contractions (`V`), and Fusion or unclassifiable beats (`Q`).

                    ---

                    ### Optical Paper Digitizer
                    Historical outpatient charts and legacy records often exist solely as physical paper printouts or scanned photographs on standard 25 mm/s grid paper. The digitization pipeline uses classical computer vision algorithms—incorporating adaptive local thresholding, Gaussian smoothing, and morphological segmentation—to extract continuous 1D voltage traces across all 12 standard leads.

                    To guarantee patient safety and algorithmic integrity, the digitization workflow operates in strict programmatic isolation from diagnostic machine learning models. Scanner grid noise, perspective distortions, and paper wrinkles are prevented from contaminating downstream classifier inputs. Extracted traces are calibrated against standard paper grid references and can be inspected via voltage heatmaps or exported as continuous CSV files with full metadata audit trails.

                    ---

                    ### Hardware Prototype AD8232 Simulation
                    To bridge algorithmic software with physical telemetry, the hardware module provides an offline simulation of an AD8232 analog front-end biopotential sensor coupled with a microcontroller analog-to-digital converter stream. The system models 10-bit integer ADC counts transmitted over a line-delimited UART serial protocol.

                    The simulation models real-world hardware operating conditions including physical electrode detachment (leads-off detection), baseline drift, somatic muscle tremors, and ADC rail clipping. A strict safety policy governs all telemetry: because uncalibrated hobbyist hardware lacks clinical isolation certificates, raw serial streams are verified for transport integrity but blocked from making automated diagnostic claims.
                    """
                )

    # --- STEP-BY-STEP WALKTHROUGH (Steps 1 through 8) ---
    else:
        step = st.session_state["demo_step"]
        step_titles = {
            1: "Clinical Problem & Scientific Pipeline",
            2: "Research Datasets & Cohort Partitions",
            3: "Anatomy of an ECG Signal",
            4: "Digital Filtering & QRS Cascade",
            5: "Model Architectures & Feature Extraction",
            6: "Prediction Generation & Calibrated Thresholds",
            7: "Evaluation Benchmarks & Confusion Matrices",
            8: "Interpretation, Boundaries & Governance",
        }

        step_labels = {
            1: "01 CLINICAL PROBLEM",
            2: "02 RESEARCH DATASETS",
            3: "03 ECG ANATOMY",
            4: "04 DIGITAL FILTERING",
            5: "05 MODEL ARCHITECTURES",
            6: "06 PREDICTION & THRESHOLDS",
            7: "07 BENCHMARK AUDITS",
            8: "08 GOVERNANCE & SAFETY",
        }

        # Step Navigation & Connected Dots Progress Stepper
        s_nav1, s_nav2, s_nav3 = st.columns([1, 4.2, 1])
        with s_nav1:
            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
            if step == 1:
                if st.button("← Back to Hero", key="btn_step_overview", use_container_width=True):
                    st.session_state["demo_step"] = 0
                    st.rerun()
            else:
                if st.button(f"← Step {step - 1}", key="btn_step_prev", use_container_width=True):
                    st.session_state["demo_step"] = step - 1
                    st.rerun()

        with s_nav2:
            dots_html = []
            for i in range(1, 9):
                if i == step:
                    # Active dot: vibrant blue with soft translucent halo ring matching reference image
                    node = (
                        f"<a href='?step={i}' target='_self' title='Step {i}: {step_titles[i]}' style='text-decoration: none; cursor: pointer; display: inline-block;'>"
                        f"<div style='width: 28px; height: 28px; border-radius: 50%; background: rgba(59, 130, 246, 0.20); display: flex; align-items: center; justify-content: center; transition: all 0.3s ease;'>"
                        f"<div style='width: 13px; height: 13px; border-radius: 50%; background: #3B82F6;'></div>"
                        f"</div></a>"
                    )
                elif i < step:
                    # Completed dot: neutral filled circle
                    node = (
                        f"<a href='?step={i}' target='_self' title='Step {i}: {step_titles[i]}' style='text-decoration: none; cursor: pointer; display: inline-block;'>"
                        f"<div style='width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; transition: all 0.2s ease;'>"
                        f"<div style='width: 12px; height: 12px; border-radius: 50%; background: #94A3B8;'></div>"
                        f"</div></a>"
                    )
                else:
                    # Inactive upcoming dot: light gray circle
                    node = (
                        f"<a href='?step={i}' target='_self' title='Step {i}: {step_titles[i]}' style='text-decoration: none; cursor: pointer; display: inline-block;'>"
                        f"<div style='width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; transition: all 0.2s ease;'>"
                        f"<div style='width: 12px; height: 12px; border-radius: 50%; background: #CBD5E1;'></div>"
                        f"</div></a>"
                    )
                dots_html.append(node)
                if i < 8:
                    line_color = "#94A3B8" if i < step else "#E2E8F0"
                    dots_html.append(
                        f"<div style='flex: 1; max-width: 38px; min-width: 14px; height: 2px; background: {line_color}; transition: background 0.3s ease;'></div>"
                    )

            stepper_html = f"""
            <div style='display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 0.2rem 0 0.6rem 0;'>
                <div style='display: flex; align-items: center; justify-content: center; max-width: 440px; width: 100%; margin: 0 auto;'>
                    {''.join(dots_html)}
                </div>
                <div style='margin-top: 10px; font-size: 0.80rem; font-weight: 600; color: #64748B; letter-spacing: 0.08em; text-transform: uppercase; text-align: center;'>
                    {step_labels[step]}
                </div>
                <div style='margin-top: 2px; font-size: 1.15rem; font-weight: 700; color: #111111; text-align: center; letter-spacing: -0.01em;'>
                    {step_titles[step]}
                </div>
            </div>
            """
            st.markdown(stepper_html, unsafe_allow_html=True)

        with s_nav3:
            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
            if step < 8:
                if st.button(f"Step {step + 1} →", key="btn_step_next", use_container_width=True):
                    st.session_state["demo_step"] = step + 1
                    st.rerun()
            else:
                if st.button("Practical Lab →", key="btn_step_finish", use_container_width=True):
                    st.session_state["nav_mode"] = "Live Experiment"
                    st.session_state["demo_step"] = 0
                    st.session_state["_pending_nav_mode"] = "Live Experiment"
                    st.rerun()

        st.markdown("<hr style='border: none; border-bottom: 1px solid #ECECEC; margin: 0.6rem 0 1.25rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div class='step-content-container'>", unsafe_allow_html=True)

        def render_image_card(image_path: Union[str, Path], max_height: int = 380, caption: str = ""):
            p = Path(image_path)
            if p.is_file():
                b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
                ext = "jpeg" if p.suffix.lower() in [".jpg", ".jpeg"] else ("gif" if p.suffix.lower() == ".gif" else "png")
                cap_html = f"<div style='font-size: 0.82rem; color: #555555; text-align: center; margin-top: 0.45rem; font-weight: 500;'>{caption}</div>" if caption else ""
                img_html = (
                    f"<div style='text-align: center; margin: 1rem 0 1.25rem 0;'>"
                    f"<img src='data:image/{ext};base64,{b64}' style='max-width: 100%; max-height: {max_height}px; object-fit: contain; border-radius: 8px; border: 1px solid #EBEBEB; box-shadow: 0 2px 8px rgba(0,0,0,0.04); background-color: #FFFFFF;' />"
                    f"{cap_html}"
                    f"</div>"
                )
                st.markdown(img_html, unsafe_allow_html=True)

        # Step 1: Clinical Problem & Real-World Motivation

        if step == 1:
            st.markdown(
                "<div style='font-size: 1.05rem; line-height: 1.55; color: #222222; margin-bottom: 0.8rem;'>"
                "<b>Cardiovascular diseases represent the leading cause of global mortality (~17.9 million deaths annually).</b> "
                "While the standard 12-lead ECG is the clinical gold standard for detecting myocardial infarction and arrhythmias, "
                "expert cardiologists are frequently unavailable in emergency triage. CardioSense delivers transparent, leak-free machine-learning triage directly from voltage signals."
                "</div>",
                unsafe_allow_html=True,
            )

            # High-resolution scientific 5-stage pipeline infographic
            render_image_card(
                "docs/assets/five_stage_pipeline_infographic.jpg",
                max_height=360,
            )

            # Elegant horizontal scientific pipeline track
            st.markdown(
                """
                <div class='pipeline-track'>
                    <div class='pipeline-step'>
                        <div class='pipeline-dot'>01</div>
                        <div class='pipeline-label'>Cardiac Origin</div>
                        <div class='pipeline-sub'>SA node depolarization spreads through atria & ventricles.</div>
                    </div>
                    <div class='pipeline-connector'></div>
                    <div class='pipeline-step'>
                        <div class='pipeline-dot'>02</div>
                        <div class='pipeline-label'>Dual Ingestion</div>
                        <div class='pipeline-sub'>12-Lead PTB-XL (100 Hz) & Single-Lead MIT-BIH (360 Hz).</div>
                    </div>
                    <div class='pipeline-connector'></div>
                    <div class='pipeline-step'>
                        <div class='pipeline-dot'>03</div>
                        <div class='pipeline-label'>Zero-Phase Filter</div>
                        <div class='pipeline-sub'>Cleans drift & mains hum with zero ST-phase distortion (θ=0°).</div>
                    </div>
                    <div class='pipeline-connector'></div>
                    <div class='pipeline-step'>
                        <div class='pipeline-dot'>04</div>
                        <div class='pipeline-label'>Decoupled ML</div>
                        <div class='pipeline-sub'>Multi-Output Logistic Regression & Balanced Random Forest.</div>
                    </div>
                    <div class='pipeline-connector'></div>
                    <div class='pipeline-step'>
                        <div class='pipeline-dot'>05</div>
                        <div class='pipeline-label'>Calibrated Output</div>
                        <div class='pipeline-sub'>Validation-tuned thresholds balancing sensitivity & specificity.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Statement introducing the three interactive research laboratories
            st.markdown(
                """
                <div style='margin: 1.8rem 0 0.8rem 0;'>
                    <span class='step-badge' style='font-size: 0.72rem; letter-spacing: 0.08em;'>RESEARCH LABORATORIES</span>
                    <div style='font-size: 0.94rem; color: #333333; font-weight: 500; margin-top: 0.35rem;'>
                        The platform provides three specialized research laboratories for hands-on clinical and signal experimentation:
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_s1_a, col_s1_b, col_s1_c = st.columns(3)
            with col_s1_a:
                st.markdown(
                    """
                    <div class='clean-card' style='height: 100%;'>
                    <span class='concept-badge'>LAB 1 · 12-LEAD CLINICAL</span><br>
                    <strong style='font-size: 0.95rem;'>Multi-Label Pathology (PTB-XL)</strong>
                    <p style='font-size: 0.83rem; color: #555; margin-top: 0.4rem; line-height: 1.45;'>
                    Detects acute Myocardial Infarction (MI), ST/T Changes (STTC), Conduction Disturbance (CD), and Hypertrophy (HYP).
                    </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_s1_b:
                st.markdown(
                    """
                    <div class='clean-card' style='height: 100%;'>
                    <span class='concept-badge'>LAB 2 · SINGLE-LEAD HOLTER</span><br>
                    <strong style='font-size: 0.95rem;'>Beat-by-Beat Arrhythmia (MIT-BIH)</strong>
                    <p style='font-size: 0.83rem; color: #555; margin-top: 0.4rem; line-height: 1.45;'>
                    Autonomous Pan-Tompkins QRS locator isolates Premature Ventricular Contractions (PVCs) and Supraventricular Ectopics.
                    </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_s1_c:
                st.markdown(
                    """
                    <div class='clean-card' style='height: 100%;'>
                    <span class='concept-badge'>LAB 3 · ARCHIVAL ISOLATION</span><br>
                    <strong style='font-size: 0.95rem;'>Optical Paper Digitizer</strong>
                    <p style='font-size: 0.83rem; color: #555; margin-top: 0.4rem; line-height: 1.45;'>
                    Digitizes paper records via adaptive thresholding. Kept strictly isolated from diagnostic ML models to prevent image artifacts.
                    </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            render_step_footer(
                prev_txt="Open Practical Lab",
                prev_key="btn_s1_lab",
                prev_target_step=0,
                prev_target_mode="Live Experiment",
                next_txt="Continue to Step 2: Research Datasets →",
                next_key="btn_s1_next",
                next_target_step=2,
            )

        # Step 2: Research Datasets & Leak-Free Cohort Splits
        elif step == 2:
            st.markdown(
                "<div style='font-size: 1.05rem; line-height: 1.55; color: #222222; margin-bottom: 0.8rem;'>"
                "<b>Patient-level data leakage</b> is a critical pitfall in biomedical AI. "
                "If records from the same subject appear in both training and testing, models memorize patient baselines rather than generalized pathology. "
                "CardioSense enforces strict, peer-reviewed patient-disjoint partitions."
                "</div>",
                unsafe_allow_html=True,
            )

            # Side-by-side Plotly bar chart
            fig_ds = create_dataset_comparison_figure()
            st.plotly_chart(fig_ds, width='stretch', theme=None)

            col_ds1, col_ds2 = st.columns(2)
            with col_ds1:
                st.markdown(
                    """
                    <div class='clean-card'>
                    <strong style='font-size: 1rem;'>PTB-XL v1.0.3 (12-Lead Clinical Benchmark)</strong><br>
                    <span style='font-size: 0.85rem; color: #555; line-height: 1.5;'>
                    • <b>Total Cohort:</b> 21,799 ten-second records from 18,885 distinct patients<br>
                    • <b>Channels:</b> Standard 12 leads (I, II, III, aVR, aVL, aVF, V1-V6) at 100 Hz ($\text{mV}$)<br>
                    • <b>Diagnostic Classes:</b> NORM (9,528), MI (5,486), STTC (5,250), CD (4,907), HYP (2,655)<br>
                    • <b>Split Protocol:</b> Official 10-fold patient-stratified split; Fold 10 (2,198 records) held-out exclusively for testing
                    </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_ds2:
                st.markdown(
                    """
                    <div class='clean-card'>
                    <strong style='font-size: 1rem;'>MIT-BIH Arrhythmia Database (Single-Lead Holter)</strong><br>
                    <span style='font-size: 0.85rem; color: #555; line-height: 1.5;'>
                    • <b>Total Cohort:</b> 48 two-channel 30-minute ambulatory Holter recordings (~109,000 beats) at 360 Hz<br>
                    • <b>Taxonomy:</b> N (Normal), S (Supraventricular), V (Ventricular), Q (Fusion & Other)<br>
                    • <b>Split Protocol:</b> Inter-patient split (de Chazal et al.): DS1 (22 records) train/val vs DS2 (21 records) test<br>
                    • <b>Quarantine:</b> Paced records (102, 104, 107, 217) quarantined; records 201/202 grouped to sub_201_202
                    </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            render_step_footer(
                prev_txt="← Previous Step",
                prev_key="btn_s2_prev",
                prev_target_step=1,
                next_txt="Continue to Step 3: ECG Anatomy →",
                next_key="btn_s2_next",
                next_target_step=3,
            )

        # Step 3: Anatomy of an ECG Signal & Electrophysiology
        elif step == 3:
            st.markdown(
                "<div style='font-size: 1.05rem; line-height: 1.55; color: #222222; margin-bottom: 0.8rem;'>"
                "An electrocardiogram records continuous millivolt potentials generated by myocardial depolarization and repolarization. "
                "Each electrical deflection corresponds to a precise stage in cardiac excitation-contraction coupling."
                "</div>",
                unsafe_allow_html=True,
            )

            # Cardiac Anatomy Waveform Figure with direct fiducial annotations
            fig_anat = create_cardiac_anatomy_figure()
            st.plotly_chart(fig_anat, width='stretch', theme=None)

            col_w1, col_w2, col_w3, col_w4 = st.columns(4)
            with col_w1:
                st.markdown("<div class='clean-card'><strong style='font-size: 0.88rem;'>P Wave (80–100 ms)</strong><br><span style='font-size: 0.80rem; color: #555;'>Sinoatrial node firing initiates atrial depolarization, pumping blood through AV valves.</span></div>", unsafe_allow_html=True)
            with col_w2:
                st.markdown("<div class='clean-card'><strong style='font-size: 0.88rem;'>PR Segment (80 ms)</strong><br><span style='font-size: 0.80rem; color: #555;'>Conduction delay at the atrioventricular (AV) node allows ventricles time to fill.</span></div>", unsafe_allow_html=True)
            with col_w3:
                st.markdown("<div class='clean-card'><strong style='font-size: 0.88rem;'>QRS Complex (80–120 ms)</strong><br><span style='font-size: 0.80rem; color: #555;'>Rapid ventricular depolarization via His bundle & Purkinje fibers, triggering systole.</span></div>", unsafe_allow_html=True)
            with col_w4:
                st.markdown("<div class='clean-card'><strong style='font-size: 0.88rem;'>ST-T Wave (160 ms)</strong><br><span style='font-size: 0.80rem; color: #555;'>Ventricular repolarization. ST displacement >0.1 mV signals acute transmural ischemia (STEMI).</span></div>", unsafe_allow_html=True)

            # Real Data Waveform Viewer
            st.markdown("<br><strong style='font-size: 0.95rem; color: #111;'>Calibrated Real-World Signal Explorer</strong>", unsafe_allow_html=True)
            demo_ptb = load_demo_ptbxl_records()
            if demo_ptb:
                rec_sample = demo_ptb[0]
                col_sel1, col_sel2 = st.columns([1, 2])
                with col_sel1:
                    avail_leads = rec_sample.get("lead_names", WF_LEAD_NAMES)
                    chosen_lead = st.selectbox("Select ECG Lead Channel", avail_leads, index=1, key="step3_lead_select")
                lead_idx = avail_leads.index(chosen_lead) if chosen_lead in avail_leads else 1
                sig_raw = np.array(rec_sample["signal_mv"])[:, lead_idx]
                time_ax = np.arange(len(sig_raw)) / 100.0

                fig_real = go.Figure()
                fig_real.add_trace(go.Scatter(x=time_ax, y=sig_raw, mode="lines", line=dict(color="#000000", width=1.5), name=f"{chosen_lead} (mV)"))
                fig_real.update_layout(
                    title=f"Calibrated 10-Second Digital Voltage Trace — {chosen_lead} (Record #{rec_sample['ecg_id']}, 100 Hz)",
                    xaxis_title="Time (seconds)",
                    yaxis_title="Voltage (mV)",
                    height=280,
                    margin=dict(l=45, r=20, t=35, b=30),
                )
                fig_real = apply_clean_scientific_theme(fig_real)
                st.plotly_chart(fig_real, width='stretch', theme=None)

            render_step_footer(
                prev_txt="← Previous Step",
                prev_key="btn_s3_prev",
                prev_target_step=2,
                next_txt="Continue to Step 4: Digital Filters →",
                next_key="btn_s3_next",
                next_target_step=4,
            )

        # Step 4: Digital Signal Conditioning & Pan-Tompkins QRS Cascade
        elif step == 4:
            st.markdown(
                "<div style='font-size: 1.05rem; line-height: 1.55; color: #222222; margin-bottom: 0.8rem;'>"
                "Raw physiological recordings contain respiration baseline drift (&lt;0.5 Hz) and 50/60 Hz electromagnetic mains hum. "
                "Preserving cardiac morphology requires <b>zero-phase forward-backward filtering</b> (<code>filtfilt</code>), "
                "which guarantees exactly 0° phase distortion (&theta;(&omega;) = 0) and prevents artificial ST-segment elevation. "
                "For beat-by-beat QRS detection, the <b>Pan-Tompkins algorithm</b> extracts peak energy across 5 stages."
                "</div>",
                unsafe_allow_html=True,
            )

            # Real or demo MIT-BIH slice for Pan-Tompkins cascade
            demo_mit = load_demo_mitbih_slice()
            if demo_mit:
                sig_4s = np.array(demo_mit["signal_mv"][:360 * 4])  # 4 seconds at 360 Hz
                fs_mit = demo_mit["sampling_rate_hz"]
            else:
                synth_t = np.linspace(0, 4.0, int(4.0 * 360.0))
                sig_4s = np.sin(2 * np.pi * 1.2 * synth_t) + 1.2 * np.exp(-((synth_t % 0.8 - 0.2) ** 2) / 0.001)
                fs_mit = 360.0

            cascade_stages = compute_pan_tompkins_cascade(sig_4s, fs=fs_mit)
            t_axis_cas = np.arange(len(sig_4s)) / fs_mit
            fig_cas = create_pan_tompkins_cascade_figure(t_axis_cas, cascade_stages)
            st.plotly_chart(fig_cas, width='stretch', theme=None)

            # Before / After Filter Trace
            filt_4s = apply_bandpass_notch_filter(sig_4s, fs=fs_mit, lowcut=1.0, highcut=40.0, notch_freq=50.0)
            fig_ba = create_before_after_filter_figure(t_axis_cas, sig_4s, filt_4s, lead_name="MLII Lead")
            st.plotly_chart(fig_ba, width='stretch', theme=None)

            with st.expander("Filter Frequency & Phase Response (Bode Plot)", expanded=False):
                w_b, m_b, p_b = compute_filter_bode_data(lowcut=1.0, highcut=40.0, fs=fs_mit, notch_freq=50.0)
                fig_bode = create_bode_plot(w_b, m_b, p_b, lowcut=1.0, highcut=40.0, notch_freq=50.0)
                st.plotly_chart(fig_bode, width='stretch', theme=None)

            render_step_footer(
                prev_txt="← Previous Step",
                prev_key="btn_s4_prev",
                prev_target_step=3,
                next_txt="Continue to Step 5: Model Architectures →",
                next_key="btn_s4_next",
                next_target_step=5,
            )

        # Step 5: Model Architectures & Feature Extraction
        elif step == 5:
            st.markdown(
                "<div style='font-size: 1.05rem; line-height: 1.55; color: #222222; margin-bottom: 0.8rem;'>"
                "CardioSense pairs physiological feature extraction with two decoupled, specialized machine-learning architectures:"
                "</div>",
                unsafe_allow_html=True,
            )

            # High-resolution architecture comparison illustration
            render_image_card(
                "docs/assets/decoupled_ml_architectures.jpg",
                max_height=380,
                caption="Decoupled Architectures: 12-Lead Multi-Output Diagnostic Model (PTB-XL) vs Single-Lead Arrhythmia Random Forest (MIT-BIH)",
            )

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(
                    """
                    <div class='clean-card'>
                    <span class='concept-badge'>PTB-XL PIPELINE</span><br>
                    <strong style='font-size: 1rem;'>12-Lead Multi-Output Diagnostic Classifier</strong>
                    <p style='font-size: 0.85rem; color: #555; line-height: 1.5; margin-top: 0.5rem;'>
                    • <b>Input:</b> 12-lead calibrated voltage traces ($100\text{ Hz}$, $10\text{ s} = 1,000 \times 12$ samples)<br>
                    • <b>Classifier:</b> L2-regularized Multi-Output Logistic Regression<br>
                    • <b>Outputs:</b> 5 independent sigmoid probabilities for NORM, MI, STTC, CD, and HYP<br>
                    • <b>Key Advantage:</b> Convex optimization guarantees global optimality, calibrated probabilities, and zero catastrophic forgetting
                    </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_m2:
                st.markdown(
                    """
                    <div class='clean-card'>
                    <span class='concept-badge'>MIT-BIH PIPELINE</span><br>
                    <strong style='font-size: 1rem;'>Single-Lead Arrhythmia Random Forest</strong>
                    <p style='font-size: 0.85rem; color: #555; line-height: 1.5; margin-top: 0.5rem;'>
                    • <b>Input:</b> Ambulatory stream ($360\text{ Hz}$, MLII preferred, V5 fallback)<br>
                    • <b>Features:</b> 47 dimensions (morphological downsampling + RR dynamics + statistical moments)<br>
                    • <b>Classifier:</b> Balanced Random Forest (100 estimators, balanced subsampling to counter 111:1 Normal-to-Fusion imbalance)<br>
                    • <b>Outputs:</b> 4 beat families: N (Normal), S (Supraventricular), V (Ventricular), Q (Fusion & Other)
                    </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with st.expander("Signal Processing & Model Architecture Flowchart", expanded=False):
                st.markdown(
                    """
                    ```mermaid
                    flowchart LR
                        A["12-Lead Continuous ECG (100 Hz)"] --> B["Zero-Phase Bandpass (0.5-40 Hz)"]
                        B --> C["148 Normalized Morphological Features"]
                        C --> D["Multi-Output Logistic Regression"]
                        D --> E["Calibrated Probabilities: NORM, MI, STTC, CD, HYP"]

                        F["Ambulatory Single-Lead (360 Hz)"] --> G["Pan-Tompkins QRS Locator"]
                        G --> H["Beat Window [-200ms, +400ms] & RR Dynamics"]
                        H --> I["47-Feature Vector"]
                        I --> J["Balanced Random Forest (100 Trees)"]
                        J --> K["Beat Family: N, S, V, Q"]
                    ```
                    """
                )

            render_step_footer(
                prev_txt="← Previous Step",
                prev_key="btn_s5_prev",
                prev_target_step=4,
                next_txt="Continue to Step 6: Calibrated Thresholds →",
                next_key="btn_s5_next",
                next_target_step=6,
            )

        # Step 6: Prediction Generation & Calibrated Decision Thresholds
        elif step == 6:
            st.markdown(
                "<div style='font-size: 1.05rem; line-height: 1.55; color: #222222; margin-bottom: 0.8rem;'>"
                "Because clinical conditions have varying natural prevalences, a default $0.5$ decision threshold produces "
                "poor clinical sensitivity for rare pathologies (e.g. Hypertrophy at 12.2% prevalence). "
                "CardioSense applies decision thresholds calibrated on held-out validation data (Fold 9) to maximize balanced F1:"
                "</div>",
                unsafe_allow_html=True,
            )

            # Crystal clear, high-contrast grouped horizontal bar chart
            fig_thresh = create_threshold_calibration_figure()
            st.plotly_chart(fig_thresh, width='stretch', theme=None)

            t_df = pd.DataFrame([
                {"Superclass": "NORM", "Name": "Normal Sinus Rhythm", "Prevalence": "43.7%", "Tuned Threshold": 0.448},
                {"Superclass": "MI", "Name": "Myocardial Infarction", "Prevalence": "25.2%", "Tuned Threshold": 0.334},
                {"Superclass": "STTC", "Name": "ST/T Wave Changes", "Prevalence": "24.1%", "Tuned Threshold": 0.309},
                {"Superclass": "CD", "Name": "Conduction Disturbance", "Prevalence": "22.5%", "Tuned Threshold": 0.311},
                {"Superclass": "HYP", "Name": "Hypertrophy", "Prevalence": "12.2%", "Tuned Threshold": 0.280},
            ])
            st.table(t_df)

            st.markdown(
                """
                <div class='clean-card' style='margin-top: 1rem;'>
                <strong>Why Threshold Tuning is Essential in Medical Decision Support:</strong>
                <p style='font-size: 0.85rem; color: #555; margin-top: 0.4rem; line-height: 1.45;'>
                In acute cardiology, missing an active Myocardial Infarction (False Negative) has catastrophic clinical consequences compared to a false alarm (False Positive). 
                Lowering the decision threshold for MI from 0.500 to 0.334 elevates recall by over 18%, catching atypical subendocardial infarctions without causing excessive false positives.
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            render_step_footer(
                prev_txt="← Previous Step",
                prev_key="btn_s6_prev",
                prev_target_step=5,
                next_txt="Continue to Step 7: Audited Benchmarks →",
                next_key="btn_s6_next",
                next_target_step=7,
            )

        # Step 7: Evaluation Benchmarks & Confusion Matrices
        elif step == 7:
            st.markdown(
                "<div style='font-size: 1.05rem; line-height: 1.6; color: #2E2E2E; margin-bottom: 0.75rem;'>"
                "All benchmarks are evaluated strictly on independent, patient-separated test cohorts. "
                "Below is the audited 4×4 confusion matrix and verified per-class performance metrics:"
                "</div>",
                unsafe_allow_html=True,
            )

            # Audited Confusion Matrix Heatmap
            audited_cm = np.array([
                [38608, 541, 555, 2179],
                [1404, 315, 38, 0],
                [182, 29, 2806, 1],
                [56, 1, 207, 112],
            ])
            class_labels = ["N", "S", "V", "Q"]
            class_descs = {
                "N": "Normal & Bundle Branch",
                "S": "Supraventricular Ectopic",
                "V": "Ventricular Ectopic",
                "Q": "Fusion & Other",
            }

            cm_col1, cm_col2 = st.columns([1, 2])
            with cm_col1:
                norm_mode = st.radio(
                    "Confusion Matrix Display Mode",
                    options=["Raw Beat Counts", "Recall (True Class %)", "Precision (Predicted Class %)"],
                    index=0,
                    key="cm_norm_mode_radio",
                )
            norm_param = "none" if norm_mode == "Raw Beat Counts" else ("true" if "Recall" in norm_mode else "pred")
            fig_cm = create_confusion_matrix_figure(
                audited_cm,
                class_labels,
                title="MIT-BIH Matched Detections (Held-Out Test DS2, N = 47,034)",
                normalize=norm_param,
                descriptions=class_descs,
            )
            st.plotly_chart(fig_cm, width='stretch', theme=None)

            b_c1, b_c2 = st.columns(2)
            with b_c1:
                st.markdown(
                    """
                    <div class='clean-card'>
                    <strong>12-Lead PTB-XL Test Benchmark (Fold 10, N = 2,198)</strong><br>
                    <span style='font-size: 0.88rem; color: #444;'>
                    • <b>Macro ROC-AUC:</b> 0.887 | <b>Macro F1:</b> 0.704<br>
                    • <b>NORM AUC:</b> 0.918 | <b>MI AUC:</b> 0.898<br>
                    • <b>STTC AUC:</b> 0.892 | <b>CD AUC:</b> 0.891 | <b>HYP AUC:</b> 0.835
                    </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with b_c2:
                st.markdown(
                    """
                    <div class='clean-card'>
                    <strong>Matched-Detection Conditional Beat-Classification Metrics (N = 47,034)</strong><br>
                    <span style='font-size: 0.88rem; color: #444;'>
                    • <b>Overall Accuracy:</b> 94.69% | <b>Macro F1:</b> 0.5274<br>
                    • <b>Class N:</b> Support 41,883 | Prec 0.9592 | Rec 0.9218 | F1 0.9401<br>
                    • <b>Class S:</b> Support 1,757 | Prec 0.3555 | Rec 0.1793 | F1 0.2384<br>
                    • <b>Class V:</b> Support 3,018 | Prec 0.7781 | Rec 0.9298 | F1 0.8472<br>
                    • <b>Class Q:</b> Support 376 | Prec 0.0489 | Rec 0.2979 | F1 0.0840
                    </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with st.expander("Benchmark Protocol & Taxonomy Disclosures", expanded=True):
                st.markdown(
                    """
                    Matched-detection conditional beat-classification metrics — classification F1 is calculated only for detector peaks matched to a reference annotation and retained after feature-window boundary checks. Detector false positives (115 peaks) and missed reference beats (542 beats, including boundary-dropped matched beats) are reported separately in detector/count-reconciliation metrics and are not included as errors in this classifier confusion matrix.
                    <br><br>
                    Project-defined 4-class beat grouping: N, S, V, Q, with ventricular fusion beats (F) merged into Q. This is not a standard five-class AAMI EC57 implementation.<br>
                    Target groupings: Class N (Normal & Bundle Branch), Class S (Supraventricular Ectopic), Class V (Premature Ventricular Contraction), Class Q (Fusion & Other) / Ventricular Fusion (F) & Unclassifiable (Q).<br>
                    Reference baseline: Secondary Benchmark: Annotation-Centred (Oracle) Classifier Reference (Macro F1 = 0.5396, N = 47,561).
                    """,
                    unsafe_allow_html=True,
                )

            render_step_footer(
                prev_txt="← Previous Step",
                prev_key="btn_s7_prev",
                prev_target_step=6,
                next_txt="Continue to Step 8: Governance & Safety →",
                next_key="btn_s7_next",
                next_target_step=8,
            )

        # Step 8: Interpretation, Safety Boundaries & Scientific Governance
        elif step == 8:
            st.markdown(
                """
                <div style='margin-bottom: 1.2rem;'>
                    <span class='step-badge' style='font-size: 0.72rem; letter-spacing: 0.08em;'>SCIENTIFIC GOVERNANCE & RIGOR</span>
                    <div style='font-size: 1.12rem; font-weight: 700; color: #111111; margin-top: 0.4rem;'>
                        Algorithmic Transparency & Clinical Safety Guardrails
                    </div>
                    <div style='font-size: 0.90rem; color: #555555; margin-top: 0.25rem; line-height: 1.5;'>
                        CardioSense operates under three binding scientific and regulatory constraints to guarantee clinical reproducibility, eliminate data contamination, and prevent diagnostic overreach.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_g1, col_g2, col_g3 = st.columns(3)
            with col_g1:
                st.markdown(
                    """
                    <div class='gov-card'>
                        <span class='gov-badge'>TIER 1 · REGULATORY</span>
                        <div class='gov-title'>Non-Diagnostic Academic System</div>
                        <ul class='gov-list'>
                            <li class='gov-item'><strong>Research Exploration:</strong> Developed strictly for biomedical signal processing, benchmark evaluation, and algorithmic transparency.</li>
                            <li class='gov-item'><strong>Regulatory Clearance:</strong> Not a medical device. Not certified or cleared by FDA, CE, or CDSCO for clinical diagnosis or triage.</li>
                            <li class='gov-item'><strong>Clinical Protocol:</strong> Algorithmic inferences are non-prescriptive and must never supersede board-certified cardiological evaluation.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_g2:
                st.markdown(
                    """
                    <div class='gov-card'>
                        <span class='gov-badge'>TIER 2 · ARCHITECTURE</span>
                        <div class='gov-title'>Decoupled Optical Workflow</div>
                        <ul class='gov-list'>
                            <li class='gov-item'><strong>Programmatic Air-Gap:</strong> Optical paper digitizer operates in strict isolation from downstream diagnostic ML inference engines.</li>
                            <li class='gov-item'><strong>Artifact Prevention:</strong> Eliminates image perspective distortion, paper fold skew, and scanner grid noise from contaminating voltage classifiers.</li>
                            <li class='gov-item'><strong>Inference Standard:</strong> Machine learning models process only calibrated digital time-series arrays conditioned with zero-phase filters.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_g3:
                st.markdown(
                    f"""
                    <div class='gov-card'>
                        <span class='gov-badge'>TIER 3 · HARDWARE BOUNDARY</span>
                        <div class='gov-title'>Validated Benchmark Replay</div>
                        <ul class='gov-list'>
                            <li class='gov-item'><strong>Benchmark Replay:</strong> Single-lead Holter streaming executes via recorded PhysioNet MIT-BIH recordings (360 Hz) — <code>{DISCLAIMER_RECORDED_REPLAY}</code>.</li>
                            <li class='gov-item'><strong>Sensor Emulation:</strong> AD8232 analog front-end and ESP32 drivers serve as reproducible ingestion models, not physical claims.</li>
                            <li class='gov-item'><strong>Hardware Scope:</strong> No live patient electrodes, wearable patches, or physical bio-potential sensors are connected to this instance.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Institutional Quality Gate Matrix
            st.markdown(
                """
                <div style='margin-top: 1.5rem; background: #FFFFFF; border: 1.5px solid #000000; border-radius: 8px; padding: 0.85rem 1.25rem;'>
                    <div style='font-size: 0.76rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #111; margin-bottom: 0.5rem;'>
                        Institutional Quality Gates & Research Invariants
                    </div>
                    <div style='display: flex; flex-wrap: wrap; gap: 1.5rem; font-size: 0.82rem; color: #333;'>
                        <div><strong>✓ Leak-Free Partition:</strong> Zero patient overlap across training and test folds</div>
                        <div><strong>✓ Zero-Phase Filtering:</strong> θ(ω) = 0° prevents artificial ST elevation</div>
                        <div><strong>✓ Deterministic Cutoffs:</strong> Validation-tuned decision thresholds</div>
                        <div><strong>✓ Ephemeral Memory:</strong> Zero patient ECG signals retained on disk</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br><br>", unsafe_allow_html=True)
            col_cta1, col_cta2, col_cta3 = st.columns([1, 1.6, 1])
            with col_cta2:
                if st.button("Enter Practical Lab (Live Experiment)", key="btn_s8_enter_lab", use_container_width=True):
                    st.session_state["nav_mode"] = "Live Experiment"
                    st.session_state["demo_step"] = 0
                    st.session_state["_pending_nav_mode"] = "Live Experiment"
                    st.rerun()

            render_step_footer(
                prev_txt="← Previous Step",
                prev_key="btn_s8_prev",
                prev_target_step=7,
                next_txt="Return to Overview →",
                next_key="btn_s8_to_hero",
                next_target_step=0,
            )

        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# VIEW 2: LIVE EXPERIMENT (Practical Lab)
# ==============================================================================

elif st.session_state["nav_mode"] == "Live Experiment":

    st.markdown(
        """
        <style>
        /* Modern Streamlit Tab Styling — Oval Pill Buttons */
        div[data-testid="stTabs"] div[role="tablist"]::after,
        div[data-testid="stTabs"] [role="tablist"]::after,
        div[data-testid="stTabs"] .react-aria-SelectionIndicator,
        .stTabs [role="tablist"]::after,
        .stTabs .react-aria-SelectionIndicator,
        .stTabs [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-border"] {
            display: none !important;
            height: 0 !important;
            opacity: 0 !important;
            visibility: hidden !important;
            border: none !important;
            background: transparent !important;
        }
        div[data-testid="stTabs"] div[role="tablist"],
        div[data-testid="stTabs"] [role="tablist"],
        .stTabs [role="tablist"],
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.85rem !important;
            border: none !important;
            border-bottom: none !important;
            padding-bottom: 0.85rem !important;
            padding-top: 0.35rem !important;
            background: transparent !important;
            display: flex !important;
            flex-wrap: wrap !important;
            align-items: center !important;
        }
        div[data-testid="stTabs"] button[role="tab"],
        div[data-testid="stTabs"] div[role="tab"],
        div[data-testid="stTabs"] [role="tab"],
        div[data-testid="stTabs"] [data-testid="stTab"],
        .stTabs [role="tab"],
        .stTabs [data-testid="stTab"],
        .stTabs [data-baseweb="tab"] {
            background-color: #FFFFFF !important;
            background: #FFFFFF !important;
            border: 1.5px solid #000000 !important;
            border-radius: 9999px !important;
            font-size: 0.90rem !important;
            font-weight: 600 !important;
            color: #111111 !important;
            padding: 0.65rem 1.6rem !important;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
            height: auto !important;
            min-height: 2.5rem !important;
            box-sizing: border-box !important;
            white-space: nowrap !important;
            outline: none !important;
            cursor: pointer !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            margin: 2px 0 !important;
        }
        div[data-testid="stTabs"] button[role="tab"] *,
        div[data-testid="stTabs"] [role="tab"] *,
        div[data-testid="stTabs"] [data-testid="stTab"] *,
        .stTabs [role="tab"] *,
        .stTabs [data-testid="stTab"] *,
        .stTabs [data-baseweb="tab"] * {
            color: #111111 !important;
            font-weight: 600 !important;
            font-size: 0.90rem !important;
        }
        div[data-testid="stTabs"] button[role="tab"] p,
        div[data-testid="stTabs"] [role="tab"] p,
        div[data-testid="stTabs"] [data-testid="stTab"] p {
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1.2 !important;
        }
        div[data-testid="stTabs"] button[role="tab"]:hover,
        div[data-testid="stTabs"] [role="tab"]:hover,
        div[data-testid="stTabs"] [data-testid="stTab"]:hover,
        div[data-testid="stTabs"] [data-testid="stTab"][data-hovered],
        .stTabs [role="tab"]:hover,
        .stTabs [data-testid="stTab"]:hover,
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #F8F9FA !important;
            background: #F8F9FA !important;
            border-color: #000000 !important;
            border-radius: 9999px !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 3px 8px rgba(0, 0, 0, 0.10) !important;
        }
        div[data-testid="stTabs"] button[role="tab"]:focus,
        div[data-testid="stTabs"] [role="tab"]:focus,
        div[data-testid="stTabs"] [data-testid="stTab"]:focus {
            outline: none !important;
            border-radius: 9999px !important;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
        div[data-testid="stTabs"] [role="tab"][aria-selected="true"],
        div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"],
        div[data-testid="stTabs"] button[role="tab"][data-selected],
        div[data-testid="stTabs"] [role="tab"][data-selected],
        div[data-testid="stTabs"] [data-testid="stTab"][data-selected],
        .stTabs [role="tab"][aria-selected="true"],
        .stTabs [data-testid="stTab"][aria-selected="true"],
        .stTabs [data-testid="stTab"][data-selected],
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #E8F5E9 !important;
            background: #E8F5E9 !important;
            border: 2px solid #2E7D32 !important;
            border-radius: 9999px !important;
            box-shadow: 0 0 14px rgba(46, 125, 50, 0.35), 0 2px 6px rgba(0, 0, 0, 0.06) !important;
            transform: translateY(-1px) !important;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] *,
        div[data-testid="stTabs"] [role="tab"][aria-selected="true"] *,
        div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] *,
        div[data-testid="stTabs"] button[role="tab"][data-selected] *,
        div[data-testid="stTabs"] [role="tab"][data-selected] *,
        div[data-testid="stTabs"] [data-testid="stTab"][data-selected] *,
        .stTabs [role="tab"][aria-selected="true"] *,
        .stTabs [data-testid="stTab"][aria-selected="true"] *,
        .stTabs [data-baseweb="tab"][aria-selected="true"] * {
            color: #1B5E20 !important;
            font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    tab_12lead, tab_single, tab_ad8232, tab_digitizer = st.tabs([
        "12-Lead Diagnostic Classification (PTB-XL)",
        "Single-Lead Lab",
        "Hardware Prototype AD8232",
        "Optical Paper Digitizer",
    ])

    # --------------------------------------------------------------------------
    # Tab 1: 12-Lead Diagnostic Classification (PTB-XL)
    # --------------------------------------------------------------------------
    with tab_12lead:
        db_path = Path(WF_DATA_DIR) / "ptbxl_database.csv"
        data_path = Path(WF_DATA_DIR)
        use_cloud_demo_ptb = False
        avail_ids = []
        meta_df = None
        demo_recs_by_id = {}

        if db_path.is_file():
            try:
                meta_df = load_ptbxl_metadata(WF_DATA_DIR)
                avail_ids = [
                    eid for eid, row in meta_df.iterrows()
                    if (data_path / f"{row['filename_lr']}.hea").is_file()
                ]
            except Exception:
                avail_ids = []

        if not avail_ids:
            demo_list = load_demo_ptbxl_records()
            if demo_list:
                use_cloud_demo_ptb = True
                demo_recs_by_id = {int(r["ecg_id"]): r for r in demo_list}
                meta_df = pd.DataFrame(demo_list).set_index("ecg_id")
                avail_ids = list(demo_recs_by_id.keys())
                st.info(
                    "**Cloud Demo Mode:** Displaying 5 pre-packaged PTB-XL benchmark records covering all diagnostic superclasses "
                    "(NORM, MI, CD, STTC, HYP). Full 21,837-record database requires local disk mounting in `data/ptb-xl`."
                )

        if not avail_ids or meta_df is None:
            st.error(f"PTB-XL database file not found at `{db_path}` and demo fixtures unavailable.")
        else:
            try:
                # 1. Select Data
                s_c1, s_c2 = st.columns([1, 2])
                with s_c1:
                    fold_options = ["Held-Out Test Fold (10)"] if use_cloud_demo_ptb else ["Held-Out Test Fold (10)", "Validation Fold (9)", "Train Folds (1-8)", "All Folds"]
                    fold_choice = st.selectbox("Partition Fold", fold_options)
                filt_df = meta_df.loc[avail_ids]
                if not use_cloud_demo_ptb:
                    if fold_choice == "Held-Out Test Fold (10)":
                        filt_df = filt_df[filt_df["strat_fold"] == 10]
                    elif fold_choice == "Validation Fold (9)":
                        filt_df = filt_df[filt_df["strat_fold"] == 9]
                    elif fold_choice == "Train Folds (1-8)":
                        filt_df = filt_df[filt_df["strat_fold"].isin(range(1, 9))]

                with s_c2:
                    def fmt_ptb_rec(eid: int) -> str:
                        r = filt_df.loc[eid]
                        sc = ", ".join(r["diagnostic_superclasses"]) if r["diagnostic_superclasses"] else "Unmapped"
                        suffix = " [Pre-packaged Demo]" if use_cloud_demo_ptb else ""
                        return f"Record #{int(eid):05d} [Fold {r['strat_fold']}] — Pt #{r['patient_id']} ({sc}){suffix}"
                    selected_eid = st.selectbox("Select Patient Record", options=filt_df.index.tolist(), format_func=fmt_ptb_rec)

                rec_row = filt_df.loc[selected_eid]
                if use_cloud_demo_ptb:
                    d_rec = demo_recs_by_id[int(selected_eid)]
                    raw_sig = np.array(d_rec["signal_mv"], dtype=np.float32)
                    rec_meta = {"fs": int(d_rec.get("fs", 100)), "sig_len": len(raw_sig), "sig_name": d_rec.get("lead_names", WF_LEAD_NAMES)}
                else:
                    rec_path = data_path / str(rec_row["filename_lr"])
                    raw_sig, rec_meta = read_wfdb_record(rec_path)

                    # Metadata metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Patient ID", f"Pt #{rec_row['patient_id']}")
                    m2.metric("Partition", f"Fold {rec_row['strat_fold']}")
                    m3.metric("Sampling Rate", f"{rec_meta['fs']} Hz")
                    gt_label = ", ".join(rec_row["diagnostic_superclasses"]) if rec_row["diagnostic_superclasses"] else "None"
                    m4.metric("Ground Truth", gt_label)

                    # 2. Configure Processing
                    with st.expander("Filter Parameters & Bode Response", expanded=False):
                        fc1, fc2, fc3 = st.columns(3)
                        with fc1:
                            bp_l = st.slider("Bandpass Lowcut (Hz)", 0.1, 2.0, 0.5, 0.1, key="ptb_l")
                        with fc2:
                            bp_h = st.slider("Bandpass Highcut (Hz)", 20.0, 48.0, 40.0, 1.0, key="ptb_h")
                        with fc3:
                            notch_s = st.selectbox("Mains Notch", ["50 Hz", "60 Hz", "Off"], index=0, key="ptb_n")
                        notch_v = 50.0 if notch_s == "50 Hz" else (60.0 if notch_s == "60 Hz" else None)

                        w_b, m_b, p_b = compute_filter_bode_data(lowcut=bp_l, highcut=bp_h, fs=rec_meta['fs'], notch_freq=notch_v)
                        fig_bode = create_bode_plot(w_b, m_b, p_b, lowcut=bp_l, highcut=bp_h, notch_freq=notch_v)
                        st.plotly_chart(fig_bode, width='stretch', theme=None)

                    # Filtered signal
                    filt_sig = np.zeros_like(raw_sig)
                    for l_idx in range(12):
                        filt_sig[:, l_idx] = apply_bandpass_notch_filter(raw_sig[:, l_idx], fs=rec_meta['fs'], lowcut=bp_l, highcut=bp_h, notch_freq=notch_v)

                    # 3. Inspect Signals & Matrices
                    viz_sel = st.radio("Signal View", ["12-Lead Multi-Channel Grid", "Continuous Voltage Matrix Heatmap", "Cross-Lead Correlation Matrix"], horizontal=True)
                    t_ax = np.arange(len(filt_sig)) / float(rec_meta['fs'])

                    if viz_sel == "12-Lead Multi-Channel Grid":
                        fig_leads = make_subplots(rows=4, cols=3, subplot_titles=WF_LEAD_NAMES, vertical_spacing=0.08, horizontal_spacing=0.05)
                        for l_i, l_name in enumerate(WF_LEAD_NAMES):
                            fig_leads.add_trace(go.Scatter(x=t_ax, y=filt_sig[:, l_i], mode="lines", name=l_name, line=dict(color="#000000", width=1.2)), row=(l_i//3)+1, col=(l_i%3)+1)
                        fig_leads.update_layout(height=580, showlegend=False, margin=dict(l=30, r=20, t=30, b=25))
                        fig_leads = apply_clean_scientific_theme(fig_leads)
                        st.plotly_chart(fig_leads, width='stretch', theme=None)
                    elif viz_sel == "Continuous Voltage Matrix Heatmap":
                        st.plotly_chart(create_12lead_matrix_heatmap(filt_sig, fs=rec_meta['fs']), width='stretch', theme=None)
                    else:
                        st.plotly_chart(create_lead_correlation_heatmap(filt_sig), width='stretch', theme=None)

                    # 4. Run Model
                    st.markdown("<br>", unsafe_allow_html=True)
                    m_btn_c1, m_btn_c2, m_btn_c3 = st.columns([1, 1, 1])
                    with m_btn_c2:
                        run_clicked = st.button("Run Diagnostic Classification", type="primary", key="btn_run_ptb_ml", use_container_width=True)

                    if run_clicked:
                        model_path = Path(WF_ARTIFACT_DIR) / "waveform_lr_model.joblib"
                        with st.spinner("Extracting features and running model inference..."):
                            model, manifest = load_verified_model(model_path)
                            feats = extract_record_features(filt_sig).reshape(1, -1)
                            probs = model.predict_proba(feats)
                            scs = manifest.get("target_superclasses", list(TARGET_SUPERCLASSES))
                            cuts = {"NORM": 0.448, "MI": 0.334, "STTC": 0.309, "CD": 0.311, "HYP": 0.280}

                            pred_list = []
                            for idx, sc in enumerate(scs):
                                p_val = float(probs[idx][0, 1]) if isinstance(probs, list) else float(probs[0, idx])
                                cutoff = cuts.get(sc, 0.5)
                                is_pos = p_val >= cutoff
                                pred_list.append({
                                    "superclass": sc,
                                    "probability": round(p_val, 4),
                                    "cutoff": cutoff,
                                    "status": "POS" if is_pos else "NEG",
                                })

                            st.session_state["ptb_ml_results"] = {
                                "selected_eid": selected_eid,
                                "strat_fold": int(rec_row["strat_fold"]),
                                "ground_truth": rec_row["diagnostic_superclasses"] or [],
                                "fs": rec_meta["fs"],
                                "pred_list": pred_list,
                                "scs": scs,
                            }

                    if "ptb_ml_results" in st.session_state and st.session_state["ptb_ml_results"].get("selected_eid") == selected_eid:
                        ptb_saved = st.session_state["ptb_ml_results"]
                        p_list = ptb_saved["pred_list"]
                        st.markdown("#### Diagnostic Prediction Results")
                        res_cols = st.columns(len(p_list))
                        for idx, p_item in enumerate(p_list):
                            with res_cols[idx]:
                                tag = "POS" if p_item["status"] == "POS" else "NEG"
                                st.metric(f"{p_item['superclass']} ({tag})", f"{p_item['probability']:.1%}", delta=f"Cutoff: {p_item['cutoff']:.1%}")

                        with st.expander("Diagnostic Contingency Benchmark Matrices (Fold 10)", expanded=False):
                            c_tabs = st.tabs(ptb_saved["scs"])
                            for idx, sc in enumerate(ptb_saved["scs"]):
                                with c_tabs[idx]:
                                    st.plotly_chart(create_multilabel_contingency_figure(sc), width='stretch', theme=None)

                        # Explain Results Action
                        ptb_payload = {
                            "workflow": "workflow_b_12lead",
                            "source_category": f"PTB-XL Record (Fold {ptb_saved['strat_fold']})",
                            "is_simulation": False,
                            "sampling_rate_hz": ptb_saved["fs"],
                            "units_and_calibration": "Calibrated 12-lead digital ECG (mV)",
                            "summary_metrics": {p["superclass"]: f"{p['probability']:.1%}" for p in p_list},
                            "class_predictions": p_list,
                            "data_quality_flags": [
                                "Standard 12 leads processed (I, II, III, aVR, aVL, aVF, V1-V6)",
                                "Zero-phase forward-backward Butterworth filter applied",
                            ],
                            "evaluation_context": {
                                "partition_fold": ptb_saved["strat_fold"],
                                "ground_truth_labels": ptb_saved["ground_truth"],
                            },
                            "known_limitations": [
                                "Multi-output logistic regression model; research prototype only",
                                "Decision cutoffs calibrated on validation Fold 9",
                                "Predicted labels are statistical pattern classes, NOT clinical diagnoses",
                            ],
                        }
                        render_explain_results_button(
                            workflow="workflow_b_12lead",
                            payload_data=ptb_payload,
                            result_id=f"ptb_{selected_eid}",
                            title="12-Lead Diagnostic Model Explanation",
                        )

            except Exception as e:
                st.error(f"Error accessing PTB-XL records: {e}")

    # --------------------------------------------------------------------------
    # Tab 2: Single-Lead Lab
    # --------------------------------------------------------------------------
    with tab_single:
        st.markdown("### Single-Lead Lab")
        st.markdown(
            "<span style='font-size: 0.85rem; color: #666;'>MIT-BIH Arrhythmia Database · 360 Hz Single-Lead MLII/V1 · "
            "<strong>RECORDED DATA REPLAY — NOT LIVE HARDWARE</strong></span>",
            unsafe_allow_html=True,
        )
        st.caption(f"Research Notice: {DISCLAIMER_SINGLE_LEAD_RESEARCH}")

        mit_recs = []
        try:
            mit_recs = get_available_records()
        except Exception:
            mit_recs = []

        use_demo_mit = False
        if not mit_recs:
            demo_mit_dict = load_demo_mitbih_slice()
            if demo_mit_dict:
                use_demo_mit = True
                mit_recs = ["100 (Cloud Demo)"]
                st.info(
                    "**Cloud Demo Mode:** Displaying pre-packaged MIT-BIH benchmark record #100 (Lead MLII, 360 Hz). "
                    "Full 48-record database requires local disk mounting in `data/mitdb`."
                )

        if not mit_recs:
            st.error("MIT-BIH records not found at `data/mitdb` and demo fixtures unavailable.")
        else:
            try:
                c_sel1, c_sel2, c_sel3 = st.columns(3)
                with c_sel1:
                    sel_m = st.selectbox("MIT-BIH Record", options=mit_recs, index=0, key="sel_mit")
                with c_sel2:
                    start_s = st.number_input("Window Start (seconds)", 0.0, 1750.0 if not use_demo_mit else 0.0, 10.0 if not use_demo_mit else 0.0, 5.0 if not use_demo_mit else 1.0, disabled=use_demo_mit)
                with c_sel3:
                    dur_s = st.selectbox("Duration (seconds)", [5.0, 10.0, 20.0, 30.0] if not use_demo_mit else [10.0], index=1 if not use_demo_mit else 0, disabled=use_demo_mit)

                if use_demo_mit:
                    pb_res = get_demo_playback_slice()
                else:
                    pb_res = get_playback_slice(record_id=sel_m, start_second=start_s, duration_seconds=dur_s)

                with st.expander("Filter & Pan-Tompkins Parameters", expanded=False):
                    f1, f2, f3, f4 = st.columns(4)
                    with f1:
                        sl_l = st.slider("Lowcut (Hz)", 1.0, 10.0, 5.0, 0.5, key="sl_l")
                    with f2:
                        sl_h = st.slider("Highcut (Hz)", 10.0, 40.0, 15.0, 1.0, key="sl_h")
                    with f3:
                        sl_w = st.slider("Kernel (ms)", 50, 300, 150, 10, key="sl_w")
                    with f4:
                        sl_n = st.selectbox("Notch Filter", ["60 Hz", "50 Hz", "Off"], index=0, key="sl_n")
                    sl_nv = 60.0 if "60 Hz" in sl_n else (50.0 if "50 Hz" in sl_n else None)

                sl_view = st.radio("Signal View", ["Annotated R-Peaks & Playback", "Before-and-After Filter Comparison", "Pan-Tompkins 5-Stage Filter Cascade (Raw → Bandpass → Derivative → Squared → Integrated)"], horizontal=True)

                if sl_view == "Annotated R-Peaks & Playback":
                    fig_p = go.Figure()
                    fig_p.add_trace(go.Scatter(x=pb_res.time_axis, y=pb_res.signal_mv, mode="lines", name="ECG (mV)", line=dict(color="#000000", width=1.6)))
                    if pb_res.detected_beats:
                        b_t = [b.time_seconds for b in pb_res.detected_beats]
                        b_v = [pb_res.signal_mv[max(0, min(len(pb_res.signal_mv)-1, int((b.time_seconds - pb_res.start_second)*pb_res.sampling_rate_hz)))] for b in pb_res.detected_beats]
                        b_c = [b.predicted_class for b in pb_res.detected_beats]
                        fig_p.add_trace(go.Scatter(
                            x=b_t, y=b_v, mode="markers+text", text=b_c, textposition="top center",
                            name="Detected Beats",
                            textfont=dict(color="#000000", size=11, family="Inter, sans-serif"),
                            marker=dict(symbol="circle-dot", size=8, color="#9E3D34"),
                        ))
                    fig_p.update_layout(height=340, margin=dict(l=35, r=20, t=30, b=25))
                    fig_p = apply_clean_scientific_theme(fig_p)
                    st.plotly_chart(fig_p, width='stretch', theme=None)

                elif sl_view == "Before-and-After Filter Comparison":
                    f_sl = apply_bandpass_notch_filter(pb_res.signal_mv, fs=pb_res.sampling_rate_hz, lowcut=sl_l, highcut=sl_h, notch_freq=sl_nv)
                    st.plotly_chart(create_before_after_filter_figure(pb_res.time_axis, pb_res.signal_mv, f_sl, lead_name=pb_res.lead_name), width='stretch', theme=None)

                else:
                    casc = compute_pan_tompkins_cascade(pb_res.signal_mv, fs=pb_res.sampling_rate_hz, lowcut=sl_l, highcut=sl_h, window_sec=float(sl_w/1000.0), notch_freq=sl_nv)
                    pk_t = [b.time_seconds for b in pb_res.detected_beats] if pb_res.detected_beats else None
                    st.plotly_chart(create_pan_tompkins_cascade_figure(pb_res.time_axis, casc, peak_times=pk_t), width='stretch', theme=None)

                # Classification Summary
                if pb_res.detected_beats:
                    bc = pd.Series([b.predicted_class for b in pb_res.detected_beats]).value_counts().to_dict()
                    cn, cs, cv, cq = st.columns(4)
                    cn.metric("Class N (Normal)", bc.get("N", 0))
                    cs.metric("Class S (Supraventricular)", bc.get("S", 0))
                    cv.metric("Class V (Premature Ventricular)", bc.get("V", 0))
                    cq.metric("Class Q (Fusion & Other)", bc.get("Q", 0))

                    st.markdown("##### Beat-by-Beat Classification Results")
                    sl_rows = []
                    for idx, b in enumerate(pb_res.detected_beats, 1):
                        r_data = {
                            "Beat #": idx,
                            "Time (s)": round(b.time_seconds, 3),
                            "Peak Sample": b.peak_sample,
                            "Predicted Class": b.predicted_class,
                            "Class Description": AAMI_CLASS_DESCRIPTIONS.get(b.predicted_class, b.predicted_class),
                        }
                        if b.class_probabilities:
                            r_data["Confidence (Uncalibrated)"] = round(max(b.class_probabilities.values()), 4)
                            for c_k in AAMI_TARGET_CLASSES:
                                r_data[f"P({c_k})"] = round(b.class_probabilities.get(c_k, 0.0), 4)
                        if b.nearest_reference_class:
                            r_data["Ref Class"] = b.nearest_reference_class
                            r_data["Ref Symbol"] = b.nearest_reference_symbol or ""
                            r_data["Agreement"] = "Match" if b.predicted_class == b.nearest_reference_class else "Mismatch"
                        sl_rows.append(r_data)
                    df_sl_beats = pd.DataFrame(sl_rows)
                    st.dataframe(df_sl_beats, use_container_width=True, hide_index=True)

                    # Explain Results Action
                    sl_payload = {
                        "workflow": "workflow_c_single_lead",
                        "source_category": f"MIT-BIH Holter Record (Record {sel_m}, window {start_s}s-{start_s+dur_s}s)",
                        "is_simulation": False,
                        "sampling_rate_hz": float(pb_res.sampling_rate_hz),
                        "units_and_calibration": "Calibrated single-lead Holter ECG (mV)",
                        "summary_metrics": {
                            "total_detected_beats": len(pb_res.detected_beats),
                            "class_counts": bc,
                            "lead_name": pb_res.lead_name,
                        },
                        "class_predictions": [
                            {
                                "beat_index": idx,
                                "time_seconds": round(b.time_seconds, 3),
                                "predicted_class": b.predicted_class,
                                "class_description": AAMI_CLASS_DESCRIPTIONS.get(b.predicted_class, b.predicted_class),
                                "confidence": round(max(b.class_probabilities.values()), 4) if b.class_probabilities else None,
                            }
                            for idx, b in enumerate(pb_res.detected_beats, 1)
                        ],
                        "data_quality_flags": [
                            "RECORDED DATA REPLAY — NOT LIVE HARDWARE",
                            "Zero-phase Butterworth bandpass filter applied",
                            "Pan-Tompkins autonomous QRS peak detector cascade",
                            "Balanced Random Forest beat-window classifier (100 trees)",
                        ],
                        "evaluation_context": {
                            "taxonomy": "Project-defined 4-class beat grouping: N, S, V, Q (F fusion beats merged into Q)",
                            "benchmark_test_support": "Class N: 41,883 | Class S: 1,757 | Class V: 3,018 | Class Q: 376 (DS2)",
                            "metric_type": "Matched-detection conditional beat-classification metrics",
                        },
                        "known_limitations": [
                            "Academic research prototype; not certified for clinical diagnosis or acute coronary triage",
                            "Ventricular fusion (F) is merged into Q; this is not a standard 5-class AAMI EC57 setup",
                            "Confidence values are uncalibrated Random Forest tree voting fractions",
                        ],
                    }
                    render_explain_results_button(
                        workflow="workflow_c_single_lead",
                        payload_data=sl_payload,
                        result_id=f"sl_{sel_m}_{start_s}_{dur_s}",
                        title="Single-Lead Beat Classification Explanation",
                    )

                with st.expander("Interactive Beat Confusion Matrix Benchmark (N = 47,034)", expanded=False):
                    st.markdown("##### Matched-Detection Conditional Beat-Classification Metrics (N = 47,034)")
                    bm_opt = st.radio("Benchmark", ["Matched-Detection Conditional (N = 47,034)", "Secondary Benchmark: Annotation-Centred (Oracle) Classifier Reference"], horizontal=True)
                    norm_opt = st.selectbox("Normalization", ["None (Raw Beat Counts)", "True Class (Recall %)", "Predicted Class (Precision %)"])
                    n_code = "none" if "None" in norm_opt else ("true" if "True" in norm_opt else "pred")
                    st.plotly_chart(create_confusion_matrix_figure(is_oracle="Oracle" in bm_opt, normalize=n_code), width='stretch', theme=None)

                    st.markdown(
                        "<p style='font-size: 0.82rem; color: #777;'>"
                        "Project-defined 4-class beat grouping: N, S, V, Q, with ventricular fusion beats (F) merged into Q. This is not a standard five-class AAMI EC57 implementation.<br>"
                        "Target breakdown: Ventricular Fusion (F) & Unclassifiable (Q).<br>"
                        "Audited Test Supports: Class N: 41,883 (Precision: 0.9592, Recall: 0.9218) | Class S: 1,757 (Precision: 0.3555, Recall: 0.1793) | "
                        "Class V: 3,018 (Precision: 0.7781, Recall: 0.9298) | Class Q: 376 (Precision: 0.0489, Recall: 0.2979).<br>"
                        "Matched-detection conditional beat-classification metrics — classification F1 is calculated only for detector peaks matched to a reference annotation and retained after feature-window boundary checks. "
                        "Detector false positives (115 peaks) and missed reference beats (542 beats, including boundary-dropped matched beats) are reported separately in detector/count-reconciliation metrics and are not included as errors in this classifier confusion matrix."
                        "</p>",
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.error(f"Error accessing MIT-BIH recordings: {e}")

    # --------------------------------------------------------------------------
    # Tab 3: Hardware Prototype AD8232
    # --------------------------------------------------------------------------
    with tab_ad8232:
        st.markdown("### Hardware Prototype AD8232")

        sim_conn = st.radio(
            "Acquisition Interface",
            ["Offline Software Simulation (Recommended)", "Physical Serial Port (Not available in this environment)"],
            horizontal=True,
            key="ad8232_sim_conn",
        )

        if sim_conn == "Physical Serial Port (Not available in this environment)":
            st.error(
                "Status: Port Disconnected / Unavailable. No physical serial adapter detected in this environment. "
                "Inference blocked: hardware calibration unknown. Physical ADC streams cannot be fed into trained models "
                "without verified analog front-end calibration."
            )

        col_src, col_preset = st.columns(2)
        with col_src:
            sim_source_type = st.selectbox(
                "ECG Signal Source",
                ["MIT-BIH recorded ECG", "Deterministic synthetic ECG fixture"],
                key="ad8232_src_type",
            )
        with col_preset:
            sim_preset_choice = st.selectbox(
                "Simulation Preset",
                ["Near-lossless software round trip", "Noisy sensor-like stress test", "Custom parameters"],
                key="ad8232_preset_choice",
            )

        # Preset default parameters
        preset_info = SIMULATION_PRESETS.get(sim_preset_choice, {})
        default_baseline = preset_info.get("baseline_count", 512.0)
        default_gain = preset_info.get("gain_counts_per_mv", 150.0)
        default_noise = preset_info.get("noise_amplitude_mv", 0.0)
        default_drift = preset_info.get("baseline_wander_amplitude_mv", 0.0)
        default_hum = preset_info.get("mains_hum_amplitude_mv", 0.0)

        avail_recs = []
        try:
            avail_recs = get_available_records()
        except Exception:
            avail_recs = []

        if sim_source_type == "MIT-BIH recorded ECG":
            c_m1, c_m2, c_m3 = st.columns(3)
            with c_m1:
                sel_sim_rec = st.selectbox(
                    "MIT-BIH Record",
                    options=avail_recs if avail_recs else ["100"],
                    index=0,
                    key="ad8232_sel_rec",
                )
            with c_m2:
                sim_start_sec = st.number_input("Window Start (seconds)", 0.0, 1750.0, 10.0, 5.0, key="ad8232_start_sec")
            with c_m3:
                sim_duration_sec = st.selectbox("Duration (seconds)", [5.0, 10.0, 20.0, 30.0], index=1, key="ad8232_dur_sec")
            synth_bpm = 72.0
            synth_seed = 42
        else:
            c_s1, c_s2, c_s3 = st.columns(3)
            with c_s1:
                synth_bpm = st.slider("Heart Rate (BPM)", 45.0, 150.0, 72.0, 1.0, key="ad8232_synth_bpm")
            with c_s2:
                synth_seed = st.number_input("Random Seed", 0, 999999, 42, 1, key="ad8232_synth_seed")
            with c_s3:
                sim_duration_sec = st.selectbox("Duration (seconds)", [5.0, 10.0, 20.0, 30.0], index=1, key="ad8232_dur_sec")
            sel_sim_rec = "SYNTH"
            sim_start_sec = 0.0

        with st.expander("Simulation ADC & Perturbation Parameters", expanded=(sim_preset_choice == "Custom parameters")):
            p_c1, p_c2, p_c3 = st.columns(3)
            with p_c1:
                sim_baseline = st.number_input("ADC Baseline (counts @ 0 mV)", 0.0, 1023.0, float(default_baseline), 1.0, key="ad8232_base")
                sim_noise = st.slider("Additive Noise (mV)", 0.0, 0.20, float(default_noise), 0.01, key="ad8232_noise")
            with p_c2:
                sim_gain = st.number_input("ADC Gain (counts/mV)", 10.0, 500.0, float(default_gain), 10.0, key="ad8232_gain")
                sim_drift = st.slider("Baseline Wander (mV @ 0.25 Hz)", 0.0, 0.50, float(default_drift), 0.05, key="ad8232_drift")
            with p_c3:
                st.caption(f"Preset: **{sim_preset_choice}**")
                st.caption(preset_info.get("description", "User-customized simulation parameters."))
                sim_hum = st.slider("Mains Hum (mV @ 50 Hz)", 0.0, 0.30, float(default_hum), 0.02, key="ad8232_hum")

        b_c1, b_c2, _ = st.columns([4.2, 1.8, 1.2])
        with b_c1:
            run_sim_clicked = st.button("Run Offline Sensor Simulation & Predict Beats", type="primary", key="btn_run_ad8232_sim")
        with b_c2:
            clear_sim_clicked = st.button("Clear Results", key="btn_clear_ad8232_sim")

        if clear_sim_clicked:
            st.session_state.pop("ad8232_sim_result", None)
            st.session_state.pop("ad8232_sim_prov", None)
            st.session_state.pop("ad8232_sim_recon", None)
            st.session_state.pop("ad8232_sim_raw_adc", None)
            st.session_state.pop("ad8232_sim_time", None)
            st.session_state.pop("ad8232_sim_parser", None)
            st.rerun()

        if run_sim_clicked:
            with st.spinner("Simulating AD8232 ADC transport, parsing stream, and executing inference..."):
                try:
                    # 1. Source Signal
                    if sim_source_type == "MIT-BIH recorded ECG":
                        if not avail_recs or "100" in str(sel_sim_rec):
                            try:
                                rec_obj, all_anns = load_record_with_annotations(sel_sim_rec)
                                fs = rec_obj.sampling_rate_hz
                                start_idx = int(sim_start_sec * fs)
                                n_samp = int(sim_duration_sec * fs)
                                end_idx = min(len(rec_obj.signal_mv), start_idx + n_samp)
                                src_signal_mv = rec_obj.signal_mv[start_idx:end_idx]
                                lead_used = rec_obj.lead_name
                                ref_anns = [a for a in all_anns if start_idx <= a.sample_index < end_idx]
                                slice_ref_anns = [
                                    BeatAnnotation(
                                        sample_index=a.sample_index - start_idx,
                                        original_symbol=a.original_symbol,
                                        aami_class=a.aami_class,
                                        time_seconds=(a.sample_index - start_idx) / float(fs),
                                    )
                                    for a in ref_anns
                                ]
                            except Exception:
                                demo_dict = load_demo_mitbih_slice()
                                if demo_dict:
                                    fs = int(demo_dict["sampling_rate_hz"])
                                    src_signal_mv = np.array(demo_dict["signal_mv"], dtype=np.float64)
                                    lead_used = demo_dict.get("lead_name", "MLII")
                                    slice_ref_anns = [
                                        BeatAnnotation(
                                            sample_index=ann["sample_index"],
                                            original_symbol=ann["symbol"],
                                            aami_class=ann["aami_class"],
                                            time_seconds=ann["time_seconds"],
                                        )
                                        for ann in demo_dict.get("reference_annotations", [])
                                    ]
                                else:
                                    raise
                        else:
                            rec_obj, all_anns = load_record_with_annotations(sel_sim_rec)
                            fs = rec_obj.sampling_rate_hz
                            start_idx = int(sim_start_sec * fs)
                            n_samp = int(sim_duration_sec * fs)
                            end_idx = min(len(rec_obj.signal_mv), start_idx + n_samp)
                            src_signal_mv = rec_obj.signal_mv[start_idx:end_idx]
                            lead_used = rec_obj.lead_name

                            # Reference annotations in this slice (offset relative to slice start)
                            ref_anns = [a for a in all_anns if start_idx <= a.sample_index < end_idx]
                            slice_ref_anns = [
                                BeatAnnotation(
                                    sample_index=a.sample_index - start_idx,
                                    original_symbol=a.original_symbol,
                                    aami_class=a.aami_class,
                                    time_seconds=(a.sample_index - start_idx) / float(fs),
                                )
                                for a in ref_anns
                            ]
                    else:
                        fs = 360
                        src_signal_mv = generate_synthetic_ecg(
                            duration_seconds=sim_duration_sec,
                            sampling_rate_hz=fs,
                            heart_rate_bpm=synth_bpm,
                            noise_amplitude_mv=0.0,
                            seed=synth_seed,
                        )
                        slice_ref_anns = None
                        lead_used = "SYNTHETIC"

                    # 2. Forward ADC Simulation
                    samples, prov = simulate_adc_stream(
                        source_type="mitbih" if sim_source_type == "MIT-BIH recorded ECG" else "synthetic",
                        record_id=sel_sim_rec,
                        lead_name=lead_used,
                        duration_seconds=sim_duration_sec,
                        sampling_rate_hz=fs,
                        baseline_count=sim_baseline,
                        gain_counts_per_mv=sim_gain,
                        noise_amplitude_mv=sim_noise,
                        baseline_wander_amplitude_mv=sim_drift,
                        mains_hum_amplitude_mv=sim_hum,
                        seed=synth_seed,
                    )

                    # 3. Protocol Serialization & Parsing
                    lines = [format_serial_line(s) for s in samples]
                    parser = SerialStreamParser()
                    for line in lines:
                        parser.feed_line(line)

                    parsed_df = parser.to_dataframe()
                    parsed_adc = parsed_df["raw_adc"].to_numpy(dtype=np.int32)
                    time_axis = parsed_df["time_seconds"].to_numpy(dtype=np.float64)

                    # 4. Calibration Inversion
                    recon_res = simulated_adc_to_voltage(
                        adc_counts=parsed_adc,
                        baseline_count=sim_baseline,
                        gain_counts_per_mv=sim_gain,
                        reference_signal_mv=src_signal_mv,
                    )
                    reconstructed_mv = recon_res["reconstructed_mv"]

                    # 5. Model Inference (Ground truth NOT provided to detector or feature extractor)
                    inf_res = run_single_lead_inference(
                        signal=reconstructed_mv,
                        sampling_rate_hz=fs,
                        units="mV",
                        calibration_provenance="SIMULATION PRESET — NOT HARDWARE CALIBRATION",
                        source_kind="offline AD8232-like simulation",
                        reference_annotations=slice_ref_anns,
                    )

                    st.session_state["ad8232_sim_result"] = inf_res
                    st.session_state["ad8232_sim_prov"] = prov
                    st.session_state["ad8232_sim_recon"] = recon_res
                    st.session_state["ad8232_sim_raw_adc"] = parsed_adc
                    st.session_state["ad8232_sim_time"] = time_axis
                    st.session_state["ad8232_sim_parser"] = {
                        "sample_count": parser.sample_count,
                        "corrupted_lines": parser.corrupted_lines_count,
                        "sequence_gaps": getattr(parser, "dropped_samples_count", getattr(parser, "sequence_gaps_count", 0)),
                    }

                except Exception as exc:
                    st.error(f"Offline simulation error: {exc}")

        # Render Stored Simulation Results
        if "ad8232_sim_result" in st.session_state:
            inf_res: SingleLeadInferenceResult = st.session_state["ad8232_sim_result"]
            prov = st.session_state.get("ad8232_sim_prov", {})
            recon_res = st.session_state.get("ad8232_sim_recon", {})
            parsed_adc = st.session_state.get("ad8232_sim_raw_adc")
            time_axis = st.session_state.get("ad8232_sim_time")
            parser_info = st.session_state.get("ad8232_sim_parser", {})

            if not inf_res.success:
                st.error(f"Inference Failure: {inf_res.error_message}")
            else:
                # 1. Status / Provenance Panel
                st.markdown("#### Simulation Pipeline Status & Provenance")
                s_c1, s_c2, s_c3, s_c4 = st.columns(4)
                s_c1.metric("Signal Source", f"{prov.get('source_type', 'N/A')} ({prov.get('lead_name', 'N/A')})")
                s_c2.metric("Total Samples", f"{prov.get('total_samples', len(time_axis))} @ 360 Hz")
                adc_range_str = f"[{prov.get('min_adc', 0)}, {prov.get('max_adc', 1023)}]"
                s_c3.metric("ADC Counts Range", adc_range_str)
                clip_pct = prov.get('clipping_percentage', 0.0)
                s_c4.metric("ADC Clipping", f"{prov.get('clipping_samples', 0)} ({clip_pct:.1f}%)")

                p_c1, p_c2, p_c3, p_c4 = st.columns(4)
                p_c1.metric("Simulator Bias / Gain", f"{prov.get('baseline_count', 512):.0f} cts / {prov.get('gain_counts_per_mv', 150):.0f} cts/mV")
                mae_str = f"{recon_res.get('mae_mv', 0.0):.4f} mV" if recon_res.get('mae_mv') is not None else "N/A"
                p_c2.metric("Reconstruction MAE", mae_str)
                p_c3.metric("Candidate QRS Peaks", inf_res.detected_peaks_count)
                p_c4.metric("Classifiable Windows", inf_res.classifiable_windows_count)

                if inf_res.warnings:
                    for w in inf_res.warnings:
                        st.warning(w)

                # 2. Dual Plotly Charts
                st.markdown("#### Dual Waveform Inspection")
                chart_view = st.radio(
                    "Waveform Inspection View",
                    [
                        "Synchronized Dual View (Raw ADC Counts & Reconstructed Simulated ECG)",
                        "Reconstructed Simulated ECG with Detected QRS Peaks",
                        "Raw 10-Bit ADC Transport Counts",
                    ],
                    horizontal=True,
                    key="ad8232_chart_view",
                )

                if chart_view == "Synchronized Dual View (Raw ADC Counts & Reconstructed Simulated ECG)":
                    fig_dual = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.10,
                        subplot_titles=[
                            "Raw Simulated ADC Stream (ADC counts [0–1023])",
                            "Reconstructed Simulated ECG (mV-equivalent; not hardware-calibrated)",
                        ],
                    )
                    # Subplot 1: Raw ADC
                    fig_dual.add_trace(
                        go.Scatter(
                            x=time_axis,
                            y=parsed_adc,
                            mode="lines",
                            name="ADC Counts",
                            line=dict(color="#444444", width=1.4),
                        ),
                        row=1,
                        col=1,
                    )
                    # Subplot 2: Reconstructed mV
                    fig_dual.add_trace(
                        go.Scatter(
                            x=time_axis,
                            y=inf_res.raw_signal_copy,
                            mode="lines",
                            name="Reconstructed ECG (mV)",
                            line=dict(color="#000000", width=1.6),
                        ),
                        row=2,
                        col=1,
                    )
                    # Overlay detected peaks on Subplot 2
                    if inf_res.predictions:
                        pk_times = [p.time_seconds for p in inf_res.predictions]
                        pk_vals = [
                            inf_res.raw_signal_copy[min(len(inf_res.raw_signal_copy) - 1, p.peak_sample)]
                            for p in inf_res.predictions
                        ]
                        pk_labels = [p.predicted_class for p in inf_res.predictions]
                        fig_dual.add_trace(
                            go.Scatter(
                                x=pk_times,
                                y=pk_vals,
                                mode="markers+text",
                                text=pk_labels,
                                textposition="top center",
                                name="Predicted Beats",
                                textfont=dict(color="#000000", size=11, family="Inter, sans-serif"),
                                marker=dict(symbol="circle-dot", size=8, color="#9E3D34"),
                            ),
                            row=2,
                            col=1,
                        )

                    fig_dual.update_layout(height=480, margin=dict(l=40, r=20, t=35, b=25))
                    fig_dual.update_yaxes(title_text="ADC counts", range=[-10, 1034], row=1, col=1)
                    fig_dual.update_yaxes(title_text="mV-equivalent", row=2, col=1)
                    fig_dual.update_xaxes(title_text="Time (seconds)", row=2, col=1)
                    fig_dual = apply_clean_scientific_theme(fig_dual)
                    st.plotly_chart(fig_dual, width='stretch', theme=None)

                elif chart_view == "Reconstructed Simulated ECG with Detected QRS Peaks":
                    fig_recon = go.Figure()
                    fig_recon.add_trace(
                        go.Scatter(
                            x=time_axis,
                            y=inf_res.raw_signal_copy,
                            mode="lines",
                            name="Reconstructed Simulated ECG (mV)",
                            line=dict(color="#000000", width=1.6),
                        )
                    )
                    if inf_res.predictions:
                        pk_times = [p.time_seconds for p in inf_res.predictions]
                        pk_vals = [
                            inf_res.raw_signal_copy[min(len(inf_res.raw_signal_copy) - 1, p.peak_sample)]
                            for p in inf_res.predictions
                        ]
                        pk_labels = [p.predicted_class for p in inf_res.predictions]
                        fig_recon.add_trace(
                            go.Scatter(
                                x=pk_times,
                                y=pk_vals,
                                mode="markers+text",
                                text=pk_labels,
                                textposition="top center",
                                name="Predicted Beats",
                                textfont=dict(color="#000000", size=11, family="Inter, sans-serif"),
                                marker=dict(symbol="circle-dot", size=8, color="#9E3D34"),
                            )
                        )
                    fig_recon.update_layout(height=340, margin=dict(l=35, r=20, t=30, b=25))
                    fig_recon.update_yaxes(title_text="reconstructed simulated ECG (mV-equivalent; not hardware-calibrated)")
                    fig_recon.update_xaxes(title_text="Time (seconds)")
                    fig_recon = apply_clean_scientific_theme(fig_recon)
                    st.plotly_chart(fig_recon, width='stretch', theme=None)

                else:
                    fig_adc = go.Figure()
                    fig_adc.add_trace(
                        go.Scatter(
                            x=time_axis,
                            y=parsed_adc,
                            mode="lines",
                            name="Raw ADC Counts",
                            line=dict(color="#333333", width=1.5),
                        )
                    )
                    fig_adc.update_layout(height=340, margin=dict(l=35, r=20, t=30, b=25))
                    fig_adc.update_yaxes(title_text="ADC counts [0–1023]", range=[-10, 1034])
                    fig_adc.update_xaxes(title_text="Time (seconds)")
                    fig_adc = apply_clean_scientific_theme(fig_adc)
                    st.plotly_chart(fig_adc, width='stretch', theme=None)

                # 3. Beat Predictions Summary & Visible Results Table
                st.markdown("#### Beat-by-Beat Classification Predictions")
                bc = inf_res.predictions_summary
                m_n, m_s, m_v, m_q = st.columns(4)
                m_n.metric("Class N (Normal)", bc.get("N", 0))
                m_s.metric("Class S (Supraventricular)", bc.get("S", 0))
                m_v.metric("Class V (Premature Ventricular)", bc.get("V", 0))
                m_q.metric("Class Q (Fusion & Other)", bc.get("Q", 0))

                df_predictions = inf_res.to_dataframe()
                if not df_predictions.empty:
                    st.dataframe(df_predictions, use_container_width=True, hide_index=True)
                else:
                    st.info("No beats were classifiable in this signal segment.")

                st.markdown(
                    "<p style='font-size: 0.82rem; color: #777; margin-top: 0.3rem;'>"
                    "<strong>Project-Defined 4-Class Taxonomy:</strong> Target classes are N, S, V, Q, with ventricular fusion beats (F) merged into Q. "
                    "This is an academic research grouping, not an unqualified 5-class AAMI EC57 implementation.<br>"
                    "<strong>Model Scores:</strong> Confidence scores are uncalibrated Random Forest voting fractions; they represent model decision certainty, not calibrated diagnostic probabilities."
                    "</p>",
                    unsafe_allow_html=True,
                )

                # 4. Evaluation Area Against MIT-BIH Annotations (Decoupled post-hoc layer)
                if inf_res.evaluation:
                    eval_info = inf_res.evaluation
                    det_eval = eval_info["detector_evaluation"]
                    cls_eval = eval_info.get("classifier_evaluation")

                    with st.expander("Evaluation Against MIT-BIH Ground-Truth Annotations (Offline Post-Hoc Only)", expanded=False):
                        st.markdown("##### Autonomous QRS Detector Evaluation (±50 ms Matching Window)")
                        d1, d2, d3, d4, d5 = st.columns(5)
                        d1.metric("Reference Beats", det_eval["n_reference"])
                        d2.metric("Detected Peaks", det_eval["n_detected"])
                        d3.metric("True Positives (TP)", det_eval["true_positives"])
                        d4.metric("Sensitivity", f"{det_eval['sensitivity']:.1%}")
                        d5.metric("Positive Pred. Value", f"{det_eval['positive_predictive_value']:.1%}")

                        if cls_eval:
                            st.markdown("##### Matched-Detection Conditional Beat Classification")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Matched Beats Evaluated", eval_info["matched_beats_count"])
                            c2.metric("Window Macro F1", f"{cls_eval['macro_f1']:.4f}")
                            c3.metric("Window Weighted F1", f"{cls_eval['weighted_f1']:.4f}")

                            # Display per-class metrics
                            per_cls_data = []
                            for c_k in AAMI_TARGET_CLASSES:
                                if c_k in cls_eval["per_class"]:
                                    c_info = cls_eval["per_class"][c_k]
                                    per_cls_data.append({
                                        "Class": c_k,
                                        "Description": c_info["description"],
                                        "Support": c_info["support"],
                                        "Precision": f"{c_info['precision']:.3f}",
                                        "Recall": f"{c_info['recall']:.3f}",
                                        "F1-Score": f"{c_info['f1_score']:.3f}",
                                    })
                            st.dataframe(pd.DataFrame(per_cls_data), use_container_width=True, hide_index=True)

                        st.markdown(
                            "<p style='font-size: 0.82rem; color: #777;'>"
                            "<strong>Integrity Disclosure:</strong> Ground-truth annotations were NOT provided to the Pan-Tompkins detector or feature extraction module. "
                            "Matching was performed strictly post-hoc within a ±50 ms window.<br>"
                            "Previously reported benchmark performance is matched-detection conditional on held-out test partition DS2 (N=47,034, Macro F1 0.5274, Weighted F1 0.9011). "
                            "The values above reflect the current offline simulated slice only."
                            "</p>",
                            unsafe_allow_html=True,
                        )

                # Explain Results Action for Offline AD8232 Simulation
                eval_ctx = {}
                if inf_res.evaluation:
                    e_info = inf_res.evaluation
                    d_eval = e_info.get("detector_evaluation", {})
                    c_eval = e_info.get("classifier_evaluation", {})
                    eval_ctx = {
                        "matched_beats_count": e_info.get("matched_beats_count", 0),
                        "detector_sensitivity": round(d_eval.get("sensitivity", 0.0), 4),
                        "detector_ppv": round(d_eval.get("positive_predictive_value", 0.0), 4),
                        "macro_f1": round(c_eval.get("macro_f1", 0.0), 4) if c_eval else None,
                        "weighted_f1": round(c_eval.get("weighted_f1", 0.0), 4) if c_eval else None,
                    }

                ad8232_payload = {
                    "workflow": "hardware_simulation",
                    "source_category": f"Offline Simulation ({prov.get('source_type', 'N/A')}, Lead {prov.get('lead_name', 'N/A')})",
                    "is_simulation": True,
                    "sampling_rate_hz": float(inf_res.sampling_rate_hz),
                    "units_and_calibration": "Simulated 10-bit integer ADC counts (0-1023) mapped to mV-equivalent; not hardware-calibrated",
                    "summary_metrics": {
                        "total_detected_peaks": inf_res.detected_peaks_count,
                        "classifiable_windows": inf_res.classifiable_windows_count,
                        "class_counts": bc,
                        "clipping_samples": prov.get("clipping_samples", 0),
                        "clipping_percentage": round(prov.get("clipping_percentage", 0.0), 2),
                        "reconstruction_mae_mv": round(recon_res.get("mae_mv", 0.0), 4) if recon_res.get("mae_mv") is not None else None,
                        "simulated_baseline_counts": round(prov.get("baseline_count", 512.0), 1),
                        "simulated_gain_counts_per_mv": round(prov.get("gain_counts_per_mv", 150.0), 1),
                    },
                    "class_predictions": [
                        {
                            "beat_index": p.beat_index,
                            "time_seconds": round(p.time_seconds, 3),
                            "predicted_class": p.predicted_class,
                            "class_description": AAMI_CLASS_DESCRIPTIONS.get(p.predicted_class, p.predicted_class),
                            "confidence": round(p.confidence_score, 4) if getattr(p, "confidence_score", None) is not None else None,
                        }
                        for p in inf_res.predictions
                    ],
                    "data_quality_flags": [
                        "OFFLINE SIMULATION — NO PHYSICAL SENSOR CONNECTED",
                        f"Calibration Provenance: {inf_res.calibration_provenance}",
                        f"Parser integrity: {parser_info.get('corrupted_lines', 0)} corrupted lines, {parser_info.get('sequence_gaps', 0)} sequence gaps",
                        *(inf_res.warnings or []),
                    ],
                    "evaluation_context": eval_ctx,
                    "known_limitations": [
                        "Offline software simulation only; no physical sensor connected",
                        "Simulator gain and bias are user approximations, not certified device calibration",
                        "Inference is blocked on physical serial ADC streams lacking verified calibration certificates",
                        "Project-defined 4-class taxonomy: N, S, V, Q (ventricular fusion F merged into Q)",
                        "Scores are uncalibrated Random Forest tree voting fractions",
                    ],
                }
                render_explain_results_button(
                    workflow="hardware_simulation",
                    payload_data=ad8232_payload,
                    result_id=f"ad8232_{prov.get('source_type', 'sim')}_{prov.get('lead_name', 'lead')}_{inf_res.detected_peaks_count}",
                    title="Offline Hardware Simulation Explanation",
                )

    # --------------------------------------------------------------------------
    # Tab 4: Optical Paper Digitizer
    # --------------------------------------------------------------------------
    with tab_digitizer:
        st.markdown("### Optical Paper Digitizer")
        st.markdown("Extract continuous 1D waveforms from paper ECG scans. *Decoupled from diagnostic ML models.*")

        src_mode = st.radio("Source Image", ["Deterministic Benchmark Sample", "Upload Scanned ECG Image"], horizontal=True)
        img_b, img_n = None, None

        if src_mode == "Upload Scanned ECG Image":
            f_up = st.file_uploader("Select Image", type=["png", "jpg", "jpeg", "tif", "tiff"])
            if f_up:
                img_b, img_n = f_up.read(), f_up.name
        else:
            bench_p = Path("sample_data/synthetic_12lead.png")
            if bench_p.is_file():
                img_b, img_n = bench_p.read_bytes(), "synthetic_12lead.png"

        if img_b:
            val_res = validate_image_bytes(img_b, filename=img_n)
            if val_res.is_valid:
                st.image(Image.open(BytesIO(img_b)), caption=f"Document: {img_n}", width=500)

                with st.expander("Morphological Filter Settings", expanded=False):
                    i_c1, i_c2 = st.columns(2)
                    with i_c1:
                        b_sig = st.slider("Gaussian Blur (σ)", 0.5, 4.0, 1.2, 0.1)
                    with i_c2:
                        t_pct = st.slider("Threshold Percentile", 50, 99, 85, 1)

                if st.button("Digitize 12 Leads", type="primary", key="btn_dig_run"):
                    with st.spinner("Extracting traces..."):
                        res = digitize_ecg_image(img_b, options={"filename": img_n})
                        st.session_state["digitizer_results"] = res
                        st.session_state["digitizer_img_name"] = img_n

                if "digitizer_results" in st.session_state:
                    res = st.session_state["digitizer_results"]
                    cur_img = st.session_state.get("digitizer_img_name", img_n or "ecg_document")
                    if res.status == DigitizationStatus.SUCCESS:
                        st.success(f"Extracted {len(res.leads)} leads.")
                        sig_mat = np.zeros((255, 12))
                        for idx, ln in enumerate(WF_LEAD_NAMES):
                            if ln in res.leads:
                                sig_mat[:, idx] = res.leads[ln].samples[:255]
                        st.plotly_chart(create_12lead_matrix_heatmap(sig_mat, fs=100.0), width='stretch', theme=None)

                        # Explain Results Action
                        dig_payload = {
                            "workflow": "workflow_a_digitization",
                            "source_category": f"Optical ECG Scanned Document ({cur_img})",
                            "is_simulation": False,
                            "sampling_rate_hz": float(res.sampling_rate_hz or 100.0),
                            "units_and_calibration": f"Amplitude: {res.amplitude_unit.value} ({res.calibration_status.value}); Standard 25 mm/s paper reference",
                            "summary_metrics": {
                                "extracted_leads_count": len(res.leads),
                                "detected_leads": res.detected_leads or list(res.leads.keys()),
                                "missing_leads": res.missing_leads or [],
                                "calibration_status": res.calibration_status.value,
                                "amplitude_unit": res.amplitude_unit.value,
                                "runtime_ms": round(res.runtime_ms, 2),
                            },
                            "class_predictions": [],
                            "data_quality_flags": [
                                "AIR-GAPPED WORKFLOW: Completely decoupled from diagnostic machine learning models",
                                f"Calibration Status: {res.calibration_status.value}",
                                f"Amplitude Unit: {res.amplitude_unit.value}",
                                *(res.warnings or []),
                            ],
                            "evaluation_context": {
                                "method": res.method,
                                "method_version": res.method_version,
                                "isolation": "Decoupled archival digitizer — no diagnostic inference attempted",
                            },
                            "known_limitations": [
                                "Archival optical digitizer only; outputs relative uncalibrated units or pixel centroids",
                                "Not calibrated for clinical diagnosis; amplitude cannot be used for ST-segment elevation measurement",
                                "Optical paper scanning artifacts, perspective skew, and paper wrinkles may distort waveform traces",
                                "Decoupled from diagnostic ML models to prevent image artifact contamination",
                            ],
                        }
                        render_explain_results_button(
                            workflow="workflow_a_digitization",
                            payload_data=dig_payload,
                            result_id=f"dig_{cur_img}",
                            title="Optical Paper Digitization Explanation",
                        )
                    elif res.status in (DigitizationStatus.REJECTED, DigitizationStatus.ERROR):
                        st.error(f"Digitization failed: {res.error_message}")
