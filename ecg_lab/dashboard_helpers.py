"""Dashboard analytical helpers, filter details, and matrix visualization utilities.

Provides:
1. Signal filter computation and frequency/phase response (Bode plot) calculations.
2. Pan-Tompkins 5-stage filter cascade extraction (Raw -> Bandpass -> Derivative -> Squared -> Integrated).
3. 12-lead signal matrix, cross-lead correlation matrix, and image preprocessing pipeline extraction.
4. Interactive confusion matrix heatmaps (Matched-Detection 4x4, Oracle 4x4, Multi-label 2x2).
5. Seamless offline demo dataset loaders for Mode A (Hosted Demo / Vercel).
"""

from io import BytesIO
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal
import scipy.ndimage


# ==============================================================================
# 0. High-Contrast Scientific Styling Utilities
# ==============================================================================

def apply_clean_scientific_theme(fig: go.Figure) -> go.Figure:
    """Enforce high-contrast scientific theme: pure white backgrounds, solid black text, titles, axes, and ticks."""
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color="#000000",
            size=12,
        ),
        title=dict(
            font=dict(
                family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                color="#000000",
                size=13,
            ),
        ),
        legend=dict(
            font=dict(
                family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                color="#000000",
                size=11,
            ),
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="#CCCCCC",
            borderwidth=1,
        ),
    )

    # Apply solid black text and clean gridlines across all subplots & axes
    fig.for_each_xaxis(lambda ax: ax.update(
        showgrid=True,
        gridcolor="#EBEBEB",
        linecolor="#000000",
        zerolinecolor="#CCCCCC",
        tickfont=dict(color="#000000", size=11, family="Inter, sans-serif"),
        title_font=dict(color="#000000", size=12, family="Inter, sans-serif"),
    ))

    fig.for_each_yaxis(lambda ax: ax.update(
        showgrid=True,
        gridcolor="#EBEBEB",
        linecolor="#000000",
        zerolinecolor="#CCCCCC",
        tickfont=dict(color="#000000", size=11, family="Inter, sans-serif"),
        title_font=dict(color="#000000", size=12, family="Inter, sans-serif"),
    ))

    # Force all layout annotations (including make_subplots titles) to solid black
    for ann in fig.layout.annotations:
        if ann.font:
            ann.font.color = "#000000"
            ann.font.family = "Inter, sans-serif"
        else:
            ann.font = dict(color="#000000", family="Inter, sans-serif")

    # Force all colorbars to have solid black tick and title fonts
    for trace in fig.data:
        if hasattr(trace, "colorbar") and trace.colorbar:
            trace.colorbar.tickfont = dict(color="#000000", size=10, family="Inter, sans-serif")
            if hasattr(trace.colorbar, "title") and trace.colorbar.title:
                trace.colorbar.title.font = dict(color="#000000", size=11, family="Inter, sans-serif")

    return fig


# ==============================================================================
# 1. Digital Filter & Signal Processing Utilities
# ==============================================================================

def compute_filter_bode_data(
    lowcut: float = 5.0,
    highcut: float = 15.0,
    fs: float = 360.0,
    order: int = 2,
    notch_freq: Optional[float] = None,
    q: float = 30.0,
    n_points: int = 1024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute frequency, magnitude (dB), and phase (degrees) response of filter cascade.

    Args:
        lowcut: Lower -3dB frequency in Hz.
        highcut: Upper -3dB frequency in Hz.
        fs: Sampling rate in Hz.
        order: Filter order.
        notch_freq: Optional mains notch frequency in Hz (e.g. 50.0 or 60.0).
        q: Quality factor for notch filter.
        n_points: Number of frequency evaluation points.

    Returns:
        Tuple of (frequencies, magnitude_db, phase_degrees).
    """
    sos = scipy.signal.butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    w, h_bp = scipy.signal.sosfreqz(sos, worN=n_points, fs=fs)

    total_h = h_bp.copy()

    if notch_freq is not None and 0 < notch_freq < (fs / 2):
        b_notch, a_notch = scipy.signal.iirnotch(notch_freq, q, fs=fs)
        _, h_notch = scipy.signal.freqz(b_notch, a_notch, worN=w, fs=fs)
        total_h *= h_notch

    mag_db = 20.0 * np.log10(np.maximum(np.abs(total_h), 1e-6))
    phase_deg = np.unwrap(np.angle(total_h)) * 180.0 / np.pi

    return w, mag_db, phase_deg


def apply_bandpass_notch_filter(
    signal: np.ndarray,
    fs: float = 360.0,
    lowcut: float = 5.0,
    highcut: float = 15.0,
    order: int = 2,
    notch_freq: Optional[float] = None,
    q: float = 30.0,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass and optional notch filter to a signal."""
    arr = np.asarray(signal, dtype=np.float64)
    if len(arr) < 15:
        return arr.copy()

    sos = scipy.signal.butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    filtered = scipy.signal.sosfiltfilt(sos, arr)

    if notch_freq is not None and 0 < notch_freq < (fs / 2):
        b_notch, a_notch = scipy.signal.iirnotch(notch_freq, q, fs=fs)
        filtered = scipy.signal.filtfilt(b_notch, a_notch, filtered)

    return filtered


def compute_pan_tompkins_cascade(
    signal_mv: np.ndarray,
    fs: float = 360.0,
    lowcut: float = 5.0,
    highcut: float = 15.0,
    window_sec: float = 0.150,
    notch_freq: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Extract intermediate signals across all 5 stages of the Pan-Tompkins QRS detector.

    Returns:
        Dictionary with keys:
            - 'raw': original input signal
            - 'bandpass': filtered 5-15 Hz bandpass
            - 'derivative': discrete derivative signal
            - 'squared': squared derivative signal
            - 'integrated': moving-window integrated signal
            - 'threshold': adaptive detection threshold estimate
    """
    raw = np.asarray(signal_mv, dtype=np.float64)
    n = len(raw)

    if n < int(fs * 0.2):
        return {
            "raw": raw,
            "bandpass": raw.copy(),
            "derivative": np.zeros_like(raw),
            "squared": np.zeros_like(raw),
            "integrated": np.zeros_like(raw),
            "threshold": np.zeros_like(raw),
        }

    # 1. Bandpass (+ optional notch)
    bandpass = apply_bandpass_notch_filter(
        raw, fs=fs, lowcut=lowcut, highcut=highcut, order=2, notch_freq=notch_freq
    )

    # 2. Derivative: discrete slope
    derivative = np.gradient(bandpass)

    # 3. Squaring
    squared = derivative ** 2

    # 4. Moving window integration
    window_samples = max(1, int(window_sec * fs))
    kernel = np.ones(window_samples) / float(window_samples)
    integrated = np.convolve(squared, kernel, mode="same")

    # 5. Adaptive threshold line
    p85 = float(np.percentile(integrated, 85)) * 0.5
    threshold_line = np.full_like(integrated, p85)

    return {
        "raw": raw,
        "bandpass": bandpass,
        "derivative": derivative,
        "squared": squared,
        "integrated": integrated,
        "threshold": threshold_line,
    }


# ==============================================================================
# 2. Plotly Analytical Visualization Constructors
# ==============================================================================

def create_bode_plot(
    freqs: np.ndarray,
    mag_db: np.ndarray,
    phase_deg: np.ndarray,
    lowcut: float,
    highcut: float,
    notch_freq: Optional[float] = None,
) -> go.Figure:
    """Create a synchronized 2-row Plotly Bode plot (Magnitude in dB, Phase in degrees)."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=[
            "<b>Filter Magnitude Response |H(f)| (dB)</b>",
            "<b>Filter Phase Response ∠H(f) (Degrees)</b>",
        ],
    )

    # Magnitude curve
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=mag_db,
            mode="lines",
            name="Magnitude (dB)",
            line=dict(color="#003366", width=2.0),
            hovertemplate="Frequency: %{x:.1f} Hz<br>Gain: %{y:.2f} dB<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # -3dB Cutoff reference lines
    fig.add_vline(x=lowcut, line=dict(color="#D9534F", width=1.2, dash="dash"), row=1, col=1)
    fig.add_vline(x=highcut, line=dict(color="#D9534F", width=1.2, dash="dash"), row=1, col=1)

    if notch_freq is not None and 0 < notch_freq < freqs[-1]:
        fig.add_vline(x=notch_freq, line=dict(color="#6A1B9A", width=1.2, dash="dot"), row=1, col=1)

    # Phase curve
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=phase_deg,
            mode="lines",
            name="Phase (°)",
            line=dict(color="#2E7D32", width=1.8),
            hovertemplate="Frequency: %{x:.1f} Hz<br>Phase: %{y:.1f}°<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=480,
        margin=dict(l=50, r=30, t=40, b=40),
        showlegend=False,
    )
    fig.update_xaxes(title_text="Frequency (Hz)", range=[0, min(65.0, freqs[-1])], row=2, col=1)
    fig.update_yaxes(title_text="Magnitude (dB)", range=[-40, 5], row=1, col=1)
    fig.update_yaxes(title_text="Phase (deg)", row=2, col=1)

    return apply_clean_scientific_theme(fig)


def create_pan_tompkins_cascade_figure(
    time_axis: np.ndarray,
    stages: Dict[str, np.ndarray],
    peak_times: Optional[List[float]] = None,
) -> go.Figure:
    """Create a 5-panel synchronized Plotly figure displaying the Pan-Tompkins cascade stages."""
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            "<b>1. Raw Single-Lead ECG (mV)</b>",
            "<b>2. Bandpass Filtered (5–15 Hz)</b>",
            "<b>3. Discrete Derivative (Slope Isolation)</b>",
            "<b>4. Squaring (Nonlinear Energy Enhancement)</b>",
            "<b>5. Moving-Window Integration & Adaptive Detection Threshold</b>",
        ],
    )

    colors = ["#003366", "#0288D1", "#6A1B9A", "#EF6C00", "#2E7D32"]

    # 1. Raw
    fig.add_trace(
        go.Scatter(x=time_axis, y=stages["raw"], mode="lines", line=dict(color=colors[0], width=1.4)),
        row=1, col=1,
    )

    # 2. Bandpass
    fig.add_trace(
        go.Scatter(x=time_axis, y=stages["bandpass"], mode="lines", line=dict(color=colors[1], width=1.4)),
        row=2, col=1,
    )

    # 3. Derivative
    fig.add_trace(
        go.Scatter(x=time_axis, y=stages["derivative"], mode="lines", line=dict(color=colors[2], width=1.4)),
        row=3, col=1,
    )

    # 4. Squaring
    fig.add_trace(
        go.Scatter(x=time_axis, y=stages["squared"], mode="lines", line=dict(color=colors[3], width=1.4)),
        row=4, col=1,
    )

    # 5. Integrated + Threshold
    fig.add_trace(
        go.Scatter(x=time_axis, y=stages["integrated"], mode="lines", name="Integrated", line=dict(color=colors[4], width=1.6)),
        row=5, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=stages["threshold"],
            mode="lines",
            name="Threshold",
            line=dict(color="#C62828", width=1.2, dash="dash"),
        ),
        row=5, col=1,
    )

    # Overlay peak markers on panel 5 if provided
    if peak_times:
        int_peaks_y = []
        for pt in peak_times:
            idx = int(np.argmin(np.abs(time_axis - pt)))
            int_peaks_y.append(stages["integrated"][idx])
        fig.add_trace(
            go.Scatter(
                x=peak_times,
                y=int_peaks_y,
                mode="markers",
                name="R-Peaks",
                marker=dict(symbol="circle-dot", size=9, color="#C62828"),
            ),
            row=5, col=1,
        )

    fig.update_layout(
        height=820,
        margin=dict(l=50, r=30, t=35, b=40),
        showlegend=False,
    )
    fig.update_xaxes(title_text="Time (seconds)", row=5, col=1)
    fig.update_yaxes(title_text="mV", row=1, col=1)
    fig.update_yaxes(title_text="mV", row=2, col=1)
    fig.update_yaxes(title_text="mV/s", row=3, col=1)
    fig.update_yaxes(title_text="(mV/s)²", row=4, col=1)
    fig.update_yaxes(title_text="Energy", row=5, col=1)

    return apply_clean_scientific_theme(fig)


def create_before_after_filter_figure(
    time_axis: np.ndarray,
    raw_signal: np.ndarray,
    filtered_signal: np.ndarray,
    lead_name: str = "Lead",
) -> go.Figure:
    """Create a before-and-after signal filtering comparison with difference / noise trace."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=[
            f"<b>Raw vs. Bandpass Filtered Signal ({lead_name})</b>",
            "<b>Extracted Noise & Baseline Wander Trace (Raw − Filtered)</b>",
        ],
    )

    # Raw signal
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=raw_signal,
            mode="lines",
            name="Raw Input Signal",
            line=dict(color="#8C8D89", width=1.3),
            opacity=0.85,
        ),
        row=1, col=1,
    )

    # Filtered signal
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=filtered_signal,
            mode="lines",
            name="Zero-Phase Filtered",
            line=dict(color="#003366", width=2.0),
        ),
        row=1, col=1,
    )

    # Noise residual
    residual = raw_signal - filtered_signal
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=residual,
            mode="lines",
            name="Residual Noise / Wander",
            line=dict(color="#CF1322", width=1.2),
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=480,
        margin=dict(l=50, r=30, t=35, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Voltage (mV)", row=1, col=1)
    fig.update_yaxes(title_text="Noise (mV)", row=2, col=1)

    return apply_clean_scientific_theme(fig)


# ==============================================================================
# 3. ECG Signal Matrices & Heatmap Constructors
# ==============================================================================

def create_confusion_matrix_figure(
    matrix: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    normalize: str = "none",  # "none", "pred", "true"
    descriptions: Optional[Dict[str, str]] = None,
    is_oracle: bool = False,
) -> go.Figure:
    """Create an interactive Plotly Confusion Matrix Heatmap with solid black count and percentage annotations."""
    if matrix is None:
        if is_oracle:
            matrix = np.array(
                [
                    [40280, 529, 1374, 185],
                    [1075, 412, 115, 172],
                    [148, 48, 2962, 58],
                    [116, 26, 128, 118],
                ],
                dtype=np.float64,
            )
            title = title or "Secondary Benchmark: Annotation-Centred (Oracle) Classifier Reference (N = 47,576)"
        else:
            matrix = np.array(
                [
                    [38608, 513, 735, 2027],
                    [1213, 315, 61, 168],
                    [153, 51, 2806, 8],
                    [242, 7, 15, 112],
                ],
                dtype=np.float64,
            )
            title = title or "Matched-Detection Conditional Beat-Classification Metrics (N = 47,034)"

    if labels is None:
        labels = ["N", "S", "V", "Q"]

    if descriptions is None:
        descriptions = {
            "N": "Normal / Bundle Branch",
            "S": "Supraventricular Ectopic",
            "V": "Ventricular Ectopic",
            "Q": "Fusion & Other",
        }

    title = title or "Confusion Matrix Heatmap"
    mat = np.array(matrix, dtype=np.float64)
    n_classes = len(labels)

    if normalize == "true":
        row_sums = np.sum(mat, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        display_mat = mat / row_sums
        text_template = [[f"<b>{int(mat[i, j]):,}</b><br>({display_mat[i, j]:.1%})" for j in range(n_classes)] for i in range(n_classes)]
        z_title = "Recall (%)"
        z_max = 1.0
    elif normalize == "pred":
        col_sums = np.sum(mat, axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        display_mat = mat / col_sums
        text_template = [[f"<b>{int(mat[i, j]):,}</b><br>({display_mat[i, j]:.1%})" for j in range(n_classes)] for i in range(n_classes)]
        z_title = "Precision (%)"
        z_max = 1.0
    else:
        display_mat = mat
        text_template = [[f"<b>{int(mat[i, j]):,}</b>" for j in range(n_classes)] for i in range(n_classes)]
        z_title = "Beats"
        z_max = float(np.max(mat))

    hover_text = []
    for i in range(n_classes):
        row_h = []
        for j in range(n_classes):
            true_desc = descriptions.get(labels[i], labels[i]) if descriptions else labels[i]
            pred_desc = descriptions.get(labels[j], labels[j]) if descriptions else labels[j]
            row_h.append(
                f"<b>True:</b> {labels[i]} ({true_desc})<br>"
                f"<b>Predicted:</b> {labels[j]} ({pred_desc})<br>"
                f"<b>Count:</b> {int(mat[i, j]):,} beats<br>"
                f"<b>Recall (%):</b> {mat[i, j] / max(1, np.sum(mat[i, :])):.1%}<br>"
                f"<b>Precision (%):</b> {mat[i, j] / max(1, np.sum(mat[:, j])):.1%}"
            )
        hover_text.append(row_h)

    fig = go.Figure(
        data=go.Heatmap(
            z=display_mat,
            x=[f"Pred {lbl}" for lbl in labels],
            y=[f"True {lbl}" for lbl in labels],
            text=text_template,
            texttemplate="%{text}",
            textfont=dict(size=12, color="#000000", family="Inter, sans-serif"),
            hoverinfo="text",
            hovertext=hover_text,
            colorscale=[[0, "#FFFFFF"], [0.2, "#E3F2FD"], [0.5, "#90CAF9"], [1.0, "#42A5F5"]],
            showscale=True,
            colorbar=dict(title=z_title, len=0.85),
            zmin=0.0,
            zmax=z_max,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#000000")),
        height=430,
        margin=dict(l=60, r=40, t=50, b=40),
        xaxis=dict(title="Predicted Beat Classification", side="bottom"),
        yaxis=dict(title="Ground-Truth Reference Class", autorange="reversed"),
    )

    return apply_clean_scientific_theme(fig)


def create_12lead_matrix_heatmap(
    signal_data: Union[pd.DataFrame, np.ndarray],
    fs: float = 100.0,
    lead_names: Optional[List[str]] = None,
) -> go.Figure:
    """Create a 2D Heatmap visualization of the 12-lead ECG Signal Matrix across time."""
    standard_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    if isinstance(signal_data, pd.DataFrame):
        lead_cols = [c for c in signal_data.columns if c != "sample_index"]
        matrix_vals = signal_data[lead_cols].values.T  # 12 x N
        samples = signal_data["sample_index"].values if "sample_index" in signal_data.columns else (np.arange(matrix_vals.shape[1]) / float(fs))
    else:
        arr = np.array(signal_data, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 12:
            matrix_vals = arr.T
        elif arr.ndim == 2 and arr.shape[0] == 12:
            matrix_vals = arr
        else:
            matrix_vals = arr.reshape(12, -1) if arr.size % 12 == 0 else arr
        lead_cols = lead_names if (lead_names and len(lead_names) == matrix_vals.shape[0]) else standard_leads[:matrix_vals.shape[0]]
        samples = np.arange(matrix_vals.shape[1]) / float(fs)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_vals,
            x=samples,
            y=lead_cols,
            colorscale="Viridis",
            colorbar=dict(title="Amplitude (mV)", len=0.85),
            hovertemplate="Lead: %{y}<br>Time: %{x:.2f} s<br>Amplitude: %{z:.3f} mV<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>12-Lead Continuous Waveform Voltage Matrix</b>",
        height=380,
        margin=dict(l=50, r=30, t=40, b=40),
        xaxis_title="Time (seconds)",
        yaxis_title="Lead Channel",
        yaxis=dict(autorange="reversed"),
    )

    return apply_clean_scientific_theme(fig)


def create_lead_correlation_heatmap(
    signal_data: Union[pd.DataFrame, np.ndarray],
    lead_names: Optional[List[str]] = None,
) -> go.Figure:
    """Create an interactive 12x12 Pearson Correlation Matrix heatmap across all 12 leads."""
    standard_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    if isinstance(signal_data, pd.DataFrame):
        lead_cols = [c for c in signal_data.columns if c != "sample_index"]
        corr = signal_data[lead_cols].corr().values
    else:
        arr = np.array(signal_data, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 12:
            df = pd.DataFrame(arr, columns=lead_names or standard_leads)
        elif arr.ndim == 2 and arr.shape[0] == 12:
            df = pd.DataFrame(arr.T, columns=lead_names or standard_leads)
        else:
            df = pd.DataFrame(arr)
        lead_cols = list(df.columns)
        corr = df.corr().values

    text_vals = np.round(corr, 2)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr,
            x=lead_cols,
            y=lead_cols,
            text=text_vals,
            texttemplate="%{text:.2f}",
            textfont=dict(size=10, color="#000000", family="Inter, sans-serif"),
            colorscale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            colorbar=dict(title="Pearson r", len=0.85),
            hovertemplate="Lead X: %{x}<br>Lead Y: %{y}<br>Correlation r: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Cross-Lead Correlation Matrix (Physiological Coherence)</b>",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(title="Lead Channel"),
        yaxis=dict(title="Lead Channel", autorange="reversed"),
    )

    return apply_clean_scientific_theme(fig)


DEFAULT_PTBXL_CONTINGENCY = {
    "NORM": {"true_positives": 772, "false_positives": 182, "false_negatives": 181, "true_negatives": 1063},
    "MI": {"true_positives": 421, "false_positives": 115, "false_negatives": 133, "true_negatives": 1529},
    "STTC": {"true_positives": 388, "false_positives": 139, "false_negatives": 142, "true_negatives": 1529},
    "CD": {"true_positives": 361, "false_positives": 110, "false_negatives": 134, "true_negatives": 1593},
    "HYP": {"true_positives": 148, "false_positives": 78, "false_negatives": 120, "true_negatives": 1852},
}


def create_multilabel_contingency_figure(
    per_label_metrics: Optional[Union[Dict[str, Any], str]] = None,
) -> go.Figure:
    """Create a high-contrast 2x2 Confusion Matrix figure for PTB-XL diagnostic superclasses."""
    classes = ["NORM", "MI", "STTC", "CD", "HYP"]

    if isinstance(per_label_metrics, str):
        c = per_label_metrics
        info = DEFAULT_PTBXL_CONTINGENCY.get(c, {"true_positives": 0, "false_positives": 0, "false_negatives": 0, "true_negatives": 0})
        tp = info.get("true_positives", 0)
        fp = info.get("false_positives", 0)
        fn = info.get("false_negatives", 0)
        tn = info.get("true_negatives", 0)
        mat_2x2 = np.array([[tp, fp], [fn, tn]])
        text_annot = [
            [f"<b>TP (True Pos)</b><br>{tp:,}", f"<b>FP (False Alarm)</b><br>{fp:,}"],
            [f"<b>FN (Missed)</b><br>{fn:,}", f"<b>TN (True Neg)</b><br>{tn:,}"],
        ]
        fig = go.Figure(
            data=go.Heatmap(
                z=mat_2x2,
                x=["Predicted Positive", "Predicted Negative"],
                y=["True Positive", "True Negative"],
                text=text_annot,
                texttemplate="%{text}",
                textfont=dict(size=12, color="#000000", family="Inter, sans-serif"),
                colorscale=[[0, "#FFFFFF"], [1.0, "#90CAF9"]],
                showscale=False,
                hoverinfo="skip",
            )
        )
        fig.update_layout(
            title=f"<b>2×2 Diagnostic Contingency Matrix — {c} (Test Fold 10, N = 2,198)</b>",
            height=300,
            margin=dict(l=50, r=30, t=40, b=30),
            yaxis=dict(autorange="reversed"),
        )
        return apply_clean_scientific_theme(fig)

    metrics_dict = per_label_metrics if isinstance(per_label_metrics, dict) else DEFAULT_PTBXL_CONTINGENCY
    fig = make_subplots(
        rows=1,
        cols=5,
        subplot_titles=[f"<b>{c}</b>" for c in classes],
        horizontal_spacing=0.04,
    )

    for idx, c in enumerate(classes):
        info = metrics_dict.get(c, {})
        tp = info.get("true_positives", 0)
        fp = info.get("false_positives", 0)
        fn = info.get("false_negatives", 0)
        tn = info.get("true_negatives", 0)

        mat_2x2 = np.array([[tp, fp], [fn, tn]])
        text_annot = [
            [f"<b>TP</b><br>{tp:,}", f"<b>FP</b><br>{fp:,}"],
            [f"<b>FN</b><br>{fn:,}", f"<b>TN</b><br>{tn:,}"],
        ]

        fig.add_trace(
            go.Heatmap(
                z=mat_2x2,
                x=["Pred +", "Pred −"],
                y=["True +", "True −"],
                text=text_annot,
                texttemplate="%{text}",
                textfont=dict(size=11, color="#000000", family="Inter, sans-serif"),
                colorscale=[[0, "#FFFFFF"], [1.0, "#90CAF9"]],
                showscale=False,
                hoverinfo="skip",
            ),
            row=1,
            col=idx + 1,
        )

    fig.update_layout(
        height=260,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    for c_idx in range(1, 6):
        fig.update_yaxes(autorange="reversed", row=1, col=c_idx)

    return apply_clean_scientific_theme(fig)


# ==============================================================================
# 4. ECG Image Preprocessing Pipeline Inspector
# ==============================================================================

def extract_image_preprocessing_stages(
    image_bytes: bytes,
    lead_key: str = "I",
    blur_sigma: float = 0.9,
    threshold_pct: float = 70.0,
) -> Dict[str, Any]:
    """Extract intermediate visual representations of the ECG image digitization pipeline.

    Stages:
        1. Original Cropped Lead Region (RGB)
        2. Normalized Ink Intensity Array (Paper background removed)
        3. Gaussian Blurred Smoothing
        4. Binarized Ink Mask
        5. Column-wise Centroid Tracking Trajectory
        6. Resampled 1D Waveform Trace
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    from ecg_lab.digitization.baseline import STANDARD_12LEAD_LAYOUT_BOXES

    box = STANDARD_12LEAD_LAYOUT_BOXES.get(lead_key, (0.18, 0.38, 0.05, 0.28))
    ymin_f, ymax_f, xmin_f, xmax_f = box
    y0, y1 = int(ymin_f * h), int(ymax_f * h)
    x0, x1 = int(xmin_f * w), int(xmax_f * w)

    cropped = img.crop((x0, y0, x1, y1))
    crop_arr = np.array(cropped, dtype=np.float32)

    # Grayscale
    gray = 0.2989 * crop_arr[:, :, 0] + 0.5870 * crop_arr[:, :, 1] + 0.1140 * crop_arr[:, :, 2]

    # Invert so ink is high
    max_v, min_v = np.max(gray), np.min(gray)
    if max_v > min_v:
        norm_inv = (max_v - gray) / (max_v - min_v)
    else:
        norm_inv = np.zeros_like(gray)

    # Gaussian blur
    blurred = scipy.ndimage.gaussian_filter(norm_inv, sigma=blur_sigma)

    # Threshold
    thresh_val = float(np.percentile(blurred, threshold_pct))
    binary_mask = (blurred > thresh_val).astype(np.uint8)

    # Centroids
    crop_h, crop_w = binary_mask.shape
    y_idx = np.arange(crop_h)
    centroids = []
    for x in range(crop_w):
        col_w = blurred[:, x] * binary_mask[:, x]
        s = np.sum(col_w)
        if s > 0.05:
            centroids.append(float(np.sum(y_idx * col_w) / s))
        else:
            centroids.append(np.nan)

    # Interpolate
    centroids_arr = np.array(centroids, dtype=np.float32)
    nans = np.isnan(centroids_arr)
    if np.any(nans) and not np.all(nans):
        x_all = np.arange(len(centroids_arr))
        centroids_arr[nans] = np.interp(x_all[nans], x_all[~nans], centroids_arr[~nans])
    elif np.all(nans):
        centroids_arr = np.zeros_like(centroids_arr)

    # Waveform inversion and normalization
    baseline_y = float(np.median(centroids_arr))
    waveform_1d = baseline_y - centroids_arr
    amp = np.max(np.abs(waveform_1d))
    if amp > 1e-4:
        waveform_1d = waveform_1d / amp

    return {
        "cropped_image": cropped,
        "grayscale": gray,
        "norm_inv": norm_inv,
        "blurred": blurred,
        "binary_mask": binary_mask,
        "threshold_val": thresh_val,
        "centroids": centroids_arr,
        "waveform_1d": waveform_1d,
        "lead_name": lead_key,
        "dimensions": f"{crop_w} × {crop_h} px",
    }


# ==============================================================================
# 5. Demo Fixture Loaders for Mode A (Hosted Demo / Vercel)
# ==============================================================================

def load_demo_ptbxl_records() -> List[Dict[str, Any]]:
    """Load bundled PTB-XL demo patient records for offline demo mode."""
    fixture_path = Path("sample_data/ptbxl_demo_records.json")
    if not fixture_path.is_file():
        return []
    try:
        with open(fixture_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def load_demo_mitbih_slice() -> Optional[Dict[str, Any]]:
    """Load bundled MIT-BIH demo record 100 slice for offline demo mode."""
    fixture_path = Path("sample_data/mitbih_demo_record_100.json")
    if not fixture_path.is_file():
        return None
    try:
        with open(fixture_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_demo_playback_slice() -> Optional[Any]:
    """Retrieve verified 10.0s demo slice from MIT-BIH record 100 with live detector & RF inference."""
    demo_dict = load_demo_mitbih_slice()
    if not demo_dict:
        return None

    sig = np.array(demo_dict["signal_mv"], dtype=np.float64)
    fs = int(demo_dict["sampling_rate_hz"])
    time_axis = np.arange(len(sig)) / float(fs)

    from ecg_lab.single_lead_ml.detector import pan_tompkins_qrs_detector
    peaks = pan_tompkins_qrs_detector(sig, sampling_rate_hz=fs)

    from ecg_lab.single_lead_ml.dataset import BeatAnnotation
    from ecg_lab.single_lead_ml.playback import PlaybackSegmentResult, PlaybackBeatResult
    from ecg_lab.single_lead_ml.features import extract_beat_features
    from ecg_lab.single_lead_ml.models import load_verified_single_lead_model
    from ecg_lab.single_lead_ml.config import DEFAULT_ARTIFACT_DIR, MODEL_NAME, AAMI_TARGET_CLASSES, DISCLAIMER_RECORDED_REPLAY

    ref_anns = [
        BeatAnnotation(
            sample_index=ann["sample_index"],
            original_symbol=ann["symbol"],
            aami_class=ann["aami_class"],
            time_seconds=ann["time_seconds"],
        )
        for ann in demo_dict.get("reference_annotations", [])
    ]

    detected_beats: List[PlaybackBeatResult] = []
    model_path = Path(DEFAULT_ARTIFACT_DIR) / MODEL_NAME
    if len(peaks) > 0 and model_path.is_file():
        try:
            model, manifest = load_verified_single_lead_model(model_path)
            valid_mask, features = extract_beat_features(sig, peaks, sampling_rate_hz=fs)
            if np.any(valid_mask):
                preds = model.predict(features)
                probs = model.predict_proba(features)
                classes = list(manifest.get("target_classes", AAMI_TARGET_CLASSES))

                v_idx = 0
                for i, p_samp in enumerate(peaks):
                    if not valid_mask[i]:
                        continue
                    p_cls = str(preds[v_idx])
                    p_prob = probs[v_idx]
                    prob_d = {cls_k: float(p_prob[c_i]) for c_i, cls_k in enumerate(classes)}

                    near_sym, near_cls = None, None
                    if ref_anns:
                        dists = [abs(ann.sample_index - p_samp) for ann in ref_anns]
                        min_i = int(np.argmin(dists))
                        if dists[min_i] <= int(0.05 * fs):
                            near_sym = ref_anns[min_i].original_symbol
                            near_cls = ref_anns[min_i].aami_class

                    detected_beats.append(
                        PlaybackBeatResult(
                            peak_sample=int(p_samp),
                            time_seconds=float(p_samp / fs),
                            predicted_class=p_cls,
                            class_probabilities=prob_d,
                            nearest_reference_symbol=near_sym,
                            nearest_reference_class=near_cls,
                        )
                    )
                    v_idx += 1
        except Exception:
            pass

    return PlaybackSegmentResult(
        record_id=demo_dict.get("record_id", "100"),
        lead_name=demo_dict.get("lead_name", "MLII"),
        start_second=0.0,
        duration_seconds=float(len(sig) / fs),
        sampling_rate_hz=fs,
        time_axis=time_axis,
        signal_mv=sig,
        disclaimer=DISCLAIMER_RECORDED_REPLAY,
        detected_beats=detected_beats,
        reference_annotations=ref_anns,
    )


# ==============================================================================
# 6. Educational Guided Walkthrough Visualizations
# ==============================================================================

def create_cardiac_anatomy_figure() -> go.Figure:
    """Create an educational annotated single-beat ECG waveform showing P-QRS-T components."""
    fs = 500
    t = np.linspace(0, 1.0, fs)
    ecg = (
        0.16 * np.exp(-((t - 0.20) ** 2) / (2 * 0.024 ** 2))
        - 0.12 * np.exp(-((t - 0.36) ** 2) / (2 * 0.008 ** 2))
        + 1.25 * np.exp(-((t - 0.40) ** 2) / (2 * 0.012 ** 2))
        - 0.35 * np.exp(-((t - 0.44) ** 2) / (2 * 0.012 ** 2))
        + 0.32 * np.exp(-((t - 0.65) ** 2) / (2 * 0.048 ** 2))
    )

    fig = go.Figure()

    # Physiological shaded background regions
    fig.add_vrect(x0=0.14, x1=0.26, fillcolor="rgba(2, 136, 209, 0.08)", line_width=0)
    fig.add_vrect(x0=0.26, x1=0.35, fillcolor="rgba(255, 193, 7, 0.08)", line_width=0)
    fig.add_vrect(x0=0.35, x1=0.48, fillcolor="rgba(198, 40, 40, 0.08)", line_width=0)
    fig.add_vrect(x0=0.48, x1=0.55, fillcolor="rgba(46, 125, 50, 0.08)", line_width=0)
    fig.add_vrect(x0=0.55, x1=0.78, fillcolor="rgba(106, 27, 154, 0.08)", line_width=0)

    # Lead II ECG continuous line
    fig.add_trace(
        go.Scatter(
            x=t,
            y=ecg,
            mode="lines",
            line=dict(color="#000000", width=2.6),
            name="Calibrated Lead II (mV)",
            hovertemplate="Time: %{x:.3f} s<br>Voltage: %{y:.3f} mV<extra></extra>",
        )
    )

    # Direct callout annotations with high-contrast white badges, black borders, and solid black text
    direct_annotations = [
        dict(
            x=0.20, y=0.18,
            text="<b>P Wave</b><br><span style='font-size:9.5px;'>Atrial Depol</span>",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#000000",
            ax=0, ay=-48,
            bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="#000000", borderwidth=1.2, borderpad=4,
            font=dict(size=11, color="#000000", family="Inter, sans-serif"),
        ),
        dict(
            x=0.30, y=0.00,
            text="<b>PR Segment</b><br><span style='font-size:9.5px;'>AV Nodal Delay</span>",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#000000",
            ax=0, ay=42,
            bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="#000000", borderwidth=1.2, borderpad=4,
            font=dict(size=11, color="#000000", family="Inter, sans-serif"),
        ),
        dict(
            x=0.36, y=-0.12,
            text="<b>Q</b>",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2, arrowcolor="#000000",
            ax=-22, ay=30,
            bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="#000000", borderwidth=1.2, borderpad=3,
            font=dict(size=11, color="#000000", family="Inter, sans-serif"),
        ),
        dict(
            x=0.40, y=1.25,
            text="<b>R Peak</b><br><span style='font-size:9.5px;'>Ventricular Depol</span>",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#000000",
            ax=0, ay=-45,
            bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="#000000", borderwidth=1.2, borderpad=4,
            font=dict(size=11, color="#000000", family="Inter, sans-serif"),
        ),
        dict(
            x=0.44, y=-0.35,
            text="<b>S</b>",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2, arrowcolor="#000000",
            ax=22, ay=30,
            bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="#000000", borderwidth=1.2, borderpad=3,
            font=dict(size=11, color="#000000", family="Inter, sans-serif"),
        ),
        dict(
            x=0.51, y=0.00,
            text="<b>ST Segment</b><br><span style='font-size:9.5px;'>STEMI Marker</span>",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#000000",
            ax=0, ay=42,
            bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="#000000", borderwidth=1.2, borderpad=4,
            font=dict(size=11, color="#000000", family="Inter, sans-serif"),
        ),
        dict(
            x=0.65, y=0.32,
            text="<b>T Wave</b><br><span style='font-size:9.5px;'>Ventricular Repol</span>",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#000000",
            ax=0, ay=-48,
            bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="#000000", borderwidth=1.2, borderpad=4,
            font=dict(size=11, color="#000000", family="Inter, sans-serif"),
        ),
    ]
    for ann in direct_annotations:
        fig.add_annotation(**ann)

    # Interval measurement brackets below baseline
    intervals = [
        dict(x0=0.14, x1=0.36, y=-0.50, label="PR Interval (120–200 ms)"),
        dict(x0=0.36, x1=0.46, y=-0.62, label="QRS (80–120 ms)"),
        dict(x0=0.36, x1=0.76, y=-0.74, label="QT Interval (<440 ms)"),
    ]
    for inter in intervals:
        fig.add_shape(
            type="line",
            x0=inter["x0"], x1=inter["x1"],
            y0=inter["y"], y1=inter["y"],
            line=dict(color="#000000", width=1.4),
        )
        fig.add_shape(
            type="line",
            x0=inter["x0"], x1=inter["x0"],
            y0=inter["y"] - 0.03, y1=inter["y"] + 0.03,
            line=dict(color="#000000", width=1.4),
        )
        fig.add_shape(
            type="line",
            x0=inter["x1"], x1=inter["x1"],
            y0=inter["y"] - 0.03, y1=inter["y"] + 0.03,
            line=dict(color="#000000", width=1.4),
        )
        fig.add_annotation(
            x=(inter["x0"] + inter["x1"]) / 2,
            y=inter["y"] - 0.05,
            text=f"<span style='font-size: 10px; color: #000000; font-weight: 600;'>{inter['label']}</span>",
            showarrow=False,
            yanchor="top",
            font=dict(color="#000000"),
        )

    fig.update_layout(
        title="Anatomy of a Cardiac Cycle — Direct Electrophysiological Annotations (Lead II)",
        xaxis_title="Time (seconds)",
        yaxis_title="Voltage (mV)",
        height=450,
        margin=dict(l=45, r=20, t=45, b=35),
        showlegend=False,
    )
    fig.update_xaxes(range=[0.05, 0.95])
    fig.update_yaxes(range=[-0.85, 1.45])
    return apply_clean_scientific_theme(fig)


def create_dataset_comparison_figure() -> go.Figure:
    """Create a side-by-side comparison of PTB-XL vs MIT-BIH class distributions."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "<b>PTB-XL v1.0.3 Superclasses (21,799 Records)</b>",
            "<b>MIT-BIH Matched Test Cohort (47,034 Beats)</b>",
        ],
        horizontal_spacing=0.12,
    )

    # PTB-XL class counts
    ptb_classes = ["NORM", "MI", "STTC", "CD", "HYP"]
    ptb_counts = [9528, 5486, 5250, 4907, 2655]
    fig.add_trace(
        go.Bar(
            x=ptb_classes,
            y=ptb_counts,
            marker=dict(color="rgba(33, 150, 243, 0.25)", line=dict(color="#000000", width=1.5)),
            text=[f"<b>{c:,}</b>" for c in ptb_counts],
            textposition="outside",
            textfont=dict(color="#000000", size=11, family="Inter, sans-serif"),
            name="PTB-XL",
        ),
        row=1, col=1,
    )

    # MIT-BIH beat counts
    mit_classes = ["N (Normal)", "S (Supravent.)", "V (Ventricular)", "Q (Fusion/Other)"]
    mit_counts = [41883, 1757, 3018, 376]
    fig.add_trace(
        go.Bar(
            x=mit_classes,
            y=mit_counts,
            marker=dict(color="rgba(76, 175, 80, 0.25)", line=dict(color="#000000", width=1.5)),
            text=[f"<b>{c:,}</b>" for c in mit_counts],
            textposition="outside",
            textfont=dict(color="#000000", size=11, family="Inter, sans-serif"),
            name="MIT-BIH",
        ),
        row=1, col=2,
    )

    fig.update_layout(
        height=360,
        margin=dict(l=40, r=20, t=40, b=30),
        showlegend=False,
    )
    fig.update_yaxes(title_text="Record Count", row=1, col=1)
    fig.update_yaxes(title_text="Beat Count", row=1, col=2)
    return apply_clean_scientific_theme(fig)


def create_threshold_calibration_figure(
    sample_probs: Optional[Dict[str, float]] = None,
) -> go.Figure:
    """Create an educational, high-contrast grouped horizontal bar chart demonstrating threshold calibration."""
    if sample_probs is None:
        # Example patient with acute Myocardial Infarction + ST/T Changes
        sample_probs = {
            "NORM": 0.12,
            "MI": 0.68,
            "STTC": 0.44,
            "CD": 0.18,
            "HYP": 0.09,
        }

    raw_keys = ["HYP", "CD", "STTC", "MI", "NORM"]  # inverted for bottom-up order
    class_labels = [
        "HYP (Hypertrophy)",
        "CD (Conduction Dist.)",
        "STTC (ST/T Changes)",
        "MI (Myocardial Infarct.)",
        "NORM (Normal Rhythm)",
    ]
    tuned_cutoffs = {
        "NORM": 0.448,
        "MI": 0.334,
        "STTC": 0.309,
        "CD": 0.311,
        "HYP": 0.280,
    }

    probs = [sample_probs[k] for k in raw_keys]
    cutoffs = [tuned_cutoffs[k] for k in raw_keys]

    prob_colors = ["#D32F2F" if probs[i] >= cutoffs[i] else "#90A4AE" for i in range(len(raw_keys))]
    prob_text = [
        f"<b>{probs[i]:.1%}</b> (POSITIVE)" if probs[i] >= cutoffs[i] else f"<b>{probs[i]:.1%}</b> (Negative)"
        for i in range(len(raw_keys))
    ]
    cutoff_text = [f"Cutoff: {cutoffs[i]:.1%}" for i in range(len(raw_keys))]

    fig = go.Figure()

    # Bar 1: Decision Cutoff (Validation Fold 9)
    fig.add_trace(
        go.Bar(
            name="Validation-Tuned Decision Threshold",
            y=class_labels,
            x=cutoffs,
            orientation="h",
            marker=dict(color="rgba(0, 0, 0, 0.08)", line=dict(color="#000000", width=1.5)),
            text=cutoff_text,
            textposition="outside",
            textfont=dict(size=11, color="#000000", family="Inter, sans-serif"),
        )
    )

    # Bar 2: Patient Output Probability
    fig.add_trace(
        go.Bar(
            name="Patient Output Probability",
            y=class_labels,
            x=probs,
            orientation="h",
            marker=dict(color=prob_colors),
            text=prob_text,
            textposition="outside",
            textfont=dict(size=11, color="#000000", family="Inter, sans-serif"),
        )
    )

    # Naive 0.5 decision threshold reference line
    fig.add_shape(
        type="line",
        x0=0.5, x1=0.5,
        y0=-0.5, y1=4.5,
        line=dict(color="#D32F2F", width=2, dash="dot"),
    )
    fig.add_annotation(
        x=0.5, y=4.5,
        text="<b>Naive 50% Cutoff (Misses STTC!)</b>",
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.95)",
        bordercolor="#D32F2F",
        borderwidth=1.2,
        borderpad=4,
        font=dict(size=11, color="#D32F2F", family="Inter, sans-serif"),
        xanchor="center",
        yanchor="bottom",
    )

    fig.update_layout(
        barmode="group",
        title="Validation-Tuned Decision Thresholds vs. Patient Output Probabilities (Sample Case)",
        xaxis=dict(
            title="Probability / Threshold Scale",
            range=[0, 1.18],
            tickformat=".0%",
        ),
        yaxis=dict(showgrid=False),
        height=440,
        margin=dict(l=170, r=40, t=65, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return apply_clean_scientific_theme(fig)


