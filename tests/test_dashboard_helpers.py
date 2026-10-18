"""Unit tests for dashboard analytical helpers, filter details, matrices, and demo loaders."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

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
    load_demo_ptbxl_records,
    load_demo_mitbih_slice,
)


class TestFilterAndSignalHelpers:
    """Test filter frequency response and signal cascade utilities."""

    def test_compute_filter_bode_data(self):
        w, mag, phase = compute_filter_bode_data(lowcut=5.0, highcut=15.0, fs=360.0, notch_freq=50.0)
        assert len(w) == 1024
        assert len(mag) == 1024
        assert len(phase) == 1024
        assert np.all(np.isfinite(mag))
        assert np.all(np.isfinite(phase))

        # Passband gain around 10 Hz should be close to 0 dB (> -3 dB)
        idx_10hz = int(np.argmin(np.abs(w - 10.0)))
        assert mag[idx_10hz] > -3.0

        # Stopband gain at 50 Hz with notch should be strongly attenuated (< -15 dB)
        idx_50hz = int(np.argmin(np.abs(w - 50.0)))
        assert mag[idx_50hz] < -15.0

    def test_apply_bandpass_notch_filter(self):
        t = np.linspace(0, 1.0, 360)
        sig = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)  # 10 Hz signal + 50 Hz hum
        filtered = apply_bandpass_notch_filter(sig, fs=360.0, lowcut=5.0, highcut=15.0, notch_freq=50.0)
        assert len(filtered) == len(sig)
        assert np.all(np.isfinite(filtered))
        # 50 Hz hum amplitude should be reduced
        assert np.std(filtered) < np.std(sig)

    def test_compute_pan_tompkins_cascade(self):
        sig = np.random.randn(360)
        stages = compute_pan_tompkins_cascade(sig, fs=360.0)
        assert "raw" in stages
        assert "bandpass" in stages
        assert "derivative" in stages
        assert "squared" in stages
        assert "integrated" in stages
        assert "threshold" in stages
        assert len(stages["bandpass"]) == len(sig)
        assert len(stages["integrated"]) == len(sig)
        assert np.all(stages["squared"] >= 0)
        assert np.all(stages["integrated"] >= 0)


class TestPlotlyFigureConstructors:
    """Test interactive visualization builders."""

    def test_bode_plot_figure(self):
        w, mag, phase = compute_filter_bode_data(lowcut=5.0, highcut=15.0, fs=360.0)
        fig = create_bode_plot(w, mag, phase, lowcut=5.0, highcut=15.0)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Mag and phase traces

    def test_pan_tompkins_cascade_figure(self):
        sig = np.random.randn(360)
        stages = compute_pan_tompkins_cascade(sig, fs=360.0)
        time_axis = np.arange(360) / 360.0
        fig = create_pan_tompkins_cascade_figure(time_axis, stages, peak_times=[0.5])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 5

    def test_before_after_figure(self):
        t = np.arange(100) / 100.0
        raw = np.sin(t)
        filt = raw * 0.9
        fig = create_before_after_filter_figure(t, raw, filt, lead_name="Lead II")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # Raw, Filtered, Residual

    def test_confusion_matrix_figure_modes(self):
        mat = np.array([[100, 10], [5, 85]])
        labels = ["A", "B"]
        fig_none = create_confusion_matrix_figure(mat, labels, title="Raw Counts", normalize="none")
        assert isinstance(fig_none, go.Figure)

        fig_true = create_confusion_matrix_figure(mat, labels, title="Recall", normalize="true")
        assert isinstance(fig_true, go.Figure)

        fig_pred = create_confusion_matrix_figure(mat, labels, title="Precision", normalize="pred")
        assert isinstance(fig_pred, go.Figure)

    def test_signal_matrix_and_correlation_heatmaps(self):
        data = {"sample_index": list(range(10))}
        for lead in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]:
            data[lead] = np.random.randn(10).tolist()
        df = pd.DataFrame(data)

        fig_mat = create_12lead_matrix_heatmap(df)
        assert isinstance(fig_mat, go.Figure)

        fig_corr = create_lead_correlation_heatmap(df)
        assert isinstance(fig_corr, go.Figure)

    def test_multilabel_contingency_figure(self):
        metrics = {
            c: {"true_positives": 100, "false_positives": 20, "false_negatives": 15, "true_negatives": 300}
            for c in ["NORM", "MI", "STTC", "CD", "HYP"]
        }
        fig = create_multilabel_contingency_figure(metrics)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 5


class TestImagePreprocessingAndDemoLoaders:
    """Test image pipeline stages and demo dataset loaders."""

    def test_extract_image_preprocessing_stages(self):
        bench_path = Path("sample_data/synthetic_12lead.png")
        if not bench_path.is_file():
            pytest.skip("synthetic_12lead.png fixture missing")
        img_bytes = bench_path.read_bytes()
        stages = extract_image_preprocessing_stages(img_bytes, lead_key="I")
        assert "cropped_image" in stages
        assert "grayscale" in stages
        assert "blurred" in stages
        assert "binary_mask" in stages
        assert "centroids" in stages
        assert "waveform_1d" in stages
        assert len(stages["waveform_1d"]) > 0

    def test_load_demo_ptbxl_records(self):
        records = load_demo_ptbxl_records()
        assert len(records) >= 1
        r = records[0]
        assert "ecg_id" in r
        assert "patient_id" in r
        assert "diagnostic_superclasses" in r
        assert "signal_mv" in r
        assert len(r["signal_mv"]) == 1000  # 10s at 100 Hz
        assert len(r["signal_mv"][0]) == 12  # 12 leads

    def test_load_demo_mitbih_slice(self):
        slice_data = load_demo_mitbih_slice()
        assert slice_data is not None
        assert slice_data["record_id"] == "100"
        assert slice_data["lead_name"] == "MLII"
        assert slice_data["sampling_rate_hz"] == 360
        assert len(slice_data["signal_mv"]) == 3600  # 10s at 360 Hz
        assert len(slice_data["reference_annotations"]) > 0
