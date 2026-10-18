"""Hardware Simulation Standalone Validation & Benchmark Script.

Executes reproducible offline simulation, protocol transport, inversion, and
beat-by-beat inference evaluation across:
1. Near-lossless software round trip (MIT-BIH record 100, Lead MLII)
2. Noisy sensor-like stress test (MIT-BIH record 100 with noise, drift, and hum)
3. Deterministic synthetic ECG fixture round trip

Saves machine-readable results artifact at:
hardware/reports/simulation_validation_results.json
"""

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecg_lab.single_lead_ml.config import (
    AAMI_TARGET_CLASSES,
    DEFAULT_ARTIFACT_DIR,
    MODEL_NAME,
    NATIVE_SAMPLING_RATE_HZ,
)
from ecg_lab.single_lead_ml.dataset import (
    BeatAnnotation,
    load_record_with_annotations,
)
from ecg_lab.single_lead_ml.inference import run_single_lead_inference
from hardware.host.serial_protocol import (
    SerialStreamParser,
    format_serial_line,
)
from hardware.host.simulate_adc_stream import (
    SIMULATION_PRESETS,
    generate_synthetic_ecg,
    simulate_adc_stream,
    simulated_adc_to_voltage,
)

EXPECTED_MODEL_SHA256 = "3889a5f5ab97803da339710d4ae9fde8ea50f8e5a5ff76ed2015a0fb0b6ec8e9"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_validation():
    print("[*] Running Phase 7.1 Offline Simulation & Inference Validation...")

    # 1. Model Checksum Verification
    model_path = DEFAULT_ARTIFACT_DIR / MODEL_NAME
    assert model_path.is_file(), f"Model file not found: {model_path}"
    actual_sha = compute_sha256(model_path)
    print(f"[*] Model SHA-256 Checksum: {actual_sha}")
    assert actual_sha == EXPECTED_MODEL_SHA256, (
        f"Model SHA mismatch: expected {EXPECTED_MODEL_SHA256}, got {actual_sha}"
    )

    results: Dict[str, Any] = {
        "phase": "7.1",
        "model_file": str(model_path),
        "model_sha256": actual_sha,
        "model_sha256_verified": True,
        "runs": {},
    }

    # 2. Benchmark 1: MIT-BIH Record 100 — Near-lossless software round trip
    print("\n[+] Benchmark 1: MIT-BIH Record 100 — Near-lossless software round trip (10.0 s)")
    rec_obj, all_anns = load_record_with_annotations("100")
    fs = rec_obj.sampling_rate_hz
    duration_s = 10.0
    n_samples = int(duration_s * fs)
    source_mv = rec_obj.signal_mv[:n_samples]
    sub_anns = [
        BeatAnnotation(
            sample_index=a.sample_index,
            original_symbol=a.original_symbol,
            aami_class=a.aami_class,
            time_seconds=a.sample_index / float(fs),
        )
        for a in all_anns if a.sample_index < n_samples
    ]

    p_near = SIMULATION_PRESETS["Near-lossless software round trip"]
    samples_near, prov_near = simulate_adc_stream(
        source_type="mitbih",
        record_id="100",
        duration_seconds=duration_s,
        sampling_rate_hz=fs,
        baseline_count=p_near["baseline_count"],
        gain_counts_per_mv=p_near["gain_counts_per_mv"],
        noise_amplitude_mv=p_near["noise_amplitude_mv"],
        baseline_wander_amplitude_mv=p_near["baseline_wander_amplitude_mv"],
        mains_hum_amplitude_mv=p_near["mains_hum_amplitude_mv"],
    )

    # Transport serialization & parse back
    lines = [format_serial_line(s) for s in samples_near]
    parser_near = SerialStreamParser()
    for l in lines:
        parser_near.feed_line(l)

    parsed_df_near = parser_near.to_dataframe()
    parsed_adc_near = parsed_df_near["raw_adc"].to_numpy(dtype=np.int32)

    recon_near = simulated_adc_to_voltage(
        adc_counts=parsed_adc_near,
        baseline_count=p_near["baseline_count"],
        gain_counts_per_mv=p_near["gain_counts_per_mv"],
        reference_signal_mv=source_mv,
    )

    inf_near = run_single_lead_inference(
        signal=recon_near["reconstructed_mv"],
        sampling_rate_hz=fs,
        units="mV",
        calibration_provenance=prov_near["calibration_provenance"],
        source_kind="offline AD8232-like simulation",
        reference_annotations=sub_anns,
    )

    print(f"    - Samples: {len(parsed_adc_near)}, ADC Range: [{prov_near['min_adc']}, {prov_near['max_adc']}]")
    print(f"    - Clipping Count: {prov_near['clipping_samples']} ({prov_near['clipping_percentage']:.2f}%)")
    print(f"    - Quantization Max Error: {recon_near['max_abs_error_mv']:.4f} mV, MAE: {recon_near['mae_mv']:.4f} mV")
    print(f"    - Candidate Peaks: {inf_near.detected_peaks_count}, Predictions: {len(inf_near.predictions)}")
    print(f"    - Summary: {inf_near.predictions_summary}")
    if inf_near.evaluation:
        det = inf_near.evaluation["detector_evaluation"]
        print(f"    - Detector TP: {det['true_positives']}, Sensitivity: {det['sensitivity']:.2%}, PPV: {det['positive_predictive_value']:.2%}")
        cls_ev = inf_near.evaluation["classifier_evaluation"]
        if cls_ev:
            print(f"    - Matched Classification Macro F1: {cls_ev['macro_f1']:.4f}, Weighted F1: {cls_ev['weighted_f1']:.4f}")

    results["runs"]["mitbih_100_near_lossless"] = {
        "preset": "Near-lossless software round trip",
        "source": "MIT-BIH Record 100",
        "duration_seconds": duration_s,
        "samples_count": len(parsed_adc_near),
        "adc_min": prov_near["min_adc"],
        "adc_max": prov_near["max_adc"],
        "clipping_samples": prov_near["clipping_samples"],
        "clipping_pct": prov_near["clipping_percentage"],
        "reconstruction_max_abs_error_mv": recon_near["max_abs_error_mv"],
        "reconstruction_mae_mv": recon_near["mae_mv"],
        "detected_peaks": inf_near.detected_peaks_count,
        "classifiable_windows": inf_near.classifiable_windows_count,
        "predictions_summary": inf_near.predictions_summary,
        "evaluation": inf_near.evaluation,
    }

    # 3. Benchmark 2: MIT-BIH Record 100 — Noisy sensor-like stress test
    print("\n[+] Benchmark 2: MIT-BIH Record 100 — Noisy sensor-like stress test (10.0 s)")
    p_noisy = SIMULATION_PRESETS["Noisy sensor-like stress test"]
    samples_noisy, prov_noisy = simulate_adc_stream(
        source_type="mitbih",
        record_id="100",
        duration_seconds=duration_s,
        sampling_rate_hz=fs,
        baseline_count=p_noisy["baseline_count"],
        gain_counts_per_mv=p_noisy["gain_counts_per_mv"],
        noise_amplitude_mv=p_noisy["noise_amplitude_mv"],
        baseline_wander_amplitude_mv=p_noisy["baseline_wander_amplitude_mv"],
        mains_hum_amplitude_mv=p_noisy["mains_hum_amplitude_mv"],
    )

    lines_noisy = [format_serial_line(s) for s in samples_noisy]
    parser_noisy = SerialStreamParser()
    for l in lines_noisy:
        parser_noisy.feed_line(l)

    parsed_df_noisy = parser_noisy.to_dataframe()
    parsed_adc_noisy = parsed_df_noisy["raw_adc"].to_numpy(dtype=np.int32)

    recon_noisy = simulated_adc_to_voltage(
        adc_counts=parsed_adc_noisy,
        baseline_count=p_noisy["baseline_count"],
        gain_counts_per_mv=p_noisy["gain_counts_per_mv"],
        reference_signal_mv=source_mv,
    )

    inf_noisy = run_single_lead_inference(
        signal=recon_noisy["reconstructed_mv"],
        sampling_rate_hz=fs,
        units="mV",
        calibration_provenance=prov_noisy["calibration_provenance"],
        source_kind="offline AD8232-like simulation",
        reference_annotations=sub_anns,
    )

    print(f"    - Samples: {len(parsed_adc_noisy)}, ADC Range: [{prov_noisy['min_adc']}, {prov_noisy['max_adc']}]")
    print(f"    - Clipping Count: {prov_noisy['clipping_samples']} ({prov_noisy['clipping_percentage']:.2f}%)")
    print(f"    - Total Noise/Drift Error MAE: {recon_noisy['mae_mv']:.4f} mV")
    print(f"    - Candidate Peaks: {inf_noisy.detected_peaks_count}, Predictions: {len(inf_noisy.predictions)}")
    print(f"    - Summary: {inf_noisy.predictions_summary}")
    if inf_noisy.evaluation:
        det = inf_noisy.evaluation["detector_evaluation"]
        print(f"    - Detector TP: {det['true_positives']}, Sensitivity: {det['sensitivity']:.2%}, PPV: {det['positive_predictive_value']:.2%}")
        cls_ev = inf_noisy.evaluation["classifier_evaluation"]
        if cls_ev:
            print(f"    - Matched Classification Macro F1: {cls_ev['macro_f1']:.4f}, Weighted F1: {cls_ev['weighted_f1']:.4f}")

    results["runs"]["mitbih_100_noisy_stress"] = {
        "preset": "Noisy sensor-like stress test",
        "source": "MIT-BIH Record 100",
        "duration_seconds": duration_s,
        "samples_count": len(parsed_adc_noisy),
        "adc_min": prov_noisy["min_adc"],
        "adc_max": prov_noisy["max_adc"],
        "clipping_samples": prov_noisy["clipping_samples"],
        "clipping_pct": prov_noisy["clipping_percentage"],
        "reconstruction_mae_mv": recon_noisy["mae_mv"],
        "detected_peaks": inf_noisy.detected_peaks_count,
        "classifiable_windows": inf_noisy.classifiable_windows_count,
        "predictions_summary": inf_noisy.predictions_summary,
        "evaluation": inf_noisy.evaluation,
    }

    # 4. Benchmark 3: Deterministic Synthetic ECG Fixture
    print("\n[+] Benchmark 3: Deterministic Synthetic ECG Fixture (10.0 s @ 72 BPM)")
    synth_source_mv = generate_synthetic_ecg(duration_seconds=10.0, sampling_rate_hz=360, seed=42)
    samples_synth, prov_synth = simulate_adc_stream(
        source_type="synthetic",
        duration_seconds=10.0,
        sampling_rate_hz=360,
        baseline_count=512.0,
        gain_counts_per_mv=150.0,
        seed=42,
    )
    lines_synth = [format_serial_line(s) for s in samples_synth]
    parser_synth = SerialStreamParser()
    for l in lines_synth:
        parser_synth.feed_line(l)

    parsed_df_synth = parser_synth.to_dataframe()
    parsed_adc_synth = parsed_df_synth["raw_adc"].to_numpy(dtype=np.int32)
    recon_synth = simulated_adc_to_voltage(parsed_adc_synth, baseline_count=512.0, gain_counts_per_mv=150.0, reference_signal_mv=synth_source_mv)

    inf_synth = run_single_lead_inference(
        signal=recon_synth["reconstructed_mv"],
        sampling_rate_hz=360,
        units="mV",
        calibration_provenance=prov_synth["calibration_provenance"],
        source_kind="offline AD8232-like simulation",
    )

    print(f"    - Samples: {len(parsed_adc_synth)}, ADC Range: [{prov_synth['min_adc']}, {prov_synth['max_adc']}]")
    print(f"    - Candidate Peaks: {inf_synth.detected_peaks_count}, Predictions: {len(inf_synth.predictions)}")
    print(f"    - Summary: {inf_synth.predictions_summary}")

    results["runs"]["synthetic_fixture"] = {
        "preset": "Default Synthetic Fixture",
        "duration_seconds": 10.0,
        "samples_count": len(parsed_adc_synth),
        "adc_min": prov_synth["min_adc"],
        "adc_max": prov_synth["max_adc"],
        "reconstruction_mae_mv": recon_synth["mae_mv"],
        "detected_peaks": inf_synth.detected_peaks_count,
        "classifiable_windows": inf_synth.classifiable_windows_count,
        "predictions_summary": inf_synth.predictions_summary,
    }

    # Save to JSON
    out_dir = Path("hardware/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "simulation_validation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Validation artifact written to: {out_file}")
    return results


if __name__ == "__main__":
    run_validation()
