"""Offline ADC / Serial Stream Simulator for Hardware Prototype Verification.

Generates realistic 10-bit ADC streams formatted identically to the Arduino Uno firmware protocol.
Sources from either:
1. MIT-BIH Arrhythmia Database records (via ecg_lab.single_lead_ml.dataset)
2. Deterministic mathematical synthetic ECG generator (fallback/standalone fixture)

Provides complete offline acceptance testing without requiring physical Arduino, AD8232, or UART hardware.
"""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from hardware.host.serial_protocol import (
    ADC_MAX_COUNT,
    ADC_MIN_COUNT,
    BAUD_RATE,
    NOMINAL_INTERVAL_MICROS,
    NOMINAL_SAMPLING_RATE_HZ,
    PROTOCOL_VERSION,
    SerialSample,
    format_serial_line,
)

# Default Simulator Transformation Parameters
DEFAULT_ADC_BASELINE: float = 512.0  # Midpoint of 10-bit range [0, 1023]
DEFAULT_ADC_GAIN_COUNTS_PER_MV: float = 200.0  # 200 counts/mV ~ 1 V dynamic swing for ±2.5 mV

# Named Simulation Presets
SIMULATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "Near-lossless software round trip": {
        "name": "Near-lossless software round trip",
        "description": "Zero simulation noise/drift. Demonstrates numerical ADC quantization error only.",
        "baseline_count": 512.0,
        "gain_counts_per_mv": 150.0,
        "noise_amplitude_mv": 0.0,
        "baseline_wander_amplitude_mv": 0.0,
        "mains_hum_amplitude_mv": 0.0,
    },
    "Noisy sensor-like stress test": {
        "name": "Noisy sensor-like stress test",
        "description": "Additive Gaussian noise (0.05 mV), low-frequency baseline drift (0.15 mV @ 0.25 Hz), and 50 Hz mains hum (0.08 mV).",
        "baseline_count": 512.0,
        "gain_counts_per_mv": 150.0,
        "noise_amplitude_mv": 0.05,
        "baseline_wander_amplitude_mv": 0.15,
        "mains_hum_amplitude_mv": 0.08,
    },
}


def generate_synthetic_ecg(
    duration_seconds: float = 10.0,
    sampling_rate_hz: int = NOMINAL_SAMPLING_RATE_HZ,
    heart_rate_bpm: float = 72.0,
    noise_amplitude_mv: float = 0.03,
    seed: int = 42,
) -> np.ndarray:
    """Generate deterministic synthetic ECG waveform in millivolts.

    Uses a Gaussian superposition model (P, Q, R, S, T waves) for offline testing.
    """
    rng = np.random.RandomState(seed)
    total_samples = int(duration_seconds * sampling_rate_hz)
    t = np.arange(total_samples) / sampling_rate_hz

    # Cardiac cycle period in seconds
    rr_interval = 60.0 / heart_rate_bpm
    phase = (t % rr_interval) / rr_interval  # Phase from 0 to 1

    # Gaussian wave components (center_phase, amplitude_mv, width_sigma)
    components = [
        (0.18, 0.15, 0.025),   # P-wave
        (0.28, -0.15, 0.010),  # Q-wave
        (0.32, 1.20, 0.012),   # R-wave (dominant spike)
        (0.36, -0.25, 0.012),  # S-wave
        (0.55, 0.30, 0.045),   # T-wave
    ]

    ecg_mv = np.zeros(total_samples, dtype=np.float64)
    for c_phase, amp, sigma in components:
        dist = np.abs(phase - c_phase)
        dist = np.minimum(dist, 1.0 - dist)
        ecg_mv += amp * np.exp(-0.5 * (dist / sigma) ** 2)

    # Add small deterministic high-frequency noise
    if noise_amplitude_mv > 0.0:
        noise = rng.normal(0.0, noise_amplitude_mv, total_samples)
        ecg_mv += noise
    return ecg_mv


def voltage_to_simulated_adc(
    signal_mv: np.ndarray,
    baseline_count: float = DEFAULT_ADC_BASELINE,
    gain_counts_per_mv: float = DEFAULT_ADC_GAIN_COUNTS_PER_MV,
) -> Tuple[np.ndarray, int]:
    """Map analog voltage array in mV to quantized 10-bit integer ADC counts with clipping.

    Returns:
        Tuple of (adc_counts_array, clipping_occurrences_count)
    """
    raw_float = baseline_count + signal_mv * gain_counts_per_mv
    rounded = np.rint(raw_float).astype(np.int64)

    # Detect clipping
    clipped_below = rounded < ADC_MIN_COUNT
    clipped_above = rounded > ADC_MAX_COUNT
    total_clipping = int(np.sum(clipped_below) + np.sum(clipped_above))

    # Clamp to 10-bit range
    clamped = np.clip(rounded, ADC_MIN_COUNT, ADC_MAX_COUNT).astype(np.int32)
    return clamped, total_clipping


def simulated_adc_to_voltage(
    adc_counts: Union[np.ndarray, List[int]],
    baseline_count: float = DEFAULT_ADC_BASELINE,
    gain_counts_per_mv: float = DEFAULT_ADC_GAIN_COUNTS_PER_MV,
    reference_signal_mv: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Inverse transformation: convert simulated 10-bit ADC counts back to simulated mV.

    Formula:
        reconstructed_mv = (adc_counts - baseline_count) / gain_counts_per_mv

    Notice:
        This conversion is valid ONLY for offline software simulation tests with known simulator
        parameters. It MUST NOT be applied to physical, uncalibrated hardware serial data!

    Args:
        adc_counts: Array or list of integer ADC counts.
        baseline_count: Configured simulator ADC offset for 0.0 mV.
        gain_counts_per_mv: Configured simulator scale factor (counts/mV).
        reference_signal_mv: Optional source signal before ADC quantization for error metrics.

    Returns:
        Dictionary with:
            - 'reconstructed_mv': 1D numpy array of reconstructed millivolt values.
            - 'clipping_count': Number of samples at rails (0 or 1023).
            - 'clipping_percentage': Percentage of clipped samples.
            - 'max_abs_error_mv': Maximum absolute reconstruction error (if reference provided).
            - 'mae_mv': Mean absolute reconstruction error (if reference provided).
            - 'rmse_mv': Root mean squared reconstruction error (if reference provided).
    """
    counts = np.asarray(adc_counts, dtype=np.float64)
    if gain_counts_per_mv <= 0:
        raise ValueError(f"gain_counts_per_mv must be > 0, got {gain_counts_per_mv}")

    reconstructed_mv = (counts - baseline_count) / float(gain_counts_per_mv)

    # Clipping detection on input counts
    clipped_mask = (counts <= ADC_MIN_COUNT) | (counts >= ADC_MAX_COUNT)
    clipping_count = int(np.sum(clipped_mask))
    clipping_pct = (clipping_count / float(len(counts)) * 100.0) if len(counts) > 0 else 0.0

    result: Dict[str, Any] = {
        "reconstructed_mv": reconstructed_mv,
        "clipping_count": clipping_count,
        "clipping_percentage": clipping_pct,
        "max_abs_error_mv": None,
        "mae_mv": None,
        "rmse_mv": None,
    }

    if reference_signal_mv is not None:
        ref_arr = np.asarray(reference_signal_mv, dtype=np.float64)
        min_len = min(len(reconstructed_mv), len(ref_arr))
        if min_len > 0:
            diff = np.abs(reconstructed_mv[:min_len] - ref_arr[:min_len])
            result["max_abs_error_mv"] = float(np.max(diff))
            result["mae_mv"] = float(np.mean(diff))
            result["rmse_mv"] = float(np.sqrt(np.mean(diff ** 2)))

    return result


def simulate_adc_stream(
    source_type: str = "mitbih",
    record_id: str = "100",
    lead_name: Optional[str] = None,
    duration_seconds: float = 10.0,
    sampling_rate_hz: int = NOMINAL_SAMPLING_RATE_HZ,
    baseline_count: float = DEFAULT_ADC_BASELINE,
    gain_counts_per_mv: float = DEFAULT_ADC_GAIN_COUNTS_PER_MV,
    noise_amplitude_mv: float = 0.0,
    baseline_wander_amplitude_mv: float = 0.0,
    mains_hum_amplitude_mv: float = 0.0,
    lead_off_start_s: Optional[float] = None,
    lead_off_duration_s: Optional[float] = None,
    seed: int = 42,
) -> Tuple[List[SerialSample], Dict[str, object]]:
    """Simulate complete serial stream of SerialSample objects and metadata.

    Args:
        source_type: 'mitbih' or 'synthetic'.
        record_id: MIT-BIH record ID (if source_type == 'mitbih').
        lead_name: Specific lead name (e.g. 'MLII').
        duration_seconds: Total duration to simulate in seconds.
        sampling_rate_hz: Sample rate (default 360 Hz).
        baseline_count: ADC offset for 0.0 mV.
        gain_counts_per_mv: Scale factor in ADC counts per mV.
        noise_amplitude_mv: Amplitude of additive Gaussian noise (mV).
        baseline_wander_amplitude_mv: Amplitude of 0.25 Hz drift (mV).
        mains_hum_amplitude_mv: Amplitude of 50 Hz powerline hum (mV).
        lead_off_start_s: Start second for simulated lead-off fault.
        lead_off_duration_s: Duration in seconds for simulated lead-off fault.
        seed: Random seed for noise/synthetic generation.

    Returns:
        Tuple of (list of SerialSample instances, provenance metadata dict).
    """
    provenance: Dict[str, object] = {
        "simulator_version": PROTOCOL_VERSION,
        "source_type": source_type.upper(),
        "sampling_rate_hz": sampling_rate_hz,
        "sample_interval_micros": NOMINAL_INTERVAL_MICROS,
        "baseline_count": baseline_count,
        "gain_counts_per_mv": gain_counts_per_mv,
        "duration_seconds": duration_seconds,
        "noise_amplitude_mv": noise_amplitude_mv,
        "baseline_wander_amplitude_mv": baseline_wander_amplitude_mv,
        "mains_hum_amplitude_mv": mains_hum_amplitude_mv,
        "random_seed": seed,
        "calibration_provenance": "SIMULATION PRESET — NOT HARDWARE CALIBRATION",
        "disclaimer": (
            "RAW SIMULATED ADC COUNTS FOR TRANSPORT/SOFTWARE TESTING ONLY — "
            "NOT REAL SENSOR ACQUISITIONS OR CALIBRATED HARDWARE TRANSFER FUNCTION. "
            "DO NOT USE FOR CLINICAL DIAGNOSTICS."
        ),
    }

    signal_mv: np.ndarray
    lead_used = "SYNTHETIC"

    if source_type.lower() == "mitbih":
        try:
            from ecg_lab.single_lead_ml.dataset import load_single_lead_signal
            rec = load_single_lead_signal(record_id=record_id, preferred_lead=lead_name)
            max_avail_samples = len(rec.signal_mv)
            req_samples = int(duration_seconds * sampling_rate_hz)
            take_samples = min(req_samples, max_avail_samples)

            signal_mv = rec.signal_mv[:take_samples]
            lead_used = rec.lead_name
            provenance["mitbih_record_id"] = record_id
            provenance["lead_name"] = lead_used
            provenance["source_dataset"] = "MIT-BIH Arrhythmia Database v1.0.0"
        except Exception as e:
            # Fallback to synthetic if MIT-BIH loading fails
            print(f"[!] Warning: Could not load MIT-BIH record {record_id} ({e}). Falling back to synthetic waveform.")
            signal_mv = generate_synthetic_ecg(duration_seconds=duration_seconds, sampling_rate_hz=sampling_rate_hz, seed=seed)
            provenance["source_type"] = "SYNTHETIC_FALLBACK"
            provenance["fallback_reason"] = str(e)
    else:
        signal_mv = generate_synthetic_ecg(
            duration_seconds=duration_seconds,
            sampling_rate_hz=sampling_rate_hz,
            noise_amplitude_mv=noise_amplitude_mv,
            seed=seed,
        )
        provenance["source_type"] = "SYNTHETIC"
        provenance["synthetic_seed"] = seed

    # Apply deterministic simulation-only perturbations
    perturbed_signal = np.copy(signal_mv)
    rng = np.random.RandomState(seed)

    if noise_amplitude_mv > 0.0 and source_type.lower() == "mitbih":
        noise = rng.normal(0.0, noise_amplitude_mv, len(perturbed_signal))
        perturbed_signal += noise

    if baseline_wander_amplitude_mv > 0.0:
        t_arr = np.arange(len(perturbed_signal)) / float(sampling_rate_hz)
        wander = baseline_wander_amplitude_mv * np.sin(2.0 * np.pi * 0.25 * t_arr)
        perturbed_signal += wander

    if mains_hum_amplitude_mv > 0.0:
        t_arr = np.arange(len(perturbed_signal)) / float(sampling_rate_hz)
        hum = mains_hum_amplitude_mv * np.sin(2.0 * np.pi * 50.0 * t_arr)
        perturbed_signal += hum

    adc_counts, clipping_count = voltage_to_simulated_adc(
        perturbed_signal, baseline_count=baseline_count, gain_counts_per_mv=gain_counts_per_mv
    )

    provenance["total_samples"] = len(adc_counts)
    provenance["clipping_samples"] = clipping_count
    provenance["clipping_percentage"] = (clipping_count / float(len(adc_counts)) * 100.0) if len(adc_counts) > 0 else 0.0
    provenance["min_adc"] = int(np.min(adc_counts)) if len(adc_counts) > 0 else 0
    provenance["max_adc"] = int(np.max(adc_counts)) if len(adc_counts) > 0 else 0

    # Build SerialSample objects
    samples: List[SerialSample] = []
    base_micros = 100_000  # Realistic startup timer offset

    for i, adc in enumerate(adc_counts):
        t_sec = i / float(sampling_rate_hz)
        cur_micros = (base_micros + int(i * NOMINAL_INTERVAL_MICROS)) % (2**32)

        # Lead-off simulation
        lo_plus = 0
        lo_minus = 0
        if lead_off_start_s is not None and lead_off_duration_s is not None:
            if lead_off_start_s <= t_sec < (lead_off_start_s + lead_off_duration_s):
                lo_plus = 1
                lo_minus = 1
                # When leads detach, ADC typically floats toward rail or zero
                adc = ADC_MAX_COUNT

        sample = SerialSample(
            seq=i,
            timestamp_micros=cur_micros,
            raw_adc=int(adc),
            lo_plus=lo_plus,
            lo_minus=lo_minus,
            lead_off=bool(lo_plus or lo_minus),
            time_seconds=t_sec,
        )
        samples.append(sample)

    return samples, provenance


def write_simulated_stream_files(
    samples: List[SerialSample],
    provenance: Dict[str, object],
    output_base_path: Union[str, Path],
    write_raw_protocol: bool = True,
    write_dataframe_csv: bool = True,
) -> Tuple[Path, Path]:
    """Write simulated samples to raw protocol text file, CSV dataframe, and metadata JSON.

    Returns:
        Tuple of (path_to_data_file, path_to_metadata_json)
    """
    base_p = Path(output_base_path)
    base_p.parent.mkdir(parents=True, exist_ok=True)

    csv_path = base_p.with_suffix(".csv")
    meta_path = base_p.with_suffix(".meta.json")

    # Write raw line-oriented UART protocol stream
    if write_raw_protocol:
        raw_stream_path = base_p.with_suffix(".stream")
        with open(raw_stream_path, "w", encoding="utf-8") as f:
            f.write("# CardioSense AD8232 Arduino Uno v1.0\n")
            f.write(f"# baud={BAUD_RATE}, nominal_fs={NOMINAL_SAMPLING_RATE_HZ}, interval_us={NOMINAL_INTERVAL_MICROS}\n")
            f.write("# fields=seq,micros,adc,lo_plus,lo_minus\n")
            f.write("# NOTICE: RAW 10-BIT ADC COUNTS (0-1023). NOT CALIBRATED ECG MV. PROTOTYPE ONLY.\n")
            for s in samples:
                f.write(format_serial_line(s))

    # Write DataFrame CSV
    if write_dataframe_csv:
        data = [s.to_dict() for s in samples]
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

    # Write metadata JSON
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    return csv_path, meta_path


def main():
    parser = argparse.ArgumentParser(description="Simulate AD8232 Arduino Uno ADC stream offline.")
    parser.add_argument("--source", choices=["mitbih", "synthetic"], default="mitbih", help="Signal source (default: mitbih)")
    parser.add_argument("--record", type=str, default="100", help="MIT-BIH record ID (default: 100)")
    parser.add_argument("--lead", type=str, default="MLII", help="Preferred lead name")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration in seconds (default: 10.0)")
    parser.add_argument("--output", type=str, default="hardware/simulation/simulated_stream.csv", help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic random seed")
    parser.add_argument("--lead-off-start", type=float, default=None, help="Start second for simulated lead detachment")
    parser.add_argument("--lead-off-duration", type=float, default=None, help="Duration of simulated lead detachment in seconds")
    parser.add_argument("--preset", choices=list(SIMULATION_PRESETS.keys()), default=None, help="Named preset")

    args = parser.parse_args()

    # Apply preset if specified
    baseline = DEFAULT_ADC_BASELINE
    gain = DEFAULT_ADC_GAIN_COUNTS_PER_MV
    noise = 0.0
    drift = 0.0
    hum = 0.0

    if args.preset and args.preset in SIMULATION_PRESETS:
        p = SIMULATION_PRESETS[args.preset]
        baseline = p["baseline_count"]
        gain = p["gain_counts_per_mv"]
        noise = p["noise_amplitude_mv"]
        drift = p["baseline_wander_amplitude_mv"]
        hum = p["mains_hum_amplitude_mv"]

    try:
        samples, prov = simulate_adc_stream(
            source_type=args.source,
            record_id=args.record,
            lead_name=args.lead,
            duration_seconds=args.duration,
            baseline_count=baseline,
            gain_counts_per_mv=gain,
            noise_amplitude_mv=noise,
            baseline_wander_amplitude_mv=drift,
            mains_hum_amplitude_mv=hum,
            seed=args.seed,
            lead_off_start_s=args.lead_off_start,
            lead_off_duration_s=args.lead_off_duration,
        )

        csv_p, meta_p = write_simulated_stream_files(
            samples=samples,
            provenance=prov,
            output_base_path=Path(args.output),
        )

        print(f"[+] Simulation Complete:")
        print(f"    - Source: {prov['source_type']} (Record: {prov.get('mitbih_record_id', 'N/A')}, Lead: {prov.get('lead_name', 'N/A')})")
        print(f"    - Generated {len(samples)} samples ({args.duration} s @ {NOMINAL_SAMPLING_RATE_HZ} Hz).")
        print(f"    - ADC Range: [{prov['min_adc']}, {prov['max_adc']}], Clipped samples: {prov['clipping_samples']}")
        print(f"    - Saved CSV to: {csv_p}")
        print(f"    - Saved Metadata to: {meta_p}")

    except Exception as e:
        print(f"[ERROR] Simulation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
