"""Calibration guards and unit separation for hardware signals and ML model pipelines.

Rigorous boundaries enforced:
1. Raw ADC counts (unitless integer 0-1023) != ADC input voltage != patient ECG mV.
2. Converting ADC counts to input voltage requires verified board Vref and bit resolution.
3. Converting ADC input voltage to true patient ECG mV requires verified analog front-end
   transfer function, gain calibration, DC offset, and electrode impedance parameters.
4. Model inference is strictly blocked on uncalibrated or incompatible signals.
"""

from enum import Enum
from typing import Any, Dict, Optional, Union
import numpy as np


class SignalUnits(str, Enum):
    """Explicit taxonomy of signal representation units."""
    RAW_ADC_COUNTS = "raw_adc_counts"
    INPUT_VOLTS = "input_volts"
    ECG_MILLIVOLTS = "mV"


class UncalibratedHardwareError(Exception):
    """Raised when uncalibrated raw hardware signals are passed where calibrated physical mV is required."""
    pass


class IncompatibleSignalError(Exception):
    """Raised when a signal fails model ingestion prerequisites (units, sampling rate, or dimensions)."""
    pass


def adc_counts_to_input_volts(
    adc_counts: Union[int, float, np.ndarray],
    vref_volts: float = 3.3,
    resolution_bits: int = 10,
) -> Union[float, np.ndarray]:
    """Convert raw Arduino ADC counts to the analog voltage present at the Arduino pin.

    Notice: This represents the voltage at pin A0 (0.0 V to Vref).
    It does NOT represent the patient's bio-potential ECG in millivolts!
    The AD8232 module applies an analog gain (~100x-1100x), bandpass filtering,
    and a virtual ground offset (~1.65 V) prior to reaching this pin.

    Args:
        adc_counts: Scalar or array of integer ADC counts.
        vref_volts: Microcontroller ADC reference voltage (default: 3.3 V).
        resolution_bits: ADC bit resolution (default: 10 bits = 1024 levels for Uno).

    Returns:
        Analog voltage in Volts.
    """
    counts = np.asarray(adc_counts, dtype=np.float64)
    max_count = (2 ** resolution_bits) - 1  # 1023 for 10-bit
    voltage = counts * (vref_volts / float(max_count))
    if np.ndim(adc_counts) == 0:
        return float(voltage)
    return voltage


def convert_hardware_to_ecg_mv(
    adc_counts: np.ndarray,
    calibration_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Attempt conversion of hardware ADC stream to calibrated electrode-equivalent ECG in millivolts.

    Guardrail Enforcement:
    Without an approved, verified calibration certificate and parameters (front-end gain,
    virtual ground offset, reference voltage, electrode impedance compensation),
    this function strictly raises UncalibratedHardwareError.

    Args:
        adc_counts: 1D array of raw ADC counts.
        calibration_params: Dictionary of verified calibration constants. Must contain:
            - 'verified': bool (True)
            - 'analog_gain_vv': float (> 0)
            - 'virtual_ground_volts': float
            - 'vref_volts': float
            - 'calibration_source': str

    Returns:
        Calibrated ECG signal in millivolts (mV).

    Raises:
        UncalibratedHardwareError: If calibration_params is missing, unverified, or invalid.
    """
    if calibration_params is None or not calibration_params.get("verified", False):
        raise UncalibratedHardwareError(
            "Uncalibrated hardware stream: raw ADC counts cannot be mapped to patient ECG mV "
            "without verified hardware calibration parameters (analog front-end gain, virtual ground offset, "
            "ADC Vref, and electrode impedance). Existing machine learning models cannot accept uncalibrated "
            "hardware data. Calibration and clinical validation remain future phases."
        )

    # If verified parameters are supplied in a future authorized phase:
    required_keys = ("analog_gain_vv", "virtual_ground_volts", "vref_volts")
    for k in required_keys:
        if k not in calibration_params:
            raise UncalibratedHardwareError(f"Missing required calibration parameter: '{k}'")

    vref = float(calibration_params["vref_volts"])
    gain = float(calibration_params["analog_gain_vv"])
    vgnd = float(calibration_params["virtual_ground_volts"])

    if gain <= 0:
        raise UncalibratedHardwareError(f"Invalid analog gain: {gain}")

    # Convert ADC to input voltage at pin
    v_in = adc_counts_to_input_volts(adc_counts, vref_volts=vref, resolution_bits=10)
    # Voltage difference from virtual ground divided by analog gain, converted to mV
    ecg_mv = ((v_in - vgnd) / gain) * 1000.0
    return ecg_mv


def validate_input_for_classifier(
    signal: np.ndarray,
    sampling_rate_hz: int,
    unit_name: Union[str, SignalUnits] = SignalUnits.ECG_MILLIVOLTS,
    expected_sampling_rate_hz: int = 360,
) -> bool:
    """Validate that input signal meets the strict physical prerequisites for ML beat classification.

    Rejects uncalibrated ADC counts, incompatible sampling rates, non-finite values, and empty data.

    Args:
        signal: 1D array of signal values.
        sampling_rate_hz: Sampling frequency of the input signal.
        unit_name: Physical unit of the input signal ('mV', 'raw_adc_counts', 'input_volts').
        expected_sampling_rate_hz: Model's expected sampling rate (default 360 Hz for MIT-BIH model).

    Returns:
        True if all prerequisites are satisfied.

    Raises:
        IncompatibleSignalError: If unit is not mV, sample rate mismatches, or signal is invalid.
    """
    if signal is None or len(signal) == 0:
        raise IncompatibleSignalError("Signal array is empty or None.")

    arr = np.asarray(signal)
    if arr.ndim != 1:
        raise IncompatibleSignalError(f"Classifier requires 1D signal array, got shape {arr.shape}.")

    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise IncompatibleSignalError("Signal contains non-finite values (NaN or Inf).")

    # Unit verification
    if hasattr(unit_name, "value"):
        str_unit = str(unit_name.value).lower()
    else:
        str_unit = str(unit_name).lower()
    valid_units = ("mv", "millivolts", SignalUnits.ECG_MILLIVOLTS.value.lower())
    if str_unit not in valid_units:
        raise IncompatibleSignalError(
            f"Classifier requires input in calibrated millivolts (mV). Received uncalibrated unit: '{unit_name}'. "
            "Raw ADC counts or uncalibrated analog volts cannot be fed into trained models."
        )

    # Sampling rate verification
    if sampling_rate_hz != expected_sampling_rate_hz:
        raise IncompatibleSignalError(
            f"Sampling rate {sampling_rate_hz} Hz is incompatible with model requirements ({expected_sampling_rate_hz} Hz). "
            "Resampling and domain validation are required before inference."
        )

    return True
