"""Data structures and types for the ECG digitization subsystem."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import json
import pandas as pd


class DigitizationStatus(str, Enum):
    """Execution status for the digitization pipeline."""
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"


class CalibrationStatus(str, Enum):
    """Calibration certainty state for extracted signals."""
    UNRECOVERED = "UNRECOVERED"  # Amplitudes are uncalibrated relative units
    ESTIMATED = "ESTIMATED"      # Approximate grid-based calibration
    CALIBRATED = "CALIBRATED"    # Rigorously verified voltage and time calibration


class AmplitudeUnit(str, Enum):
    """Units of measurement for extracted waveform amplitudes."""
    RELATIVE = "relative_units"      # Uncalibrated normalized amplitude
    PIXEL = "pixels"                 # Raw pixel vertical displacement
    MILLIVOLTS = "mV"                # Physical voltage in millivolts
    MICROVOLTS = "uV"                # Physical voltage in microvolts


@dataclass
class LeadSignal:
    """Represents a digitized 1D continuous waveform for a single ECG lead."""
    lead_name: str
    samples: List[float]
    sample_indices: List[int]
    confidence: float = 1.0
    is_valid: bool = True
    time_seconds: Optional[List[float]] = None
    quality_flags: List[str] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lead_name": self.lead_name,
            "sample_count": self.sample_count,
            "confidence": round(self.confidence, 3),
            "is_valid": self.is_valid,
            "quality_flags": self.quality_flags,
            "samples": [round(float(s), 4) for s in self.samples],
            "sample_indices": self.sample_indices,
            "time_seconds": [round(float(t), 4) for t in self.time_seconds] if self.time_seconds else None,
        }


@dataclass
class DigitizationResult:
    """Complete structured output from an ECG digitization run."""
    status: DigitizationStatus
    method: str
    method_version: str
    calibration_status: CalibrationStatus
    amplitude_unit: AmplitudeUnit
    leads: Dict[str, LeadSignal] = field(default_factory=dict)
    detected_leads: List[str] = field(default_factory=list)
    missing_leads: List[str] = field(default_factory=list)
    sampling_rate_hz: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    runtime_ms: float = 0.0

    @property
    def lead_count(self) -> int:
        return len(self.detected_leads)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "status": self.status.value,
            "method": self.method,
            "method_version": self.method_version,
            "calibration_status": self.calibration_status.value,
            "amplitude_unit": self.amplitude_unit.value,
            "sampling_rate_hz": self.sampling_rate_hz,
            "lead_count": self.lead_count,
            "detected_leads": self.detected_leads,
            "missing_leads": self.missing_leads,
            "runtime_ms": round(self.runtime_ms, 2),
            "warnings": self.warnings,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "leads": {name: lead.to_dict() for name, lead in self.leads.items()},
        }

    def to_long_dataframe(self) -> pd.DataFrame:
        """Export all detected lead signals into a tidy long-form pandas DataFrame."""
        rows = []
        for lead_name, lead_sig in self.leads.items():
            if not lead_sig.is_valid:
                continue
            for idx, amp in enumerate(lead_sig.samples):
                time_val = lead_sig.time_seconds[idx] if lead_sig.time_seconds and idx < len(lead_sig.time_seconds) else None
                rows.append({
                    "lead": lead_name,
                    "sample_index": lead_sig.sample_indices[idx] if idx < len(lead_sig.sample_indices) else idx,
                    "time_seconds": time_val,
                    "amplitude": round(amp, 5),
                    "amplitude_unit": self.amplitude_unit.value,
                    "calibration_status": self.calibration_status.value,
                    "method": self.method,
                })
        return pd.DataFrame(rows)

    def to_wide_dataframe(self) -> pd.DataFrame:
        """Export lead waveforms into wide columnar DataFrame (samples x leads)."""
        if not self.leads:
            return pd.DataFrame()
        
        # Determine maximum sample length across valid leads
        valid_leads = {k: v for k, v in self.leads.items() if v.is_valid}
        if not valid_leads:
            return pd.DataFrame()

        data: Dict[str, Any] = {}
        # Use first lead's indices as reference index
        first_lead = next(iter(valid_leads.values()))
        data["sample_index"] = first_lead.sample_indices
        if first_lead.time_seconds:
            data["time_seconds"] = first_lead.time_seconds

        for name, lead_sig in valid_leads.items():
            data[name] = lead_sig.samples

        return pd.DataFrame(data)

    def to_json(self, indent: int = 2) -> str:
        """Serialize result and metadata to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
