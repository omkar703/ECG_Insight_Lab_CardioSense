"""Line-oriented serial communication protocol definitions and robust stream parser.

Target: AD8232 + Arduino Uno R3 prototype stream
Protocol: Line-oriented CSV over UART (115200 baud, 360 Hz nominal)
Field order: seq,micros,adc,lo_plus,lo_minus
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Protocol Constants
PROTOCOL_VERSION: str = "1.0.0"
NOMINAL_SAMPLING_RATE_HZ: int = 360
NOMINAL_INTERVAL_MICROS: int = 2778
BAUD_RATE: int = 115200

# 10-bit ADC Boundaries (Arduino Uno ATmega328P)
ADC_MIN_COUNT: int = 0
ADC_MAX_COUNT: int = 1023

# Timer Rollover: 32-bit unsigned micros() wraps at 2^32 microseconds (~71.58 minutes)
MICROS_ROLLOVER_THRESHOLD: int = 4_294_967_296


@dataclass
class SerialSample:
    """Represents a single parsed sample point from the hardware serial stream."""
    seq: int
    timestamp_micros: int
    raw_adc: int
    lo_plus: int
    lo_minus: int
    lead_off: bool
    time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "seq": self.seq,
            "timestamp_micros": self.timestamp_micros,
            "time_seconds": round(self.time_seconds, 6),
            "raw_adc": self.raw_adc,
            "lo_plus": self.lo_plus,
            "lo_minus": self.lo_minus,
            "lead_off": self.lead_off,
        }


def parse_serial_line(line: str) -> Optional[SerialSample]:
    """Parse a single line from the serial stream into a SerialSample.

    Args:
        line: Raw text line received from UART or exported CSV file.

    Returns:
        SerialSample if line is valid data, or None if line is a header/comment/empty.

    Raises:
        ValueError: If line is malformed (wrong field count, non-integers, out of ADC bounds).
    """
    clean_line = line.strip()
    if not clean_line or clean_line.startswith("#") or clean_line.startswith("seq,"):
        return None

    tokens = clean_line.split(",")
    if len(tokens) == 5:
        # Standard line-oriented UART protocol: seq,micros,adc,lo_plus,lo_minus
        try:
            seq = int(tokens[0].strip())
            micros_ts = int(tokens[1].strip())
            raw_adc = int(tokens[2].strip())
            lo_plus = int(tokens[3].strip())
            lo_minus = int(tokens[4].strip())
        except ValueError as e:
            raise ValueError(f"Failed to parse integer tokens in serial line: '{clean_line}' ({e})") from e
    elif len(tokens) == 7:
        # Exported DataFrame CSV format: seq,timestamp_micros,time_seconds,raw_adc,lo_plus,lo_minus,lead_off
        try:
            seq = int(tokens[0].strip())
            micros_ts = int(tokens[1].strip())
            raw_adc = int(tokens[3].strip())
            lo_plus = int(tokens[4].strip())
            lo_minus = int(tokens[5].strip())
        except ValueError as e:
            raise ValueError(f"Failed to parse integer tokens in CSV row: '{clean_line}' ({e})") from e
    else:
        raise ValueError(
            f"Invalid field count in serial line: expected 5 tokens (seq,micros,adc,lo_plus,lo_minus), "
            f"got {len(tokens)} in line: '{clean_line}'"
        )

    # Validate ADC quantization bounds (0 to 1023 for 10-bit Uno)
    if not (ADC_MIN_COUNT <= raw_adc <= ADC_MAX_COUNT):
        raise ValueError(
            f"ADC count {raw_adc} out of valid 10-bit range [{ADC_MIN_COUNT}, {ADC_MAX_COUNT}]"
        )

    # Validate lead-off status bits (must be 0 or 1)
    if lo_plus not in (0, 1) or lo_minus not in (0, 1):
        raise ValueError(f"Lead-off flags must be 0 or 1, got lo_plus={lo_plus}, lo_minus={lo_minus}")

    lead_off = bool(lo_plus or lo_minus)

    return SerialSample(
        seq=seq,
        timestamp_micros=micros_ts,
        raw_adc=raw_adc,
        lo_plus=lo_plus,
        lo_minus=lo_minus,
        lead_off=lead_off,
        time_seconds=0.0,
    )


def format_serial_line(sample: SerialSample) -> str:
    """Format a SerialSample into the canonical line-oriented protocol string."""
    return f"{sample.seq},{sample.timestamp_micros},{sample.raw_adc},{sample.lo_plus},{sample.lo_minus}\n"


class SerialStreamParser:
    """Stateful parser for serial stream with sequence continuity and timestamp unwrap."""

    def __init__(self, nominal_sampling_rate_hz: int = NOMINAL_SAMPLING_RATE_HZ):
        self.nominal_fs = nominal_sampling_rate_hz
        self.samples: List[SerialSample] = []
        
        # State tracking
        self.total_lines_read: int = 0
        self.header_lines_count: int = 0
        self.corrupted_lines_count: int = 0
        self.dropped_samples_count: int = 0
        self.lead_off_count: int = 0
        self.clipping_count: int = 0

        self._last_seq: Optional[int] = None
        self._last_raw_micros: Optional[int] = None
        self._micros_wrap_offset: int = 0
        self._base_micros: Optional[int] = None

    def feed_line(self, line: str) -> Optional[SerialSample]:
        """Feed a raw serial line to the parser, handling tracking and statistics.

        Returns:
            SerialSample if line was parsed as valid data, or None if skipped/corrupted.
        """
        self.total_lines_read += 1
        clean = line.strip()
        if not clean or clean.startswith("#"):
            self.header_lines_count += 1
            return None

        try:
            sample = parse_serial_line(clean)
        except ValueError:
            self.corrupted_lines_count += 1
            return None

        if sample is None:
            return None

        # Sequence gap detection
        if self._last_seq is not None:
            expected_seq = self._last_seq + 1
            if sample.seq != expected_seq:
                # Calculate dropped frames accounting for 32-bit wrap
                if sample.seq > expected_seq:
                    gap = sample.seq - expected_seq
                else:
                    gap = (2**32 - expected_seq) + sample.seq
                self.dropped_samples_count += gap
        self._last_seq = sample.seq

        # Timestamp unwrap tracking (micros wraps every ~71.58 min)
        raw_micros = sample.timestamp_micros
        if self._last_raw_micros is not None:
            if raw_micros < self._last_raw_micros:
                # Timer wrapped around 2^32
                self._micros_wrap_offset += MICROS_ROLLOVER_THRESHOLD
        self._last_raw_micros = raw_micros

        unwrapped_micros = raw_micros + self._micros_wrap_offset
        if self._base_micros is None:
            self._base_micros = unwrapped_micros

        elapsed_micros = unwrapped_micros - self._base_micros
        sample.time_seconds = elapsed_micros / 1_000_000.0

        # Clipping tracking (hit 0 or 1023)
        if sample.raw_adc in (ADC_MIN_COUNT, ADC_MAX_COUNT):
            self.clipping_count += 1

        # Lead-off tracking
        if sample.lead_off:
            self.lead_off_count += 1

        self.samples.append(sample)
        return sample

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def duration_seconds(self) -> float:
        if not self.samples:
            return 0.0
        return self.samples[-1].time_seconds - self.samples[0].time_seconds

    @property
    def sequence_gaps_count(self) -> int:
        """Alias for dropped_samples_count representing detected continuity gaps."""
        return self.dropped_samples_count

    def to_dataframe(self) -> pd.DataFrame:
        """Export parsed samples to a pandas DataFrame with strict column typing.

        Notice: The ADC column is named 'raw_adc' (integer counts), NEVER 'mV'.
        """
        if not self.samples:
            return pd.DataFrame(
                columns=[
                    "seq",
                    "timestamp_micros",
                    "time_seconds",
                    "raw_adc",
                    "lo_plus",
                    "lo_minus",
                    "lead_off",
                ]
            )

        data = [s.to_dict() for s in self.samples]
        df = pd.DataFrame(data)
        df["seq"] = df["seq"].astype("int64")
        df["timestamp_micros"] = df["timestamp_micros"].astype("int64")
        df["time_seconds"] = df["time_seconds"].astype("float64")
        df["raw_adc"] = df["raw_adc"].astype("int32")
        df["lo_plus"] = df["lo_plus"].astype("int8")
        df["lo_minus"] = df["lo_minus"].astype("int8")
        df["lead_off"] = df["lead_off"].astype("bool")
        return df

    def get_quality_summary(self) -> Dict[str, object]:
        """Return diagnostic dictionary of stream transmission quality."""
        return {
            "total_lines_read": self.total_lines_read,
            "valid_samples_count": len(self.samples),
            "header_lines_count": self.header_lines_count,
            "corrupted_lines_count": self.corrupted_lines_count,
            "dropped_samples_count": self.dropped_samples_count,
            "lead_off_count": self.lead_off_count,
            "clipping_count": self.clipping_count,
            "duration_seconds": round(self.duration_seconds, 3),
            "packet_loss_rate": (
                self.dropped_samples_count / (len(self.samples) + self.dropped_samples_count)
                if (len(self.samples) + self.dropped_samples_count) > 0
                else 0.0
            ),
        }
