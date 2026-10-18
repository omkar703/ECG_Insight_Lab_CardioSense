"""Tests for UART line-oriented serial protocol and stateful stream parser."""

import pytest
import pandas as pd

from hardware.host.serial_protocol import (
    ADC_MAX_COUNT,
    ADC_MIN_COUNT,
    MICROS_ROLLOVER_THRESHOLD,
    NOMINAL_SAMPLING_RATE_HZ,
    SerialSample,
    SerialStreamParser,
    format_serial_line,
    parse_serial_line,
)


class TestSerialLineParsing:
    """Test line-level parsing of individual serial stream rows."""

    def test_parse_valid_nominal_line(self):
        line = "42, 116676, 512, 0, 0\n"
        sample = parse_serial_line(line)
        assert sample is not None
        assert sample.seq == 42
        assert sample.timestamp_micros == 116676
        assert sample.raw_adc == 512
        assert sample.lo_plus == 0
        assert sample.lo_minus == 0
        assert sample.lead_off is False

    def test_parse_lead_off_conditions(self):
        # LO+ asserted
        s1 = parse_serial_line("100, 200000, 1023, 1, 0")
        assert s1 is not None and s1.lead_off is True and s1.lo_plus == 1 and s1.lo_minus == 0

        # LO- asserted
        s2 = parse_serial_line("101, 202778, 1023, 0, 1")
        assert s2 is not None and s2.lead_off is True and s2.lo_plus == 0 and s2.lo_minus == 1

        # Both asserted
        s3 = parse_serial_line("102, 205556, 1023, 1, 1")
        assert s3 is not None and s3.lead_off is True and s3.lo_plus == 1 and s3.lo_minus == 1

    def test_skip_header_and_empty_lines(self):
        assert parse_serial_line("") is None
        assert parse_serial_line("   \n") is None
        assert parse_serial_line("# CardioSense AD8232 Arduino Uno v1.0") is None
        assert parse_serial_line("# baud=115200, nominal_fs=360") is None

    def test_malformed_token_count_raises_error(self):
        with pytest.raises(ValueError, match="Invalid field count"):
            parse_serial_line("0, 2778, 512, 0")  # Only 4 tokens

        with pytest.raises(ValueError, match="Invalid field count"):
            parse_serial_line("0, 2778, 512, 0, 0, 99")  # 6 tokens

    def test_malformed_non_integer_tokens_raises_error(self):
        with pytest.raises(ValueError, match="Failed to parse integer"):
            parse_serial_line("0, 2778, NOT_A_NUMBER, 0, 0")

    def test_adc_out_of_bounds_raises_error(self):
        with pytest.raises(ValueError, match="out of valid 10-bit range"):
            parse_serial_line("0, 2778, -1, 0, 0")

        with pytest.raises(ValueError, match="out of valid 10-bit range"):
            parse_serial_line("0, 2778, 1024, 0, 0")

    def test_invalid_lead_off_values_raises_error(self):
        with pytest.raises(ValueError, match="Lead-off flags must be 0 or 1"):
            parse_serial_line("0, 2778, 512, 2, 0")

    def test_format_serial_line_roundtrip(self):
        orig = SerialSample(seq=15, timestamp_micros=41670, raw_adc=505, lo_plus=0, lo_minus=0, lead_off=False)
        formatted = format_serial_line(orig)
        parsed = parse_serial_line(formatted)
        assert parsed is not None
        assert parsed.seq == orig.seq
        assert parsed.timestamp_micros == orig.timestamp_micros
        assert parsed.raw_adc == orig.raw_adc
        assert parsed.lead_off == orig.lead_off


class TestSerialStreamParser:
    """Test stateful stream processing, sequence continuity, and timestamp unwrapping."""

    def test_continuous_stream_parsing(self):
        parser = SerialStreamParser(nominal_sampling_rate_hz=360)
        lines = [
            "# Header comment 1\n",
            "# Header comment 2\n",
            "0, 100000, 512, 0, 0\n",
            "1, 102778, 520, 0, 0\n",
            "2, 105556, 515, 0, 0\n",
        ]
        for l in lines:
            parser.feed_line(l)

        assert parser.total_lines_read == 5
        assert parser.header_lines_count == 2
        assert parser.sample_count == 3
        assert parser.dropped_samples_count == 0
        assert parser.corrupted_lines_count == 0

        # DataFrame verification
        df = parser.to_dataframe()
        assert len(df) == 3
        assert "raw_adc" in df.columns
        assert "mV" not in df.columns  # Critical guardrail
        assert list(df["raw_adc"]) == [512, 520, 515]

    def test_sequence_gap_detection(self):
        parser = SerialStreamParser()
        parser.feed_line("0, 100000, 500, 0, 0")
        parser.feed_line("1, 102778, 501, 0, 0")
        # Drop packets 2, 3, 4
        parser.feed_line("5, 113890, 502, 0, 0")

        assert parser.sample_count == 3
        assert parser.dropped_samples_count == 3

    def test_micros_timer_rollover_handling(self):
        parser = SerialStreamParser()
        # Feed near 32-bit wrap boundary (2^32 - 1 = 4,294,967,295 us)
        near_wrap = MICROS_ROLLOVER_THRESHOLD - 5000
        after_wrap = 2000  # Wrapped around to small positive micros

        parser.feed_line(f"0, {near_wrap}, 500, 0, 0")
        sample2 = parser.feed_line(f"1, {after_wrap}, 505, 0, 0")

        assert sample2 is not None
        # Monotonically increasing continuous time despite raw timer wrap
        assert sample2.time_seconds > parser.samples[0].time_seconds

    def test_clipping_and_lead_off_statistics(self):
        parser = SerialStreamParser()
        parser.feed_line("0, 100000, 0, 0, 0")      # Clip low
        parser.feed_line("1, 102778, 1023, 1, 0")   # Clip high + lead off
        parser.feed_line("2, 105556, 512, 0, 0")    # Normal

        summary = parser.get_quality_summary()
        assert summary["clipping_count"] == 2
        assert summary["lead_off_count"] == 1
        assert summary["valid_samples_count"] == 3
