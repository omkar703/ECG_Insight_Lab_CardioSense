"""Static specification tests for the Arduino Uno firmware sketch."""

from pathlib import Path
import pytest


class TestFirmwareSourceSpecification:
    """Validate firmware code structure, pin naming, timing, and safety notices."""

    @pytest.fixture
    def firmware_code(self) -> str:
        sketch_path = Path("hardware/firmware/ad8232_arduino_uno/ad8232_arduino_uno.ino")
        assert sketch_path.is_file(), f"Missing firmware sketch: {sketch_path}"
        return sketch_path.read_text(encoding="utf-8")

    def test_firmware_pin_constants(self, firmware_code: str):
        assert "PIN_ECG_OUTPUT = A0;" in firmware_code
        assert "PIN_LO_PLUS    = 10;" in firmware_code or "PIN_LO_PLUS = 10;" in firmware_code
        assert "PIN_LO_MINUS   = 11;" in firmware_code or "PIN_LO_MINUS = 11;" in firmware_code

    def test_firmware_timing_and_baud_constants(self, firmware_code: str):
        assert "SAMPLE_INTERVAL_MICROS = 2778UL;" in firmware_code
        assert "NOMINAL_SAMPLING_RATE_HZ = 360UL;" in firmware_code
        assert "BAUD_RATE = 115200UL;" in firmware_code

    def test_no_dynamic_string_allocation_in_loop(self, firmware_code: str):
        # Locate loop() section
        loop_start = firmware_code.find("void loop()")
        assert loop_start != -1
        loop_body = firmware_code[loop_start:]
        assert "String(" not in loop_body
        assert "String " not in loop_body

    def test_firmware_electrical_safety_header(self, firmware_code: str):
        assert "ELECTRICAL SAFETY NOTICE" in firmware_code
        assert "NEVER CONNECT ELECTRODES TO HUMAN SUBJECTS" in firmware_code
        assert "IEC 60601-1" in firmware_code
