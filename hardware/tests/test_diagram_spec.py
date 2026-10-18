"""Tests for hardware wiring diagrams, pin maps, and safety documentation."""

from pathlib import Path
import xml.etree.ElementTree as ET
import pytest


class TestDiagramAndPinoutSpec:
    """Validate SVG diagram validity, pin naming completeness, and safety banners."""

    def test_svg_diagram_exists_and_is_valid_xml(self):
        svg_path = Path("hardware/diagrams/ad8232_arduino_uno_wiring.svg")
        assert svg_path.is_file(), f"Missing wiring diagram SVG: {svg_path}"

        content = svg_path.read_text(encoding="utf-8")
        # Must parse cleanly as XML
        root = ET.fromstring(content)
        assert "svg" in root.tag.lower()

        # Check required pin names are labelled
        assert "3.3V" in content
        assert "GND" in content
        assert "A0" in content
        assert "D10" in content
        assert "D11" in content

        # Check mandatory safety warnings in SVG
        assert "CAUTION: PROTOTYPE REFERENCE ONLY" in content
        assert "NO HUMAN ELECTRODE TEST" in content
        assert "MAINS/USB ISOLATION REQUIRED" in content

    def test_pin_map_markdown_spec(self):
        map_path = Path("hardware/diagrams/pin_map.md")
        assert map_path.is_file()
        content = map_path.read_text(encoding="utf-8")
        assert "AD8232 Pin Label" in content
        assert "Arduino Uno R3 Header" in content
        assert "A0" in content
        assert "D10" in content
        assert "D11" in content
        assert "SIMULATION / SPECIFICATION ONLY" in content

    def test_pinout_and_safety_document(self):
        safety_path = Path("hardware/PINOUT_AND_SAFETY.md")
        assert safety_path.is_file()
        content = safety_path.read_text(encoding="utf-8")
        assert "DO NOT CONNECT ELECTRODES TO HUMAN SUBJECTS" in content
        assert "IEC 60601-1" in content
        assert "Lack of Galvanic Isolation" in content
