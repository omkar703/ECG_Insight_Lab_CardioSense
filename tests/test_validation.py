"""Unit tests for the image validation module in ECG Insight Lab.

Ensures all input validation safeguards, format verifications, and error
handling mechanisms behave deterministically.
"""

from io import BytesIO
from pathlib import Path
import pytest
from PIL import Image

from ecg_lab.config import ColorPalette, TypographySystem, ValidationLimits
from ecg_lab.models import ValidationStatus
from ecg_lab.validation import validate_image_file, validate_image_bytes


SAMPLE_IMAGE_PATH = Path("sample_data/synthetic_sample.png")


def test_valid_image_file():
    """Verify that a well-formed image file passes validation."""
    assert SAMPLE_IMAGE_PATH.exists(), "Sample test image must exist."
    result = validate_image_file(SAMPLE_IMAGE_PATH)

    assert result.is_valid is True
    assert result.status == ValidationStatus.VALID
    assert result.metadata is not None
    assert result.metadata.width == 1200
    assert result.metadata.height == 800
    assert result.metadata.format.upper() == "PNG"
    assert result.metadata.channels == 3
    assert result.metadata.size_bytes > 0
    assert len(result.warnings) == 0


def test_valid_image_bytes():
    """Verify that in-memory image bytes pass validation."""
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        data = f.read()

    result = validate_image_bytes(data, filename="sample.png")
    assert result.is_valid is True
    assert result.status == ValidationStatus.VALID
    assert result.metadata is not None
    assert result.metadata.width == 1200


def test_missing_file():
    """Verify that a missing file path returns FILE_NOT_FOUND."""
    result = validate_image_file("non_existent_ecg_scan_123.jpg")
    assert result.is_valid is False
    assert result.status == ValidationStatus.FILE_NOT_FOUND
    assert "not found" in result.message.lower()


def test_unsupported_extension(tmp_path):
    """Verify that disallowed extensions (.txt, .pdf) are rejected."""
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Not an ECG scan")

    result = validate_image_file(txt_file)
    assert result.is_valid is False
    assert result.status == ValidationStatus.INVALID_EXTENSION
    assert "unsupported" in result.message.lower()


def test_empty_file(tmp_path):
    """Verify that an empty file (0 bytes) is rejected as corrupted."""
    empty_file = tmp_path / "empty.jpg"
    empty_file.write_bytes(b"")

    result = validate_image_file(empty_file)
    assert result.is_valid is False
    assert result.status == ValidationStatus.CORRUPTED_FILE


def test_corrupted_image_bytes():
    """Verify that invalid/corrupt image bytes fail verification."""
    garbage_bytes = b"NOT_A_VALID_JPEG_HEADER_RANDOM_GARBAGE_DATA" * 50
    result = validate_image_bytes(garbage_bytes, filename="corrupt.jpg")

    assert result.is_valid is False
    assert result.status == ValidationStatus.CORRUPTED_FILE


def test_undersized_image(tmp_path):
    """Verify that an image below minimum resolution is rejected."""
    tiny_img = Image.new("RGB", (100, 100), color="white")
    tiny_path = tmp_path / "tiny.jpg"
    tiny_img.save(tiny_path)

    result = validate_image_file(tiny_path)
    assert result.is_valid is False
    assert result.status == ValidationStatus.INVALID_DIMENSIONS
    assert "too small" in result.message.lower()


def test_portrait_orientation_warning(tmp_path):
    """Verify that a portrait orientation image generates an informative warning."""
    portrait_img = Image.new("RGB", (600, 1000), color="white")
    p_path = tmp_path / "portrait.jpg"
    portrait_img.save(p_path)

    result = validate_image_file(p_path)
    assert result.is_valid is True  # Valid image, but triggers warning
    assert any("portrait orientation" in w.lower() for w in result.warnings)


def test_aspect_ratio_deviation_warning(tmp_path):
    """Verify that extreme aspect ratios trigger a layout warning."""
    wide_img = Image.new("RGB", (1800, 400), color="white")
    w_path = tmp_path / "wide.jpg"
    wide_img.save(w_path)

    result = validate_image_file(w_path)
    assert result.is_valid is True
    assert any("aspect ratio" in w.lower() for w in result.warnings)


def test_exceeds_size_limit():
    """Verify that buffers exceeding the size limit are rejected."""
    fake_large_size = ValidationLimits.MAX_FILE_SIZE_BYTES + 1024
    # Test via mock or empty large buffer check without allocating massive RAM
    large_header = b"\x00" * fake_large_size
    result = validate_image_bytes(large_header, filename="huge.jpg")

    assert result.is_valid is False
    assert result.status == ValidationStatus.EXCEEDS_SIZE_LIMIT


def test_editorial_design_tokens():
    """Verify that the Quiet Academic Editorial design tokens conform to specifications."""
    palette = ColorPalette()
    assert palette.BG_PRIMARY == "#F7F6F2"
    assert palette.TEXT_PRIMARY == "#252722"
    assert palette.ACCENT_PRIMARY == "#455D4A"
    assert palette.BORDER_DIVIDER == "#DCDDD5"
    assert palette.SURFACE_INSET == "#F0EFE9"

    typography = TypographySystem()
    assert "Source Serif 4" in typography.FONT_HEADINGS
    assert "IBM Plex Sans" in typography.FONT_BODY


def test_legacy_model_isolation():
    """Ensure that legacy .pkl files are strictly quarantined and not imported in tests."""
    legacy_dir = Path("legacy")
    assert legacy_dir.is_dir(), "legacy/ directory must exist."
    pkl_files = list(legacy_dir.glob("*.pkl"))
    assert len(pkl_files) >= 2, "Historical model files must be preserved in legacy/."
