"""Input validation module for ECG images in ECG Insight Lab.

Performs robust, non-destructive safety checks on ECG scans and mobile photos
prior to digitization and signal processing.
"""

import os
from io import BytesIO
from pathlib import Path
from typing import Union, List

from PIL import Image, UnidentifiedImageError

from ecg_lab.config import ValidationLimits
from ecg_lab.models import ValidationResult, ValidationStatus, ImageMetadata


def validate_image_file(file_path: Union[str, Path]) -> ValidationResult:
    """Validate an ECG image from a filesystem path.

    Args:
        file_path: Path to the image file.

    Returns:
        ValidationResult containing validation status, structured metadata,
        and informative warning/error messages.
    """
    path = Path(file_path)

    # 1. Existence Check
    if not path.is_file():
        return ValidationResult(
            is_valid=False,
            status=ValidationStatus.FILE_NOT_FOUND,
            message=f"File not found or inaccessible: '{path}'."
        )

    # 2. Extension Check
    ext = path.suffix.lower().lstrip(".")
    if ext not in ValidationLimits.ALLOWED_EXTENSIONS:
        return ValidationResult(
            is_valid=False,
            status=ValidationStatus.INVALID_EXTENSION,
            message=(
                f"Unsupported file extension '.{ext}'. Allowed formats: "
                f"{', '.join(ValidationLimits.ALLOWED_EXTENSIONS)}."
            )
        )

    # 3. File Size Check
    try:
        size_bytes = path.stat().st_size
    except OSError as err:
        return ValidationResult(
            is_valid=False,
            status=ValidationStatus.FILE_NOT_FOUND,
            message=f"Could not read file metadata: {err}"
        )

    if size_bytes == 0:
        return ValidationResult(
            is_valid=False,
            status=ValidationStatus.CORRUPTED_FILE,
            message="Image file is completely empty (0 bytes)."
        )

    if size_bytes > ValidationLimits.MAX_FILE_SIZE_BYTES:
        max_mb = ValidationLimits.MAX_FILE_SIZE_BYTES // (1024 * 1024)
        curr_mb = round(size_bytes / (1024 * 1024), 2)
        return ValidationResult(
            is_valid=False,
            status=ValidationStatus.EXCEEDS_SIZE_LIMIT,
            message=f"File size ({curr_mb} MB) exceeds maximum allowed limit ({max_mb} MB)."
        )

    # 4. Binary Read & Pillow Verification
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception as err:
        return ValidationResult(
            is_valid=False,
            status=ValidationStatus.CORRUPTED_FILE,
            message=f"Failed to read file bytes: {err}"
        )

    return validate_image_bytes(data, filename=path.name)


def validate_image_bytes(image_data: bytes, filename: str = "image.jpg") -> ValidationResult:
    """Validate in-memory image bytes (e.g. from web upload).

    Args:
        image_data: Raw bytes of the image file.
        filename: Optional filename for logging and extension checks.

    Returns:
        ValidationResult with validation diagnostics.
    """
    size_bytes = len(image_data)

    if size_bytes == 0:
        return ValidationResult(
            is_valid=False,
            status=ValidationStatus.CORRUPTED_FILE,
            message="Provided image byte buffer is empty (0 bytes)."
        )

    if size_bytes > ValidationLimits.MAX_FILE_SIZE_BYTES:
        max_mb = ValidationLimits.MAX_FILE_SIZE_BYTES // (1024 * 1024)
        curr_mb = round(size_bytes / (1024 * 1024), 2)
        return ValidationResult(
            is_valid=False,
            status=ValidationStatus.EXCEEDS_SIZE_LIMIT,
            message=f"File size ({curr_mb} MB) exceeds maximum limit ({max_mb} MB)."
        )

    # Integrity verification
    try:
        buffer = BytesIO(image_data)
        with Image.open(buffer) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as err:
        return ValidationResult(
            is_valid=False,
            status=ValidationStatus.CORRUPTED_FILE,
            message=f"Corrupt or invalid image format: {err}"
        )

    # Re-open to read dimensions and mode (verify closes stream in Pillow)
    buffer.seek(0)
    try:
        with Image.open(buffer) as img:
            width, height = img.size
            img_format = img.format or "UNKNOWN"
            mode = img.mode
    except Exception as err:
        return ValidationResult(
            is_valid=False,
            status=ValidationStatus.CORRUPTED_FILE,
            message=f"Error inspecting image headers: {err}"
        )

    # Channel computation
    mode_channels = {
        "1": 1, "L": 1, "P": 1,
        "RGB": 3, "YCbCr": 3, "LAB": 3, "HSV": 3,
        "RGBA": 4, "CMYK": 4
    }
    channels = mode_channels.get(mode, len(mode))

    # Dimension Checks
    if width < ValidationLimits.MIN_IMAGE_WIDTH or height < ValidationLimits.MIN_IMAGE_HEIGHT:
        return ValidationResult(
            is_valid=False,
            status=ValidationStatus.INVALID_DIMENSIONS,
            message=(
                f"Image dimensions ({width}x{height}) are too small. "
                f"Minimum required resolution is {ValidationLimits.MIN_IMAGE_WIDTH}x"
                f"{ValidationLimits.MIN_IMAGE_HEIGHT} for reliable ECG lead extraction."
            )
        )

    # Construct Metadata
    metadata = ImageMetadata(
        filename=filename,
        format=img_format,
        width=width,
        height=height,
        mode=mode,
        channels=channels,
        size_bytes=size_bytes
    )

    # Generate Helpful Warnings (without failing validation)
    warnings: List[str] = []
    aspect = metadata.aspect_ratio
    canonical_aspect = round(
        ValidationLimits.CANONICAL_TEMPLATE_WIDTH / ValidationLimits.CANONICAL_TEMPLATE_HEIGHT, 3
    )

    if aspect < 1.0:
        warnings.append("Image is in portrait orientation. Standard clinical 12-lead ECGs are typically landscape.")
    elif abs(aspect - canonical_aspect) > 0.4:
        warnings.append(
            f"Image aspect ratio ({aspect}) deviates from standard 12-lead ECG layout (~{canonical_aspect})."
        )

    if width < 1200 or height < 800:
        warnings.append("Image resolution is below 1200x800; fine ECG grid lines and waveforms may lack precision.")

    return ValidationResult(
        is_valid=True,
        status=ValidationStatus.VALID,
        message="Image passed all structural and integrity checks.",
        metadata=metadata,
        warnings=warnings
    )
