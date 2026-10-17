"""Data models and type definitions for ECG Insight Lab."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class ValidationStatus(str, Enum):
    """Enumeration of possible validation states."""
    VALID = "VALID"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    INVALID_EXTENSION = "INVALID_EXTENSION"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    INVALID_DIMENSIONS = "INVALID_DIMENSIONS"
    EXCEEDS_SIZE_LIMIT = "EXCEEDS_SIZE_LIMIT"
    CORRUPTED_FILE = "CORRUPTED_FILE"
    NON_IMAGE_DATA = "NON_IMAGE_DATA"


@dataclass
class ImageMetadata:
    """Detailed structural metadata for an evaluated ECG image."""
    filename: str
    format: str
    width: int
    height: int
    mode: str
    channels: int
    size_bytes: int

    @property
    def aspect_ratio(self) -> float:
        """Calculate width-to-height ratio."""
        return round(self.width / max(1, self.height), 3)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation."""
        return {
            "filename": self.filename,
            "format": self.format,
            "dimensions": f"{self.width}x{self.height}",
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "channels": self.channels,
            "size_bytes": self.size_bytes,
            "size_kb": round(self.size_bytes / 1024, 2),
            "aspect_ratio": self.aspect_ratio
        }


@dataclass
class ValidationResult:
    """Complete diagnostic outcome of image input validation."""
    is_valid: bool
    status: ValidationStatus
    message: str
    metadata: Optional[ImageMetadata] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to structured dictionary."""
        return {
            "is_valid": self.is_valid,
            "status": self.status.value,
            "message": self.message,
            "warnings": self.warnings,
            "metadata": self.metadata.to_dict() if self.metadata else None
        }
