"""Configuration module for ECG Insight Lab.

Contains design tokens for the Quiet Academic Editorial visual identity,
system constants, image validation constraints, and research disclaimers.
"""

from dataclasses import dataclass
from typing import Tuple, List

# ==============================================================================
# Editorial Design System: "Quiet Academic Editorial"
# ==============================================================================

@dataclass(frozen=True)
class ColorPalette:
    """Restrained, academic editorial color tokens."""
    BG_PRIMARY: str = "#F7F6F2"        # Warm ivory background
    TEXT_PRIMARY: str = "#252722"      # Deep charcoal body & heading text
    TEXT_SECONDARY: str = "#6F716B"    # Muted neutral gray for captions/meta
    ACCENT_PRIMARY: str = "#455D4A"    # Subdued forest green (primary accent)
    BORDER_DIVIDER: str = "#DCDDD5"    # Soft neutral gray for lines & dividers
    SURFACE_INSET: str = "#F0EFE9"     # Slightly darker warm neutral for cards/panels
    ERROR_ALERT: str = "#9E3D34"       # Subdued rust red for validation errors
    SUCCESS_ALERT: str = "#3D6E52"     # Deep moss green for successful states


@dataclass(frozen=True)
class TypographySystem:
    """Typographic hierarchy tokens."""
    FONT_HEADINGS: str = "Source Serif 4, Newsreader, Georgia, serif"
    FONT_BODY: str = "IBM Plex Sans, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"
    FONT_MONO: str = "IBM Plex Mono, SFMono-Regular, Menlo, monospace"
    LINE_HEIGHT_BODY: float = 1.65
    LINE_HEIGHT_HEADINGS: float = 1.25


# ==============================================================================
# Image Validation & Signal Processing Constraints
# ==============================================================================

@dataclass(frozen=True)
class ValidationLimits:
    """Safeguards for image ingestion and lead extraction."""
    ALLOWED_EXTENSIONS: Tuple[str, ...] = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")
    ALLOWED_MIME_TYPES: Tuple[str, ...] = ("image/jpeg", "image/png", "image/bmp", "image/tiff")
    MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB max upload limit
    MIN_IMAGE_WIDTH: int = 400
    MIN_IMAGE_HEIGHT: int = 300
    CANONICAL_TEMPLATE_WIDTH: int = 2213
    CANONICAL_TEMPLATE_HEIGHT: int = 1572
    EXPECTED_LEAD_COUNT: int = 12
    EXPECTED_POINTS_PER_LEAD: int = 255
    TOTAL_FEATURE_DIMENSION: int = 3060  # 12 leads * 255 timepoints
    PCA_COMPONENT_COUNT: int = 400       # Target components from PCA dimensionality reduction


# ==============================================================================
# Academic Scope & Integrity Disclaimers
# ==============================================================================

DISCLAIMER_RESEARCH_ONLY = (
    "Academic research prototype for educational and exploratory signal analysis only. "
    "Not evaluated or approved by the FDA/CE for medical diagnosis, triage, or clinical care."
)

DISCLAIMER_HARDWARE_ABSENT = (
    "Software-only environment. No physical ECG sensors (AD8232/ESP32) are connected. "
    "All displayed signals are loaded from public benchmarks or user-uploaded scans."
)
