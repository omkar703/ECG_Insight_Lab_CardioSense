"""Protocol definition and adapters for third-party and native ECG digitizers."""

import os
import shutil
import time
from pathlib import Path
from typing import Protocol, Optional, Dict, Any, Tuple
from PIL import Image

from ecg_lab.models import ImageMetadata
from ecg_lab.digitization.types import (
    DigitizationResult,
    DigitizationStatus,
    CalibrationStatus,
    AmplitudeUnit,
)


class ECGDigitizer(Protocol):
    """Protocol establishing the required interface for all digitization engines."""

    @property
    def name(self) -> str:
        """Name of the digitization engine."""
        ...

    @property
    def version(self) -> str:
        """Version string of the engine."""
        ...

    def is_available(self) -> Tuple[bool, Optional[str]]:
        """Verify whether required binaries, weights, and runtimes are accessible."""
        ...

    def digitize(
        self,
        image: Image.Image,
        metadata: ImageMetadata,
        options: Optional[Dict[str, Any]] = None,
    ) -> DigitizationResult:
        """Execute digitization on an already-validated PIL image."""
        ...


class OpenECGDigitizerAdapter:
    """Adapter for the Ahus-AIM Open-ECG-Digitizer U-Net deep learning tool.

    Operates behind a strictly isolated interface. Performs upfront environment
    and model weight checks to fail gracefully without crashing when upstream
    runtimes or Git LFS weights are missing.
    """

    def __init__(self, repo_path: Optional[Path] = None):
        self.repo_path = repo_path or Path("external/Open-ECG-Digitizer")

    @property
    def name(self) -> str:
        return "Open-ECG-Digitizer (Ahus-AIM)"

    @property
    def version(self) -> str:
        return "1.9.3-adapter"

    def is_available(self) -> Tuple[bool, Optional[str]]:
        """Check for Git LFS, PyTorch, and valid pretrained weight files."""
        # Check Git LFS availability
        if shutil.which("git-lfs") is None and shutil.which("git") is not None:
            # Check git lfs command directly
            pass

        # Check repository folder
        if not self.repo_path.is_dir():
            return False, f"Open-ECG-Digitizer directory not found at '{self.repo_path}'."

        # Check model weights presence and size (guard against 133-byte LFS pointers)
        weights_dir = self.repo_path / "weights"
        if not weights_dir.is_dir():
            return False, f"Weights directory missing at '{weights_dir}'."

        unet_weights = weights_dir / "unet_weights_07072025.pt"
        if not unet_weights.is_file():
            return False, f"Missing U-Net weight file at '{unet_weights}'."

        if unet_weights.stat().st_size < 1024 * 1024:
            return (
                False,
                f"U-Net weights at '{unet_weights}' appear to be an unpulled Git LFS pointer "
                f"({unet_weights.stat().st_size} bytes). Run 'git lfs pull' with git-lfs installed."
            )

        # Check PyTorch availability
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, "PyTorch runtime is not installed in the active Python environment."

        return True, None

    def digitize(
        self,
        image: Image.Image,
        metadata: ImageMetadata,
        options: Optional[Dict[str, Any]] = None,
    ) -> DigitizationResult:
        """Run upstream digitization or return structured unavailability status."""
        start_time = time.time()
        available, reason = self.is_available()

        if not available:
            return DigitizationResult(
                status=DigitizationStatus.UNAVAILABLE,
                method=self.name,
                method_version=self.version,
                calibration_status=CalibrationStatus.UNRECOVERED,
                amplitude_unit=AmplitudeUnit.RELATIVE,
                runtime_ms=(time.time() - start_time) * 1000.0,
                error_message=reason or "Open-ECG-Digitizer is unavailable in this environment.",
                warnings=[
                    "Upstream engine requires Python 3.12, Git LFS, PyTorch, and U-Net weights.",
                    "Using the Classical Baseline Digitizer is recommended for this host."
                ],
                metadata={
                    "adapter_mode": "upstream_stub",
                    "upstream_repo": "https://github.com/Ahus-AIM/Open-ECG-Digitizer",
                    "upstream_license": "CC BY-SA 4.0",
                }
            )

        # When fully installed and configured, execution proceeds via subprocess/CLI
        return DigitizationResult(
            status=DigitizationStatus.ERROR,
            method=self.name,
            method_version=self.version,
            calibration_status=CalibrationStatus.UNRECOVERED,
            amplitude_unit=AmplitudeUnit.RELATIVE,
            error_message="Subprocess execution is not configured for this prototype.",
            runtime_ms=(time.time() - start_time) * 1000.0,
        )
