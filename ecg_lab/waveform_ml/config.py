"""Configuration, constants, and schema versions for Waveform ML."""

from pathlib import Path
from typing import Dict, Tuple

# Official PTB-XL diagnostic superclasses (multilabel targets)
TARGET_SUPERCLASSES: Tuple[str, ...] = ("NORM", "MI", "STTC", "CD", "HYP")

# Standard clinical 12 leads in order
LEAD_NAMES: Tuple[str, ...] = (
    "I",
    "II",
    "III",
    "AVR",
    "AVL",
    "AVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)

# Benchmark sampling parameters (100 Hz subset for memory-bounded processing)
SAMPLING_RATE_HZ: int = 100
RECORD_DURATION_SEC: float = 10.0
EXPECTED_SAMPLES_PER_RECORD: int = 1000
EXPECTED_LEADS_COUNT: int = 12

# Official patient-separated split recommendation (Folds 1-8 Train, 9 Val, 10 Test)
TRAIN_FOLDS: Tuple[int, ...] = tuple(range(1, 9))
VAL_FOLDS: Tuple[int, ...] = (9,)
TEST_FOLDS: Tuple[int, ...] = (10,)

# Provenance and caching versions
FEATURE_SCHEMA_VERSION: str = "1.0.0"
MODEL_ARCHITECTURE_VERSION: str = "1.0.0"
DEFAULT_RANDOM_SEED: int = 42

# Official PhysioNet & AWS Open Data sources
PHYSIONET_DATASET_NAME: str = "PTB-XL v1.0.3"
PHYSIONET_CITATION: str = (
    "Wagner et al., Scientific Data 2020. PTB-XL, a large publicly available "
    "electrocardiography dataset. DOI: 10.1038/s41597-020-0386-4"
)
PHYSIONET_LICENSE: str = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
PHYSIONET_BASE_URL: str = "https://physionet.org/files/ptb-xl/1.0.3/"
PHYSIONET_S3_BASE_URL: str = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"

# Verified cryptographic checksums from PhysioNet SHA256SUMS.txt
OFFICIAL_CHECKSUMS: Dict[str, str] = {
    "ptbxl_database.csv": "7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b",
    "scp_statements.csv": "ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f",
}

# Project default paths
DEFAULT_DATA_DIR = Path("data/ptb-xl")
DEFAULT_CACHE_DIR = Path("data/cache/waveform_ml")
DEFAULT_ARTIFACT_DIR = Path("artifacts/waveform_ml")
