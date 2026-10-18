"""Configuration, constants, AAMI taxonomies, and split definitions for MIT-BIH single-lead ML."""

from pathlib import Path
from typing import Dict, List, Set, Tuple

# Dataset paths
DEFAULT_MITDB_DIR = Path("data/mitdb/raw")
FALLBACK_MITDB_DIR = Path("../.data/MIT-BHI")
DEFAULT_ARTIFACT_DIR = Path("artifacts/single_lead_ml")

# Physiological signal parameters
NATIVE_SAMPLING_RATE_HZ: int = 360
ADC_GAIN_UNITS_PER_MV: float = 200.0
ADC_ZERO_BASELINE: float = 1024.0

# Channel selection preference order
PREFERRED_LEADS_ORDER: Tuple[str, ...] = ("MLII", "V5", "V1", "V2", "V4")

# Beat windowing in samples at 360 Hz (approx. -250 ms to +305 ms around R-peak)
PRE_R_SAMPLES: int = 90
POST_R_SAMPLES: int = 110
BEAT_WINDOW_TOTAL_SAMPLES: int = PRE_R_SAMPLES + POST_R_SAMPLES  # 200 samples

# AAMI EC57 Beat Mapping
AAMI_TARGET_CLASSES: Tuple[str, ...] = ("N", "S", "V", "Q")

AAMI_CLASS_DESCRIPTIONS: Dict[str, str] = {
    "N": "Normal / Bundle Branch Block Beat",
    "S": "Supraventricular Ectopic Beat (SVEB / PAC)",
    "V": "Ventricular Ectopic Beat (VEB / PVC)",
    "Q": "Fusion / Paced / Unclassifiable Beat",
}

# Mapping from MIT-BIH raw annotation symbols to AAMI classes
SYMBOL_TO_AAMI: Dict[str, str] = {
    # Normal and Bundle Branch Block Beats -> N
    "N": "N",  # Normal beat
    "L": "N",  # Left bundle branch block beat
    "R": "N",  # Right bundle branch block beat
    "e": "N",  # Atrial escape beat
    "j": "N",  # Nodal (junctional) escape beat
    # Supraventricular Ectopic Beats -> S
    "A": "S",  # Atrial premature beat
    "a": "S",  # Aberrated atrial premature beat
    "J": "S",  # Nodal (junctional) premature beat
    "S": "S",  # Supraventricular premature or ectopic beat
    # Ventricular Ectopic Beats -> V
    "V": "V",  # Premature ventricular contraction
    "E": "V",  # Ventricular escape beat
    # Fusion Beats -> Q (or F if 5-class)
    "F": "Q",  # Fusion of ventricular and normal beat
    # Paced and Unclassifiable Beats -> Q
    "/": "Q",  # Paced beat
    "f": "Q",  # Fusion of paced and normal beat
    "Q": "Q",  # Unclassifiable beat
}

# Non-beat events that must be filtered out
NON_BEAT_SYMBOLS: Set[str] = {"+", "~", "!", '"', "x", "|", "[", "]"}

# Paced records quarantined from natural arrhythmia evaluation per AAMI EC57
PACED_RECORDS: Tuple[str, ...] = ("102", "104", "107", "217")

# Subject mapping: Records 201 and 202 are from the same subject!
def get_subject_id(record_id: str) -> str:
    """Return subject ID, forcing records 201 and 202 into the same subject."""
    if record_id in ("201", "202"):
        return "sub_201_202"
    return f"sub_{record_id}"

# Deterministic Patient-Disjoint Partitions
TRAIN_RECORDS: Tuple[str, ...] = (
    "101", "108", "109", "112", "114", "115", "116", "119",
    "122", "124", "201", "202", "203", "208", "209", "215",
)  # 16 records, 15 subjects (includes 201 & 202)

VAL_RECORDS: Tuple[str, ...] = (
    "106", "118", "205", "207", "220", "223", "230",
)  # 7 records, 7 subjects

TEST_RECORDS: Tuple[str, ...] = (
    "100", "103", "105", "111", "113", "117", "121", "123",
    "200", "210", "212", "213", "214", "219", "221", "222",
    "228", "231", "232", "233", "234",
)  # 21 records, 21 subjects

ALL_EVALUATED_RECORDS: Tuple[str, ...] = TRAIN_RECORDS + VAL_RECORDS + TEST_RECORDS  # 44 records
ALL_DATASET_RECORDS: Tuple[str, ...] = ALL_EVALUATED_RECORDS + PACED_RECORDS  # 48 records

MODEL_NAME: str = "single_lead_rf_model.joblib"
MODEL_VERSION: str = "1.0.0"
FEATURE_SCHEMA_VERSION: str = "1.0.0"
DEFAULT_RANDOM_SEED: int = 42

DISCLAIMER_RECORDED_REPLAY: str = "RECORDED DATA REPLAY — NOT LIVE HARDWARE"
DISCLAIMER_SINGLE_LEAD_RESEARCH: str = (
    "Academic research prototype for single-channel beat classification. "
    "Not a medical device. Not calibrated for clinical diagnosis or acute coronary syndrome triage. "
    "No physical sensors (AD8232/ESP32) are connected."
)
