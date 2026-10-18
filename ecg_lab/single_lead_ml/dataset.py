"""Dataset ingestion, channel selection, and expert annotation parsing for MIT-BIH."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from ecg_lab.single_lead_ml.config import (
    ADC_GAIN_UNITS_PER_MV,
    ADC_ZERO_BASELINE,
    DEFAULT_MITDB_DIR,
    FALLBACK_MITDB_DIR,
    NATIVE_SAMPLING_RATE_HZ,
    NON_BEAT_SYMBOLS,
    PACED_RECORDS,
    PREFERRED_LEADS_ORDER,
    SYMBOL_TO_AAMI,
    TEST_RECORDS,
    TRAIN_RECORDS,
    VAL_RECORDS,
    get_subject_id,
)


@dataclass
class BeatAnnotation:
    """Represents a single expert-reviewed ECG beat annotation."""
    sample_index: int
    original_symbol: str
    aami_class: str
    time_seconds: float
    aux_note: Optional[str] = None


@dataclass
class SingleLeadRecord:
    """Represents an ingested single-channel ECG recording from MIT-BIH."""
    record_id: str
    lead_name: str
    signal_mv: np.ndarray
    raw_adc: np.ndarray
    sampling_rate_hz: int = NATIVE_SAMPLING_RATE_HZ
    available_leads: List[str] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        return len(self.signal_mv)

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sampling_rate_hz


def resolve_mitdb_dir(data_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve active directory containing MIT-BIH CSV and annotation files."""
    if data_dir is not None:
        p = Path(data_dir)
        if p.is_dir() and any(p.glob("*.csv")):
            return p

    # Check default symlink/local path
    if DEFAULT_MITDB_DIR.is_dir() and any(DEFAULT_MITDB_DIR.glob("*.csv")):
        return DEFAULT_MITDB_DIR

    # Check fallback path
    if FALLBACK_MITDB_DIR.is_dir() and any(FALLBACK_MITDB_DIR.glob("*.csv")):
        return FALLBACK_MITDB_DIR

    raise FileNotFoundError(
        "MIT-BIH Arrhythmia Database files not found. "
        f"Checked default '{DEFAULT_MITDB_DIR}' and fallback '{FALLBACK_MITDB_DIR}'. "
        "Ensure the dataset is acquired and placed in the project data directory."
    )


def get_available_records(data_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """Return sorted list of available record IDs."""
    root = resolve_mitdb_dir(data_dir)
    csv_files = root.glob("*.csv")
    records = sorted([f.stem for f in csv_files])
    return records


def load_single_lead_signal(
    record_id: str,
    data_dir: Optional[Union[str, Path]] = None,
    preferred_lead: Optional[str] = None,
) -> SingleLeadRecord:
    """Load single ECG channel from record CSV, converting ADC to millivolts."""
    root = resolve_mitdb_dir(data_dir)
    csv_path = root / f"{record_id}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Record CSV not found: {csv_path}")

    # Read header line first to identify channels
    with open(csv_path, "r", encoding="utf-8") as f:
        raw_header = f.readline().strip().split(",")
    
    clean_cols = [c.strip().strip("'\" ") for c in raw_header]
    available_channels = clean_cols[1:]  # First column is sample #

    # Select lead
    selected_lead = None
    if preferred_lead and preferred_lead in available_channels:
        selected_lead = preferred_lead
    else:
        for pref in PREFERRED_LEADS_ORDER:
            if pref in available_channels:
                selected_lead = pref
                break
        if not selected_lead:
            selected_lead = available_channels[0]

    # Load dataframe with selected column
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [c.strip().strip("'\" ") for c in df.columns]

    raw_adc = df[selected_lead].values.astype(np.float64)
    # Physical voltage conversion: (ADC - baseline) / gain
    signal_mv = (raw_adc - ADC_ZERO_BASELINE) / ADC_GAIN_UNITS_PER_MV

    return SingleLeadRecord(
        record_id=str(record_id),
        lead_name=selected_lead,
        signal_mv=signal_mv,
        raw_adc=raw_adc,
        sampling_rate_hz=NATIVE_SAMPLING_RATE_HZ,
        available_leads=available_channels,
    )


def load_beat_annotations(
    record_id: str,
    data_dir: Optional[Union[str, Path]] = None,
    include_non_beat: bool = False,
) -> List[BeatAnnotation]:
    """Parse expert annotations for a record, mapping symbols to AAMI classes."""
    root = resolve_mitdb_dir(data_dir)
    ann_path = root / f"{record_id}annotations.txt"
    if not ann_path.is_file():
        raise FileNotFoundError(f"Record annotation text file not found: {ann_path}")

    annotations: List[BeatAnnotation] = []
    with open(ann_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip header
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            sample_idx = int(parts[1])
            symbol = parts[2]
            aux_note = parts[6] if len(parts) >= 7 else None
        except (ValueError, IndexError):
            continue

        if symbol in NON_BEAT_SYMBOLS:
            if include_non_beat:
                annotations.append(BeatAnnotation(
                    sample_index=sample_idx,
                    original_symbol=symbol,
                    aami_class="NON_BEAT",
                    time_seconds=sample_idx / NATIVE_SAMPLING_RATE_HZ,
                    aux_note=aux_note,
                ))
            continue

        if symbol in SYMBOL_TO_AAMI:
            annotations.append(BeatAnnotation(
                sample_index=sample_idx,
                original_symbol=symbol,
                aami_class=SYMBOL_TO_AAMI[symbol],
                time_seconds=sample_idx / NATIVE_SAMPLING_RATE_HZ,
                aux_note=aux_note,
            ))

    return annotations


def load_record_with_annotations(
    record_id: str,
    data_dir: Optional[Union[str, Path]] = None,
    preferred_lead: Optional[str] = None,
) -> Tuple[SingleLeadRecord, List[BeatAnnotation]]:
    """Load both single-lead signal and expert annotations for a record."""
    rec = load_single_lead_signal(record_id, data_dir=data_dir, preferred_lead=preferred_lead)
    anns = load_beat_annotations(record_id, data_dir=data_dir)
    return rec, anns


def verify_split_integrity() -> Dict[str, Any]:
    """Assert patient-disjoint split boundaries and record 201/202 grouping."""
    train_subs = {get_subject_id(r) for r in TRAIN_RECORDS}
    val_subs = {get_subject_id(r) for r in VAL_RECORDS}
    test_subs = {get_subject_id(r) for r in TEST_RECORDS}
    paced_subs = {get_subject_id(r) for r in PACED_RECORDS}

    # Integrity assertions
    assert "201" in TRAIN_RECORDS and "202" in TRAIN_RECORDS, "Records 201 and 202 must be in the same partition!"
    assert get_subject_id("201") == get_subject_id("202") == "sub_201_202"
    assert len(train_subs.intersection(val_subs)) == 0, "Train and Val subjects must have zero overlap!"
    assert len(train_subs.intersection(test_subs)) == 0, "Train and Test subjects must have zero overlap!"
    assert len(val_subs.intersection(test_subs)) == 0, "Val and Test subjects must have zero overlap!"

    return {
        "is_valid": True,
        "n_train_records": len(TRAIN_RECORDS),
        "n_train_subjects": len(train_subs),
        "n_val_records": len(VAL_RECORDS),
        "n_val_subjects": len(val_subs),
        "n_test_records": len(TEST_RECORDS),
        "n_test_subjects": len(test_subs),
        "n_paced_records": len(PACED_RECORDS),
        "total_records": len(TRAIN_RECORDS) + len(VAL_RECORDS) + len(TEST_RECORDS) + len(PACED_RECORDS),
        "total_subjects": len(train_subs) + len(val_subs) + len(test_subs) + len(paced_subs),
    }
