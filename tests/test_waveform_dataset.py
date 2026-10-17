"""Unit tests for dataset metadata loading, SCP statement mapping, and record validation.

Uses small mock metadata fixtures; does NOT require network access or full PTB-XL download.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from ecg_lab.waveform_ml.config import TARGET_SUPERCLASSES
from ecg_lab.waveform_ml.dataset import (
    MalformedRecordError,
    load_ptbxl_metadata,
    load_scp_mapping,
    parse_scp_codes_to_superclasses,
    read_wfdb_record,
)


@pytest.fixture
def mock_scp_file(tmp_path: Path) -> Path:
    """Create a minimal mock scp_statements.csv fixture."""
    scp_path = tmp_path / "scp_statements.csv"
    content = (
        ",description,diagnostic,form,rhythm,diagnostic_class,diagnostic_subclass\n"
        "NORM,normal ecg,1.0,,,NORM,NORM\n"
        "IMI,inferior myocardial infarction,1.0,,,MI,IMI\n"
        "STTC,ST-T changes,1.0,,,STTC,STTC\n"
        "LAFB,left anterior fascicular block,1.0,,,CD,LAFB\n"
        "LVH,left ventricular hypertrophy,1.0,,,HYP,LVH\n"
        "SR,sinus rhythm,,1.0,1.0,,\n"
        "AFIB,atrial fibrillation,,1.0,1.0,,\n"
    )
    scp_path.write_text(content, encoding="utf-8")
    return scp_path


@pytest.fixture
def mock_dataset_dir(tmp_path: Path, mock_scp_file: Path) -> Path:
    """Create a minimal mock PTB-XL directory with database and scp statements."""
    data_dir = tmp_path / "ptb_xl_mock"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Copy scp statements
    (data_dir / "scp_statements.csv").write_text(mock_scp_file.read_text())

    # Create mock ptbxl_database.csv
    db_content = (
        "ecg_id,patient_id,age,sex,strat_fold,filename_lr,scp_codes\n"
        "1,101,65,1,1,records100/00000/00001_lr,\"{'NORM': 100.0, 'SR': 100.0}\"\n"
        "2,102,54,0,2,records100/00000/00002_lr,\"{'IMI': 80.0, 'LAFB': 50.0}\"\n"
        "3,103,72,1,9,records100/00000/00003_lr,\"{'LVH': 100.0, 'STTC': 50.0}\"\n"
        "4,104,45,0,10,records100/00000/00004_lr,\"{'SR': 100.0, 'AFIB': 0.0}\"\n"
        "5,105,50,1,1,records100/00000/00005_lr,\"{'IMI': 0.0}\"\n"
    )
    (data_dir / "ptbxl_database.csv").write_text(db_content, encoding="utf-8")
    return data_dir


def test_load_scp_mapping(mock_scp_file: Path):
    """Test that only diagnostic == 1.0 codes are mapped to target superclasses."""
    mapping = load_scp_mapping(mock_scp_file)
    assert mapping["NORM"] == "NORM"
    assert mapping["IMI"] == "MI"
    assert mapping["STTC"] == "STTC"
    assert mapping["LAFB"] == "CD"
    assert mapping["LVH"] == "HYP"
    # Rhythm statements must be excluded
    assert "SR" not in mapping
    assert "AFIB" not in mapping


def test_parse_scp_codes_to_superclasses(mock_scp_file: Path):
    """Test parsing scp_codes string into multi-label superclasses."""
    mapping = load_scp_mapping(mock_scp_file)

    # Single label with rhythm
    res1 = parse_scp_codes_to_superclasses("{'NORM': 100.0, 'SR': 100.0}", mapping)
    assert res1 == ["NORM"]

    # Multiple diagnostic labels
    res2 = parse_scp_codes_to_superclasses("{'IMI': 80.0, 'LAFB': 50.0}", mapping)
    assert sorted(res2) == ["CD", "MI"]

    # Zero score should be ignored
    res3 = parse_scp_codes_to_superclasses("{'IMI': 0.0, 'LAFB': 100.0}", mapping)
    assert res3 == ["CD"]

    # Only non-diagnostic rhythm codes
    res4 = parse_scp_codes_to_superclasses("{'SR': 100.0}", mapping)
    assert res4 == []

    # Malformed or empty
    assert parse_scp_codes_to_superclasses("", mapping) == []
    assert parse_scp_codes_to_superclasses(None, mapping) == []


def test_load_ptbxl_metadata(mock_dataset_dir: Path):
    """Test loading and column generation in metadata DataFrame."""
    df = load_ptbxl_metadata(mock_dataset_dir, verify_checksums=False)
    assert len(df) == 5
    assert "diagnostic_superclasses" in df.columns
    assert "num_diagnostic_labels" in df.columns
    assert "has_diagnostic_label" in df.columns

    for sc in TARGET_SUPERCLASSES:
        assert f"y_{sc}" in df.columns

    # Check ecg_id 1: NORM=1, others=0
    assert df.loc[1, "y_NORM"] == 1
    assert df.loc[1, "y_MI"] == 0

    # Check ecg_id 2: MI=1, CD=1, others=0
    assert df.loc[2, "y_MI"] == 1
    assert df.loc[2, "y_CD"] == 1
    assert df.loc[2, "y_NORM"] == 0

    # Check ecg_id 4: no diagnostic label
    assert df.loc[4, "num_diagnostic_labels"] == 0
    assert df.loc[4, "has_diagnostic_label"] == False


def test_read_wfdb_record_missing_file():
    """Test that missing WFDB header file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_wfdb_record("/non/existent/path/00001_lr")
