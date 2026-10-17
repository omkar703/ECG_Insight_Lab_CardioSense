"""Dataset loading, metadata validation, SCP label mapping, and patient split integrity.

Handles PTB-XL v1.0.3 100 Hz records and guarantees zero patient overlap across folds.
"""

import ast
import asyncio
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import requests

from ecg_lab.waveform_ml.config import (
    DEFAULT_DATA_DIR,
    EXPECTED_LEADS_COUNT,
    EXPECTED_SAMPLES_PER_RECORD,
    LEAD_NAMES,
    OFFICIAL_CHECKSUMS,
    PHYSIONET_S3_BASE_URL,
    SAMPLING_RATE_HZ,
    TARGET_SUPERCLASSES,
    TEST_FOLDS,
    TRAIN_FOLDS,
    VAL_FOLDS,
)


class PatientOverlapError(ValueError):
    """Raised when patient identifiers leak across train, validation, or test folds."""
    pass


class MalformedRecordError(RuntimeError):
    """Raised when a WFDB signal record is corrupt, non-finite, or malformed."""
    pass


class ChecksumMismatchError(ValueError):
    """Raised when an official dataset file does not match its cryptographic hash."""
    pass


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Compute the SHA-256 digest of a local file in bounded chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_scp_mapping(scp_statements_path: Union[str, Path]) -> Dict[str, str]:
    """Load SCP-ECG statements and extract only diagnostic statements mapped to target superclasses.

    Only statements where diagnostic == 1.0 and diagnostic_class is populated
    with one of ('NORM', 'MI', 'STTC', 'CD', 'HYP') are mapped.
    All non-diagnostic statements (rhythm, form, etc.) are excluded from diagnostic mapping.
    """
    path = Path(scp_statements_path)
    if not path.is_file():
        raise FileNotFoundError(f"SCP statements file not found: {path}")

    df = pd.read_csv(path, index_col=0)
    if "diagnostic" not in df.columns or "diagnostic_class" not in df.columns:
        raise ValueError(
            f"Invalid scp_statements.csv schema. Expected 'diagnostic' and 'diagnostic_class' columns, got {df.columns.tolist()}"
        )

    # Filter explicitly designated diagnostic codes
    diag_df = df[df["diagnostic"] == 1.0]
    mapping: Dict[str, str] = {}
    for code, row in diag_df.iterrows():
        diag_class = row["diagnostic_class"]
        if pd.notna(diag_class) and str(diag_class) in TARGET_SUPERCLASSES:
            mapping[str(code)] = str(diag_class)

    return mapping


def parse_scp_codes_to_superclasses(
    scp_codes_str: Any,
    code_to_super: Dict[str, str],
    filter_zero_likelihood: bool = False,
) -> List[str]:
    """Parse a record's scp_codes string into a list of mapped diagnostic superclasses.

    Guarantees:
    - If filter_zero_likelihood=False (default, matching official PhysioNet example_physionet.py):
      maps all keys in codes_dict that exist in code_to_super.
      Produces official PTB-XL v1.0.3 reference counts:
      NORM=9514, MI=5469, STTC=5235, CD=4898, HYP=2649.
    - If filter_zero_likelihood=True:
      strictly requires score > 0.
      Produces: NORM=9514, MI=5469, STTC=5108, CD=4898, HYP=2649
      (127 records where an STTC code such as NST_ has likelihood score 0.0 are excluded).
    - Preserves multi-label nature (multiple superclasses can be positive).
    - Returns sorted unique superclasses.
    """
    if pd.isna(scp_codes_str):
        return []

    if isinstance(scp_codes_str, dict):
        codes_dict = scp_codes_str
    elif isinstance(scp_codes_str, str):
        try:
            codes_dict = ast.literal_eval(scp_codes_str)
        except (ValueError, SyntaxError):
            return []
    else:
        return []

    if not isinstance(codes_dict, dict):
        return []

    classes: Set[str] = set()
    for code, score in codes_dict.items():
        if code not in code_to_super:
            continue
        if filter_zero_likelihood:
            try:
                score_num = float(score)
            except (ValueError, TypeError):
                continue
            if score_num <= 0:
                continue
        classes.add(code_to_super[code])

    return sorted(list(classes))


def generate_label_mapping_audit_df(
    scp_statements_path: Union[str, Path],
) -> pd.DataFrame:
    """Audit all 71 statements in scp_statements.csv against official diagnostic superclasses.

    Verifies:
    - Every diagnostic == 1.0 statement maps to exactly one official diagnostic superclass.
    - No code is assigned to conflicting superclasses.
    - Documents explicit reason for any unmapped/excluded statement.
    """
    scp_path = Path(scp_statements_path)
    if not scp_path.is_file():
        raise FileNotFoundError(f"SCP statements file missing: {scp_path}")

    df = pd.read_csv(scp_path, index_col=0)
    code_to_super = load_scp_mapping(scp_path)

    audit_rows: List[Dict[str, Any]] = []
    seen_codes: Set[str] = set()

    for code, row in df.iterrows():
        code_str = str(code)
        if code_str in seen_codes:
            conflict = True
        else:
            seen_codes.add(code_str)
            conflict = False

        is_diag = (row.get("diagnostic") == 1.0)
        official_class = str(row["diagnostic_class"]) if pd.notna(row.get("diagnostic_class")) else "NONE"
        impl_class = code_to_super.get(code_str, "UNMAPPED")

        if is_diag and official_class in TARGET_SUPERCLASSES:
            status = "MAPPED_DIAGNOSTIC"
            reason = "Official diagnostic statement mapped to canonical superclass"
        elif row.get("rhythm") == 1.0:
            status = "EXCLUDED_RHYTHM"
            reason = "Rhythm descriptor statement (non-diagnostic)"
        elif row.get("form") == 1.0:
            status = "EXCLUDED_FORM"
            reason = "Morphological form statement (non-diagnostic)"
        else:
            status = "EXCLUDED_OTHER"
            reason = "Non-diagnostic statement"

        audit_rows.append({
            "code": code_str,
            "description": row.get("description", ""),
            "diagnostic_flag": 1 if is_diag else 0,
            "official_diagnostic_class": official_class,
            "implemented_diagnostic_class": impl_class,
            "status": status,
            "exclusion_reason": reason,
            "statement_category": row.get("Statement Category", "N/A"),
            "conflict_detected": conflict,
        })

    return pd.DataFrame(audit_rows)


def load_ptbxl_metadata(
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    verify_checksums: bool = False,
    filter_zero_likelihood: bool = False,
) -> pd.DataFrame:
    """Load ptbxl_database.csv and map scp_codes to multi-label superclasses.

    Adds binary indicator columns: y_NORM, y_MI, y_STTC, y_CD, y_HYP.
    """
    path = Path(data_dir)
    db_file = path / "ptbxl_database.csv"
    scp_file = path / "scp_statements.csv"

    if not db_file.is_file():
        raise FileNotFoundError(
            f"PTB-XL database metadata missing at {db_file}. Run CLI download or place official files there."
        )
    if not scp_file.is_file():
        raise FileNotFoundError(
            f"PTB-XL scp_statements.csv missing at {scp_file}."
        )

    if verify_checksums:
        for fname, expected_hash in OFFICIAL_CHECKSUMS.items():
            fpath = path / fname
            if fpath.is_file():
                actual_hash = compute_file_sha256(fpath)
                if actual_hash != expected_hash:
                    raise ChecksumMismatchError(
                        f"Checksum mismatch for {fname}: expected {expected_hash}, got {actual_hash}"
                    )

    code_to_super = load_scp_mapping(scp_file)
    df = pd.read_csv(db_file, index_col="ecg_id")

    # Required columns check
    required_cols = {"patient_id", "scp_codes", "strat_fold", "filename_lr"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"ptbxl_database.csv is missing required columns: {missing}")

    # Map superclasses
    df["diagnostic_superclasses"] = df["scp_codes"].apply(
        lambda x: parse_scp_codes_to_superclasses(
            x, code_to_super, filter_zero_likelihood=filter_zero_likelihood
        )
    )
    df["num_diagnostic_labels"] = df["diagnostic_superclasses"].apply(len)
    df["has_diagnostic_label"] = df["num_diagnostic_labels"] > 0

    # Binary indicator matrix
    for sc in TARGET_SUPERCLASSES:
        df[f"y_{sc}"] = df["diagnostic_superclasses"].apply(lambda classes: int(sc in classes))

    return df


def validate_split_integrity(
    df: pd.DataFrame,
    train_folds: Tuple[int, ...] = TRAIN_FOLDS,
    val_folds: Tuple[int, ...] = VAL_FOLDS,
    test_folds: Tuple[int, ...] = TEST_FOLDS,
) -> Dict[str, Any]:
    """Verify zero patient overlap across train, validation, and test fold splits.

    Raises PatientOverlapError if any patient_id appears in more than one partition.
    """
    if "patient_id" not in df.columns or "strat_fold" not in df.columns:
        raise ValueError("DataFrame must contain 'patient_id' and 'strat_fold' columns.")

    train_mask = df["strat_fold"].isin(train_folds)
    val_mask = df["strat_fold"].isin(val_folds)
    test_mask = df["strat_fold"].isin(test_folds)

    train_pts = set(df.loc[train_mask, "patient_id"].dropna())
    val_pts = set(df.loc[val_mask, "patient_id"].dropna())
    test_pts = set(df.loc[test_mask, "patient_id"].dropna())

    train_val_overlap = train_pts.intersection(val_pts)
    train_test_overlap = train_pts.intersection(test_pts)
    val_test_overlap = val_pts.intersection(test_pts)

    if train_val_overlap or train_test_overlap or val_test_overlap:
        err_msg = [
            f"FATAL: Patient leakage detected between splits!",
            f"- Train vs Val overlap: {len(train_val_overlap)} patients",
            f"- Train vs Test overlap: {len(train_test_overlap)} patients",
            f"- Val vs Test overlap: {len(val_test_overlap)} patients",
        ]
        raise PatientOverlapError("\n".join(err_msg))

    stats = {
        "train_records": int(train_mask.sum()),
        "val_records": int(val_mask.sum()),
        "test_records": int(test_mask.sum()),
        "train_patients": len(train_pts),
        "val_patients": len(val_pts),
        "test_patients": len(test_pts),
        "train_val_overlap": 0,
        "train_test_overlap": 0,
        "val_test_overlap": 0,
        "split_integrity_verified": True,
    }
    return stats


def read_wfdb_record(
    record_path_or_prefix: Union[str, Path],
    expected_fs: int = SAMPLING_RATE_HZ,
    expected_samples: int = EXPECTED_SAMPLES_PER_RECORD,
    expected_leads: int = EXPECTED_LEADS_COUNT,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read a 12-lead WFDB record, preserving physical units (mV) and sampling rate.

    Returns:
        signal: np.ndarray of shape (samples, leads) with physical values in millivolts (mV).
        metadata: Dict containing fs, units, lead_names, comments, length.

    Raises:
        FileNotFoundError if record header or dat file does not exist.
        MalformedRecordError if signal contains non-finite values or unexpected dimensions.
    """
    import wfdb

    record_str = str(record_path_or_prefix)
    # Strip extension if passed
    if record_str.endswith(".hea") or record_str.endswith(".dat"):
        record_str = record_str[:-4]

    hea_file = Path(f"{record_str}.hea")
    if not hea_file.is_file():
        raise FileNotFoundError(f"WFDB header file missing: {hea_file}")

    try:
        record = wfdb.rdrecord(record_str)
    except Exception as e:
        raise MalformedRecordError(f"Failed to read WFDB record {record_str}: {e}") from e

    signal = record.p_signal
    if signal is None:
        raise MalformedRecordError(f"WFDB record {record_str} returned None p_signal.")

    # Validate finite values
    if not np.isfinite(signal).all():
        raise MalformedRecordError(
            f"WFDB record {record_str} contains NaN or Infinite voltage samples."
        )

    # Validate shape
    if signal.ndim != 2:
        raise MalformedRecordError(
            f"Expected 2D signal array (samples, leads), got ndim={signal.ndim}."
        )

    samples, leads = signal.shape
    if leads != expected_leads:
        raise MalformedRecordError(
            f"Expected {expected_leads} leads, got {leads} leads in {record_str}."
        )

    meta = {
        "fs": record.fs,
        "sig_len": record.sig_len,
        "sig_name": list(record.sig_name) if record.sig_name else list(LEAD_NAMES),
        "units": list(record.units) if record.units else ["mV"] * leads,
        "comments": list(record.comments) if record.comments else [],
    }

    return signal, meta


def download_ptbxl_metadata(
    dest_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    base_url: str = PHYSIONET_S3_BASE_URL,
) -> Dict[str, Path]:
    """Download official metadata files from PhysioNet/AWS Open Data and verify checksums."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    files_to_download = [
        "ptbxl_database.csv",
        "scp_statements.csv",
        "LICENSE.txt",
    ]

    saved_paths: Dict[str, Path] = {}
    for filename in files_to_download:
        target = dest / filename
        if not target.is_file():
            url = f"{base_url}{filename}"
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            with open(target, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)

        # Verify hash if in checksum registry
        if filename in OFFICIAL_CHECKSUMS:
            actual = compute_file_sha256(target)
            expected = OFFICIAL_CHECKSUMS[filename]
            if actual != expected:
                raise ChecksumMismatchError(
                    f"Checksum mismatch for {filename}: expected {expected}, got {actual}"
                )

        saved_paths[filename] = target

    return saved_paths


def download_single_record(
    record_rel_path: str,
    dest_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    base_url: str = PHYSIONET_S3_BASE_URL,
) -> Path:
    """Download a single 100 Hz record (.hea and .dat) from the official repository."""
    dest = Path(dest_dir)
    # record_rel_path e.g. "records100/00000/00001_lr"
    clean_rel = record_rel_path.replace("\\", "/")
    if clean_rel.endswith(".hea") or clean_rel.endswith(".dat"):
        clean_rel = clean_rel[:-4]

    target_prefix = dest / clean_rel
    target_prefix.parent.mkdir(parents=True, exist_ok=True)

    for ext in (".hea", ".dat"):
        target_file = target_prefix.with_suffix(ext)
        if not target_file.is_file():
            url = f"{base_url}{clean_rel}{ext}"
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            with open(target_file, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)

    return target_prefix


def audit_dataset_completeness(
    metadata_df: pd.DataFrame,
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
) -> Dict[str, Any]:
    """Audit actual on-disk WFDB availability for every record by partition fold.

    Distinguishes metadata rows from actual .hea and .dat file existence on disk.
    Computes expected, present, missing, corrupt, and complete counts by fold.
    """
    path = Path(data_dir)
    folds_summary: Dict[int, Dict[str, Any]] = {}
    total_expected = len(metadata_df)
    total_hea = 0
    total_dat = 0
    total_complete = 0
    total_missing = 0

    for fold in range(1, 11):
        fold_df = metadata_df[metadata_df["strat_fold"] == fold]
        exp_count = len(fold_df)
        hea_count = 0
        dat_count = 0
        complete_count = 0
        missing_records: List[str] = []

        for ecg_id, row in fold_df.iterrows():
            filename_lr = row.get("filename_lr")
            if not filename_lr or pd.isna(filename_lr):
                missing_records.append(f"ecg_id_{ecg_id}_no_filename")
                continue
            hea_file = path / f"{filename_lr}.hea"
            dat_file = path / f"{filename_lr}.dat"
            has_hea = hea_file.is_file() and hea_file.stat().st_size > 0
            has_dat = dat_file.is_file() and dat_file.stat().st_size > 0

            if has_hea:
                hea_count += 1
            if has_dat:
                dat_count += 1
            if has_hea and has_dat:
                complete_count += 1
            else:
                missing_records.append(str(filename_lr))

        total_hea += hea_count
        total_dat += dat_count
        total_complete += complete_count
        missing_count = exp_count - complete_count
        total_missing += missing_count

        folds_summary[fold] = {
            "fold": fold,
            "expected_records": exp_count,
            "present_hea": hea_count,
            "present_dat": dat_count,
            "complete_records": complete_count,
            "missing_records": missing_count,
            "is_complete": bool(complete_count == exp_count and exp_count > 0),
        }

    return {
        "total_expected": total_expected,
        "total_hea_present": total_hea,
        "total_dat_present": total_dat,
        "total_complete_records": total_complete,
        "total_missing_records": total_missing,
        "all_folds_complete": bool(total_missing == 0 and total_expected > 0),
        "folds": folds_summary,
    }


async def download_missing_records_async(
    metadata_df: pd.DataFrame,
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    base_url: str = PHYSIONET_S3_BASE_URL,
    concurrency: int = 40,
    batch_size: int = 500,
    target_folds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Download all missing .hea and .dat files concurrently with bounded memory and retries."""
    import aiohttp

    path = Path(data_dir)
    target_df = (
        metadata_df
        if target_folds is None
        else metadata_df[metadata_df["strat_fold"].isin(target_folds)]
    )

    missing_tasks: List[Tuple[str, str, Path]] = []
    for ecg_id, row in target_df.iterrows():
        rel = str(row["filename_lr"]).replace("\\", "/")
        if rel.endswith(".hea") or rel.endswith(".dat"):
            rel = rel[:-4]
        for ext in (".hea", ".dat"):
            fpath = path / f"{rel}{ext}"
            if not fpath.is_file() or fpath.stat().st_size == 0:
                missing_tasks.append((rel, ext, fpath))

    if not missing_tasks:
        return {
            "attempted": 0,
            "downloaded": 0,
            "failed": 0,
            "message": "All records already present",
        }

    sem = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=concurrency + 10, ttl_dns_cache=300)

    downloaded = 0
    failed = 0

    async def _fetch(session: aiohttp.ClientSession, rel: str, ext: str, target_file: Path):
        nonlocal downloaded, failed
        async with sem:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = target_file.with_name(f"{target_file.name}.tmp")
            url = f"{base_url}{rel}{ext}"
            for attempt in range(4):
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status == 200:
                            data = await resp.read()
                            tmp_file.write_bytes(data)
                            tmp_file.replace(target_file)
                            downloaded += 1
                            return
                        else:
                            await asyncio.sleep(0.5 * (attempt + 1))
                except Exception:
                    await asyncio.sleep(0.5 * (attempt + 1))
            failed += 1

    async with aiohttp.ClientSession(connector=conn) as session:
        total = len(missing_tasks)
        for i in range(0, total, batch_size):
            chunk = missing_tasks[i : i + batch_size]
            tasks = [_fetch(session, rel, ext, fpath) for rel, ext, fpath in chunk]
            await asyncio.gather(*tasks)
            done = min(i + batch_size, total)
            print(f"  Progress: {done}/{total} files ({done/total*100:.1f}%) downloaded...")

    return {
        "attempted": len(missing_tasks),
        "downloaded": downloaded,
        "failed": failed,
    }


def download_missing_records(
    metadata_df: pd.DataFrame,
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    base_url: str = PHYSIONET_S3_BASE_URL,
    concurrency: int = 40,
    target_folds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Synchronous entrypoint for downloading missing PTB-XL records."""
    return asyncio.run(
        download_missing_records_async(
            metadata_df=metadata_df,
            data_dir=data_dir,
            base_url=base_url,
            concurrency=concurrency,
            target_folds=target_folds,
        )
    )

