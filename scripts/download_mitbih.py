"""Download and extract the official PhysioNet MIT-BIH Arrhythmia Database v1.0.0 archive.

Source: https://physionet.org/content/mitdb/get-zip/1.0.0/
Target directory: data/mitdb/1.0.0/
"""

import hashlib
from pathlib import Path
import urllib.request
import zipfile
import sys
import shutil

ZIP_URL = "https://physionet.org/content/mitdb/get-zip/1.0.0/"
TARGET_DIR = Path("data/mitdb/1.0.0")
ZIP_DEST = Path("data/mitdb/mit-bih-arrhythmia-database-1.0.0.zip")


def download_with_progress(url: str, dest_path: Path) -> None:
    print(f"Connecting to {url}...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CardioSense-Research/1.0 (Academic verification)"}
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        print(f"Downloading {total_size / (1024 * 1024):.2f} MB to {dest_path}...")

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        block_size = 65536
        sha256 = hashlib.sha256()

        with open(dest_path, "wb") as f:
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                sha256.update(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    if downloaded % (block_size * 40) == 0 or downloaded == total_size:
                        sys.stdout.write(f"\rProgress: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({pct:.1f}%)")
                        sys.stdout.flush()

        print(f"\nDownload complete. SHA-256: {sha256.hexdigest()}")


def verify_and_extract(zip_path: Path, extract_dir: Path) -> None:
    print(f"Verifying zip integrity: {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_file = zf.testzip()
        if bad_file:
            raise ValueError(f"Corrupt file found in zip archive: {bad_file}")
        
        infolist = zf.infolist()
        print(f"Archive is intact. Contains {len(infolist)} files/folders.")
        
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting to {extract_dir}...")
        zf.extractall(extract_dir)

    # Check extracted structure
    # Some archives extract into a subfolder like mit-bih-arrhythmia-database-1.0.0/
    extracted_subdirs = [p for p in extract_dir.iterdir() if p.is_dir()]
    print(f"Extraction complete. Subdirectories: {[p.name for p in extracted_subdirs]}")


def main():
    if not ZIP_DEST.is_file() or ZIP_DEST.stat().st_size < 70000000:
        download_with_progress(ZIP_URL, ZIP_DEST)
    else:
        print(f"ZIP archive already present: {ZIP_DEST} ({ZIP_DEST.stat().st_size:,} bytes)")

    verify_and_extract(ZIP_DEST, TARGET_DIR)
    print("MIT-BIH download and extraction verification successful!")


if __name__ == "__main__":
    main()
