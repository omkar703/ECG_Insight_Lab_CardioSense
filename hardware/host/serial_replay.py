"""Serial replay utility for offline testing and protocol validation.

Replays captured CSV or simulated stream files at real-time (360 Hz) or accelerated pacing.
Enables end-to-end testing of parsers and UI visualization without physical hardware.
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Callable, Iterator, Optional
import pandas as pd

from hardware.host.serial_protocol import (
    NOMINAL_SAMPLING_RATE_HZ,
    SerialSample,
    format_serial_line,
    parse_serial_line,
)


def stream_samples_from_file(
    file_path: Path,
    pacing_hz: Optional[float] = NOMINAL_SAMPLING_RATE_HZ,
    max_samples: Optional[int] = None,
    callback: Optional[Callable[[SerialSample, str], None]] = None,
) -> Iterator[SerialSample]:
    """Iterate through samples in a CSV or raw protocol stream file, pacing each yield.

    Args:
        file_path: Path to CSV or raw protocol file.
        pacing_hz: Replay rate in Hz. Set to 0 or None for as-fast-as-possible.
        max_samples: Optional maximum number of samples to yield.
        callback: Optional callback receiving (SerialSample, formatted_line).

    Yields:
        SerialSample instances.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Replay source file not found: {file_path}")

    # Determine file format: raw protocol text or exported CSV DataFrame
    delay_per_sample = 1.0 / pacing_hz if (pacing_hz and pacing_hz > 0) else 0.0

    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        # Check first line
        first_line = f.readline()
        if not first_line:
            return

        is_header_csv = "timestamp_micros" in first_line or "raw_adc" in first_line
        f.seek(0)

        if is_header_csv:
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                if max_samples and count >= max_samples:
                    break

                sample = SerialSample(
                    seq=int(row["seq"]),
                    timestamp_micros=int(row["timestamp_micros"]),
                    raw_adc=int(row["raw_adc"]),
                    lo_plus=int(row["lo_plus"]),
                    lo_minus=int(row["lo_minus"]),
                    lead_off=bool(row["lead_off"]),
                    time_seconds=float(row.get("time_seconds", 0.0)),
                )
                formatted = format_serial_line(sample)
                if callback:
                    callback(sample, formatted)
                if delay_per_sample > 0:
                    time.sleep(delay_per_sample)

                yield sample
                count += 1
        else:
            # Raw protocol text
            for line in f:
                if max_samples and count >= max_samples:
                    break

                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue

                sample = parse_serial_line(line_str)
                if sample is None:
                    continue

                formatted = format_serial_line(sample)
                if callback:
                    callback(sample, formatted)
                if delay_per_sample > 0:
                    time.sleep(delay_per_sample)

                yield sample
                count += 1


def main():
    parser = argparse.ArgumentParser(description="Replay serial stream or CSV file at controlled pacing.")
    parser.add_argument("--input", type=str, required=True, help="Input CSV or raw stream file")
    parser.add_argument("--pace", type=float, default=NOMINAL_SAMPLING_RATE_HZ, help=f"Pacing in Hz (0 for max speed, default: {NOMINAL_SAMPLING_RATE_HZ})")
    parser.add_argument("--samples", type=int, default=None, help="Max samples to replay")
    parser.add_argument("--quiet", action="store_true", help="Suppress line-by-line terminal printing")

    args = parser.parse_args()

    def print_callback(sample: SerialSample, line_text: str):
        if not args.quiet:
            sys.stdout.write(line_text)
            sys.stdout.flush()

    try:
        total = 0
        t0 = time.time()
        for _ in stream_samples_from_file(
            file_path=Path(args.input),
            pacing_hz=args.pace,
            max_samples=args.samples,
            callback=print_callback,
        ):
            total += 1

        elapsed = time.time() - t0
        effective_hz = total / elapsed if elapsed > 0 else 0
        if args.quiet:
            print(f"[+] Replayed {total} samples in {elapsed:.3f} s (effective rate: {effective_hz:.1f} Hz).")
    except Exception as e:
        print(f"[ERROR] Replay failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
