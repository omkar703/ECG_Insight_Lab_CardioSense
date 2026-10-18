"""Serial port capture utility for AD8232 + Arduino Uno hardware prototype.

Captures UART serial stream to a CSV file with sidecar provenance metadata.
Requires the optional 'pyserial' package when connecting to physical hardware.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Optional

from hardware.host.serial_protocol import (
    BAUD_RATE,
    NOMINAL_SAMPLING_RATE_HZ,
    PROTOCOL_VERSION,
    SerialStreamParser,
)

# Optional dependency check
try:
    import serial  # pyserial
    PYSERIAL_AVAILABLE = True
except ImportError:
    serial = None
    PYSERIAL_AVAILABLE = False


def check_pyserial_available() -> None:
    """Raise informative ImportError if pyserial is not installed."""
    if not PYSERIAL_AVAILABLE:
        raise ImportError(
            "The 'pyserial' library is required for live hardware capture but is not installed.\n"
            "To install it, run: pip install pyserial\n"
            "Note: For offline testing and software verification, physical serial hardware is not required.\n"
            "Use 'python -m hardware.host.simulate_adc_stream' instead."
        )


def capture_serial_to_file(
    port: str,
    output_path: Path,
    baud_rate: int = BAUD_RATE,
    duration_seconds: Optional[float] = None,
    max_samples: Optional[int] = None,
) -> Path:
    """Capture serial stream from specified port to a CSV file.

    Args:
        port: Serial device identifier (e.g. '/dev/ttyACM0' or 'COM3').
        output_path: Target path for output CSV.
        baud_rate: Serial baud rate (default: 115200).
        duration_seconds: Optional finite capture duration in seconds.
        max_samples: Optional limit on total samples collected.

    Returns:
        Path to written CSV file.

    Raises:
        ImportError: If pyserial is missing.
        RuntimeError: If serial port cannot be opened or communication fails.
    """
    check_pyserial_available()

    try:
        ser = serial.Serial(port=port, baudrate=baud_rate, timeout=1.0)
    except Exception as e:
        raise RuntimeError(
            f"Failed to open serial port '{port}' at {baud_rate} baud: {e}\n"
            "Ensure the Arduino is plugged in, user has dialout/serial permissions, and no other process is using the port."
        ) from e

    parser = SerialStreamParser(nominal_sampling_rate_hz=NOMINAL_SAMPLING_RATE_HZ)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Opened serial port {port} @ {baud_rate} baud.")
    print(f"[*] Capturing to: {output_path}")
    if duration_seconds:
        print(f"[*] Target duration: {duration_seconds:.1f} s")
    print("[*] Press Ctrl+C to terminate capture safely.\n")

    start_wall_time = time.time()
    try:
        # Flush initial buffers
        ser.reset_input_buffer()
        while True:
            line_bytes = ser.readline()
            if not line_bytes:
                continue

            try:
                line_str = line_bytes.decode("utf-8", errors="replace")
            except Exception:
                continue

            parser.feed_line(line_str)

            # Check termination conditions
            elapsed = time.time() - start_wall_time
            if duration_seconds and elapsed >= duration_seconds:
                print(f"\n[+] Target duration reached ({elapsed:.2f} s).")
                break
            if max_samples and parser.sample_count >= max_samples:
                print(f"\n[+] Maximum sample count reached ({parser.sample_count}).")
                break

    except KeyboardInterrupt:
        print("\n[!] Capture interrupted by user (Ctrl+C). Finalizing data...")
    finally:
        ser.close()
        print("[*] Serial port closed.")

    # Export to DataFrame and save CSV
    df = parser.to_dataframe()
    df.to_csv(output_path, index=False)
    print(f"[+] Saved {len(df)} samples to {output_path}")

    # Write sidecar provenance JSON
    meta_path = output_path.with_suffix(".meta.json")
    sidecar_data = {
        "capture_type": "LIVE_HARDWARE_SERIAL_CAPTURE",
        "protocol_version": PROTOCOL_VERSION,
        "port": port,
        "baud_rate": baud_rate,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_time_seconds": round(time.time() - start_wall_time, 2),
        "quality_summary": parser.get_quality_summary(),
        "disclaimer": (
            "RAW EXPERIMENTAL SERIAL CAPTURE — PROTOTYPE ONLY. "
            "NOT CALIBRATED ECG VOLTAGE OR MEDICAL DEVICE DATA. "
            "RAW ADC COUNTS MUST NOT BE INTERPRETED AS PATIENT MILLIVOLTS."
        ),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(sidecar_data, f, indent=2)
    print(f"[+] Saved capture metadata to {meta_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Capture AD8232 Arduino Uno serial stream to CSV.")
    parser.add_argument("--port", type=str, required=True, help="Serial port (e.g. /dev/ttyACM0 or COM3)")
    parser.add_argument("--baud", type=int, default=BAUD_RATE, help=f"Baud rate (default: {BAUD_RATE})")
    parser.add_argument("--duration", type=float, default=None, help="Capture duration in seconds")
    parser.add_argument("--samples", type=int, default=None, help="Maximum number of samples")
    parser.add_argument("--output", type=str, default="hardware/simulation/captured_stream.csv", help="Output CSV path")

    args = parser.parse_args()

    try:
        capture_serial_to_file(
            port=args.port,
            output_path=Path(args.output),
            baud_rate=args.baud,
            duration_seconds=args.duration,
            max_samples=args.samples,
        )
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
