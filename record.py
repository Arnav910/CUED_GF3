"""
record.py  —  JOSS-F acoustic recorder.

Always saves to the received_signals folder next to this script.

Usage:
    python record.py                        # records until Ctrl+C
    python record.py rx.wav                 # custom filename
    python record.py rx.wav --duration 15   # stop after 15 seconds
"""

import argparse
import os
import sys

import numpy as np
import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 48_000


def record(file_name: str = "rx.wav", duration: float = None) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # If file_name is already an absolute path or contains a directory separator,
    # use it directly. Otherwise save into received_signals/ next to this script.
    if os.path.isabs(file_name) or os.sep in file_name or '/' in file_name:
        output_path = os.path.normpath(file_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        output_dir = os.path.join(script_dir, "received_signals")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, file_name)

    if duration:
        print(f"Recording {duration:.1f} s at {SAMPLE_RATE} Hz  (Ctrl+C to stop early)")
    else:
        print(f"Recording at {SAMPLE_RATE} Hz  (press Ctrl+C to stop)")
    print(f"Saving to: {output_path}")

    chunks = []

    def callback(indata, frames, time, status):
        if status:
            print(f"  [audio status: {status}]", file=sys.stderr)
        chunks.append(indata[:, 0].copy())

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="float32", callback=callback):
            if duration:
                import time
                end_time = time.time() + duration
                while time.time() < end_time:
                    time.sleep(0.1)
            else:
                import time
                print("Recording... press Ctrl+C to stop.")
                while True:
                    time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nRecording stopped.")

    if not chunks:
        print("No audio recorded.")
        return output_path

    audio = np.concatenate(chunks).astype(np.float32)
    peak = np.max(np.abs(audio))
    print(f"Peak amplitude: {peak:.4f}  "
          f"({'OK' if peak < 0.95 else 'WARNING: close to clipping!'})")

    sf.write(output_path, audio, SAMPLE_RATE, subtype="FLOAT")
    print(f"Saved {len(audio)} samples ({len(audio)/SAMPLE_RATE:.2f} s)  →  {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Record audio for JOSS-F decoding.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python record.py                    # record until Ctrl+C\n"
            "  python record.py rx.wav             # custom filename, stop with Ctrl+C\n"
            "  python record.py rx.wav --duration 15  # stop after 15 s\n"
        )
    )
    parser.add_argument("file_name", nargs="?", default="rx.wav",
                        help="Output filename inside received_signals/ (default: rx.wav)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Recording duration in seconds (omit to record until Ctrl+C)")
    args = parser.parse_args()
    record(args.file_name, args.duration)


if __name__ == "__main__":
    main()
