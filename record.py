
import argparse
import os

import numpy as np
import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 48_000


def record(file_name: str = "rx.wav", duration: float = 7.0) -> str:
    # Always save next to this script, not wherever Python was launched from
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "received_signals")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)

    n_samples = int(duration * SAMPLE_RATE)

    print(f"Recording {duration:.1f} s at {SAMPLE_RATE} Hz")
    print(f"Saving to: {output_path}")
    print("Press Ctrl+C to stop early.")

    try:
        audio = sd.rec(
            n_samples,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        print("\nRecording stopped early.")

    audio = audio[:, 0].astype(np.float32)

    peak = np.max(np.abs(audio))
    print(f"Peak amplitude: {peak:.4f}  "
          f"({'OK' if peak < 0.95 else 'WARNING: close to clipping!'})")

    sf.write(output_path, audio, SAMPLE_RATE, subtype="FLOAT")
    print(f"Saved {len(audio)} samples ({len(audio)/SAMPLE_RATE:.2f} s)")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Record audio for JOSS-F decoding.")
    parser.add_argument("file_name", nargs="?", default="rx.wav",
                        help="Output filename (default: rx.wav)")
    parser.add_argument("--duration", type=float, default=7.0,
                        help="Recording duration in seconds (default: 20)")
    args = parser.parse_args()
    record(args.file_name, args.duration)


if __name__ == "__main__":
    main()
