#!/usr/bin/env python3
"""
Audio Modem – Recorder
======================
Records audio from the default microphone and saves it as a WAV file.
Designed to passively capture a transmission played from a phone or speaker.

Usage:
  python recorder.py [options]

Options:
  --duration    seconds to record (default: 30; use 0 for press-Enter-to-stop)
  --samplerate  Hz (default: 48000)
  --channels    1 or 2 (default: 1, mono)
  --output      filename.wav (default: recorded.wav)
  --list-devices  show available audio devices and exit

The recorder normalises the captured audio so the loudest peak = 0.9.
"""

import argparse
import sys
import wave
import struct
import array
import time
import threading
import numpy as np

# ── Try to import sounddevice; fall back to a pyaudio attempt ───────────────
_SD_AVAILABLE = False
_PA_AVAILABLE = False

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except Exception:
    pass

try:
    import pyaudio
    _PA_AVAILABLE = True
except Exception:
    pass


def save_wav(filename: str, samples: np.ndarray, samplerate: int, channels: int = 1):
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(pcm.tobytes())
    duration = len(samples) / (samplerate * channels)
    print(f"[REC] Saved {filename}  ({duration:.2f} s, {len(pcm)} frames)")


def record_sounddevice(args):
    """Record using sounddevice (preferred)."""
    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    fs       = args.samplerate
    channels = args.channels

    if args.duration > 0:
        print(f"[REC] Recording for {args.duration} s at {fs} Hz …  (Ctrl+C to abort)")
        audio = sd.rec(int(args.duration * fs), samplerate=fs,
                       channels=channels, dtype='float32')
        sd.wait()
    else:
        print(f"[REC] Press Enter to START recording …")
        input()
        print("[REC] Recording … press Enter to STOP")
        buf   = []
        stop  = threading.Event()

        def callback(indata, frames, time_info, status):
            buf.append(indata.copy())

        with sd.InputStream(samplerate=fs, channels=channels,
                            dtype='float32', callback=callback):
            input()
        stop.set()
        audio = np.concatenate(buf, axis=0)

    data = audio if channels == 1 else audio[:, 0]
    data = data.flatten()

    peak = np.max(np.abs(data))
    if peak > 0:
        data = data * (0.9 / peak)
    print(f"[REC] Peak amplitude (normalised): {np.max(np.abs(data)):.3f}")
    save_wav(args.output, data, fs, channels)


def record_pyaudio(args):
    """Record using PyAudio fallback."""
    pa      = pyaudio.PyAudio()
    fs      = args.samplerate
    chunk   = 1024
    fmt     = pyaudio.paInt16
    channels= args.channels

    if args.list_devices:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            print(f"  [{i}] {info['name']}  (in={info['maxInputChannels']} out={info['maxOutputChannels']})")
        pa.terminate()
        sys.exit(0)

    stream = pa.open(format=fmt, channels=channels, rate=fs,
                     input=True, frames_per_buffer=chunk)

    frames = []
    if args.duration > 0:
        n_chunks = int(fs / chunk * args.duration)
        print(f"[REC] Recording for {args.duration} s …")
        for _ in range(n_chunks):
            frames.append(stream.read(chunk))
    else:
        print("[REC] Press Enter to START recording …")
        input()
        print("[REC] Recording … press Enter to STOP")
        stop_flag = [False]

        def stopper():
            input()
            stop_flag[0] = True

        t = threading.Thread(target=stopper, daemon=True)
        t.start()
        while not stop_flag[0]:
            frames.append(stream.read(chunk, exception_on_overflow=False))

    stream.stop_stream()
    stream.close()
    pa.terminate()

    raw   = b''.join(frames)
    pcm   = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    peak  = np.max(np.abs(pcm))
    if peak > 0:
        pcm = pcm * (0.9 / peak)
    save_wav(args.output, pcm, fs, channels)


def record_fallback(args):
    """Pure-stdlib fallback using os-level arecord (Linux) or afrecord (macOS)."""
    import subprocess, os, platform

    fs       = args.samplerate
    channels = args.channels
    duration = args.duration if args.duration > 0 else 30
    tmp      = '/tmp/_modem_rec_raw.wav'

    plat = platform.system()
    if plat == 'Linux':
        cmd = ['arecord', '-f', 'S16_LE', '-r', str(fs),
               '-c', str(channels), '-d', str(duration), tmp]
    elif plat == 'Darwin':
        cmd = ['rec', '-r', str(fs), '-c', str(channels), '-b', '16',
               tmp, 'trim', '0', str(duration)]
    else:
        print("[REC] ERROR: No audio recording backend found.")
        print("      Install sounddevice or pyaudio:")
        print("        pip install sounddevice   (requires PortAudio)")
        print("        pip install pyaudio")
        sys.exit(1)

    print(f"[REC] Recording via system command for {duration} s …")
    print(f"      Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Read the saved WAV back
    with wave.open(tmp, 'r') as wf:
        raw = wf.readframes(wf.getnframes())
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        actual_fs = wf.getframerate()

    os.remove(tmp)
    peak = np.max(np.abs(pcm))
    if peak > 0:
        pcm = pcm * (0.9 / peak)
    save_wav(args.output, pcm, actual_fs, channels)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Audio Modem Recorder')
    p.add_argument('--duration',     default=30,    type=float,
                   help='Recording duration in seconds (0 = press-Enter-to-stop, default: 30)')
    p.add_argument('--samplerate',   default=48000, type=int,
                   help='Sample rate Hz (default: 48000)')
    p.add_argument('--channels',     default=1,     type=int,   choices=[1, 2],
                   help='Channels 1=mono 2=stereo (default: 1)')
    p.add_argument('--output',       default='recorded.wav',
                   help='Output WAV filename (default: recorded.wav)')
    p.add_argument('--list-devices', action='store_true',
                   help='List audio input devices and exit')
    args = p.parse_args()

    print("─" * 60)
    print("  Audio Modem Recorder")
    print(f"  Sample rate : {args.samplerate} Hz")
    print(f"  Channels    : {args.channels}")
    print(f"  Duration    : {args.duration if args.duration > 0 else 'until Enter'} s")
    print(f"  Output      : {args.output}")
    print("─" * 60)

    if _SD_AVAILABLE:
        print("[REC] Backend: sounddevice")
        record_sounddevice(args)
    elif _PA_AVAILABLE:
        print("[REC] Backend: PyAudio")
        record_pyaudio(args)
    else:
        print("[REC] Backend: system fallback (arecord / rec)")
        record_fallback(args)


if __name__ == '__main__':
    main()
