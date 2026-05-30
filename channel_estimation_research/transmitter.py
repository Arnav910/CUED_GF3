#!/usr/bin/env python3
"""
Audio Modem – Transmitter
=========================
Modulates a text file into a WAV using OFDM/QPSK with Gray coding.

Frame structure (time-domain):
  [chirp] | [guard interval] | [pilot block × M (each with CP)] | [data blocks (with CP, pilots every K blocks)]

OFDM parameters
  N_FFT  = 1024   (without CP)
  N_CP   = 1024   (cyclic prefix length → total symbol = 2048 samples)

Usage:
  python transmitter.py <input_text_file> [options]

Options:
  --pilot-type   mls | noise   (default: mls)
  --seed         integer seed for noise pilots (default: 42)
  --pilot-blocks M             (default: 4)
  --pilot-every  K             (default: 8)
  --samplerate   Hz            (default: 48000)
  --output       filename.wav  (default: transmitted.wav)
  --amplitude    0–1           (default: 0.7)
"""

import argparse
import wave
import struct
import sys
import numpy as np

# ── Fixed OFDM Parameters ───────────────────────────────────────────────────
N_FFT   = 1024          # OFDM sub-carriers
N_CP    = 1024          # cyclic prefix length
N_SYM   = N_FFT + N_CP  # total samples per OFDM symbol = 2048

# Sub-carrier allocation: leave DC and Nyquist null
DATA_CARRIERS = np.arange(1, N_FFT // 2)          # indices 1..511  (511 carriers)
N_DATA_SC     = len(DATA_CARRIERS)                 # 511

# QPSK Gray-coded constellation
QPSK_MAP = {
    (0, 0): ( 1+1j),
    (0, 1): (-1+1j),
    (1, 1): (-1-1j),
    (1, 0): ( 1-1j),
}
QPSK_MAP_INV = {v: k for k, v in QPSK_MAP.items()}
NORM         = 1 / np.sqrt(2)   # normalise to unit power

# ── Helper utilities ─────────────────────────────────────────────────────────

def bits_to_qpsk(bits: np.ndarray) -> np.ndarray:
    """Convert array of bits (length must be even) to QPSK symbols."""
    assert len(bits) % 2 == 0
    pairs   = bits.reshape(-1, 2)
    symbols = np.array([QPSK_MAP[tuple(p)] for p in pairs], dtype=complex)
    return symbols * NORM


def build_ofdm_symbol(freq_domain: np.ndarray) -> np.ndarray:
    """IFFT + add cyclic prefix.  freq_domain length must equal N_FFT."""
    assert len(freq_domain) == N_FFT
    td  = np.fft.ifft(freq_domain) * np.sqrt(N_FFT)     # normalised IFFT
    td  = td.real                                          # take real part
    cp  = td[-N_CP:]
    return np.concatenate([cp, td])


def mls_sequence(length: int, seed: int = 1) -> np.ndarray:
    """Generate a ±1 MLS-like PRBS using a linear feedback shift register."""
    # 15-bit LFSR (taps 15, 14 → period 32767)
    n_bits = 15
    state  = seed & ((1 << n_bits) - 1) or 1
    out    = []
    for _ in range(length):
        bit   = ((state >> 14) ^ (state >> 13)) & 1
        state = ((state << 1) | bit) & ((1 << n_bits) - 1)
        out.append(1 if bit else -1)
    return np.array(out, dtype=float)


def generate_pilot_sequence(n_carriers: int, pilot_type: str, seed: int) -> np.ndarray:
    """Return complex pilot symbols of unit power for DATA_CARRIERS."""
    if pilot_type == 'mls':
        real_part = mls_sequence(n_carriers, seed=seed)
        imag_part = mls_sequence(n_carriers, seed=seed + 1)
    else:  # white noise
        rng       = np.random.default_rng(seed)
        real_part = rng.choice([-1.0, 1.0], size=n_carriers)
        imag_part = rng.choice([-1.0, 1.0], size=n_carriers)
    pilots = (real_part + 1j * imag_part) / np.sqrt(2)
    return pilots


def pack_freq_domain(symbols: np.ndarray) -> np.ndarray:
    """Place symbols on positive sub-carriers, conjugate-mirror for real IFFT."""
    X      = np.zeros(N_FFT, dtype=complex)
    X[DATA_CARRIERS]             = symbols
    X[N_FFT - DATA_CARRIERS]     = np.conj(symbols)   # Hermitian symmetry
    return X


def chirp_signal(duration_s: float, f0: float, f1: float, fs: int) -> np.ndarray:
    """Linear chirp from f0 to f1 over duration_s seconds."""
    t   = np.arange(int(duration_s * fs)) / fs
    k   = (f1 - f0) / duration_s
    sig = np.sin(2 * np.pi * (f0 * t + 0.5 * k * t**2))
    # raised-cosine fade in/out (20 ms)
    fade = int(0.02 * fs)
    win  = np.ones_like(sig)
    win[:fade]  = np.linspace(0, 1, fade)
    win[-fade:] = np.linspace(1, 0, fade)
    return sig * win


def save_wav(filename: str, samples: np.ndarray, samplerate: int):
    """Save float32 array (range ±1) as 16-bit PCM WAV."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(pcm.tobytes())
    print(f"[TX] Saved {filename}  ({len(samples)/samplerate:.2f} s, {len(samples)} samples)")


# ── Main transmitter logic ────────────────────────────────────────────────────

def transmit(args):
    # ── 1. Read and encode text file ──────────────────────────────────────
    with open(args.input, 'rb') as f:
        raw = f.read()
    print(f"[TX] Input file: {args.input}  ({len(raw)} bytes)")

    # Prepend 4-byte big-endian length header so receiver can strip padding
    length_bytes = len(raw).to_bytes(4, 'big')
    payload      = length_bytes + raw

    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    # Pad to multiple of 2 × N_DATA_SC (one full OFDM symbol worth of QPSK)
    bits_per_symbol = 2 * N_DATA_SC
    remainder = len(bits) % bits_per_symbol
    if remainder:
        bits = np.concatenate([bits, np.zeros(bits_per_symbol - remainder, dtype=np.uint8)])

    total_data_symbols = len(bits) // bits_per_symbol
    print(f"[TX] Bits: {len(bits)}  →  {total_data_symbols} OFDM data blocks")

    # ── 2. Generate pilot sequence ────────────────────────────────────────
    pilots = generate_pilot_sequence(N_DATA_SC, args.pilot_type, args.seed)
    print(f"[TX] Pilot type: {args.pilot_type}  seed={args.seed}")

    # ── 3. Build frame ────────────────────────────────────────────────────
    fs   = args.samplerate
    amp  = args.amplitude
    out  = []

    # 3a  Chirp (0.3 s, 100 → 8000 Hz)
    chirp = chirp_signal(0.3, 100, 8000, fs) * amp
    out.append(chirp)
    print(f"[TX] Chirp: {len(chirp)} samples")

    # 3b  Guard interval (1024 samples of silence)
    guard = np.zeros(1024)
    out.append(guard)

    # 3c  Pilot OFDM blocks × M
    pilot_td = build_ofdm_symbol(pack_freq_domain(pilots)) * amp
    for _ in range(args.pilot_blocks):
        out.append(pilot_td)
    print(f"[TX] Pilot blocks: {args.pilot_blocks}  each={len(pilot_td)} samples")

    # 3d  Data OFDM blocks (pilots inserted every K blocks)
    for blk_idx in range(total_data_symbols):
        if blk_idx % args.pilot_every == 0 and blk_idx > 0:
            out.append(pilot_td)      # periodic pilot

        bit_start = blk_idx * bits_per_symbol
        blk_bits  = bits[bit_start: bit_start + bits_per_symbol]
        qpsk_syms = bits_to_qpsk(blk_bits)
        X         = pack_freq_domain(qpsk_syms)
        td        = build_ofdm_symbol(X) * amp
        out.append(td)

    waveform = np.concatenate(out)
    print(f"[TX] Total frame: {len(waveform)} samples  ({len(waveform)/fs:.2f} s)")

    # ── 4. Save metadata sidecar (needed by receiver) ─────────────────────
    meta_file = args.output.replace('.wav', '_meta.npz')
    np.savez(meta_file,
             pilot_type      = np.array([args.pilot_type]),
             seed            = np.array([args.seed]),
             pilot_blocks    = np.array([args.pilot_blocks]),
             pilot_every     = np.array([args.pilot_every]),
             total_data_blks = np.array([total_data_symbols]),
             samplerate      = np.array([fs]),
             pilots_fd       = pilots)
    print(f"[TX] Metadata → {meta_file}")

    # ── 5. Save WAV ───────────────────────────────────────────────────────
    save_wav(args.output, waveform, fs)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Audio Modem Transmitter (OFDM/QPSK)')
    p.add_argument('input',                                   help='Input text file')
    p.add_argument('--pilot-type',  default='mls',            choices=['mls','noise'],
                   help='Pilot sequence type (default: mls)')
    p.add_argument('--seed',        default=42,  type=int,    help='PRNG seed (default: 42)')
    p.add_argument('--pilot-blocks',default=4,   type=int,    help='Number of leading pilot blocks M (default: 4)')
    p.add_argument('--pilot-every', default=8,   type=int,    help='Insert pilot every K data blocks (default: 8)')
    p.add_argument('--samplerate',  default=48000, type=int,  help='Sample rate Hz (default: 48000)')
    p.add_argument('--output',      default='transmitted.wav',help='Output WAV file (default: transmitted.wav)')
    p.add_argument('--amplitude',   default=0.7, type=float,  help='Output amplitude 0–1 (default: 0.7)')
    args = p.parse_args()
    transmit(args)

if __name__ == '__main__':
    main()
