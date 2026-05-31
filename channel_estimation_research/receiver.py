#!/usr/bin/env python3
"""
Audio Modem – Receiver
======================
Parses a recorded WAV, synchronises using the chirp + pilot OFDM structure,
demodulates QPSK symbols, separates pilot and data symbols, and saves
all relevant quantities to an NPZ file for the Analyser.

Usage:
  python receiver.py <recorded.wav> [options]

Options:
  --meta         path to the _meta.npz sidecar from the transmitter
  --output       basename for output files (default: received)
  --no-eq        skip one-tap equalisation (for debugging)

Output files:
  received.npz   – all demodulation data for the analyser
"""

import argparse
import wave
import sys
import numpy as np
from scipy.signal import correlate, firwin, lfilter

# ── Shared OFDM constants (must match transmitter) ────────────────────────────
N_FFT  = 1024
N_CP   = 1024
N_SYM  = N_FFT + N_CP   # 2048

DATA_CARRIERS = np.arange(1, N_FFT // 2)
N_DATA_SC     = len(DATA_CARRIERS)   # 511

QPSK_NORM = 1 / np.sqrt(2)
QPSK_CONSTELLATION = np.array([1+1j, -1+1j, -1-1j, 1-1j]) * QPSK_NORM

# ── WAV utilities ─────────────────────────────────────────────────────────────

def load_wav(filename: str):
    with wave.open(filename, 'r') as wf:
        fs      = wf.getframerate()
        nframes = wf.getnframes()
        nch     = wf.getnchannels()
        sw      = wf.getsampwidth()
        raw     = wf.readframes(nframes)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    pcm   = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if nch == 2:
        pcm = pcm[::2] + pcm[1::2]   # mix to mono
    pcm /= (2.0 ** (8 * sw - 1))  # Normalize to [-1, 1] using proper bit depth
    print(f"[RX] Loaded {filename}: {nframes} frames @ {fs} Hz  ({nframes/fs:.2f} s)")
    return pcm, fs


# ── OFDM symbol helpers ───────────────────────────────────────────────────────

def extract_ofdm_symbol(signal: np.ndarray, offset: int) -> np.ndarray:
    """Extract one OFDM symbol (strip CP, FFT)."""
    frame = signal[offset: offset + N_SYM]
    data  = frame[N_CP:]                    # strip CP
    return np.fft.fft(data) / np.sqrt(N_FFT)


def mls_sequence(length: int, seed: int = 1) -> np.ndarray:
    n_bits = 15
    state  = seed & ((1 << n_bits) - 1) or 1
    out    = []
    for _ in range(length):
        bit   = ((state >> 14) ^ (state >> 13)) & 1
        state = ((state << 1) | bit) & ((1 << n_bits) - 1)
        out.append(1 if bit else -1)
    return np.array(out, dtype=float)


def generate_pilot_sequence(n_carriers: int, pilot_type: str, seed: int) -> np.ndarray:
    if pilot_type == 'mls':
        real_part = mls_sequence(n_carriers, seed=seed)
        imag_part = mls_sequence(n_carriers, seed=seed + 1)
    else:
        rng       = np.random.default_rng(seed)
        real_part = rng.choice([-1.0, 1.0], size=n_carriers)
        imag_part = rng.choice([-1.0, 1.0], size=n_carriers)
    return (real_part + 1j * imag_part) / np.sqrt(2)


# ── Synchronisation ───────────────────────────────────────────────────────────

def generate_reference_chirp(duration_s: float, f0: float, f1: float, fs: int) -> np.ndarray:
    t   = np.arange(int(duration_s * fs)) / fs
    k   = (f1 - f0) / duration_s
    sig = np.sin(2 * np.pi * (f0 * t + 0.5 * k * t**2))
    fade = int(0.02 * fs)
    win  = np.ones_like(sig)
    win[:fade]  = np.linspace(0, 1, fade)
    win[-fade:] = np.linspace(1, 0, fade)
    return sig * win


def coarse_chirp_sync(signal: np.ndarray, fs: int) -> int:
    """Cross-correlate with reference chirp to find frame start."""
    ref   = generate_reference_chirp(0.3, 100, 8000, fs)
    # Downsample by 4 for speed
    ds    = 4
    sig_d = signal[::ds]
    ref_d = ref[::ds]
    xc    = correlate(sig_d, ref_d, mode='full')
    peak  = np.argmax(np.abs(xc)) - len(ref_d) + 1
    coarse = peak * ds
    print(f"[RX] Chirp sync: estimated start sample = {coarse}")
    return max(0, coarse)


def fine_cp_sync(signal: np.ndarray, start: int, search_range: int = 512) -> int:
    """Fine timing via CP correlation near the expected pilot-block position."""
    # We're looking for the start of the first pilot block
    # Expected offset after chirp (0.3 s) + guard (1024 sa)
    best_score = -np.inf
    best_off   = start
    for delta in range(-search_range, search_range):
        off   = start + delta
        if off + N_SYM > len(signal):
            break
        frame = signal[off: off + N_SYM]
        cp    = frame[:N_CP]
        body  = frame[N_SYM - N_CP:]
        score = np.real(np.dot(cp, np.conj(body)))
        if score > best_score:
            best_score = score
            best_off   = off
    print(f"[RX] Fine CP sync: refined start = {best_off} (delta={best_off - start})")
    return best_off


def estimate_cfo(signal: np.ndarray, symbol_offset: int) -> float:
    """Carrier Frequency Offset estimation from CP correlation."""
    frame  = signal[symbol_offset: symbol_offset + N_SYM]
    cp     = frame[:N_CP]
    body   = frame[N_SYM - N_CP:]
    angle  = np.angle(np.dot(body, np.conj(cp)))
    cfo    = angle / (2 * np.pi)   # normalised CFO
    print(f"[RX] CFO estimate: {cfo:.6f} (normalised, bins)")
    return cfo


def correct_cfo(signal: np.ndarray, cfo_norm: float, fs: int) -> np.ndarray:
    """Apply CFO correction to the entire signal."""
    n   = np.arange(len(signal))
    return (signal * np.exp(-1j * 2 * np.pi * cfo_norm * n / N_FFT)).real


# ── Main receiver logic ───────────────────────────────────────────────────────

def receive(args):
    signal, fs = load_wav(args.recorded_wav)

    # Load meta
    meta = np.load(args.meta, allow_pickle=True)
    pilot_type      = str(meta['pilot_type'][0])
    seed            = int(meta['seed'][0])
    M               = int(meta['pilot_blocks'][0])
    K               = int(meta['pilot_every'][0])
    total_data_blks = int(meta['total_data_blks'][0])
    pilots_fd       = meta['pilots_fd']

    print(f"[RX] Meta: pilot_type={pilot_type} seed={seed} M={M} K={K} data_blks={total_data_blks}")

    # ── Coarse sync (chirp) ───────────────────────────────────────────────
    chirp_len   = int(0.3 * fs)
    guard_len   = 1024
    coarse_start = coarse_chirp_sync(signal, fs)

    # Expected start of first pilot block
    # coarse_start points to beginning of chirp; add chirp + guard
    pilot_start_est = coarse_start + chirp_len + guard_len

    # Fine sync
    pilot_start = fine_cp_sync(signal, pilot_start_est, search_range=2056)

    # CFO estimation from first pilot block
    cfo = estimate_cfo(signal, pilot_start)
    if abs(cfo) > 1e-6:
        signal = correct_cfo(signal, cfo, fs)
        print("[RX] CFO correction applied")

    # ── Extract pilot blocks ──────────────────────────────────────────────
    pilot_rxd_fd = []
    cur = pilot_start
    for i in range(M):
        fd = extract_ofdm_symbol(signal, cur)
        pilot_rxd_fd.append(fd[DATA_CARRIERS])
        cur += N_SYM
    pilot_rxd_fd = np.array(pilot_rxd_fd)            # (M, N_DATA_SC)
    print(f"[RX] Extracted {M} pilot blocks")

    # Average channel estimate from leading pilots (for reference)
    H_est_init = np.mean(pilot_rxd_fd / pilots_fd[np.newaxis, :], axis=0)

    # ── Extract data + interleaved pilots ─────────────────────────────────
    data_syms_raw   = []   # demodulated (possibly equalized) QPSK on data subcarriers
    pilot_syms_raw  = []   # received pilot subcarriers for per-block channel tracking
    pilot_fd_ref    = []   # reference pilots at same positions
    inter_pilot_positions = []   # which block indices (in data_stream) are interleaved pilots

    H_track = H_est_init.copy()
    blk_idx = 0   # counts data blocks

    while blk_idx < total_data_blks and (cur + N_SYM) <= len(signal):
        # Insert a pilot update every K data blocks
        if blk_idx > 0 and blk_idx % K == 0:
            fd_p = extract_ofdm_symbol(signal, cur)
            p_rx = fd_p[DATA_CARRIERS]
            # Update channel estimate
            H_track = p_rx / pilots_fd
            pilot_syms_raw.append(p_rx)
            pilot_fd_ref.append(pilots_fd.copy())
            inter_pilot_positions.append(blk_idx)
            cur += N_SYM

        if cur + N_SYM > len(signal):
            break

        fd   = extract_ofdm_symbol(signal, cur)
        d_rx = fd[DATA_CARRIERS]

        if not args.no_eq:
            d_eq = d_rx / H_track
        else:
            d_eq = d_rx

        data_syms_raw.append(d_eq)
        blk_idx += 1
        cur += N_SYM

    actual_blks = len(data_syms_raw)
    print(f"[RX] Extracted {actual_blks} data blocks ({blk_idx} expected)")

    data_syms_raw  = np.array(data_syms_raw)    # (blks, N_DATA_SC)

    # ── QPSK Hard Decision ────────────────────────────────────────────────
    all_syms = data_syms_raw.reshape(-1)
    dists    = np.abs(all_syms[:, np.newaxis] - QPSK_CONSTELLATION[np.newaxis, :])
    qpsk_idx = np.argmin(dists, axis=1)
    decided  = QPSK_CONSTELLATION[qpsk_idx]

    # ── Bit recovery ──────────────────────────────────────────────────────
    # Gray decode: 0→00, 1→01, 2→11, 3→10
    gray_decode = {0: (0,0), 1: (0,1), 2: (1,1), 3: (1,0)}
    bits = []
    for idx in qpsk_idx:
        bits.extend(gray_decode[idx])
    bits = np.array(bits, dtype=np.uint8)

    # ── Reconstruct text ──────────────────────────────────────────────────
    # Bits → bytes
    n_bytes = len(bits) // 8
    byte_arr = np.packbits(bits[:n_bytes * 8])
    # First 4 bytes = length header
    if len(byte_arr) >= 4:
        payload_len = int.from_bytes(byte_arr[:4], 'big')
        text_bytes  = byte_arr[4: 4 + payload_len]
    else:
        payload_len = 0
        text_bytes  = byte_arr

    print(f"[RX] Recovered payload: {payload_len} bytes declared, {len(text_bytes)} extracted")

    # ── Save NPZ ──────────────────────────────────────────────────────────
    out_npz = args.output + '.npz'
    np.savez(out_npz,
             # Raw received OFDM symbols (complex, all sub-carriers)
             data_syms_raw       = data_syms_raw,          # (blks, 511)
             all_syms_eq         = all_syms,                # flat equalized symbols
             decided_syms        = decided,                 # flat hard-decided symbols
             # Pilot data
             pilot_rxd_fd        = pilot_rxd_fd,            # leading pilots
             pilot_syms_raw      = np.array(pilot_syms_raw) if pilot_syms_raw else np.zeros((0,N_DATA_SC), dtype=complex),
             pilot_fd_ref        = np.array(pilot_fd_ref)   if pilot_fd_ref   else np.zeros((0,N_DATA_SC), dtype=complex),
             pilots_fd_tx        = pilots_fd,               # transmitted pilot sequence
             H_est               = H_est_init,              # initial channel estimate
             # Metadata
             pilot_type          = np.array([pilot_type]),
             seed                = np.array([seed]),
             M                   = np.array([M]),
             K                   = np.array([K]),
             total_data_blks     = np.array([total_data_blks]),
             actual_data_blks    = np.array([actual_blks]),
             samplerate          = np.array([fs]),
             # Recovered payload
             payload_len         = np.array([payload_len]),
             text_bytes          = text_bytes,
             bits                = bits,
             # CFO
             cfo                 = np.array([cfo]),
             )
    print(f"[RX] Results saved → {out_npz}")

    # Quick text preview
    try:
        preview = text_bytes.tobytes().decode('utf-8', errors='replace')[:200]
        print(f"[RX] Text preview: {repr(preview)}")
    except Exception:
        pass


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Audio Modem Receiver (OFDM/QPSK)')
    p.add_argument('recorded_wav',              help='Recorded WAV file')
    p.add_argument('--meta',     default=None,  help='Transmitter metadata NPZ (default: auto-detect)')
    p.add_argument('--output',   default='received', help='Output basename (default: received)')
    p.add_argument('--no-eq',    action='store_true', help='Disable one-tap equalisation')
    args = p.parse_args()

    if args.meta is None:
        # Auto-detect: same base name as wav but _meta.npz
        guess = args.recorded_wav.replace('.wav', '').replace('recorded', 'transmitted') + '_meta.npz'
        import os
        if os.path.exists(guess):
            args.meta = guess
            print(f"[RX] Auto-detected meta: {guess}")
        else:
            # try in same directory
            args.meta = 'transmitted_meta.npz'
            print(f"[RX] Using default meta path: {args.meta}")

    receive(args)


if __name__ == '__main__':
    main()
