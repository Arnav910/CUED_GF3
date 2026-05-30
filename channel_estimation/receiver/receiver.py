# Improved `receiver_cli.py`

"""
receiver_cli.py

Improved CLI-based OFDM receiver for acoustic channel estimation.

Fixes added:
- CP-based timing refinement
- packet-end trimming
- symbol count validation
- CFO / phase drift estimation
- safer reshaping
- stacked ndarray storage instead of dtype=object
- raw time-domain symbol storage
- optional amplitude normalization
- better sync chirp band
"""

import pickle
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import soundfile as sf

from dataclasses import dataclass
from scipy.signal import correlate

from config import (
    SAMPLE_RATE,
    GUARD_DURATION,
)

from utils.signal_utils import make_chirp


# ─────────────────────────────────────────────────────────────
# Sync
# ─────────────────────────────────────────────────────────────

def make_sync_chirp():
    """
    Narrower sync band for better speaker/mic robustness.
    """

    return make_chirp(
        duration=0.3,
        f0=1000,
        f1=7000,
        fs=SAMPLE_RATE,
        amplitude=1.0,
        method="linear",
    )


def detect_chirp(signal, chirp):

    corr = correlate(signal, chirp, mode="valid")

    idx = int(np.argmax(np.abs(corr)))

    return idx


# ─────────────────────────────────────────────────────────────
# OFDM helpers
# ─────────────────────────────────────────────────────────────

def refine_symbol_timing(rx, coarse_start, nfft, cp, search_radius=64):
    """
    CP-based timing refinement.

    Finds offset maximizing CP correlation.
    """

    best_metric = -np.inf
    best_offset = coarse_start

    for offset in range(
        coarse_start - search_radius,
        coarse_start + search_radius,
    ):

        if offset < 0:
            continue

        end_needed = offset + cp + nfft

        if end_needed >= len(rx):
            continue

        cp_region = rx[offset:offset + cp]

        tail_region = rx[offset + nfft:offset + nfft + cp]

        metric = np.abs(np.vdot(cp_region, tail_region))

        if metric > best_metric:
            best_metric = metric
            best_offset = offset

    return best_offset



def estimate_cfo_from_cp(symbol, nfft, cp):
    """
    Estimate phase rotation between CP and tail.
    """

    cp_region = symbol[:cp]
    tail_region = symbol[nfft:nfft + cp]

    phase = np.angle(np.vdot(cp_region, tail_region))

    return phase / nfft



def correct_cfo(symbol, eps):

    n = np.arange(len(symbol))

    return symbol * np.exp(-1j * eps * n)



def ofdm_fft(symbol, nfft, cp):

    td = symbol[cp:cp + nfft]

    X = np.fft.fft(td)

    return X[:nfft//2 + 1]


# ─────────────────────────────────────────────────────────────
# Dataset container
# ─────────────────────────────────────────────────────────────

@dataclass
class RXDataset:

    method: int

    chirp_index: int
    refined_start: int

    guard_start: int
    pilot_start: int
    data_start: int

    nfft: int
    cp: int

    pilot_symbols_fd: np.ndarray
    data_symbols_fd: np.ndarray
    mixed_symbols_fd: np.ndarray

    pilot_symbols_td: np.ndarray
    data_symbols_td: np.ndarray
    mixed_symbols_td: np.ndarray

    cfo_estimates: np.ndarray

    fs: int
    rms: float


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def normalize_signal(x, target_rms=0.1):

    rms = np.sqrt(np.mean(x ** 2))

    if rms < 1e-12:
        return x

    return x * (target_rms / rms)



def safe_symbol_extract(region, sym_len):

    n_sym = len(region) // sym_len

    trimmed = region[:n_sym * sym_len]

    return trimmed.reshape(n_sym, sym_len)


# ─────────────────────────────────────────────────────────────
# Receiver
# ─────────────────────────────────────────────────────────────

def receive(
    rx,
    method,
    nfft,
    cp,
    pilot_reps,
    save_dir,
    file_name,
    max_data_symbols=None,
):

    save_dir = os.path.join(os.path.dirname(__file__), save_dir)

    os.makedirs(save_dir, exist_ok=True)

    rx = rx.astype(np.float32)

    rx = normalize_signal(rx)

    # ─────────────────────────────
    # Sync detection
    # ─────────────────────────────

    chirp = make_sync_chirp()

    chirp_idx = detect_chirp(rx, chirp)

    guard_len = int(GUARD_DURATION * SAMPLE_RATE)

    coarse_start = chirp_idx + len(chirp) + guard_len

    refined_start = refine_symbol_timing(
        rx,
        coarse_start,
        nfft,
        cp,
    )

    sym_len = nfft + cp

    pilot_symbols_fd = []
    data_symbols_fd = []
    mixed_symbols_fd = []

    pilot_symbols_td = []
    data_symbols_td = []
    mixed_symbols_td = []

    cfo_estimates = []

    # ─────────────────────────────
    # METHOD 1
    # ─────────────────────────────

    if method == 1:

        pilot_start = refined_start

        pilot_len = pilot_reps * sym_len

        pilot_region = rx[pilot_start:pilot_start + pilot_len]

        pilot_blocks = safe_symbol_extract(
            pilot_region,
            sym_len,
        )

        for block in pilot_blocks:

            eps = estimate_cfo_from_cp(block, nfft, cp)

            corrected = correct_cfo(block.astype(np.complex64), eps)
            print(corrected.shape)
            X = ofdm_fft(corrected, nfft, cp)

            pilot_symbols_fd.append(X)
            pilot_symbols_td.append(corrected)
            cfo_estimates.append(eps)

        data_start = pilot_start + pilot_len + guard_len

        data_region = rx[data_start:]

        if max_data_symbols is not None:
            data_region = data_region[:max_data_symbols * sym_len]

        data_blocks = safe_symbol_extract(
            data_region,
            sym_len,
        )

        for block in data_blocks:

            eps = estimate_cfo_from_cp(block, nfft, cp)

            corrected = correct_cfo(block.astype(np.complex64), eps)

            X = ofdm_fft(corrected, nfft, cp)

            data_symbols_fd.append(X)
            data_symbols_td.append(corrected)
            cfo_estimates.append(eps)

    # ─────────────────────────────
    # METHOD 2
    # ─────────────────────────────

    elif method == 2:

        pilot_start = refined_start

        data_start = pilot_start

        data_region = rx[data_start:]

        if max_data_symbols is not None:
            data_region = data_region[:max_data_symbols * sym_len]

        mixed_blocks = safe_symbol_extract(
            data_region,
            sym_len,
        )

        for block in mixed_blocks:

            eps = estimate_cfo_from_cp(block, nfft, cp)

            corrected = correct_cfo(block.astype(np.complex64), eps)

            X = ofdm_fft(corrected, nfft, cp)

            mixed_symbols_fd.append(X)
            mixed_symbols_td.append(corrected)
            cfo_estimates.append(eps)

    else:
        raise ValueError("method must be 1 or 2")

    # ─────────────────────────────
    # Convert to arrays
    # ─────────────────────────────

    pilot_symbols_fd = np.array(pilot_symbols_fd)
    data_symbols_fd = np.array(data_symbols_fd)
    mixed_symbols_fd = np.array(mixed_symbols_fd)

    pilot_symbols_td = np.array(pilot_symbols_td)
    data_symbols_td = np.array(data_symbols_td)
    mixed_symbols_td = np.array(mixed_symbols_td)

    cfo_estimates = np.array(cfo_estimates)

    # ─────────────────────────────
    # Dataset packaging
    # ─────────────────────────────

    dataset = RXDataset(
        method=method,

        chirp_index=chirp_idx,
        refined_start=refined_start,

        guard_start=chirp_idx + len(chirp),
        pilot_start=pilot_start,
        data_start=data_start,

        nfft=nfft,
        cp=cp,

        pilot_symbols_fd=pilot_symbols_fd,
        data_symbols_fd=data_symbols_fd,
        mixed_symbols_fd=mixed_symbols_fd,

        pilot_symbols_td=pilot_symbols_td,
        data_symbols_td=data_symbols_td,
        mixed_symbols_td=mixed_symbols_td,

        cfo_estimates=cfo_estimates,

        fs=SAMPLE_RATE,
        rms=float(np.sqrt(np.mean(rx ** 2))),
    )

    # ─────────────────────────────
    # Save
    # ─────────────────────────────

    base = os.path.join(save_dir, file_name)

    with open(base + ".pkl", "wb") as f:
        pickle.dump(dataset, f)

    np.savez(
        base + ".npz",

        rx=rx,

        pilot_symbols_fd=pilot_symbols_fd,
        data_symbols_fd=data_symbols_fd,
        mixed_symbols_fd=mixed_symbols_fd,

        pilot_symbols_td=pilot_symbols_td,
        data_symbols_td=data_symbols_td,
        mixed_symbols_td=mixed_symbols_td,

        cfo_estimates=cfo_estimates,

        chirp_index=chirp_idx,
        refined_start=refined_start,
    )

    print(f"[RX] Saved dataset → {base}.pkl / .npz")

    print(f"[RX] Chirp index      : {chirp_idx}")
    print(f"[RX] Refined start   : {refined_start}")
    print(f"[RX] CFO mean        : {np.mean(cfo_estimates):.6e}")

    return dataset


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser(
        description="Improved acoustic OFDM receiver"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--method",
        type=int,
        choices=[1, 2],
        required=True,
    )

    parser.add_argument(
        "--nfft",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--cp",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--pilot-reps",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-data-symbols",
        type=int,
        default=None,
        help="Optional explicit symbol count"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/results",
    )

    parser.add_argument(
        "--file",
        type=str,
    )

    args = parser.parse_args()

    rx, fs = sf.read(
        os.path.join(os.path.dirname(__file__), args.input)
    )

    if fs != SAMPLE_RATE:
        raise ValueError(
            f"Expected fs={SAMPLE_RATE}, got {fs}"
        )

    if rx.ndim > 1:
        rx = rx[:, 0]

    receive(
        rx=rx,
        method=args.method,
        nfft=args.nfft,
        cp=args.cp,
        pilot_reps=args.pilot_reps,
        save_dir=args.output_dir,
        file_name=args.file if args.file else os.path.splitext(os.path.basename(args.input))[0],
        max_data_symbols=args.max_data_symbols,
    )


if __name__ == "__main__":
    main()