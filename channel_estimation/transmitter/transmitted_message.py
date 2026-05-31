
import sys, os
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import soundfile as sf

from config_test import (
    SAMPLE_RATE,
    TX_DIR,

    F_LOW,
    F_HIGH,

    PILOT_DURATION,
    PILOT_REPS,
    GUARD_DURATION,

    SYNC_CHIRP_DURATION,
    SYNC_CHIRP_F0,
    SYNC_CHIRP_F1,
    SYNC_CHIRP_AMPLITUDE,

    MLS_ORDER,

    CHIRP_N_SWEEPS,

    WN_SEED,
    PILOT_AMPLITUDE,
)

from utils.signal_utils import (
    silence,
    make_chirp,
    make_mls,
    normalise_amplitude,
    apply_tukey_window,
)


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────

def make_active_carrier_mask(
    nfft: int,
    fs: int,
    f_low: float,
    f_high: float,
):
    freqs = np.fft.rfftfreq(nfft, d=1/fs)

    return (freqs >= f_low) & (freqs <= f_high)


def ofdm_modulate(
    freq_bins: np.ndarray,
    cp_len: int,
):
    td = np.fft.irfft(freq_bins).astype(np.float32)

    cp = td[-cp_len:]

    return np.concatenate([cp, td]).astype(np.float32)


# ────────────────────────────────────────────────────────────────────────
# File → bits
# ────────────────────────────────────────────────────────────────────────

def file_to_bits(path: str):

    with open(path, "rb") as f:
        data = f.read()

    byte_array = np.frombuffer(data, dtype=np.uint8)

    bits = np.unpackbits(byte_array)

    return bits.astype(np.uint8)


# ────────────────────────────────────────────────────────────────────────
# QPSK
# ────────────────────────────────────────────────────────────────────────

def qpsk_modulate(bits: np.ndarray):

    if len(bits) % 2:
        bits = np.append(bits, 0)

    bit_pairs = bits.reshape(-1, 2)

    symbols = np.zeros(len(bit_pairs), dtype=np.complex64)

    for i, (b0, b1) in enumerate(bit_pairs):

        if (b0, b1) == (0, 0):
            s = 1 + 1j

        elif (b0, b1) == (0, 1):
            s = -1 + 1j

        elif (b0, b1) == (1, 1):
            s = -1 - 1j

        else:
            s = 1 - 1j

        symbols[i] = s

    symbols /= np.sqrt(2)

    return symbols


# ────────────────────────────────────────────────────────────────────────
# Sync chirp
# ────────────────────────────────────────────────────────────────────────

def build_sync_chirp():

    return make_chirp(
        duration=SYNC_CHIRP_DURATION,
        f0=SYNC_CHIRP_F0,
        f1=SYNC_CHIRP_F1,
        fs=SAMPLE_RATE,
        amplitude=SYNC_CHIRP_AMPLITUDE,
        method="linear",
    )


# ────────────────────────────────────────────────────────────────────────
# Method 1 pilots
# ────────────────────────────────────────────────────────────────────────

def generate_noise_ofdm_pilots(
    M,
    nfft=1024,
    cp_len=128,
):

    rng = np.random.default_rng(WN_SEED)

    active = make_active_carrier_mask(
        nfft,
        SAMPLE_RATE,
        F_LOW,
        F_HIGH,
    )

    pilots = []

    for _ in range(M):

        X = np.zeros(nfft//2 + 1, dtype=np.complex64)

        phases = rng.uniform(
            0,
            2*np.pi,
            active.sum()
        )

        X[active] = np.exp(1j * phases)

        tx = ofdm_modulate(X, cp_len)

        tx = normalise_amplitude(
            tx,
            PILOT_AMPLITUDE
        )

        pilots.append(tx)

    return np.concatenate(pilots)


def generate_mls_ofdm_pilots(
    M,
    nfft=1024,
    cp_len=128,
):

    active = make_active_carrier_mask(
        nfft,
        SAMPLE_RATE,
        F_LOW,
        F_HIGH,
    )

    n_active = active.sum()

    mls = make_mls(MLS_ORDER)

    reps = int(np.ceil(n_active / len(mls)))

    chips = np.tile(mls, reps)[:n_active]

    pilots = []

    for _ in range(M):

        X = np.zeros(nfft//2 + 1, dtype=np.complex64)

        X[active] = chips

        tx = ofdm_modulate(X, cp_len)

        tx = normalise_amplitude(
            tx,
            PILOT_AMPLITUDE
        )

        pilots.append(tx)

    return np.concatenate(pilots)


def generate_chirp_pilot():

    chirps = []

    sweep_duration = PILOT_DURATION / CHIRP_N_SWEEPS

    for i in range(CHIRP_N_SWEEPS):

        f0 = F_LOW
        f1 = F_HIGH

        c = make_chirp(
            duration=sweep_duration,
            f0=f0,
            f1=f1,
            fs=SAMPLE_RATE,
            amplitude=PILOT_AMPLITUDE,
            method="linear",
        )

        chirps.append(c)

    return np.concatenate(chirps)


# ────────────────────────────────────────────────────────────────────────
# METHOD 2
# Comb pilots inside OFDM symbols
# ────────────────────────────────────────────────────────────────────────

def qpsk_to_ofdm_with_comb_pilots(
    qpsk_symbols,
    pilot_spacing=8,
    nfft=1024,
    cp_len=128,
):

    active = make_active_carrier_mask(
        nfft,
        SAMPLE_RATE,
        F_LOW,
        F_HIGH,
    )

    active_bins = np.where(active)[0]

    # comb pilot locations
    pilot_bins = active_bins[::pilot_spacing]

    # data bins
    data_bins = np.setdiff1d(
        active_bins,
        pilot_bins
    )

    n_data = len(data_bins)

    pad = (-len(qpsk_symbols)) % n_data

    if pad:
        qpsk_symbols = np.concatenate([
            qpsk_symbols,
            np.zeros(pad, dtype=np.complex64)
        ])

    qpsk_symbols = qpsk_symbols.reshape(-1, n_data)

    tx_blocks = []

    for block in qpsk_symbols:

        X = np.zeros(
            nfft//2 + 1,
            dtype=np.complex64
        )

        # pilots
        X[pilot_bins] = 1 + 0j

        # data
        X[data_bins] = block

        tx = ofdm_modulate(X, cp_len)

        tx = normalise_amplitude(
            tx,
            PILOT_AMPLITUDE
        )

        tx_blocks.append(tx)

    tx_signal = np.concatenate(tx_blocks)

    metadata = {
        "pilot_bins": pilot_bins,
        "data_bins": data_bins,
        "n_symbols": len(tx_blocks),
    }

    return tx_signal, metadata


# ────────────────────────────────────────────────────────────────────────
# METHOD 1
# Dedicated pilot section
# ────────────────────────────────────────────────────────────────────────

def qpsk_to_ofdm(
    qpsk_symbols,
    nfft=1024,
    cp_len=128,
):

    active = make_active_carrier_mask(
        nfft,
        SAMPLE_RATE,
        F_LOW,
        F_HIGH,
    )

    active_bins = np.where(active)[0]

    n_data = len(active_bins)

    pad = (-len(qpsk_symbols)) % n_data

    if pad:
        qpsk_symbols = np.concatenate([
            qpsk_symbols,
            np.zeros(pad, dtype=np.complex64)
        ])

    qpsk_symbols = qpsk_symbols.reshape(-1, n_data)

    tx_blocks = []

    for block in qpsk_symbols:

        X = np.zeros(
            nfft//2 + 1,
            dtype=np.complex64
        )

        X[active_bins] = block

        tx = ofdm_modulate(X, cp_len)

        tx = normalise_amplitude(
            tx,
            PILOT_AMPLITUDE
        )

        tx_blocks.append(tx)

    return np.concatenate(tx_blocks)


# ────────────────────────────────────────────────────────────────────────
# Build TX packet
# ────────────────────────────────────────────────────────────────────────

def build_tx(
    input_file,
    method=1,
    pilot_type="noise",
    pilot_spacing=8,
    nfft=1024,
    cp_len=128,
):

    bits = file_to_bits(input_file)

    qpsk = qpsk_modulate(bits)

    sync = build_sync_chirp()

    guard = silence(
        GUARD_DURATION,
        SAMPLE_RATE
    )

    # ────────────────────────────────────────────────────────────────
    # METHOD 1
    # ────────────────────────────────────────────────────────────────

    if method == 1:

        if pilot_type == "noise":

            pilots = generate_noise_ofdm_pilots(
                PILOT_REPS,
                nfft,
                cp_len,
            )

        elif pilot_type == "mls":

            pilots = generate_mls_ofdm_pilots(
                PILOT_REPS,
                nfft,
                cp_len,
            )

        elif pilot_type == "chirp":

            pilots = generate_chirp_pilot()

        else:
            raise ValueError(
                f"Unknown pilot type: {pilot_type}"
            )

        data = qpsk_to_ofdm(
            qpsk,
            nfft,
            cp_len,
        )

        tx = np.concatenate([
            silence(0.1, SAMPLE_RATE),
            sync,
            guard,
            pilots,
            guard,
            data,
            silence(0.1, SAMPLE_RATE),
        ])

    # ────────────────────────────────────────────────────────────────
    # METHOD 2
    # ────────────────────────────────────────────────────────────────

    elif method == 2:

        data, metadata = qpsk_to_ofdm_with_comb_pilots(
            qpsk,
            pilot_spacing=pilot_spacing,
            nfft=nfft,
            cp_len=cp_len,
        )

        tx = np.concatenate([
            silence(0.1, SAMPLE_RATE),
            sync,
            guard,
            data,
            silence(0.1, SAMPLE_RATE),
        ])

    else:
        raise ValueError("method must be 1 or 2")

    tx = apply_tukey_window(tx)

    tx = normalise_amplitude(
        tx,
        PILOT_AMPLITUDE
    )

    return tx.astype(np.float32)


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input file"
    )

    parser.add_argument(
        "--method",
        type=int,
        default=1,
        choices=[1, 2]
    )

    parser.add_argument(
        "--pilot",
        type=str,
        default="noise",
        choices=["noise", "mls", "chirp"]
    )

    parser.add_argument(
        "--pilot-spacing",
        type=int,
        default=8
    )

    parser.add_argument(
        "--nfft",
        type=int,
        default=1024
    )

    parser.add_argument(
        "--cp",
        type=int,
        default=128
    )

    parser.add_argument(
        "--output",
        type=str,
        default="tx.wav"
    )

    args = parser.parse_args()

    tx = build_tx(
        input_file=args.input,
        method=args.method,
        pilot_type=args.pilot,
        pilot_spacing=args.pilot_spacing,
        nfft=args.nfft,
        cp_len=args.cp,
    )

    out_path = os.path.join(
        TX_DIR,
        args.output
    )

    sf.write(
        out_path,
        tx,
        SAMPLE_RATE,
    )

    print(f"Saved TX waveform: {out_path}")
    print(f"Duration: {len(tx)/SAMPLE_RATE:.2f} sec")


if __name__ == "__main__":
    main()