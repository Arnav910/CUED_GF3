# analyze_channel.py
#
# Offline channel-estimation analysis for RXDataset
#
# Reads:
#   receiver_cli.py output (.pkl)
#
# Performs:
#   - ML / LS channel estimation
#   - equalization
#   - BER
#   - SNR
#   - EVM
#   - channel visualization
#
# Supports:
#   METHOD 1 -> dedicated pilot OFDM symbols
#   METHOD 2 -> comb pilots
#
# ------------------------------------------------------------

import os
import sys
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    SAMPLE_RATE,
    F_LOW,
    F_HIGH,
    WN_SEED,
    MLS_ORDER,
)
from receiver.receiver import RXDataset
from utils.signal_utils import make_mls


# ============================================================
# Helpers
# ============================================================

def make_active_carrier_mask(
    nfft,
    fs,
    f_low,
    f_high,
):
    freqs = np.fft.rfftfreq(nfft, d=1/fs)

    return (freqs >= f_low) & (freqs <= f_high)


def qpsk_demod(symbols):

    bits = []

    for s in symbols:

        b0 = 1 if np.imag(s) < 0 else 0
        b1 = 1 if np.real(s) < 0 else 0

        bits.extend([b0, b1])

    return np.array(bits, dtype=np.uint8)


def equalize(Y, H):
    return Y / (H + 1e-12)


# ============================================================
# Metrics
# ============================================================

def compute_ber(tx_bits, rx_bits):

    n = min(len(tx_bits), len(rx_bits))

    return np.mean(tx_bits[:n] != rx_bits[:n])


def compute_evm(tx, rx):

    err = rx - tx

    return np.sqrt(
        np.mean(np.abs(err)**2)
        /
        np.mean(np.abs(tx)**2)
    )


def estimate_snr(
    rx_pilot,
    tx_pilot,
    H,
):
    signal = H * tx_pilot

    noise = rx_pilot - signal

    Ps = np.mean(np.abs(signal)**2)
    Pn = np.mean(np.abs(noise)**2)

    return 10 * np.log10(Ps / (Pn + 1e-12))


# ============================================================
# ML / LS channel estimate
# ============================================================

def ml_channel_estimate(
    rx_pilots,
    tx_pilots,
):
    """
    H[k] = sum(X*Y) / sum(|X|^2)
    """

    numerator = np.sum(
        np.conj(tx_pilots) * rx_pilots,
        axis=0,
    )

    denominator = np.sum(
        np.abs(tx_pilots)**2,
        axis=0,
    ) + 1e-12

    return numerator / denominator


# ============================================================
# TX Pilot Reconstruction
# ============================================================

def generate_noise_pilots(
    pilot_reps,
    nfft,
    active,
):
    rng = np.random.default_rng(WN_SEED)

    pilots = []

    for _ in range(pilot_reps):

        X = np.zeros(
            nfft//2 + 1,
            dtype=np.complex64,
        )

        phases = rng.uniform(
            0,
            2*np.pi,
            active.sum(),
        )

        X[active] = np.exp(1j * phases)

        pilots.append(X)

    return np.array(pilots)


def generate_mls_pilots(
    pilot_reps,
    nfft,
    active,
):
    mls = make_mls(MLS_ORDER)

    n_active = active.sum()

    reps = int(np.ceil(n_active / len(mls)))

    chips = np.tile(
        mls,
        reps,
    )[:n_active]

    pilots = []

    for _ in range(pilot_reps):

        X = np.zeros(
            nfft//2 + 1,
            dtype=np.complex64,
        )

        X[active] = chips

        pilots.append(X)

    return np.array(pilots)


# ============================================================
# Method 1
# ============================================================

def analyze_method1(
    ds,
    pilot_type,
):

    active = make_active_carrier_mask(
        ds.nfft,
        SAMPLE_RATE,
        F_LOW,
        F_HIGH,
    )

    rx_pilots = np.array(ds.pilot_symbols_fd)

    rx_active = rx_pilots[:, active]

    # --------------------------------------------------------
    # reconstruct TX pilots
    # --------------------------------------------------------

    if pilot_type == "noise":

        tx_pilots = generate_noise_pilots(
            len(rx_pilots),
            ds.nfft,
            active,
        )

    elif pilot_type == "mls":

        tx_pilots = generate_mls_pilots(
            len(rx_pilots),
            ds.nfft,
            active,
        )

    else:
        raise ValueError("unsupported pilot type")

    tx_active = tx_pilots[:, active]

    # --------------------------------------------------------
    # ML estimate
    # --------------------------------------------------------

    H = ml_channel_estimate(
        rx_active,
        tx_active,
    )

    # --------------------------------------------------------
    # Equalize data
    # --------------------------------------------------------

    data = np.array(ds.data_symbols_fd)

    equalized = []

    for sym in data:

        Y = sym[active]

        Xhat = equalize(Y, H)

        equalized.append(Xhat)

    equalized = np.concatenate(equalized)

    # --------------------------------------------------------
    # SNR
    # --------------------------------------------------------

    snr = estimate_snr(
        rx_active,
        tx_active,
        H,
    )

    print(f"SNR: {snr:.2f} dB")

    # --------------------------------------------------------
    # plots
    # --------------------------------------------------------

    freqs = np.fft.rfftfreq(
        ds.nfft,
        d=1/SAMPLE_RATE,
    )[active]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(
        freqs,
        20*np.log10(np.abs(H) + 1e-12),
    )

    plt.title("Channel Magnitude")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)

    plt.subplot(1, 2, 2)

    plt.plot(
        freqs,
        np.unwrap(np.angle(H)),
    )

    plt.title("Channel Phase")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (rad)")
    plt.grid(True)

    plt.tight_layout()

    plt.show()

    # constellation
    h = np.fft.irfft(H, n=ds.nfft)
    plt.figure(figsize=(12, 5))
    plt.plot(h.real, label="Real")
    #plt.plot(h.imag, label="Imaginary")
    plt.title("Channel Impulse Response")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(6, 6))



    filtered = equalized[np.abs(equalized) < 5]
    plt.scatter(
        np.real(equalized),
        np.imag(equalized),
        s=4,
    )

    plt.grid(True)
    plt.axis("equal")

    plt.title("Equalized Constellation")

    plt.show()

    return {
        "H": H,
        "equalized": equalized,
        "snr_db": snr,
    }


# ============================================================
# Method 2 (comb pilots)
# ============================================================

def interpolate_channel(
    H_pilot,
    pilot_bins,
    all_bins,
):
    Hr = np.interp(
        all_bins,
        pilot_bins,
        np.real(H_pilot),
    )

    Hi = np.interp(
        all_bins,
        pilot_bins,
        np.imag(H_pilot),
    )

    return Hr + 1j * Hi


def analyze_method2(
    ds,
    pilot_spacing,
):

    active = make_active_carrier_mask(
        ds.nfft,
        SAMPLE_RATE,
        F_LOW,
        F_HIGH,
    )

    active_bins = np.where(active)[0]

    pilot_bins = active_bins[::pilot_spacing]

    equalized_all = []

    for sym in ds.mixed_symbols:

        Y = sym[active_bins]

        Hpilot = Y[::pilot_spacing]

        H = interpolate_channel(
            Hpilot,
            pilot_bins,
            active_bins,
        )

        Xhat = equalize(Y, H)

        equalized_all.append(Xhat)

    equalized_all = np.concatenate(equalized_all)

    plt.figure(figsize=(6, 6))

    plt.scatter(
        np.real(equalized_all),
        np.imag(equalized_all),
        s=4,
    )

    plt.grid(True)
    plt.axis("equal")

    plt.title("Comb Pilot Equalized Constellation")

    plt.show()

    return {
        "equalized": equalized_all,
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        help=".pkl RXDataset file",
    )

    parser.add_argument(
        "--pilot-type",
        default="noise",
        choices=["noise", "mls"],
    )

    parser.add_argument(
        "--pilot-spacing",
        type=int,
        default=8,
    )


    args = parser.parse_args()
    file = os.path.join(os.path.dirname(__file__), "receiver/output/results", args.dataset)
    with open(file, "rb") as f:
        ds = pickle.load(f)

    print("\n==============================")
    print("RX DATASET")
    print("==============================")

    print(f"Method: {ds.method}")
    print(f"NFFT: {ds.nfft}")
    print(f"CP: {ds.cp}")
    print(f"RMS: {ds.rms:.4f}")

    print()

    if ds.method == 1:

        analyze_method1(
            ds,
            pilot_type=args.pilot_type,
        )

    elif ds.method == 2:

        analyze_method2(
            ds,
            pilot_spacing=args.pilot_spacing,
        )

    else:
        raise ValueError("unsupported method")


if __name__ == "__main__":
    main()