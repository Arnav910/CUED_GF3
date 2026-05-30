#!/usr/bin/env python3
"""
Audio Modem – Analyser
======================
Reads the received.npz output from the Receiver and performs:

  1. MLE Channel Estimation   H_n = (ΣX_n* Y_n) / (ΣX_n* X_n)
  2. Text file reconstruction
  3. QPSK Constellation cloud
  4. Channel FIR – time domain & frequency domain
  5. Metrics: SNR per sub-carrier, aggregate SNR, BER

Usage:
  python analyser.py <received.npz> [options]

Options:
  --no-show     Don't open matplotlib windows (save PNGs only)
  --out-dir     Directory for output figures and reconstructed text (default: .)
"""

import argparse
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')   # headless by default; overridden below if --show
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

# ── Constants ──────────────────────────────────────────────────────────────────
N_FFT         = 1024
N_CP          = 1024
DATA_CARRIERS = np.arange(1, N_FFT // 2)
N_DATA_SC     = len(DATA_CARRIERS)

QPSK_NORM          = 1 / np.sqrt(2)
QPSK_CONSTELLATION = np.array([1+1j, -1+1j, -1-1j, 1-1j]) * QPSK_NORM

GRAY_DECODE = {0: (0,0), 1: (0,1), 2: (1,1), 3: (1,0)}


# ── Helpers ────────────────────────────────────────────────────────────────────

def mle_channel_estimate(X_pilots: np.ndarray, Y_pilots: np.ndarray) -> np.ndarray:
    """
    MLE channel estimate per sub-carrier.
      H_n = (Σ X_n* · Y_n) / (Σ X_n* · X_n)
    X_pilots, Y_pilots: shape (M, N_DATA_SC) complex
    Returns H: shape (N_DATA_SC,) complex
    """
    numer = np.sum(np.conj(X_pilots) * Y_pilots, axis=0)
    denom = np.sum(np.conj(X_pilots) * X_pilots, axis=0)
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    return numer / denom


def plot_style():
    plt.rcParams.update({
        'figure.facecolor':  '#0d0d0d',
        'axes.facecolor':    '#111111',
        'axes.edgecolor':    '#444444',
        'axes.labelcolor':   '#cccccc',
        'text.color':        '#cccccc',
        'xtick.color':       '#888888',
        'ytick.color':       '#888888',
        'grid.color':        '#2a2a2a',
        'grid.linestyle':    '--',
        'lines.linewidth':   1.5,
        'font.family':       'monospace',
    })


def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"[AN] Figure saved → {path}")


# ── Analysis ───────────────────────────────────────────────────────────────────

def analyse(args):
    plot_style()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    d = np.load(args.npz, allow_pickle=True)
    data_syms_raw   = d['data_syms_raw']         # (blks, 511)  equalized
    all_syms_eq     = d['all_syms_eq']            # flat
    decided_syms    = d['decided_syms']           # flat hard decisions
    pilot_rxd_fd    = d['pilot_rxd_fd']           # (M, 511) received leading pilots
    pilots_fd_tx    = d['pilots_fd_tx']           # (511,)   transmitted pilots
    H_est           = d['H_est']                  # (511,)   initial H estimate

    payload_len = int(d['payload_len'][0])
    text_bytes  = d['text_bytes']
    bits        = d['bits']
    cfo         = float(d['cfo'][0])
    fs          = int(d['samplerate'][0])

    M = int(d['M'][0])

    print(f"[AN] Data blocks: {data_syms_raw.shape[0]}  Sub-carriers: {data_syms_raw.shape[1]}")
    print(f"[AN] CFO: {cfo:.6f} normalised bins")

    # ── 1. MLE Channel Estimation ─────────────────────────────────────────
    print("\n── 1. MLE Channel Estimation ──────────────────────────────────────")
    # Use all M leading pilot blocks
    X_tx = np.tile(pilots_fd_tx, (M, 1))     # (M, 511)
    Y_rx = pilot_rxd_fd[:M]                   # (M, 511)
    H_mle = mle_channel_estimate(X_tx, Y_rx)

    H_mag_dB = 20 * np.log10(np.abs(H_mle) + 1e-12)
    H_phase  = np.angle(H_mle, deg=True)
    print(f"[AN] H_mle mean magnitude: {np.mean(np.abs(H_mle)):.4f}")
    print(f"[AN] H_mle mean phase (°): {np.mean(H_phase):.2f}")

    # ── 2. Reconstruct text ───────────────────────────────────────────────
    print("\n── 2. Text Reconstruction ─────────────────────────────────────────")
    try:
        text = text_bytes.tobytes().decode('utf-8', errors='replace')
        txt_path = os.path.join(out_dir, 'reconstructed.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"[AN] Reconstructed text ({len(text)} chars) → {txt_path}")
        print(f"[AN] Preview: {repr(text[:300])}")
    except Exception as e:
        print(f"[AN] Text reconstruction error: {e}")
        text = ''

    # ── 3. Constellation Cloud ────────────────────────────────────────────
    print("\n── 3. Constellation Cloud ─────────────────────────────────────────")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(all_syms_eq.real, all_syms_eq.imag,
               s=0.5, alpha=0.3, color='#00e5ff', rasterized=True, label='Received')
    ref = QPSK_CONSTELLATION
    ax.scatter(ref.real, ref.imag, s=120, marker='x', color='#ff4444',
               linewidths=2, zorder=5, label='Ideal QPSK')
    labels = ['00', '01', '11', '10']
    for i, (r, lbl) in enumerate(zip(ref, labels)):
        ax.annotate(lbl, (r.real + 0.06, r.imag + 0.06),
                    color='#ffaa00', fontsize=10, fontweight='bold')
    ax.axhline(0, color='#555', lw=0.8)
    ax.axvline(0, color='#555', lw=0.8)
    ax.set_aspect('equal')
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel('In-phase (I)'); ax.set_ylabel('Quadrature (Q)')
    ax.set_title('QPSK Constellation Cloud', fontsize=13, color='#ffffff', pad=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True)
    save_fig(fig, os.path.join(out_dir, 'constellation.png'))
    plt.close(fig)

    # ── 4. Channel FIR (time + frequency domain) ──────────────────────────
    print("\n── 4. Channel FIR ─────────────────────────────────────────────────")

    # Build full-bandwidth H (positive + mirrored)
    H_full    = np.zeros(N_FFT, dtype=complex)
    H_full[DATA_CARRIERS]         = H_mle
    H_full[N_FFT - DATA_CARRIERS] = np.conj(H_mle)

    # Time-domain channel impulse response (IFFT of H)
    h_time = np.fft.ifft(H_full)
    h_abs  = np.abs(h_time)
    t_axis = np.arange(N_FFT) / fs * 1e3    # ms

    # Frequency domain magnitude
    freq_axis = np.fft.fftfreq(N_FFT, 1/fs)
    f_pos     = freq_axis[DATA_CARRIERS]
    H_mag     = np.abs(H_mle)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Channel Estimate (MLE)', fontsize=14, color='#ffffff')

    # 4a – CIR magnitude
    ax = axes[0, 0]
    ax.plot(t_axis[:N_FFT//4], h_abs[:N_FFT//4], color='#00e5ff')
    ax.set_xlabel('Delay (ms)'); ax.set_ylabel('|h(t)|')
    ax.set_title('Channel Impulse Response (CIR)')
    ax.grid(True)

    # 4b – CIR in dB
    ax = axes[0, 1]
    h_dB = 20 * np.log10(h_abs[:N_FFT//4] + 1e-12)
    ax.plot(t_axis[:N_FFT//4], h_dB, color='#ff9900')
    ax.set_xlabel('Delay (ms)'); ax.set_ylabel('|h(t)| (dB)')
    ax.set_title('CIR (dB)')
    ax.grid(True)

    # 4c – Channel frequency response magnitude
    ax = axes[1, 0]
    ax.plot(f_pos / 1e3, 20 * np.log10(H_mag + 1e-12), color='#7fff7f')
    ax.set_xlabel('Frequency (kHz)'); ax.set_ylabel('|H(f)| (dB)')
    ax.set_title('Channel Frequency Response')
    ax.grid(True)

    # 4d – Channel phase response
    ax = axes[1, 1]
    ax.plot(f_pos / 1e3, H_phase, color='#ff77ff')
    ax.set_xlabel('Frequency (kHz)'); ax.set_ylabel('Phase (°)')
    ax.set_title('Channel Phase Response')
    ax.grid(True)

    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'channel_fir.png'))
    plt.close(fig)

    # ── 5. Metrics ────────────────────────────────────────────────────────
    print("\n── 5. Metrics ─────────────────────────────────────────────────────")

    # Per-sub-carrier SNR from pilot blocks
    Y_noise = Y_rx - X_tx * H_mle[np.newaxis, :]
    noise_var  = np.mean(np.abs(Y_noise)**2, axis=0)
    signal_var = np.mean(np.abs(X_tx)**2,    axis=0)
    snr_per_sc = np.where(noise_var > 1e-15, signal_var / noise_var, 0)
    snr_db_per_sc = 10 * np.log10(snr_per_sc + 1e-12)

    mean_snr_db  = 10 * np.log10(np.mean(snr_per_sc))
    median_snr_db = 10 * np.log10(np.median(snr_per_sc))
    print(f"[AN] Mean SNR   : {mean_snr_db:.2f} dB")
    print(f"[AN] Median SNR : {median_snr_db:.2f} dB")

    # EVM (Error Vector Magnitude)
    evm_rms = np.sqrt(np.mean(np.abs(all_syms_eq - decided_syms)**2)) / QPSK_NORM
    evm_pct = evm_rms * 100
    print(f"[AN] EVM (RMS)  : {evm_pct:.2f}%")

    # BER – compare decided bits vs. ideal
    # Hard-decision reference: re-decide on ideal symbols
    dists      = np.abs(all_syms_eq[:, np.newaxis] - QPSK_CONSTELLATION[np.newaxis, :])
    dec_idx    = np.argmin(dists, axis=1)
    dec_bits   = np.array([b for idx in dec_idx for b in GRAY_DECODE[idx]], dtype=np.uint8)
    n_compare  = min(len(bits), len(dec_bits))
    # We don't have the ground-truth transmitted bits in the receiver output,
    # so we estimate BER from the symbol error rate (SER) which is available.
    # SER → BER for QPSK: BER ≈ SER / 2  (for AWGN, Gray coded)
    # Compute SER vs ideal nearest point
    sym_errors = np.sum(np.abs(all_syms_eq - decided_syms) > QPSK_NORM * 0.5)
    ser        = sym_errors / len(all_syms_eq)
    ber_est    = ser / 2
    print(f"[AN] Symbol Error Rate  : {ser:.4f}  ({sym_errors}/{len(all_syms_eq)})")
    print(f"[AN] Estimated BER      : {ber_est:.4f}")

    # MER (Modulation Error Ratio)
    signal_power = np.mean(np.abs(decided_syms)**2)
    error_power  = np.mean(np.abs(all_syms_eq - decided_syms)**2)
    mer_db       = 10 * np.log10(signal_power / (error_power + 1e-15))
    print(f"[AN] MER                : {mer_db:.2f} dB")

    # ── 5a – SNR per sub-carrier plot ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Link Quality Metrics', fontsize=14, color='#ffffff')

    ax = axes[0]
    ax.plot(f_pos / 1e3, snr_db_per_sc, color='#00e5ff', lw=0.8)
    ax.axhline(mean_snr_db, color='#ff4444', ls='--', lw=1.2, label=f'Mean {mean_snr_db:.1f} dB')
    ax.set_xlabel('Frequency (kHz)'); ax.set_ylabel('SNR (dB)')
    ax.set_title('SNR per Sub-carrier')
    ax.legend(); ax.grid(True)

    ax = axes[1]
    ax.hist(snr_db_per_sc, bins=60, color='#7fff7f', edgecolor='#333', alpha=0.85)
    ax.axvline(mean_snr_db,   color='#ff4444', ls='--', lw=1.5, label=f'Mean {mean_snr_db:.1f} dB')
    ax.axvline(median_snr_db, color='#ffaa00', ls=':',  lw=1.5, label=f'Median {median_snr_db:.1f} dB')
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Count')
    ax.set_title('SNR Histogram')
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'snr_metrics.png'))
    plt.close(fig)

    # ── Summary report ────────────────────────────────────────────────────
    report_path = os.path.join(out_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write("Audio Modem – Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Sample rate         : {fs} Hz\n")
        f.write(f"CFO (normalised)    : {cfo:.6f} bins\n")
        f.write(f"Data blocks decoded : {data_syms_raw.shape[0]}\n")
        f.write(f"QPSK symbols        : {len(all_syms_eq)}\n\n")
        f.write("Channel (MLE)\n")
        f.write(f"  Mean |H|          : {np.mean(np.abs(H_mle)):.4f}\n")
        f.write(f"  Std  |H|          : {np.std(np.abs(H_mle)):.4f}\n\n")
        f.write("Signal Quality\n")
        f.write(f"  Mean SNR          : {mean_snr_db:.2f} dB\n")
        f.write(f"  Median SNR        : {median_snr_db:.2f} dB\n")
        f.write(f"  EVM (RMS)         : {evm_pct:.2f} %\n")
        f.write(f"  MER               : {mer_db:.2f} dB\n")
        f.write(f"  Symbol Error Rate : {ser:.4f}\n")
        f.write(f"  Estimated BER     : {ber_est:.4f}\n\n")
        f.write("Output files\n")
        f.write(f"  reconstructed.txt\n")
        f.write(f"  constellation.png\n")
        f.write(f"  channel_fir.png\n")
        f.write(f"  snr_metrics.png\n")
    print(f"\n[AN] Report → {report_path}")

    # Print summary to console
    print("\n" + "─" * 50)
    print("  SUMMARY")
    print("─" * 50)
    print(f"  Mean SNR    : {mean_snr_db:.1f} dB")
    print(f"  Median SNR  : {median_snr_db:.1f} dB")
    print(f"  EVM         : {evm_pct:.2f} %")
    print(f"  MER         : {mer_db:.1f} dB")
    print(f"  Est. BER    : {ber_est:.4f}")
    print(f"  SER         : {ser:.4f}")
    print("─" * 50)

    if not args.no_show:
        try:
            matplotlib.use('TkAgg')
            plt.show()
        except Exception:
            pass


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Audio Modem Analyser')
    p.add_argument('npz',                              help='received.npz from receiver')
    p.add_argument('--no-show',   action='store_true', help="Don't open plot windows (save PNGs only)")
    p.add_argument('--out-dir',   default='.',         help='Output directory for figures (default: .)')
    args = p.parse_args()
    analyse(args)

if __name__ == '__main__':
    main()
