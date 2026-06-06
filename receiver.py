"""
receiver.py  —  JOSS-F compliant OFDM receiver.
Matched to Transmitter.py (vE, 04/06/2026).

Uses Appendix B soft LLR deinterleaver for LDPC decoding.

Usage:
    python receiver.py received_signals/rx.wav recovered.txt
    python receiver.py received_signals/rx.wav recovered.txt --no-ldpc
"""

import argparse
import importlib.util
import struct
from pathlib import Path

import numpy as np
from scipy.io.wavfile import read as wav_read
from scipy.signal import find_peaks


# ── Constants (must match Transmitter.py exactly) ─────────────────────────────
SAMPLE_RATE               = 48_000
CHIRP_COUNT               = 10
CHIRP_LENGTH              = 4_096
CHIRP_F0                  = 750.0
CHIRP_F1                  = 18_000.0
GOLAY_ORDER               = 12
GOLAY_LENGTH              = 4_096
GOLAY_CP_LENGTH           = 2_048
GOLAY_PAIR_COUNT          = 4
OFDM_SIZE                 = 4_096
OFDM_CP_LENGTH            = 2_048
OFDM_F_LOW                = 2_000.0
OFDM_F_HIGH               = 12_000.0
PILOT_BLOCK_PERIOD        = 20
LDPC_STANDARD             = "802.16"
LDPC_RATE                 = "1/2"
LDPC_Z                    = 61
LDPC_INFO_BITS            = 732
LDPC_CODE_BITS            = 1_464
LDPC_BLOCKS_PER_GROUP     = 35
DATA_OFDM_SYMBOLS_PER_GROUP = 30
DATA_CARRIERS_PER_SYMBOL  = 854
APPENDIX_B_STRIDE         = 15_839

BLOCK_LENGTH  = OFDM_SIZE + OFDM_CP_LENGTH
CHIRPS_LENGTH = CHIRP_COUNT * CHIRP_LENGTH
GOLAY_SIGNAL_LENGTH = (GOLAY_CP_LENGTH
                       + GOLAY_PAIR_COUNT
                       * (GOLAY_LENGTH + GOLAY_CP_LENGTH
                          + GOLAY_LENGTH + GOLAY_CP_LENGTH))

# Active bins: inclusive >= and <= to match Transmitter._active_subcarrier_indices
_bins       = np.arange(1, OFDM_SIZE // 2, dtype=np.int64)
_freqs      = _bins * SAMPLE_RATE / OFDM_SIZE
ACTIVE_BINS = _bins[(_freqs >= OFDM_F_LOW) & (_freqs <= OFDM_F_HIGH)]
assert len(ACTIVE_BINS) == DATA_CARRIERS_PER_SYMBOL


# ── Golay pair ────────────────────────────────────────────────────────────────

def _golay_pair():
    a = np.array([1.0]); b = np.array([1.0])
    for _ in range(GOLAY_ORDER):
        a, b = np.concatenate([a, b]), np.concatenate([a, -b])
    return a, b


# ── Synchronisation ───────────────────────────────────────────────────────────

def find_chirp_start(signal):
    t   = np.arange(CHIRP_LENGTH, dtype=np.float64) / SAMPLE_RATE
    sw  = (CHIRP_F1 - CHIRP_F0) / (CHIRP_LENGTH / SAMPLE_RATE)
    ref = np.cos(2.0 * np.pi * (CHIRP_F0 * t + 0.5 * sw * t * t))
    corr = np.correlate(signal, ref, mode='valid')
    peaks, _ = find_peaks(np.abs(corr),
                          height=np.max(np.abs(corr)) * 0.7,
                          distance=CHIRP_LENGTH // 2)
    if len(peaks) == 0:
        raise RuntimeError("No chirp detected in signal.")
    return int(peaks[np.argmin(peaks)])


# ── Golay channel estimation ──────────────────────────────────────────────────

def estimate_channel_golay(signal, chirp_start):
    a, b = _golay_pair()
    L    = GOLAY_LENGTH
    A    = np.fft.rfft(a, n=L)
    B    = np.fft.rfft(b, n=L)
    denom = np.abs(A)**2 + np.abs(B)**2

    r     = signal[chirp_start:]
    a_off = CHIRPS_LENGTH + GOLAY_CP_LENGTH
    b_off = a_off + GOLAY_LENGTH + GOLAY_CP_LENGTH
    Ya    = np.fft.rfft(r[a_off : a_off + L], n=L)
    Yb    = np.fft.rfft(r[b_off : b_off + L], n=L)
    return (np.conj(A) * Ya + np.conj(B) * Yb) / denom


# ── OFDM equaliser ────────────────────────────────────────────────────────────

def equalise_block(block, H_rfft, lam=1e-6):
    Y = np.fft.rfft(block[OFDM_CP_LENGTH:], n=OFDM_SIZE)
    H = H_rfft[ACTIVE_BINS]
    return Y[ACTIVE_BINS] * np.conj(H) / (np.abs(H)**2 + lam)


# ── QPSK demodulator ─────────────────────────────────────────────────────────

def qpsk_demod(symbols):
    """Hard QPSK: Im<0 → b0=1, Re<0 → b1=1. Output: [b0,b1, b0,b1, ...]"""
    s    = np.asarray(symbols).reshape(-1)
    bits = np.zeros(s.size * 2, dtype=np.uint8)
    bits[0::2] = (s.imag < 0).astype(np.uint8)
    bits[1::2] = (s.real < 0).astype(np.uint8)
    return bits


# ── Appendix B deinterleaver — SOFT LLR output ───────────────────────────────

def deinterleave_group_soft(data_syms, H_active=None):
    """
    Appendix B deinterleaver producing soft LLRs for the LDPC decoder.

    data_syms : (30, 854) complex equalised symbols
    H_active  : (854,) complex channel estimate at active bins, or None for uniform weight

    Returns LLR : (35, 1464) float
        LLR[block][2*pos]   = w[bin] * Im(x)   for b0
        LLR[block][2*pos+1] = w[bin] * Re(x)   for b1
    where w[bin] = |H[bin]|^2 (per-bin SNR weight, Appendix B).
    """
    total = DATA_OFDM_SYMBOLS_PER_GROUP * DATA_CARRIERS_PER_SYMBOL  # 25620
    syms  = np.asarray(data_syms, dtype=np.complex128)
    # Per-bin weights: |H|^2, normalised so mean=1 to keep LLR scale sensible
    if H_active is not None:
        w = np.abs(np.asarray(H_active))**2
        w = w / (np.mean(w) + 1e-12)   # normalise
    else:
        w = np.ones(DATA_CARRIERS_PER_SYMBOL)
    LLR = np.zeros((LDPC_BLOCKS_PER_GROUP, LDPC_CODE_BITS), dtype=np.float64)

    for j in range(total):
        block       = j // LDPC_INFO_BITS
        pos         = j %  LDPC_INFO_BITS
        cell        = (APPENDIX_B_STRIDE * j) % total
        sym_i, bin_ = divmod(cell, DATA_CARRIERS_PER_SYMBOL)
        x           = syms[sym_i, bin_]
        wj          = w[bin_]
        LLR[block, 2 * pos    ] = wj * x.imag   # b0
        LLR[block, 2 * pos + 1] = wj * x.real   # b1

    return LLR


# ── Appendix B deinterleaver — HARD bits (for --no-ldpc) ─────────────────────

def deinterleave_group_hard(data_syms):
    """Hard-decision deinterleaver for --no-ldpc mode."""
    total = DATA_OFDM_SYMBOLS_PER_GROUP * DATA_CARRIERS_PER_SYMBOL
    syms  = np.asarray(data_syms, dtype=np.complex128)
    gathered = np.empty(total, dtype=np.complex128)
    for j in range(total):
        cell        = (APPENDIX_B_STRIDE * j) % total
        sym_i, bin_ = divmod(cell, DATA_CARRIERS_PER_SYMBOL)
        gathered[j] = syms[sym_i, bin_]
    return qpsk_demod(gathered).reshape(LDPC_BLOCKS_PER_GROUP, LDPC_CODE_BITS)


# ── LDPC loader and decoder ───────────────────────────────────────────────────

def _load_ldpc_decoder():
    module_path = Path(__file__).parent / "new_ldpc" / "py" / "ldpc.py"
    spec        = importlib.util.spec_from_file_location("joss_f_ldpc", module_path)
    if spec is None:
        raise ImportError(f"Cannot find ldpc module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    decoder = module.code(LDPC_STANDARD, LDPC_RATE, LDPC_Z)
    if decoder.K != LDPC_INFO_BITS or decoder.N != LDPC_CODE_BITS:
        raise ValueError(f"Unexpected LDPC dimensions K={decoder.K}, N={decoder.N}")
    return decoder


def apply_ldpc_decoding(LLR_blocks, decoder):
    """
    Decode (35, 1464) soft LLRs → (35, 732) info bits.
    LLR convention: positive = bit 0, negative = bit 1.
    Decoder output app: negative = bit 1 (same convention).
    Info bits are at app[1:733] due to a consistent 1-position offset
    in this decoder implementation.
    """
    info = np.empty((LDPC_BLOCKS_PER_GROUP, LDPC_INFO_BITS), dtype=np.uint8)
    for i, llr in enumerate(LLR_blocks):
        app, _  = decoder.decode(llr)
        info[i] = (app[:LDPC_INFO_BITS] < 0).astype(np.uint8)
    return info


# ── Header parser ─────────────────────────────────────────────────────────────

def parse_header(data):
    if len(data) < 6:
        raise ValueError("Too few bytes to parse header.")
    total_hdr = struct.unpack(">H", data[:2])[0]
    if total_hdr < 6 or len(data) < total_hdr:
        raise ValueError(f"Invalid header length {total_hdr}.")
    file_len  = struct.unpack(">I", data[2:6])[0]
    filename  = data[6:total_hdr].decode("utf-8", errors="replace")
    return total_hdr, file_len, filename


# ── Main decode pipeline ──────────────────────────────────────────────────────

def decode_signal(signal, use_ldpc=True, lam=1e-6, ldpc_decoder=None):
    signal = np.asarray(signal, dtype=np.float64).squeeze()

    # 1. Synchronise
    chirp_start = find_chirp_start(signal)
    data_start  = chirp_start + CHIRPS_LENGTH + GOLAY_SIGNAL_LENGTH
    print(f"Chirp found at sample {chirp_start}  ({chirp_start/SAMPLE_RATE:.3f} s)")
    print(f"OFDM data starts at sample {data_start}  ({data_start/SAMPLE_RATE:.3f} s)")

    # 2. Golay channel estimate
    H_rfft   = estimate_channel_golay(signal, chirp_start)
    h_active = H_rfft[ACTIVE_BINS]
    print(f"Golay H: mean|H|={np.mean(np.abs(h_active)):.3f}  "
          f"min={np.min(np.abs(h_active)):.4f}  max={np.max(np.abs(h_active)):.4f}")

    # 3. Extract OFDM blocks
    data_region = signal[data_start:]
    n_blocks    = len(data_region) // BLOCK_LENGTH
    if n_blocks == 0:
        raise ValueError("No complete OFDM blocks after preamble.")
    blocks = data_region[:n_blocks * BLOCK_LENGTH].reshape(n_blocks, BLOCK_LENGTH)
    print(f"OFDM blocks available: {n_blocks}")

    # 4. Separate pilots from data; equalise data blocks
    data_syms_groups = []
    current_group    = []
    output_block_idx = 0

    for block in blocks:
        output_block_idx += 1
        if output_block_idx % PILOT_BLOCK_PERIOD == 0:
            pass  # pilot — skip
        else:
            eq = equalise_block(block, H_rfft, lam)
            current_group.append(eq)
            if len(current_group) == DATA_OFDM_SYMBOLS_PER_GROUP:
                data_syms_groups.append(np.array(current_group))
                current_group = []

    if current_group:
        while len(current_group) < DATA_OFDM_SYMBOLS_PER_GROUP:
            current_group.append(np.zeros(DATA_CARRIERS_PER_SYMBOL, dtype=np.complex128))
        data_syms_groups.append(np.array(current_group))

    print(f"LDPC groups: {len(data_syms_groups)}")

    # 5. Decode
    all_info_bits = []

    if not use_ldpc:
        # TX --no-ldpc maps bits sequentially to QPSK with no interleaver.
        all_syms = np.concatenate([grp.reshape(-1) for grp in data_syms_groups])
        all_info_bits.append(qpsk_demod(all_syms))
    else:
        decoder = ldpc_decoder or _load_ldpc_decoder()
        h_active = H_rfft[ACTIVE_BINS]
        for group_syms in data_syms_groups:
            LLR_blocks  = deinterleave_group_soft(group_syms, H_active=h_active)
            info_blocks = apply_ldpc_decoding(LLR_blocks, decoder)
            all_info_bits.append(info_blocks.reshape(-1))

    info_bitstream = np.concatenate(all_info_bits).astype(np.uint8)
    usable         = (len(info_bitstream) // 8) * 8
    data_bytes     = np.packbits(info_bitstream[:usable], bitorder='big').tobytes()

    # 6. Parse header
    total_hdr, file_len, filename = parse_header(data_bytes)
    payload_end = total_hdr + file_len
    if len(data_bytes) < payload_end:
        available = len(data_bytes) - total_hdr
        print(f"Warning: header claims {file_len} bytes but only {available} available. "
              f"Saving available payload.")
        payload_end = len(data_bytes)
        file_len = available

    payload = data_bytes[total_hdr:payload_end]
    print(f"Decoded: '{filename}'  {file_len} bytes")
    return filename, file_len, payload


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="JOSS-F receiver")
    parser.add_argument("input_wav",   nargs="?", default="rx.wav")
    parser.add_argument("output_file", nargs="?", default=None)
    parser.add_argument("--no-ldpc",   action="store_true")
    parser.add_argument("--lam",       type=float, default=1e-6)
    args = parser.parse_args()

    fs, data = wav_read(args.input_wav)
    if fs != SAMPLE_RATE:
        raise ValueError(f"WAV sample rate {fs} Hz ≠ required {SAMPLE_RATE} Hz.")
    if data.dtype == np.int16:
        signal = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        signal = data.astype(np.float64) / 2147483648.0
    else:
        signal = data.astype(np.float64)   # float32 already ±1
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    print(f"Loading {args.input_wav} ...")
    print(f"Signal: {len(signal)} samples ({len(signal)/SAMPLE_RATE:.2f} s)")

    filename, file_len, payload = decode_signal(
        signal, use_ldpc=not args.no_ldpc, lam=args.lam
    )

    out_path = args.output_file
    if not out_path:
        invalid = set('\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b'
                      '\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17'
                      '\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f<>:"/\\|?*')
        if filename and not any(c in invalid for c in filename):
            out_path = filename
        else:
            out_path = "recovered_output.bin"
            print(f"Warning: filename {filename!r} invalid, saving as '{out_path}'")

    Path(out_path).write_bytes(payload)
    print(f"Saved '{out_path}'  ({file_len} bytes)")


if __name__ == "__main__":
    main()
