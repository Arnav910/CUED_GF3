import argparse
import os
import struct
from typing import Tuple

import numpy as np
from scipy.io.wavfile import write as wav_write


def pack_header(file_size: int) -> bytes:
    """Pack the header protocol: 2 bytes header length + 4 bytes file size. LSB first."""
    header_payload = struct.pack("<I", file_size)
    header_length = len(header_payload)
    return struct.pack("<H", header_length) + header_payload


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Convert bytes to a 1-D bit array (MSB-first)."""
    if len(data) == 0:
        return np.zeros(0, dtype=np.uint8)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")


def bits_to_bpsk(symbol_bits: np.ndarray) -> np.ndarray:
    """Map bits to BPSK symbols: 0 -> -1, 1 -> +1."""
    return 2 * symbol_bits.astype(np.float64) - 1.0


def bits_to_qpsk(symbol_bits: np.ndarray) -> np.ndarray:
    """Map bits to QPSK symbols using Gray coding."""
    if len(symbol_bits) % 2 != 0:
        symbol_bits = np.concatenate([symbol_bits, np.zeros(1, dtype=np.uint8)])
    pairs = symbol_bits.reshape(-1, 2)
    mapping = {
        (0, 0): 1 + 1j,
        (0, 1): -1 + 1j,
        (1, 1): -1 - 1j,
        (1, 0): 1 - 1j,
    }
    return np.array([mapping[tuple(pair)] for pair in pairs], dtype=np.complex128) / np.sqrt(2)


def generate_linear_chirp(
    f0: float,
    f1: float,
    length: int,
    fs: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a single linear chirp from f0 to f1 in the given sample length."""
    t = np.arange(length, dtype=np.float64) / fs
    k = (f1 - f0) / (length / fs)
    phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
    return amplitude * np.cos(phase)


def generate_chirp_train(
    chirp_count: int = 10,
    chirp_len: int = 1024,
    gap_len: int = 4000,
    guard_seconds: float = 1.0,
    f0: float = 20.0,
    f1: float = 20000.0,
    fs: int = 48000,
) -> np.ndarray:
    """Generate the chirp preamble train with gaps and a final guard interval."""
    chirps = []
    chirp = generate_linear_chirp(f0, f1, chirp_len, fs)
    silence_gap = np.zeros(gap_len, dtype=np.float64)
    for i in range(chirp_count):
        chirps.append(chirp)
        if i < chirp_count - 1:
            chirps.append(silence_gap)
    chirps.append(np.zeros(int(np.round(guard_seconds * fs)), dtype=np.float64))
    return np.concatenate(chirps)


def _active_subcarrier_indices(fs: int, N: int, f_low: float, f_high: float) -> np.ndarray:
    """Return the rFFT positive-frequency bin indices that fall inside the transmit band."""
    df = fs / N
    start = int(np.ceil(f_low / df))
    stop = int(np.floor(f_high / df))
    if start < 1:
        start = 1
    if stop > N // 2:
        stop = N // 2
    if stop < start:
        raise ValueError(f"Invalid OFDM band: {f_low}-{f_high} Hz with fs={fs} and N={N}")
    return np.arange(start, stop + 1, dtype=np.int64)


def generate_ofdm_symbols(
    bitstream: np.ndarray,
    fs: int = 48000,
    N: int = 1024,
    cp_len: int = 1024,
    f_low: float = 4000.0,
    f_high: float = 13000.0,
    modulation: str = "qpsk",
) -> np.ndarray:
    """Generate an OFDM waveform from a bitstream using the specified band and modulation."""
    active_bins = _active_subcarrier_indices(fs, N, f_low, f_high)
    if modulation == "bpsk":
        bits_per_carrier = 1
    elif modulation == "qpsk":
        bits_per_carrier = 2
    else:
        raise ValueError(f"Unsupported modulation: {modulation}")

    bits_per_symbol = len(active_bins) * bits_per_carrier
    if bits_per_symbol == 0:
        raise ValueError("No active OFDM subcarriers available in the selected band.")

    symbol_count = int(np.ceil(len(bitstream) / bits_per_symbol))
    padded_length = symbol_count * bits_per_symbol
    if padded_length != len(bitstream):
        bitstream = np.concatenate([bitstream, np.zeros(padded_length - len(bitstream), dtype=np.uint8)])

    symbols = []
    for i in range(symbol_count):
        block = bitstream[i * bits_per_symbol : (i + 1) * bits_per_symbol]
        if modulation == "bpsk":
            constellation = bits_to_bpsk(block)
        else:
            constellation = bits_to_qpsk(block)

        spectrum = np.zeros(N // 2 + 1, dtype=np.complex128)
        spectrum[active_bins] = constellation
        time_wave = np.fft.irfft(spectrum, n=N)
        cp = time_wave[-cp_len:]
        symbols.append(np.concatenate([cp, time_wave]))

    return np.concatenate(symbols)


def normalize_waveform(signal: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Normalize a real waveform to the full 16-bit audio range with a safety peak."""
    if np.all(signal == 0):
        return signal.astype(np.float64)
    max_val = np.max(np.abs(signal))
    return signal * (peak / max_val)


def save_waveform_to_wav(signal: np.ndarray, fs: int, path: str) -> None:
    """Save a real waveform to a 16-bit WAV file."""
    signal = normalize_waveform(signal)
    int_signal = np.asarray(np.round(signal * 32767), dtype=np.int16)
    wav_write(path, fs, int_signal)


def build_transmitter_waveform(
    file_path: str,
    fs: int = 48000,
    chirp_count: int = 10,
    chirp_len: int = 1024,
    chirp_gap: int = 4000,
    guard_seconds: float = 1.0,
    f0: float = 20.0,
    f1: float = 20000.0,
    ofdm_size: int = 1024,
    cp_len: int = 1024,
    band_low: float = 4000.0,
    band_high: float = 13000.0,
    modulation: str = "bpsk",
) -> np.ndarray:
    """Build the full transmit waveform from an input file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)

    with open(file_path, "rb") as f:
        payload = f.read()

    file_size = len(payload)
    header = pack_header(file_size)
    message = header + payload
    bitstream = bytes_to_bits(message)
    print(bitstream.size)
    if bitstream.size == 0:
        raise ValueError("Input file is empty. Transmitter requires a non-empty payload.")

    preamble = generate_chirp_train(
        chirp_count=chirp_count,
        chirp_len=chirp_len,
        gap_len=chirp_gap,
        guard_seconds=guard_seconds,
        f0=f0,
        f1=f1,
        fs=fs,
    )
    data_wave = generate_ofdm_symbols(
        bitstream,
        fs=fs,
        N=ofdm_size,
        cp_len=cp_len,
        f_low=band_low,
        f_high=band_high,
        modulation=modulation,
    )
    return np.concatenate([preamble, data_wave])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a chirp+OFDM transmitter waveform.")
    parser.add_argument("input_file", nargs="?", default="README.md", help="Path to the file to transmit. Defaults to README.md when running in VS Code.")
    parser.add_argument("output_wav", nargs="?", default="tx_output.wav", help="Output WAV path. Defaults to tx_output.wav when running in VS Code.")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Audio sample rate in Hz (48 kHz preferred).")
    parser.add_argument("--chirp-count", type=int, default=10, help="Number of linear chirps in the preamble.")
    parser.add_argument("--chirp-len", type=int, default=1024, help="Samples per chirp.")
    parser.add_argument("--chirp-gap", type=int, default=4000, help="Zero-sample gap between chirps.")
    parser.add_argument("--guard-seconds", type=float, default=1.0, help="Guard interval after chirps in seconds.")
    parser.add_argument("--chirp-f0", type=float, default=20.0, help="Start frequency of the chirp in Hz.")
    parser.add_argument("--chirp-f1", type=float, default=20000.0, help="End frequency of the chirp in Hz.")
    parser.add_argument("--ofdm-size", type=int, default=1024, help="IFFT size for OFDM.")
    parser.add_argument("--cp-len", type=int, default=1024, help="Cyclic prefix length in samples.")
    parser.add_argument("--band-low", type=float, default=4000.0, help="Lowest OFDM carrier frequency in Hz.")
    parser.add_argument("--band-high", type=float, default=13000.0, help="Highest OFDM carrier frequency in Hz.")
    parser.add_argument("--modulation", choices=["bpsk", "qpsk"], default="qpsk", help="Modulation scheme for OFDM subcarriers (default QPSK).")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    waveform = build_transmitter_waveform(
        args.input_file,
        fs=args.sample_rate,
        chirp_count=args.chirp_count,
        chirp_len=args.chirp_len,
        chirp_gap=args.chirp_gap,
        guard_seconds=args.guard_seconds,
        f0=args.chirp_f0,
        f1=args.chirp_f1,
        ofdm_size=args.ofdm_size,
        cp_len=args.cp_len,
        band_low=args.band_low,
        band_high=args.band_high,
        modulation=args.modulation,
    )
    save_waveform_to_wav(waveform, args.sample_rate, args.output_wav)
    print(f"Transmitter waveform built and saved to: {args.output_wav}")


if __name__ == "__main__":
    main()
