import argparse
from pathlib import Path
import struct

import numpy as np
from scipy.io import wavfile


def _active_subcarrier_indices(fs: int, N: int, f_low: float, f_high: float) -> np.ndarray:
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


def qpsk_demod(symbols: np.ndarray) -> np.ndarray:
    bits = np.zeros(symbols.size * 2, dtype=np.uint8)
    # Transmitter mapping uses Gray-coded QPSK:
    #   00 -> 1+1j, 01 -> -1+1j, 11 -> -1-1j, 10 -> 1-1j
    bits[0::2] = (symbols.imag < 0).astype(np.uint8)
    bits[1::2] = (symbols.real < 0).astype(np.uint8)
    return bits


def bpsk_demod(symbols: np.ndarray) -> np.ndarray:
    return (symbols.real >= 0).astype(np.uint8)


def decode_ofdm_waveform(
    waveform: np.ndarray,
    fs: int,
    N: int,
    cp_len: int,
    f_low: float,
    f_high: float,
    modulation: str,
) -> np.ndarray:
    active_bins = _active_subcarrier_indices(fs, N, f_low, f_high)
    bits_per_carrier = 1 if modulation == "bpsk" else 2
    symbol_len = N + cp_len
    symbol_count = len(waveform) // symbol_len
    if symbol_count == 0:
        raise ValueError("No OFDM symbols found in the waveform.")

    bit_chunks = []
    for sym_idx in range(symbol_count):
        start = sym_idx * symbol_len
        end = start + symbol_len
        block = waveform[start:end]
        if len(block) != symbol_len:
            break
        payload = block[cp_len:]
        spectrum = np.fft.rfft(payload, n=N)
        carriers = spectrum[active_bins]
        if modulation == "bpsk":
            bit_chunks.append(bpsk_demod(carriers))
        else:
            bit_chunks.append(qpsk_demod(carriers))

    return np.concatenate(bit_chunks, dtype=np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    if len(bits) == 0:
        return b""
    if len(bits) % 8 != 0:
        pad = 8 - (len(bits) % 8)
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits, bitorder="big").tobytes()


def parse_header(data_bytes: bytes) -> tuple[int, int]:
    if len(data_bytes) < 2:
        raise ValueError("Not enough data to read header length.")
    header_len = struct.unpack("<H", data_bytes[:2])[0]
    if len(data_bytes) < 2 + header_len:
        raise ValueError("Not enough data to read header payload.")
    payload = data_bytes[2 : 2 + header_len]
    if header_len != 4:
        raise ValueError(f"Unexpected header payload length: {header_len}")
    file_size = struct.unpack("<I", payload)[0]
    return header_len, file_size


def extract_payload(bitstream: np.ndarray) -> bytes:
    data_bytes = bits_to_bytes(bitstream)
    _, file_size = parse_header(data_bytes)
    total_header_bytes = 2 + 4
    payload = data_bytes[total_header_bytes : total_header_bytes + file_size]
    return payload


def load_wav(path: Path) -> tuple[int, np.ndarray]:
    fs, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32767.0
    elif data.dtype == np.float32:
        data = data.astype(np.float64)
    elif data.dtype == np.uint8:
        data = (data.astype(np.float64) - 128) / 128.0
    else:
        data = data.astype(np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return fs, data


def expected_preamble_length(chirp_count: int, chirp_len: int, chirp_gap: int, guard_seconds: float, fs: int) -> int:
    return chirp_count * chirp_len + (chirp_count - 1) * chirp_gap + int(np.round(guard_seconds * fs))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode a TX WAV file from the chirp+OFDM transmitter.")
    parser.add_argument("input_wav", nargs="?", default="tx_output.wav", help="Input WAV file to decode.")
    parser.add_argument("output_file", nargs="?", default="recovered.bin", help="Recovered output file path.")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Sample rate used by the transmitter.")
    parser.add_argument("--chirp-count", type=int, default=10, help="Number of preamble chirps.")
    parser.add_argument("--chirp-len", type=int, default=1024, help="Samples per chirp.")
    parser.add_argument("--chirp-gap", type=int, default=4000, help="Zero-sample gap between chirps.")
    parser.add_argument("--guard-seconds", type=float, default=1.0, help="Guard interval after chirps in seconds.")
    parser.add_argument("--ofdm-size", type=int, default=1024, help="IFFT size for OFDM.")
    parser.add_argument("--cp-len", type=int, default=1024, help="Cyclic prefix length in samples.")
    parser.add_argument("--band-low", type=float, default=4000.0, help="Lowest OFDM carrier frequency in Hz.")
    parser.add_argument("--band-high", type=float, default=13000.0, help="Highest OFDM carrier frequency in Hz.")
    parser.add_argument("--modulation", choices=["bpsk", "qpsk"], default="qpsk", help="Modulation scheme used in the transmitter.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    input_wav = Path(args.input_wav)
    output_file = Path(args.output_file)
    fs_wav, waveform = load_wav(input_wav)
    preamble_len = expected_preamble_length(args.chirp_count, args.chirp_len, args.chirp_gap, args.guard_seconds, fs_wav)
    if len(waveform) <= preamble_len:
        raise ValueError("WAV file is too short to contain a preamble plus data.")

    data_wave = waveform[preamble_len:]
    bits = decode_ofdm_waveform(
        data_wave,
        fs=fs_wav,
        N=args.ofdm_size,
        cp_len=args.cp_len,
        f_low=args.band_low,
        f_high=args.band_high,
        modulation=args.modulation,
    )
    payload = extract_payload(bits)
    output_file.write_bytes(payload)
    print(f"Decoded {len(payload)} bytes and saved to: {output_file}")
    print(f"Header expected file size: {len(payload)}")


if __name__ == "__main__":
    main()
