import argparse
import importlib.util
import os
import struct
from pathlib import Path

import numpy as np
from scipy.io.wavfile import write as wav_write


SAMPLE_RATE = 48000
PRE_ROLL_SECONDS = 0.5
PRE_ROLL_NOISE_RMS = 1e-3
PRE_ROLL_NOISE_SEED = 0
POST_ROLL_SECONDS = 0.5
POST_ROLL_NOISE_RMS = 1e-3
POST_ROLL_NOISE_SEED = 1
PREAMBLE_PEAK = 0.05
DATA_PEAK = 0.95

CHIRP_COUNT = 10
CHIRP_LENGTH = 4096
CHIRP_F0 = 750.0
CHIRP_F1 = 18000.0

GOLAY_ORDER = 12
GOLAY_LENGTH = 4096
GOLAY_CP_LENGTH = 2048
GOLAY_PAIR_COUNT = 4

OFDM_SIZE = 4096
OFDM_CP_LENGTH = 2048
OFDM_F_LOW = 2000.0
OFDM_F_HIGH = 12000.0
PILOT_BLOCK_PERIOD = 20
LDPC_STANDARD = "802.16"
LDPC_RATE = "1/2"
LDPC_Z = 61
LDPC_INFO_BITS = 732
LDPC_CODE_BITS = 1464
LDPC_BLOCKS_PER_GROUP = 35
DATA_OFDM_SYMBOLS_PER_GROUP = 30
DATA_CARRIERS_PER_SYMBOL = 854
APPENDIX_B_STRIDE = 15839
PILOT_SEED_PATH = Path(__file__).resolve().parent.parent / "seed_qpsk.npy"

def pack_header(file_size: int, filename: str) -> bytes:
    """Pack JOSS-F header fields A, B and C using big-endian integers."""
    name_bytes = filename.encode("utf-8")
    total_header_length = 2 + 4 + len(name_bytes)
    if total_header_length > 0xFFFF:
        raise ValueError("The UTF-8 filename is too long for the 2-byte header length.")
    if not 0 <= file_size <= 0xFFFFFFFF:
        raise ValueError("File size must fit in the 4-byte unsigned header field.")
    return (
        struct.pack(">H", total_header_length)
        + struct.pack(">I", file_size)
        + name_bytes
    )


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Convert bytes to a one-dimensional MSB-first bit array."""
    if len(data) == 0:
        return np.zeros(0, dtype=np.uint8)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")


def bits_to_qpsk(symbol_bits: np.ndarray) -> np.ndarray:
    """Map bits to the JOSS-F Gray-coded QPSK constellation."""
    bits = np.asarray(symbol_bits, dtype=np.uint8)
    if len(bits) % 2 != 0:
        bits = np.concatenate([bits, np.zeros(1, dtype=np.uint8)])
    pairs = bits.reshape(-1, 2)
    mapping = {
        (0, 0): 1 + 1j,
        (0, 1): -1 + 1j,
        (1, 0): 1 - 1j,
        (1, 1): -1 - 1j,
    }
    return np.array([mapping[tuple(pair)] for pair in pairs], dtype=np.complex128)


def generate_linear_chirp(
    f0: float = CHIRP_F0,
    f1: float = CHIRP_F1,
    length: int = CHIRP_LENGTH,
    fs: int = SAMPLE_RATE,
) -> np.ndarray:
    """Generate one JOSS-F linear chirp."""
    t = np.arange(length, dtype=np.float64) / fs
    sweep_rate = (f1 - f0) / (length / fs)
    phase = 2.0 * np.pi * (f0 * t + 0.5 * sweep_rate * t * t)
    return np.cos(phase)


def generate_chirp_train(
    chirp_count: int = CHIRP_COUNT,
    chirp_len: int = CHIRP_LENGTH,
    f0: float = CHIRP_F0,
    f1: float = CHIRP_F1,
    fs: int = SAMPLE_RATE,
) -> np.ndarray:
    """Generate consecutive JOSS-F chirps with no gap."""
    chirp = generate_linear_chirp(f0, f1, chirp_len, fs)
    return np.tile(chirp, chirp_count)


def generate_golay_pair(order: int = GOLAY_ORDER) -> tuple[np.ndarray, np.ndarray]:
    """Generate a Golay complementary pair from scalar seeds A=(1), B=(1)."""
    a = np.array([1.0])
    b = np.array([1.0])
    for _ in range(order):
        a, b = np.concatenate([a, b]), np.concatenate([a, -b])
    return a, b


def generate_golay_preamble(
    pair_count: int = GOLAY_PAIR_COUNT,
    gap_len: int = GOLAY_CP_LENGTH,
) -> np.ndarray:
    """Generate four A/B Golay pairs, each symbol carrying its own cyclic prefix."""
    a, b = generate_golay_pair()
    if len(a) != GOLAY_LENGTH or len(b) != GOLAY_LENGTH:
        raise ValueError("Golay order does not produce the JOSS-F length of 4096.")
    if not 0 <= gap_len <= GOLAY_LENGTH:
        raise ValueError("Invalid Golay cyclic-prefix length.")

    block = np.concatenate([a, np.zeros(gap_len), b, np.zeros(gap_len)]) 
    signal = np.concatenate([np.zeros(gap_len),np.tile(block,pair_count)]) 
    return signal


def _active_subcarrier_indices(
    fs: int = SAMPLE_RATE,
    nfft: int = OFDM_SIZE,
    f_low: float = OFDM_F_LOW,
    f_high: float = OFDM_F_HIGH,
) -> np.ndarray:
    """Return positive-frequency bins from 2 kHz through 12 kHz inclusive."""
    positive_bins = np.arange(1, nfft // 2, dtype=np.int64)
    frequencies = positive_bins * fs / nfft
    active_bins = positive_bins[
        (frequencies >= f_low) & (frequencies <= f_high)
    ]
    if len(active_bins) == 0:
        raise ValueError("No OFDM bins are inside the selected band.")
    if (
        fs == SAMPLE_RATE
        and nfft == OFDM_SIZE
        and f_low == OFDM_F_LOW
        and f_high == OFDM_F_HIGH
        and len(active_bins) != DATA_CARRIERS_PER_SYMBOL
    ):
        raise ValueError("JOSS-F Appendix B requires exactly 854 active bins.")
    return active_bins

# here is a bit complicated, as initial function has only be implemented on linux or mac
def _load_ldpc_encoder():
    """Load the new_ldpc encoder without requiring its decoder shared library."""
    module_path = Path(__file__).parent / "new_ldpc" / "py" / "ldpc.py"
    spec = importlib.util.spec_from_file_location("joss_f_new_ldpc", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load LDPC module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    encoder = module.code.__new__(module.code)
    encoder.standard = LDPC_STANDARD
    encoder.rate = LDPC_RATE
    encoder.z = LDPC_Z
    encoder.ptype = "A"
    encoder.proto = encoder.assign_proto()
    encoder.N = len(encoder.proto[0]) * encoder.z
    encoder.K = (len(encoder.proto[0]) - len(encoder.proto)) * encoder.z
    if encoder.K != LDPC_INFO_BITS or encoder.N != LDPC_CODE_BITS:
        raise ValueError(
            f"Unexpected LDPC dimensions K={encoder.K}, N={encoder.N}."
        )
    return encoder


def apply_ldpc_encoding(bitstream: np.ndarray) -> tuple[np.ndarray, int]:
    """Encode padded 732-bit blocks using new_ldpc IEEE 802.16 rate 1/2."""
    bits = np.asarray(bitstream, dtype=np.uint8)
    group_info_bits = LDPC_BLOCKS_PER_GROUP * LDPC_INFO_BITS
    group_count = int(np.ceil(len(bits) / group_info_bits))
    padded_length = group_count * group_info_bits
    padding_bits = padded_length - len(bits)
    if padding_bits:
        bits = np.concatenate(
            [bits, np.zeros(padding_bits, dtype=np.uint8)]
        )

    encoder = _load_ldpc_encoder()
    info_blocks = bits.reshape(-1, LDPC_INFO_BITS)
    coded_blocks = np.empty(
        (len(info_blocks), LDPC_CODE_BITS),
        dtype=np.uint8,
    )
    for block_index, info_block in enumerate(info_blocks):
        coded_blocks[block_index] = encoder.encode(info_block).astype(np.uint8)
    return coded_blocks, padding_bits


def apply_standard_interleaver(coded_blocks: np.ndarray) -> np.ndarray:
    """Apply Appendix B to groups of 35 LDPC codewords."""
    coded = np.asarray(coded_blocks, dtype=np.uint8)
    if coded.ndim != 2 or coded.shape[1] != LDPC_CODE_BITS:
        raise ValueError(
            f"Expected LDPC blocks with shape (n, {LDPC_CODE_BITS})."
        )
    if coded.shape[0] % LDPC_BLOCKS_PER_GROUP != 0:
        raise ValueError("LDPC block count must be a multiple of 35.")

    group_count = coded.shape[0] // LDPC_BLOCKS_PER_GROUP
    data_symbols = np.empty(
        (
            group_count * DATA_OFDM_SYMBOLS_PER_GROUP,
            DATA_CARRIERS_PER_SYMBOL,
        ),
        dtype=np.complex128,
    )

    for group_index in range(group_count):
        group = coded[
            group_index * LDPC_BLOCKS_PER_GROUP:
            (group_index + 1) * LDPC_BLOCKS_PER_GROUP
        ]
        qpsk = bits_to_qpsk(group.reshape(-1))
        interleaved = np.empty(
            (DATA_OFDM_SYMBOLS_PER_GROUP, DATA_CARRIERS_PER_SYMBOL),
            dtype=np.complex128,
        )
        for j, value in enumerate(qpsk):
            cell = (APPENDIX_B_STRIDE * j) % qpsk.size
            symbol_index, bin_index = divmod(
                cell,
                DATA_CARRIERS_PER_SYMBOL,
            )
            interleaved[symbol_index, bin_index] = value
        data_symbols[
            group_index * DATA_OFDM_SYMBOLS_PER_GROUP:
            (group_index + 1) * DATA_OFDM_SYMBOLS_PER_GROUP
        ] = interleaved

    return data_symbols


def uncoded_bits_to_data_symbols(bitstream: np.ndarray) -> np.ndarray:
    """Map uncoded bits sequentially onto 854-carrier QPSK data symbols."""
    bits = np.asarray(bitstream, dtype=np.uint8)
    bits_per_symbol = 2 * DATA_CARRIERS_PER_SYMBOL
    symbol_count = int(np.ceil(len(bits) / bits_per_symbol))
    padded_length = symbol_count * bits_per_symbol
    if padded_length > len(bits):
        bits = np.concatenate(
            [bits, np.zeros(padded_length - len(bits), dtype=np.uint8)]
        )
    return bits_to_qpsk(bits).reshape(
        symbol_count,
        DATA_CARRIERS_PER_SYMBOL,
    )


def appendix_a_pilot_values(
    seed_path: Path = PILOT_SEED_PATH,
) -> np.ndarray:
    """
    Load pilot QPSK values for positive-frequency bins 1..2047.

    seed_qpsk.npy contains the complete 2048-point positive spectrum:
    index 0 is the zero-valued DC bin and indices 1..2047 are QPSK values.
    """
    seed_path = Path(seed_path)
    if not seed_path.is_file():
        raise FileNotFoundError(f"Pilot seed file not found: {seed_path}")

    spectrum = np.load(seed_path, allow_pickle=False)
    if spectrum.shape != (OFDM_SIZE // 2,):
        raise ValueError(
            f"Expected {OFDM_SIZE // 2} pilot values in {seed_path}, "
            f"got shape {spectrum.shape}."
        )

    spectrum = np.asarray(spectrum, dtype=np.complex128)
    if not np.isclose(spectrum[0], 0.0):
        raise ValueError("Pilot seed index 0 must be the zero-valued DC bin.")

    pilot_values = spectrum[1:]
    valid_qpsk = (
        np.isin(pilot_values.real, (-1.0, 1.0))
        & np.isin(pilot_values.imag, (-1.0, 1.0))
    )
    if not np.all(valid_qpsk):
        invalid_count = int(np.count_nonzero(~valid_qpsk))
        raise ValueError(
            f"Pilot seed contains {invalid_count} non-QPSK values "
            "outside the DC bin."
        )
    return pilot_values.copy()


def _ofdm_symbol_from_positive_bins(
    bin_indices: np.ndarray,
    values: np.ndarray,
    nfft: int = OFDM_SIZE,
    cp_len: int = OFDM_CP_LENGTH,
) -> np.ndarray:
    """Build one real OFDM symbol and prepend its cyclic prefix."""
    spectrum = np.zeros(nfft // 2 + 1, dtype=np.complex128)
    spectrum[bin_indices] = values
    spectrum[0] = 0.0
    spectrum[nfft // 2] = 0.0
    time_wave = np.fft.irfft(spectrum, n=nfft)
    return np.concatenate([time_wave[-cp_len:], time_wave])


def generate_periodic_pilot_symbol(
    fs: int = SAMPLE_RATE,
    nfft: int = OFDM_SIZE,
    cp_len: int = OFDM_CP_LENGTH,
    f_low: float = OFDM_F_LOW,
    f_high: float = OFDM_F_HIGH,
) -> np.ndarray:
    """Generate the shared pilot seed on active data bins only."""
    active_bins = _active_subcarrier_indices(fs, nfft, f_low, f_high)
    appendix_values = appendix_a_pilot_values()
    if len(appendix_values) != nfft // 2 - 1:
        raise ValueError("Appendix A pilot mapping does not match the OFDM size.")
    pilot_values = appendix_values[active_bins - 1]
    return _ofdm_symbol_from_positive_bins(
        active_bins,
        pilot_values,
        nfft=nfft,
        cp_len=cp_len,
    )


def generate_ofdm_symbols(
    data_symbols: np.ndarray,
    fs: int = SAMPLE_RATE,
    nfft: int = OFDM_SIZE,
    cp_len: int = OFDM_CP_LENGTH,
    f_low: float = OFDM_F_LOW,
    f_high: float = OFDM_F_HIGH,
    pilot_period: int = PILOT_BLOCK_PERIOD,
) -> np.ndarray:
    """Generate interleaved data blocks and insert a pilot every 20th block."""
    active_bins = _active_subcarrier_indices(fs, nfft, f_low, f_high)
    cells = np.asarray(data_symbols, dtype=np.complex128)
    if cells.ndim != 2 or cells.shape[1] != len(active_bins):
        raise ValueError(
            f"Expected data symbols with shape (n, {len(active_bins)})."
        )

    pilot_symbol = generate_periodic_pilot_symbol(
        fs=fs,
        nfft=nfft,
        cp_len=cp_len,
        f_low=f_low,
        f_high=f_high,
    )
    symbols = []
    data_block_index = 0

    while data_block_index < len(cells):
        next_block_number = len(symbols) + 1
        if next_block_number % pilot_period == 0:
            symbols.append(pilot_symbol)
            continue

        symbols.append(
            _ofdm_symbol_from_positive_bins(
                active_bins,
                cells[data_block_index],
                nfft=nfft,
                cp_len=cp_len,
            )
        )
        data_block_index += 1

    if len(symbols) == 0:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(symbols)


def normalize_waveform(signal: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Attenuate a real waveform only when it exceeds the target peak."""
    signal = np.asarray(signal, dtype=np.float64)
    max_value = np.max(np.abs(signal)) if len(signal) else 0.0
    if max_value > peak:
        return signal * (peak / max_value)
    return signal


def scale_waveform_to_peak(signal: np.ndarray, peak: float) -> np.ndarray:
    """Scale one waveform section up or down to an exact target peak."""
    signal = np.asarray(signal, dtype=np.float64)
    if not 0.0 < peak <= 1.0:
        raise ValueError("Target peak must be in the interval (0, 1].")
    max_value = np.max(np.abs(signal)) if len(signal) else 0.0
    if max_value == 0.0:
        return signal
    return signal * (peak / max_value)


def save_waveform_to_wav(signal: np.ndarray, fs: int, path: str) -> None:
    """Save a real waveform as signed 16-bit PCM WAV."""
    int_signal = np.asarray(
        np.round(normalize_waveform(signal) * 32767),
        dtype=np.int16,
    )
    wav_write(path, fs, int_signal)


def build_transmitter_waveform(
    file_path: str,
    use_ldpc: bool = True,
) -> np.ndarray:
    """Build pre-roll small amplitude white noise to avoid cutoff of speaker, chirps, Golay pairs and OFDM blocks."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)

    payload = Path(file_path).read_bytes()
    if len(payload) == 0:
        raise ValueError("Input file must not be empty.")

    header = pack_header(len(payload), Path(file_path).name)
    source_bits = bytes_to_bits(header + payload)
    if use_ldpc:
        coded_blocks, _ = apply_ldpc_encoding(source_bits)
        data_symbols = apply_standard_interleaver(coded_blocks)
    else:
        data_symbols = uncoded_bits_to_data_symbols(source_bits)

    pre_roll_rng = np.random.default_rng(PRE_ROLL_NOISE_SEED)
    pre_roll = pre_roll_rng.normal(
        loc=0.0,
        scale=PRE_ROLL_NOISE_RMS,
        size=int(round(PRE_ROLL_SECONDS * SAMPLE_RATE)),
    )
    post_roll_rng = np.random.default_rng(POST_ROLL_NOISE_SEED)
    post_roll = post_roll_rng.normal(
        loc=0.0,
        scale=POST_ROLL_NOISE_RMS,
        size=int(round(POST_ROLL_SECONDS * SAMPLE_RATE)),
    )
    chirps = generate_chirp_train()
    golay = generate_golay_preamble()
    data = generate_ofdm_symbols(data_symbols)

    preamble = scale_waveform_to_peak(
        np.concatenate([chirps, golay]),
        PREAMBLE_PEAK,
    )
    data = scale_waveform_to_peak(data, DATA_PEAK)
    return np.concatenate([pre_roll, preamble, data, post_roll])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a JOSS-F chirp, Golay and OFDM waveform."
    )
    parser.add_argument("input_file", nargs="?", default="test.txt")
    parser.add_argument("output_wav", nargs="?", default="joss_f_tx.wav")
    parser.add_argument(
        "--no-ldpc",
        action="store_true",
        help="Disable LDPC and Appendix B interleaving for uncoded testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    use_ldpc = not args.no_ldpc #you can disable ldpc by type "--no-ldpc" after your input file and output file

    print(f"LDPC: {'enabled' if use_ldpc else 'disabled'}")

    waveform = build_transmitter_waveform(
        args.input_file,
        use_ldpc=use_ldpc,
    )

    save_waveform_to_wav(waveform, SAMPLE_RATE, args.output_wav)
    print(f"Transmitter waveform saved to: {args.output_wav}")
  

if __name__ == "__main__":
    main()
