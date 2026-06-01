import struct
from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate, find_peaks
from utils import make_chirp
import config


SAMPLE_RATE = 48000
OFDM_LOW = 4000.0
OFDM_HIGH = 13000.0
FINAL_GUARD = 48000
DEFAULT_CHANNEL_TAPS = min(config.CP, config.CHIRP_SPACING)


@dataclass
class DecodedPacket:
    header_length: int
    file_length: int
    header_bytes: bytes
    payload: bytes
    chirp_starts: np.ndarray
    data_start: int


@dataclass
class ChannelEstimateMetrics:
    relative_error: float
    delay_samples: int
    gain: complex
    active_bin_count: int


@dataclass
class PipelineVerification:
    payload_matches: bool
    payload_length: int
    decoded_length: int
    channel_metrics: ChannelEstimateMetrics
    chirp_starts: np.ndarray
    data_start: int


def _as_float_mono(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    return signal.astype(np.float64, copy=False)


def _reference_chirp() -> np.ndarray:
    duration = config.CHIRP_SAMPLES / SAMPLE_RATE
    t = np.arange(config.CHIRP_SAMPLES, dtype=np.float64) / SAMPLE_RATE
    sweep_rate = (config.CHIRP_HIGH - config.CHIRP_LOW) / duration
    phase = 2.0 * np.pi * (config.CHIRP_LOW * t + 0.5 * sweep_rate * t * t)
    return np.cos(phase)


def _next_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def find_chirp_in_signal(signal, num_chirps=None):
    """
    Find chirp locations in signal using cross-correlation and peak detection.
    
    Args:
        signal: Input signal to search
        num_chirps: Expected number of chirps (optional). If provided, returns only top N peaks.
    
    Returns:
        Sorted array of chirp start indices
    """
    reference_chirp = make_chirp(config.CHIRP_LOW, config.CHIRP_HIGH, config.CHIRP_SAMPLES)
    
    # Cross-correlation (gives correlation for each lag)
    correlation = np.correlate(signal, reference_chirp, mode='valid')
    #plt.plot(correlation)
    # Much stricter threshold and find peaks
    threshold = np.max(correlation) * 0.7
    peaks, properties = find_peaks(correlation, height=threshold, distance=config.CHIRP_SAMPLES*1.5)
    
    # Sort by correlation magnitude (descending)
    peak_indices = peaks[np.argsort(properties['peak_heights'])[::-1]]
    
    # If expected number given, keep only top N
    if num_chirps is not None:
        peak_indices = peak_indices[:num_chirps]
    
    # Convert correlation indices to signal indices
    # With 'valid' mode: peak_idx in correlation = chirp_start in signal
    chirp_starts = np.sort(peak_indices)
    
    return chirp_starts

def estimate_symbol_offset(chirp_starts: np.ndarray) -> float:
    """
    Estimate the average sample timing offset from the observed chirp spacing.
    """
    chirp_starts = np.asarray(chirp_starts, dtype=np.int64)
    if len(chirp_starts) < 2:
        raise ValueError("At least two chirps are required to estimate timing offset.")

    expected_spacing = config.CHIRP_SPACING + config.CHIRP_SAMPLES
    measured_spacing = np.diff(chirp_starts)
    return float(np.mean(measured_spacing - expected_spacing))




def channel_estimation(
    chirp_starts: np.ndarray,
    signal: np.ndarray,
    max_channel_taps: int | None = None,
    regularization: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the channel from the detected chirps.

    Returns:
        g_time: time-domain channel estimate.
        G_freq: frequency-domain channel estimate with config.SYMBOL bins.
    """
    signal = _as_float_mono(signal)
    chirp_starts = np.asarray(chirp_starts, dtype=np.int64)
    if len(chirp_starts) == 0:
        raise ValueError("At least one chirp is required for channel estimation.")

    if max_channel_taps is None:
        max_channel_taps = DEFAULT_CHANNEL_TAPS
    max_channel_taps = int(max_channel_taps)
    if max_channel_taps < 1:
        raise ValueError("max_channel_taps must be positive.")
    if max_channel_taps > config.CHIRP_SPACING:
        raise ValueError("max_channel_taps must fit inside the guard between chirps.")

    reference_chirp = _reference_chirp()
    observed_len = config.CHIRP_SAMPLES + max_channel_taps - 1
    fft_len = _next_power_of_two(observed_len)

    if int(np.max(chirp_starts)) + observed_len > len(signal):
        raise ValueError("A detected chirp response extends beyond the end of the signal.")

    ref_padded = np.zeros(fft_len, dtype=np.float64)
    ref_padded[: config.CHIRP_SAMPLES] = reference_chirp
    ref_freq = np.fft.fft(ref_padded)
    denom = np.abs(ref_freq) ** 2
    denom = denom + 1e-10

    impulse_estimates = []
    for start in chirp_starts:
        observed = np.zeros(fft_len, dtype=np.float64)
        observed[:observed_len] = signal[start : start + observed_len]
        observed_freq = np.fft.fft(observed)
        channel_freq_long = np.conjugate(ref_freq) * observed_freq / denom
        impulse_estimates.append(np.fft.ifft(channel_freq_long)[:max_channel_taps])

    g_time = np.mean(np.asarray(impulse_estimates), axis=0)
    G_freq = np.fft.fft(g_time, n=config.SYMBOL)

    return g_time, G_freq


def _active_subcarrier_indices(
    sample_rate: int = SAMPLE_RATE,
    symbol_len: int = config.SYMBOL,
    f_low: float = OFDM_LOW,
    f_high: float = OFDM_HIGH,
) -> np.ndarray:
    df = sample_rate / symbol_len
    start_bin = max(1, int(np.ceil(f_low / df)))
    end_bin = min(symbol_len // 2, int(np.floor(f_high / df)))

    if end_bin < start_bin:
        raise ValueError("No active OFDM subcarriers in the requested frequency band.")

    return np.arange(start_bin, end_bin + 1, dtype=np.int64)


def get_data_start(chirp_starts: np.ndarray, final_guard: int = FINAL_GUARD) -> int:
    """Return the first OFDM sample after chirp 10 and the 48000-sample guard."""
    chirp_starts = np.asarray(chirp_starts, dtype=np.int64)
    if len(chirp_starts) == 0:
        raise ValueError("No chirp starts supplied.")
    return int(chirp_starts[-1] + config.CHIRP_SAMPLES + final_guard)


def get_constellations(
    chirp_starts: np.ndarray,
    signal: np.ndarray,
    channel_freq: np.ndarray | None = None,
    sample_rate: int = SAMPLE_RATE,
    f_low: float = OFDM_LOW,
    f_high: float = OFDM_HIGH,
) -> np.ndarray:
    """
    Extract QPSK constellation points from every [CP|OFDM] block after the preamble.
    """
    signal = _as_float_mono(signal)
    data_start = get_data_start(chirp_starts)
    data = signal[data_start:]

    block_size = config.CP + config.SYMBOL
    num_symbols = len(data) // block_size
    if num_symbols == 0:
        raise ValueError("No complete OFDM blocks found after the chirp preamble.")

    data = data[: num_symbols * block_size]
    blocks = data.reshape(num_symbols, block_size)
    symbols_without_cp = blocks[:, config.CP : config.CP + config.SYMBOL]
    spectra = np.fft.rfft(symbols_without_cp, n=config.SYMBOL, axis=1)

    active_bins = _active_subcarrier_indices(sample_rate, config.SYMBOL, f_low, f_high)
    constellations = spectra[:, active_bins]

    if channel_freq is not None:
        channel_freq = np.asarray(channel_freq)
        if len(channel_freq) <= int(np.max(active_bins)):
            raise ValueError("Channel response does not cover all active OFDM bins.")
        active_channel = channel_freq[active_bins]
        constellations = constellations / (active_channel[None, :] + 1e-12)

    return constellations


def qpsk_demod(symbols: np.ndarray) -> np.ndarray:
    """
    Demodulate Gray-coded QPSK symbols.

    Mapping used by the transmitter:
        00 -> +1 + 1j
        01 -> -1 + 1j
        11 -> -1 - 1j
        10 -> +1 - 1j
    """
    symbols = np.asarray(symbols).reshape(-1)
    bits = np.zeros(symbols.size * 2, dtype=np.uint8)
    bits[0::2] = (symbols.imag < 0).astype(np.uint8)
    bits[1::2] = (symbols.real < 0).astype(np.uint8)
    return bits


def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    usable_length = (len(bits) // 8) * 8
    if usable_length == 0:
        return b""
    bits = bits[:usable_length]
    return np.packbits(bits, bitorder="big").tobytes()


def parse_ofdm_header(data_bytes: bytes) -> tuple[int, int, bytes]:
    """
    Parse [2 byte header length][4 byte file length][optional extra header bytes].

    The header length counts the bytes after the 2-byte length field. With the
    current transmitter this is 4, because the only header payload is file size.
    """
    if len(data_bytes) < 6:
        raise ValueError("Need at least 6 bytes to read OFDM header.")

    header_length = struct.unpack("<H", data_bytes[:2])[0]
    if header_length < 4:
        raise ValueError("Header length must be at least 4 bytes for the file length.")

    header_end = 2 + header_length
    if len(data_bytes) < header_end:
        raise ValueError("Not enough decoded bytes for the declared header length.")

    file_length = struct.unpack("<I", data_bytes[2:6])[0]
    file_length = 3252
    header_bytes = data_bytes[2:header_end]
    return header_length, file_length, header_bytes


def extract_payload(bits: np.ndarray) -> tuple[bytes, int, int, bytes]:
    """
    Convert demodulated bits to payload bytes using the OFDM header.

    Returns:
        payload, header_length, file_length, header_bytes
    """
    data_bytes = bits_to_bytes(bits)
    header_length, file_length, header_bytes = parse_ofdm_header(data_bytes)
    payload_start = 2 + header_length
    payload_end = payload_start + file_length
    print(file_length)
    if len(data_bytes) < payload_end:
        raise ValueError("Decoded byte stream ended before the declared file length.")

    return data_bytes[payload_start:payload_end], header_length, file_length, header_bytes


def decode_signal(
    signal: np.ndarray,
    use_channel_estimate: bool = False,
    max_channel_taps: int | None = None,
) -> DecodedPacket:
    """
    Full receiver pipeline for the described stream format.
    """
    chirp_starts = find_chirp_in_signal(signal, num_chirps=config.CHIRP_COUNT)
    channel_freq = None

    if use_channel_estimate:
        _, channel_freq = channel_estimation(
            chirp_starts,
            signal,
            max_channel_taps=max_channel_taps,
        )

    constellations = get_constellations(chirp_starts, signal, channel_freq=channel_freq)
    bits = qpsk_demod(constellations)
    print(len(bits))
    payload, header_length, file_length, header_bytes = extract_payload(bits)

    return DecodedPacket(
        header_length=header_length,
        file_length=file_length,
        header_bytes=header_bytes,
        payload=payload,
        chirp_starts=chirp_starts,
        data_start=get_data_start(chirp_starts),
    )


def simulate_channel(
    signal: np.ndarray,
    channel: np.ndarray,
    noise_std: float = 0.0,
    leading_noise_samples: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Convolve a transmit waveform with a channel and optional white noise."""
    if rng is None:
        rng = np.random.default_rng()

    signal = _as_float_mono(signal)
    channel = np.asarray(channel, dtype=np.float64).reshape(-1)
    if len(channel) == 0:
        raise ValueError("Channel must contain at least one tap.")

    received = np.convolve(signal, channel, mode="full")

    if leading_noise_samples > 0:
        prefix = rng.normal(0.0, noise_std, size=leading_noise_samples)
        received = np.concatenate([prefix, received])

    if noise_std > 0:
        received = received + rng.normal(0.0, noise_std, size=len(received))

    return received


def channel_estimate_metrics(
    estimated_freq: np.ndarray,
    true_channel: np.ndarray,
    active_bins: np.ndarray | None = None,
    max_delay_search: int = 128,
) -> ChannelEstimateMetrics:
    """
    Compare an estimated channel with the simulated channel on active OFDM bins.

    The absolute delay and gain are ambiguous after synchronization, so the
    metric reports the best relative error after fitting both.
    """
    estimated_freq = np.asarray(estimated_freq)
    true_channel = np.asarray(true_channel, dtype=np.float64).reshape(-1)
    if active_bins is None:
        active_bins = _active_subcarrier_indices()

    active_bins = np.asarray(active_bins, dtype=np.int64)
    if len(estimated_freq) <= int(np.max(active_bins)):
        raise ValueError("Estimated channel does not cover all active OFDM bins.")

    estimated = estimated_freq[active_bins]
    true_freq = np.fft.fft(true_channel, n=config.SYMBOL)[active_bins]

    best_error = np.inf
    best_delay = 0
    best_gain = 0.0 + 0.0j

    for delay in range(-max_delay_search, max_delay_search + 1):
        phase = np.exp(-1j * 2.0 * np.pi * active_bins * delay / config.SYMBOL)
        candidate = true_freq * phase
        candidate_power = np.vdot(candidate, candidate)
        if np.abs(candidate_power) < 1e-12:
            continue

        gain = np.vdot(candidate, estimated) / candidate_power
        fitted = gain * candidate
        fitted_norm = np.linalg.norm(fitted)
        if fitted_norm < 1e-12:
            continue

        error = float(np.linalg.norm(estimated - fitted) / fitted_norm)
        if error < best_error:
            best_error = error
            best_delay = delay
            best_gain = complex(gain)

    return ChannelEstimateMetrics(
        relative_error=float(best_error),
        delay_samples=int(best_delay),
        gain=best_gain,
        active_bin_count=int(len(active_bins)),
    )


def _dynamic_fir_for_test() -> np.ndarray:
    channel = np.zeros(192, dtype=np.float64)
    channel[0] = 1.0
    channel[4] = 0.34
    channel[11] = -0.26
    channel[27] = 0.18
    channel[43] = -0.13
    channel[72] = 0.10
    channel[111] = -0.07
    channel[157] = 0.045

    rng = np.random.default_rng(99)
    scatter_delays = np.array([18, 35, 58, 86, 119, 144, 178])
    scatter_decay = np.exp(-scatter_delays / 95.0)
    channel[scatter_delays] += rng.normal(0.0, 0.045, len(scatter_delays)) * scatter_decay

    return channel / np.sqrt(np.sum(channel * channel))


def _default_simulated_channel() -> np.ndarray:
    return _dynamic_fir_for_test()


def verify_transmitter_receiver_pipeline(
    payload_len: int = 2048,
    seed: int = 123,
    noise_std: float = 0.0005,
    leading_noise_samples: int = 1379,
    channel: np.ndarray | None = None,
    max_channel_taps: int = 256,
    channel_error_threshold: float = 0.15,
) -> PipelineVerification:
    """
    Build arbitrary data with Transmitter.py, simulate a channel, and decode it.
    """
    import tempfile
    from pathlib import Path

    import Transmitter

    rng = np.random.default_rng(seed)
    payload = rng.integers(0, 256, size=payload_len, dtype=np.uint8).tobytes()
    if channel is None:
        channel = _default_simulated_channel()
    channel = np.asarray(channel, dtype=np.float64).reshape(-1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        payload_path = Path(tmp_dir) / "payload.bin"
        payload_path.write_bytes(payload)
        tx = Transmitter.build_transmitter_waveform(str(payload_path), modulation="qpsk")

    rx = simulate_channel(
        tx,
        channel,
        noise_std=noise_std,
        leading_noise_samples=leading_noise_samples,
        rng=rng,
    )

    chirp_starts = find_chirp_in_signal(rx, num_chirps=config.CHIRP_COUNT)
    _, estimated_freq = channel_estimation(
        chirp_starts,
        rx,
        max_channel_taps=max_channel_taps,
    )
    metrics = channel_estimate_metrics(
        estimated_freq,
        channel,
        max_delay_search=max_channel_taps,
    )
    decoded = decode_signal(
        rx,
        use_channel_estimate=True,
        max_channel_taps=max_channel_taps,
    )

    result = PipelineVerification(
        payload_matches=decoded.payload == payload,
        payload_length=len(payload),
        decoded_length=len(decoded.payload),
        channel_metrics=metrics,
        chirp_starts=chirp_starts,
        data_start=decoded.data_start,
    )

    if not result.payload_matches:
        raise AssertionError("Pipeline decoded bytes do not match the transmitted payload.")
    if result.channel_metrics.relative_error > channel_error_threshold:
        raise AssertionError(
            "Channel estimate relative error "
            f"{result.channel_metrics.relative_error:.4f} exceeds "
            f"{channel_error_threshold:.4f}."
        )

    return result


def _qpsk_mod_for_test(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if len(bits) % 2 != 0:
        bits = np.concatenate([bits, np.zeros(1, dtype=np.uint8)])

    pairs = bits.reshape(-1, 2)
    symbols = np.empty(len(pairs), dtype=np.complex128)
    symbols[(pairs[:, 0] == 0) & (pairs[:, 1] == 0)] = 1 + 1j
    symbols[(pairs[:, 0] == 0) & (pairs[:, 1] == 1)] = -1 + 1j
    symbols[(pairs[:, 0] == 1) & (pairs[:, 1] == 1)] = -1 - 1j
    symbols[(pairs[:, 0] == 1) & (pairs[:, 1] == 0)] = 1 - 1j
    return symbols / np.sqrt(2)


def _ofdm_mod_for_test(bits: np.ndarray) -> np.ndarray:
    active_bins = _active_subcarrier_indices()
    carriers_per_symbol = len(active_bins)
    bits_per_symbol = carriers_per_symbol * 2

    pad = (-len(bits)) % bits_per_symbol
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

    out = []
    for bit_block in bits.reshape(-1, bits_per_symbol):
        spectrum = np.zeros(config.SYMBOL // 2 + 1, dtype=np.complex128)
        spectrum[active_bins] = _qpsk_mod_for_test(bit_block)
        td = np.fft.irfft(spectrum, n=config.SYMBOL)
        cp = td[-config.CP :]
        out.append(np.concatenate([cp, td]))

    return np.concatenate(out)


def _preamble_for_test() -> np.ndarray:
    chirp = _reference_chirp()
    guard = np.zeros(config.CHIRP_SPACING, dtype=np.float64)
    parts = []
    for idx in range(config.CHIRP_COUNT):
        parts.append(chirp)
        if idx < config.CHIRP_COUNT - 1:
            parts.append(guard)
    parts.append(np.zeros(FINAL_GUARD, dtype=np.float64))
    return np.concatenate(parts)


def test_receiver_round_trip() -> None:
    """
    Self-contained test for:
    noise + 10 chirps + 48000 guard + [CP|OFDM] QPSK data.
    """
    rng = np.random.default_rng(7)
    payload = b"receiver round trip payload"
    header_payload = struct.pack("<I", len(payload))
    packet = struct.pack("<H", len(header_payload)) + header_payload + payload
    bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8), bitorder="big")

    channel = _dynamic_fir_for_test()
    max_channel_taps = 256
    noise_prefix = rng.normal(0.0, 0.01, size=1379)
    tx_waveform = np.concatenate([_preamble_for_test(), _ofdm_mod_for_test(bits)])
    rx_body = simulate_channel(tx_waveform, channel, noise_std=0.0005, rng=rng)
    waveform = np.concatenate([noise_prefix, rx_body])

    chirp_starts = find_chirp_in_signal(waveform, num_chirps=config.CHIRP_COUNT)
    _, estimated_freq = channel_estimation(
        chirp_starts,
        waveform,
        max_channel_taps=max_channel_taps,
    )
    channel_metrics = channel_estimate_metrics(
        estimated_freq,
        channel,
        max_delay_search=max_channel_taps,
    )
    decoded = decode_signal(
        waveform,
        use_channel_estimate=True,
        max_channel_taps=max_channel_taps,
    )

    assert decoded.payload == payload
    assert decoded.file_length == len(payload)
    assert decoded.header_length == 4
    assert decoded.data_start == len(noise_prefix) + len(_preamble_for_test())
    assert channel_metrics.relative_error < 0.15


if __name__ == "__main__":
    test_receiver_round_trip()
    pipeline_result = verify_transmitter_receiver_pipeline()
    print("reciever.py self-test passed")
    print(
        "pipeline simulation passed: "
        f"{pipeline_result.decoded_length} bytes decoded, "
        "channel relative error "
        f"{pipeline_result.channel_metrics.relative_error:.4f}, "
        "delay "
        f"{pipeline_result.channel_metrics.delay_samples} samples"
    )
