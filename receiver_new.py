"""
receiver.py  —  JOSS-F compliant OFDM receiver.
Matched to Transmitter.py (vE, 08/06/2026).

Uses Appendix B soft LLR deinterleaver for LDPC decoding.

Usage:
    python receiver.py received_signals/rx.wav recovered.txt
    python receiver.py received_signals/rx.wav recovered.txt --no-ldpc
"""

import argparse
import importlib.util
import struct
from pathlib import Path
import os

import numpy as np
from scipy.io.wavfile import read as wav_read
from scipy.signal import find_peaks,chirp,correlate,correlation_lags

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
GUARD_LENGTH              = 0
GUARD_AFTER_GOLAY         = 0
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
LDPC_MAX_ITERATIONS       = 200
DATA_OFDM_SYMBOLS_PER_GROUP = 30
DATA_CARRIERS_PER_SYMBOL  = 854
APPENDIX_B_STRIDE         = 15_839
PILOT_SEED_PATH           = Path(__file__).resolve().parent/ "seed_qpsk.npy"
PHASE_COHERENCE_THRESHOLD = 0.30
MAX_SLOPE_INNOVATION      = 4.0e-4
SLOPE_CONTINUITY_PENALTY  = 0.20
CHANNEL_SMOOTHING_BINS    = 9
CHANNEL_REGULARISATION    = 0.20
PILOT_CHANNEL_UPDATE      = 0.75

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
assert len(ACTIVE_BINS) == DATA_CARRIERS_PER_SYMBOL ### test to verify if thsi breaks im sorry whhat??

def _load_pilot_values(seed_path=PILOT_SEED_PATH):
    """Load and validate pilot values for positive-frequency bins 1..2047."""
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
    if not np.isclose(spectrum[0], 0.0): ### verify starts from 0 and meets the standard
        raise ValueError("Pilot seed index 0 must be the zero-valued DC bin.")

    pilot_values = spectrum[1:] ### read all pilot values
    valid_qpsk = (
        np.isin(pilot_values.real, (-1.0, 1.0))
        & np.isin(pilot_values.imag, (-1.0, 1.0))
    )
    if not np.all(valid_qpsk): ### verify if all the pilots are valid
        invalid_count = int(np.count_nonzero(~valid_qpsk))
        raise ValueError(
            f"Pilot seed contains {invalid_count} non-QPSK values "
            "outside the DC bin."
        )
    return pilot_values.copy()


_PILOT_VALUES = _load_pilot_values() ### prepare pilots
ACTIVE_PILOT_VALUES = _PILOT_VALUES[ACTIVE_BINS - 1] ### nyquist - 1


# ── Golay pair ────────────────────────────────────────────────────────────────

def _golay_pair():
    a = np.array([1.0]); b = np.array([1.0])
    for _ in range(GOLAY_ORDER):
        a, b = np.concatenate([a, b]), np.concatenate([a, -b])
    return a, b


# ── Synchronisation ───────────────────────────────────────────────────────────

def find_chirp_end(signal):
    t = np.linspace(0, CHIRP_LENGTH / SAMPLE_RATE, CHIRP_LENGTH)
    single_chirp = chirp(t, f0=CHIRP_F0, f1=CHIRP_F1, t1=(CHIRP_LENGTH / SAMPLE_RATE), method='linear')

    corr = np.abs(correlate(signal, single_chirp, mode="valid"))
    peaks, properties = find_peaks(
        corr,
        height=np.max(corr) * 0.5,
        distance=int(0.7 * CHIRP_LENGTH),
    )
    lags = correlation_lags(len(signal),len(single_chirp),mode='valid')
    detected_lags = lags[peaks]
    if len(peaks) == 0:
        raise RuntimeError("No chirp detected in signal.")

    return int(detected_lags[-1])


# ── Golay channel estimation ──────────────────────────────────────────────────

def _normalised_matched_filter(signal, reference):
    """Return a valid-mode matched-filter magnitude normalised by local energy."""
    reference = np.asarray(reference, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    numerator = np.abs(correlate(signal, reference, mode="valid", method="fft"))
    local_energy = np.convolve(
        signal * signal,
        np.ones(len(reference), dtype=np.float64),
        mode="valid",
    )
    denominator = np.sqrt(
        local_energy * np.sum(reference * reference)
    ) + 1e-12
    return numerator / denominator


def refine_sync_with_golay(signal, rough_golay_start):
    """
    Refine the first Golay-A sequence start using one complementary A/B pair.

    rough_golay_start points to the beginning of the Golay signal, including
    its leading zero guard. The returned refined start points to the first
    sample of sequence A, after that guard.
    """
    a, b = _golay_pair()
    rough_first_a = rough_golay_start + GOLAY_CP_LENGTH
    search_radius = GOLAY_CP_LENGTH
    search_low = max(0, rough_first_a - search_radius)
    search_high = min(
        len(signal) - 2 * GOLAY_LENGTH - GOLAY_CP_LENGTH,
        rough_first_a + search_radius,
    )
    if search_high < search_low:
        raise ValueError("Not enough samples for Golay fine synchronisation.")

    candidate_count = search_high - search_low + 1
    a_region = signal[search_low:search_high + GOLAY_LENGTH]
    b_offset = GOLAY_LENGTH + GOLAY_CP_LENGTH
    b_region = signal[
        search_low + b_offset:
        search_high + b_offset + GOLAY_LENGTH
    ]
    a_score = _normalised_matched_filter(a_region, a)[:candidate_count]
    b_score = _normalised_matched_filter(b_region, b)[:candidate_count]
    complementary_score = a_score + b_score
    best_offset = int(np.argmax(complementary_score))
    refined_golay_start = search_low + best_offset
    peak_delay = refined_golay_start - rough_first_a

    segment_a = signal[
        refined_golay_start:
        refined_golay_start + GOLAY_LENGTH
    ]
    segment_b = signal[
        refined_golay_start + b_offset:
        refined_golay_start + b_offset + GOLAY_LENGTH
    ]
    A = np.fft.rfft(a, n=GOLAY_LENGTH)
    B = np.fft.rfft(b, n=GOLAY_LENGTH)
    Ya = np.fft.rfft(segment_a, n=GOLAY_LENGTH)
    Yb = np.fft.rfft(segment_b, n=GOLAY_LENGTH)
    denominator = np.abs(A) ** 2 + np.abs(B) ** 2
    H0 = (np.conj(A) * Ya + np.conj(B) * Yb) / denominator
    h0 = np.fft.irfft(H0, n=GOLAY_LENGTH)
    return refined_golay_start, peak_delay, H0, h0


def estimate_channel_golay_pairs(signal, golay_start):
    """
    Estimate the channel from every Golay pair.

    golay_start is the refined first sample of the first A sequence.
    """
    a, b = _golay_pair()
    L = GOLAY_LENGTH
    A = np.fft.rfft(a, n=L)
    B = np.fft.rfft(b, n=L)
    denom = np.abs(A) ** 2 + np.abs(B) ** 2
    pair_stride = 2 * (GOLAY_LENGTH + GOLAY_CP_LENGTH)
    H_estimates = []
    h_estimates = []
    centres = []

    for pair_index in range(GOLAY_PAIR_COUNT):
        a_start = golay_start + pair_index * pair_stride
        b_start = a_start + GOLAY_LENGTH + GOLAY_CP_LENGTH
        if b_start + L > len(signal):
            raise ValueError("Not enough samples for all Golay channel estimates.")
        Ya = np.fft.rfft(signal[a_start:a_start + L], n=L)
        Yb = np.fft.rfft(signal[b_start:b_start + L], n=L)
        H = (np.conj(A) * Ya + np.conj(B) * Yb) / denom
        H_estimates.append(H)
        h_estimates.append(np.fft.irfft(H, n=L))
        centres.append((a_start + b_start + L) / 2.0)

    return (
        np.asarray(H_estimates),
        np.asarray(h_estimates),
        np.asarray(centres, dtype=np.float64),
    )


def _fit_common_and_slope(phase, weight):
    """Fit phase = common + slope * centred_bin on reliable carriers."""
    phase = np.asarray(phase, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    finite = np.isfinite(phase) & np.isfinite(weight)
    if not np.any(finite):
        return 0.0, 0.0
    reliable = finite & (weight > 0.03 * np.max(weight[finite]))
    if np.count_nonzero(reliable) < 16:
        return 0.0, 0.0

    x = ACTIVE_BINS.astype(np.float64)
    x -= np.mean(x) ## centred x
    design = np.column_stack([np.ones(np.count_nonzero(reliable)), x[reliable]])
    root_weight = np.sqrt(weight[reliable] / np.max(weight[reliable]))
    coefficients = np.linalg.lstsq(
        design * root_weight[:, None],
        phase[reliable] * root_weight,
        rcond=None,
    )[0]
    return float(coefficients[0]), float(coefficients[1])


def _smooth_channel(channel, window=CHANNEL_SMOOTHING_BINS):
    """Apply short complex-frequency smoothing without changing array length."""
    channel = np.asarray(channel, dtype=np.complex128)
    if window <= 1:
        return channel.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.hanning(window)
    kernel /= np.sum(kernel)
    padding = window // 2
    padded = np.pad(channel, padding, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _blend_channel(old_channel, pilot_channel, update=PILOT_CHANNEL_UPDATE):
    """Update the channel to the current pilot, after frequency smoothing."""
    old_channel = np.asarray(old_channel, dtype=np.complex128)
    pilot_channel = _smooth_channel(pilot_channel)
    phase_change = _intelligent_unwrap_phase_difference(
        old_channel,
        pilot_channel,
    )
    old_magnitude = np.maximum(np.abs(old_channel), 1e-12)
    pilot_magnitude = np.maximum(np.abs(pilot_channel), 1e-12)
    magnitude = np.exp(
        (1.0 - update) * np.log(old_magnitude)
        + update * np.log(pilot_magnitude)
    )
    phase = np.angle(old_channel) + update * phase_change
    return magnitude * np.exp(1j * phase)


def estimate_initial_channel(signal, golay_start, data_start):
    """Estimate initial H and phase-rate priors from all Golay pairs."""
    H_estimates, h_estimates, centres = estimate_channel_golay_pairs(
        signal,
        golay_start,
    )
    reference = H_estimates[0, ACTIVE_BINS]
    common = []
    slope = []
    for estimate in H_estimates:
        active = estimate[ACTIVE_BINS]
        phase_change = _intelligent_unwrap_phase_difference(
            reference,
            active,
        )
        common_value, slope_value = _fit_common_and_slope(
            phase_change,
            np.abs(reference) * np.abs(active),
        )
        common.append(common_value)
        slope.append(slope_value)

    common = np.unwrap(np.asarray(common, dtype=np.float64))
    slope = np.asarray(slope, dtype=np.float64)
    common_rate, _ = np.polyfit(centres, common, 1)
    slope_rate, _ = np.polyfit(centres, slope, 1)
    implied_sampling_ppm = (
        -slope_rate * OFDM_SIZE / (2.0 * np.pi) * 1e6
    )
    if abs(implied_sampling_ppm) > 1000.0:
        slope_rate = 0.0

    x = ACTIVE_BINS.astype(np.float64)
    x -= np.mean(x)

    H_avg = np.mean(H_estimates, axis=0)
    aligned_active = []
    for estimate, common_value, slope_value in zip(
        H_estimates,
        common,
        slope,
    ):
        correction = -common_value - slope_value * x
        aligned_active.append(
            estimate[ACTIVE_BINS] * np.exp(1j * correction)
        )
    H_avg[ACTIVE_BINS] = _smooth_channel(
        np.mean(aligned_active, axis=0)
    )
    h_avg = np.fft.irfft(H_avg, n=GOLAY_LENGTH)
    return (
        H_avg,
        h_avg,
        H_estimates,
        h_estimates,
        centres,
        float(common_rate),
        float(slope_rate),
    )


# ── OFDM equaliser ────────────────────────────────────────────────────────────

def equalise_block(block, H_rfft, lam=1e-6):
    Y = np.fft.rfft(block[OFDM_CP_LENGTH:], n=OFDM_SIZE)
    H = H_rfft[ACTIVE_BINS]
    return Y[ACTIVE_BINS] * np.conj(H) / (np.abs(H)**2 + lam)


def equalise_block_active(block, H_active, lam=1e-6):
    """Conservative MMSE equalisation without inverting deep channel fades."""
    Y = np.fft.rfft(block[OFDM_CP_LENGTH:], n=OFDM_SIZE)[ACTIVE_BINS]
    H = np.asarray(H_active, dtype=np.complex128)
    channel_power = np.abs(H) ** 2
    typical_power = np.median(channel_power[np.isfinite(channel_power)])
    regularisation = max(
        float(lam),
        CHANNEL_REGULARISATION * float(typical_power),
    )
    return Y * np.conj(H) / (channel_power + regularisation)


def _nearest_qpsk(symbols):
    """Return hard QPSK decisions with the transmitter's unnormalised levels."""
    symbols = np.asarray(symbols, dtype=np.complex128)
    real = np.where(symbols.real < 0.0, -1.0, 1.0)
    imag = np.where(symbols.imag < 0.0, -1.0, 1.0)
    return real + 1j * imag


def track_qpsk_symbol_phase(
    symbols,
    predicted_common=0.0,
    predicted_slope=0.0,
    slope_search_span=0.04,
):
    """
    Remove residual common phase and linear phase slope from one QPSK symbol.

    Raising QPSK to the fourth power removes the unknown transmitted data.
    The remaining phase is four times the common phase and bin slope. The
    pi/2 common-phase ambiguity is resolved using the preceding symbol.
    """
    symbols = np.asarray(symbols, dtype=np.complex128)
    magnitude = np.abs(symbols)
    finite = np.isfinite(symbols) & np.isfinite(magnitude)
    if not np.any(finite):
        return symbols.copy(), predicted_common, predicted_slope, 0.0

    median_magnitude = np.median(magnitude[finite])
    valid = finite & (magnitude > 0.2 * median_magnitude)
    if np.count_nonzero(valid) < 16:
        return symbols.copy(), predicted_common, predicted_slope, 0.0

    unit = symbols / np.maximum(magnitude, 1e-12)
    fourth = -(unit**4)
    weights = np.minimum(
        magnitude / (median_magnitude + 1e-12),
        3.0,
    ) ** 2

    x = ACTIVE_BINS.astype(np.float64)
    x -= np.mean(x)

    # Direct adjacent-bin unwrapping is fragile when the channel contains deep
    # fades. Search for the slope that makes the fourth-power phases globally
    # coherent instead. The search span corresponds to about +/-26 samples of
    # FFT-window timing error for a 4096-point OFDM symbol.
    valid_x = x[valid]
    valid_fourth = fourth[valid]
    valid_weights = weights[valid]
    coarse_step = 2e-4
    slope_grid = predicted_slope + np.arange(
        -slope_search_span,
        slope_search_span + coarse_step,
        coarse_step,
    )
    coherence_grid = np.abs(
        np.exp(-4j * np.outer(slope_grid, valid_x))
        @ (valid_weights * valid_fourth)
    ) / (np.sum(valid_weights) + 1e-12)
    slope_distance = (
        (slope_grid - predicted_slope)
        / max(slope_search_span, coarse_step)
    )
    slope_score = (
        coherence_grid
        - SLOPE_CONTINUITY_PENALTY * slope_distance**2
    )
    slope = float(slope_grid[np.argmax(slope_score)])

    fine_step = coarse_step / 20.0
    fine_grid = slope + np.arange(-coarse_step, coarse_step + fine_step, fine_step)
    fine_coherence = np.abs(
        np.exp(-4j * np.outer(fine_grid, valid_x))
        @ (valid_weights * valid_fourth)
    ) / (np.sum(valid_weights) + 1e-12)
    fine_distance = (
        (fine_grid - predicted_slope)
        / max(slope_search_span, coarse_step)
    )
    fine_score = (
        fine_coherence
        - SLOPE_CONTINUITY_PENALTY * fine_distance**2
    )
    slope = float(fine_grid[np.argmax(fine_score)])
    maximum_innovation = MAX_SLOPE_INNOVATION * max(
        1.0,
        slope_search_span / 0.004,
    )
    slope = float(
        np.clip(
            slope,
            predicted_slope - maximum_innovation,
            predicted_slope + maximum_innovation,
        )
    )

    common_vector = fourth * np.exp(-4j * slope * x)
    weighted_sum = np.sum(weights[valid] * common_vector[valid])
    common_base = 0.25 * np.angle(weighted_sum)
    common_candidates = common_base + 0.5 * np.pi * np.arange(-3, 4)
    common = common_candidates[
        np.argmin(np.abs(common_candidates - predicted_common))
    ]
    coherence = float(
        np.abs(weighted_sum) / (np.sum(weights[valid]) + 1e-12)
    )

    corrected = symbols * np.exp(-1j * (common + slope * x))

    # Once the fourth-power estimate is close, hard QPSK decisions provide a
    # higher-resolution common-phase estimate without requiring extra pilots.
    decisions = _nearest_qpsk(corrected)
    decision_ratio = corrected * np.conj(decisions)
    decision_unit = decision_ratio / np.maximum(np.abs(decision_ratio), 1e-12)
    residual_common = np.angle(
        np.sum(weights[valid] * decision_unit[valid])
    )
    residual_common = float(np.clip(residual_common, -0.25, 0.25))
    corrected *= np.exp(-1j * residual_common)

    return (
        corrected,
        common + residual_common,
        slope,
        coherence,
    )


def estimate_channel_from_pilot(block, lam=1e-12):
    """Estimate the channel on active data bins from one periodic pilot."""
    Y = np.fft.rfft(block[OFDM_CP_LENGTH:], n=OFDM_SIZE)[ACTIVE_BINS]
    X = ACTIVE_PILOT_VALUES
    return Y * np.conj(X) / (np.abs(X)**2 + lam)


def bootstrap_channel_from_periodic_pilots(blocks):
    """
    Extrapolate the channel at block zero from the first two periodic pilots.

    This is a fallback for recordings where the Golay-to-OFDM phase evolution
    is not representative of the phase drift during the OFDM frame.
    """
    first_index = PILOT_BLOCK_PERIOD - 1
    second_index = 2 * PILOT_BLOCK_PERIOD - 1
    if len(blocks) <= second_index:
        raise ValueError("Need two periodic pilots for channel bootstrap.")

    first_H = _smooth_channel(
        estimate_channel_from_pilot(blocks[first_index])
    )
    second_H = _smooth_channel(
        estimate_channel_from_pilot(blocks[second_index])
    )
    phase_change = _intelligent_unwrap_phase_difference(first_H, second_H)
    common_change, slope_change = _fit_common_and_slope(
        phase_change,
        np.abs(first_H) * np.abs(second_H),
    )
    pilot_separation = second_index - first_index
    common_rate = common_change / pilot_separation
    slope_rate = slope_change / pilot_separation

    x = ACTIVE_BINS.astype(np.float64)
    x -= np.mean(x)
    intervals_to_zero = first_index + 1
    channel_zero = first_H * np.exp(
        -1j
        * intervals_to_zero
        * (common_rate + slope_rate * x)
    )
    return channel_zero, float(common_rate), float(slope_rate)


def _robust_timing_line(elapsed_samples, timing_offsets):
    """Fit timing drift with a Theil-Sen slope and MAD outlier rejection."""
    elapsed_samples = np.asarray(elapsed_samples, dtype=np.float64)
    timing_offsets = np.asarray(timing_offsets, dtype=np.float64)
    count = len(elapsed_samples)
    if count < 2:
        return None

    pair_slopes = []
    for left in range(count - 1):
        dx = elapsed_samples[left + 1:] - elapsed_samples[left]
        valid = dx != 0.0
        pair_slopes.extend(
            (
                (timing_offsets[left + 1:][valid] - timing_offsets[left])
                / dx[valid]
            ).tolist()
        )
    if not pair_slopes:
        return None

    slope = float(np.median(pair_slopes))
    intercept = float(np.median(timing_offsets - slope * elapsed_samples))
    residuals = timing_offsets - (intercept + slope * elapsed_samples)
    residual_median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - residual_median)))
    residual_scale = 1.4826 * mad
    threshold = max(0.75, 3.0 * residual_scale)
    inliers = np.abs(residuals - residual_median) <= threshold

    if np.count_nonzero(inliers) >= 2 and not np.all(inliers):
        inlier_x = elapsed_samples[inliers]
        inlier_y = timing_offsets[inliers]
        pair_slopes = []
        for left in range(len(inlier_x) - 1):
            dx = inlier_x[left + 1:] - inlier_x[left]
            valid = dx != 0.0
            pair_slopes.extend(
                (
                    (inlier_y[left + 1:][valid] - inlier_y[left])
                    / dx[valid]
                ).tolist()
            )
        slope = float(np.median(pair_slopes))
        intercept = float(np.median(inlier_y - slope * inlier_x))
        residuals = timing_offsets - (intercept + slope * elapsed_samples)

    inlier_residuals = residuals[inliers]
    rms = float(np.sqrt(np.mean(inlier_residuals**2)))
    inlier_y = timing_offsets[inliers]
    denominator = float(np.sum((inlier_y - np.mean(inlier_y))**2))
    if denominator > 1e-12:
        r_squared = 1.0 - float(np.sum(inlier_residuals**2)) / denominator
    else:
        r_squared = 1.0 if rms < 1e-9 else 0.0

    reliable = (
        np.count_nonzero(inliers) >= 3
        and np.count_nonzero(inliers) >= int(np.ceil(0.7 * count))
        and rms <= 1.0
        and r_squared >= 0.80
    )
    return {
        "slope": slope,
        "intercept": intercept,
        "residuals": residuals,
        "inliers": inliers,
        "rms": rms,
        "r_squared": r_squared,
        "reliable": reliable,
    }


def estimate_sampling_mismatch_from_pilots(blocks):
    """
    Estimate accumulated timing drift from periodic pilot channel phases.

    Returns one row per pilot and a fitted sampling mismatch in ppm. This is
    diagnostic only: the waveform is not resampled automatically.
    """
    pilot_rows = []
    previous_H = None
    reference_block = None
    accumulated_common = 0.0
    accumulated_slope = 0.0

    for block_index, block in enumerate(blocks):
        if (block_index + 1) % PILOT_BLOCK_PERIOD != 0:
            continue

        pilot_H = _smooth_channel(estimate_channel_from_pilot(block))
        if previous_H is None:
            reference_block = block_index
            delta_common = 0.0
            delta_slope = 0.0
        else:
            phase_change = _intelligent_unwrap_phase_difference(
                previous_H,
                pilot_H,
            )
            delta_common, delta_slope = _fit_common_and_slope(
                phase_change,
                np.abs(previous_H) * np.abs(pilot_H),
            )
            accumulated_common += delta_common
            accumulated_slope += delta_slope

        previous_H = pilot_H

        timing_samples = -accumulated_slope * OFDM_SIZE / (2.0 * np.pi)
        pilot_rows.append(
            {
                "block": block_index + 1,
                "sample": block_index * BLOCK_LENGTH,
                "common": accumulated_common,
                "slope": accumulated_slope,
                "delta_common": delta_common,
                "delta_slope": delta_slope,
                "timing_samples": timing_samples,
            }
        )

    sampling_ppm = None
    fit = None
    if len(pilot_rows) >= 2:
        elapsed_samples = np.asarray(
            [
                row["sample"]
                - reference_block * BLOCK_LENGTH
                for row in pilot_rows
            ],
            dtype=np.float64,
        )
        timing_offsets = np.asarray(
            [row["timing_samples"] for row in pilot_rows],
            dtype=np.float64,
        )
        fit = _robust_timing_line(elapsed_samples, timing_offsets)
        if fit is not None:
            sampling_ppm = float(fit["slope"] * 1e6)
            for row, residual, inlier in zip(
                pilot_rows,
                fit["residuals"],
                fit["inliers"],
            ):
                row["fit_residual"] = float(residual)
                row["inlier"] = bool(inlier)

    return pilot_rows, sampling_ppm, fit


def print_pilot_timing_diagnostics(
    pilot_rows,
    sampling_ppm,
    fit=None,
    title="Pilot timing diagnostics",
):
    """Print periodic-pilot timing drift and the fitted clock mismatch."""
    if not pilot_rows:
        print(f"{title}: no periodic pilots available.")
        return
    print(f"{title}:")
    print(" block   common(rad)  slope(rad/bin)  timing(samples)  fit")
    for row in pilot_rows:
        marker = ""
        if "inlier" in row:
            marker = " ok" if row["inlier"] else " OUTLIER"
        print(
            f" {row['block']:5d}  "
            f"{row['common']:+11.4f}  "
            f"{row['slope']:+14.6e}  "
            f"{row['timing_samples']:+15.3f}  "
            f"{marker}"
        )
    if sampling_ppm is not None:
        print(f"Robust pilot-fitted sampling mismatch: {sampling_ppm:+.1f} ppm")
    if fit is not None:
        quality = "reliable" if fit["reliable"] else "NOT reliable"
        print(
            "Pilot timing fit: "
            f"R^2={fit['r_squared']:.3f}, "
            f"RMS residual={fit['rms']:.3f} samples, "
            f"{np.count_nonzero(fit['inliers'])}/{len(pilot_rows)} inliers "
            f"({quality})"
        )


def _intelligent_unwrap_phase_difference(left_H, right_H):
    """
    Unwrap channel phase change across frequency while ignoring deep fades.

    Each new reliable bin chooses the 2*pi branch nearest a local linear
    prediction from preceding reliable bins. Unreliable bins are filled by
    interpolation after the unwrap, so random phase in spectral nulls cannot
    introduce permanent branch errors.
    """
    left_H = np.asarray(left_H, dtype=np.complex128)
    right_H = np.asarray(right_H, dtype=np.complex128)
    ratio = right_H * np.conj(left_H)
    weight = np.abs(left_H) * np.abs(right_H)
    finite = np.isfinite(ratio) & np.isfinite(weight)
    if not np.any(finite):
        return np.zeros_like(weight)

    reliable = finite & (weight > 0.03 * np.max(weight[finite]))
    reliable_indices = np.flatnonzero(reliable)
    if len(reliable_indices) < 3:
        return np.unwrap(np.angle(ratio))

    wrapped = np.angle(ratio)
    unwrapped_reliable = np.empty(len(reliable_indices), dtype=np.float64)
    for position, bin_index in enumerate(reliable_indices):
        value = wrapped[bin_index]
        if position == 0:
            unwrapped_reliable[position] = value
            continue

        history_start = max(0, position - 12)
        history_indices = reliable_indices[history_start:position]
        history_phase = unwrapped_reliable[history_start:position]
        if len(history_indices) >= 2:
            phase_steps = np.diff(history_phase)
            bin_steps = np.diff(history_indices)
            local_slope = np.median(phase_steps / bin_steps)
            local_slope = float(np.clip(local_slope, -0.2, 0.2))
        else:
            local_slope = 0.0
        prediction = (
            history_phase[-1]
            + local_slope * (bin_index - history_indices[-1])
        )

        branch = np.round((prediction - value) / (2.0 * np.pi))
        branch = np.clip(branch, -64, 64)
        unwrapped_reliable[position] = value + 2.0 * np.pi * branch

    all_indices = np.arange(len(ratio), dtype=np.float64)
    return np.interp(
        all_indices,
        reliable_indices.astype(np.float64),
        unwrapped_reliable,
    )


def _cp_timing_metric(data_region, start):
    if start < 0 or start + BLOCK_LENGTH > len(data_region):
        return -np.inf
    block = data_region[start:start + BLOCK_LENGTH]
    prefix = block[:OFDM_CP_LENGTH]
    repeated = block[OFDM_SIZE:OFDM_SIZE + OFDM_CP_LENGTH]
    denom = np.linalg.norm(prefix) * np.linalg.norm(repeated)
    if denom == 0.0:
        return -np.inf
    return float(np.abs(np.vdot(prefix, repeated)) / denom)


def refine_block_start(data_region, expected_start, search_radius):
    """Refine an OFDM block start using cyclic-prefix correlation."""
    low = max(0, int(round(expected_start)) - search_radius)
    high = min(
        len(data_region) - BLOCK_LENGTH,
        int(round(expected_start)) + search_radius,
    )
    if high < low:
        raise ValueError("Not enough samples to extract another OFDM block.")

    candidates = np.arange(low, high + 1, dtype=np.int64)
    metrics = np.array(
        [_cp_timing_metric(data_region, int(start)) for start in candidates]
    )
    best = np.max(metrics)
    near_best = candidates[metrics >= best * 0.999]
    return int(near_best[np.argmin(np.abs(near_best - expected_start))])


def extract_timing_tracked_blocks(data_region):
    """
    Estimate the received block period from CP correlation, then fractionally
    resample every block back onto the transmitter's nominal sample grid.
    """
    if len(data_region) < BLOCK_LENGTH:
        raise ValueError("No complete OFDM blocks after preamble.")

    measured_starts = []
    cp_metrics = []
    block_rms = []
    expected = 0.0
    timing_step = float(BLOCK_LENGTH)

    while True:
        radius = 16 if measured_starts else OFDM_CP_LENGTH // 4
        try:
            start = refine_block_start(data_region, expected, radius)
        except ValueError:
            break
        if measured_starts and start <= measured_starts[-1]:
            break
        measured_starts.append(start)
        cp_metrics.append(_cp_timing_metric(data_region, start))
        block = data_region[start:start + BLOCK_LENGTH]
        block_rms.append(float(np.sqrt(np.mean(block * block))))
        if len(measured_starts) >= 2:
            measured_step = measured_starts[-1] - measured_starts[-2]
            timing_step = 0.9 * timing_step + 0.1 * measured_step
        expected = start + timing_step
        if expected + BLOCK_LENGTH > len(data_region) + radius:
            break

    measured_starts = np.asarray(measured_starts, dtype=np.float64)
    cp_metrics = np.asarray(cp_metrics, dtype=np.float64)
    block_rms = np.asarray(block_rms, dtype=np.float64)

    # Do not infer the frame endpoint from CP correlation or energy. In a
    # noisy acoustic channel both can fall sharply inside a valid frame. The
    # decoded header later supplies the exact number of OFDM blocks.
    block_indices = np.arange(len(measured_starts), dtype=np.float64)
    reliable = cp_metrics >= 0.5
    if np.count_nonzero(reliable) < 2:
        reliable = np.ones(len(measured_starts), dtype=bool)

    # A multipath channel can move the strongest CP-correlation peak around
    # inside the CP even when the sampling clocks match. Estimate clock drift
    # only from long-baseline pairs, where that local peak motion is small
    # relative to accumulated sampling error.
    reliable_indices = np.flatnonzero(reliable)
    minimum_separation = max(4, len(measured_starts) // 3)
    long_slopes = []
    for left_offset, left_index in enumerate(reliable_indices):
        for right_index in reliable_indices[left_offset + 1:]:
            separation = right_index - left_index
            if separation >= minimum_separation:
                long_slopes.append(
                    (measured_starts[right_index] - measured_starts[left_index])
                    / separation
                )
    measured_step = (
        float(np.median(long_slopes))
        if long_slopes
        else float(BLOCK_LENGTH)
    )
    # data_region already begins at the protocol-defined end of the chirp and
    # Golay preamble. A multipath CP peak can wander hundreds of samples while
    # remaining inside the valid CP, so it must not redefine block zero.
    first_start = 0.0
    received_step = float(BLOCK_LENGTH)
    sample_scale = 1.0
    source_axis = np.arange(len(data_region), dtype=np.float64)
    blocks = []
    fitted_starts = []

    for block_index in block_indices:
        start = first_start + block_index * received_step
        sample_positions = (
            start + np.arange(BLOCK_LENGTH, dtype=np.float64) * sample_scale
        )
        if sample_positions[-1] > len(data_region) + 64:
            break
        sample_positions = np.minimum(sample_positions, len(data_region) - 1)
        blocks.append(np.interp(sample_positions, source_axis, data_region))
        fitted_starts.append(start)

    return (
        np.asarray(blocks),
        np.asarray(fitted_starts, dtype=np.float64),
        sample_scale,
    )


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
    # Saturating reliability weights: weak bins contribute little, while very
    # strong bins cannot dominate the LDPC decoder.
    if H_active is not None:
        power = np.abs(np.asarray(H_active)) ** 2
        typical_power = np.median(power) + 1e-12
        w = power / (power + CHANNEL_REGULARISATION * typical_power)
        w[power < 0.05 * typical_power] = 0.0
        w = np.clip(w / (np.mean(w) + 1e-12), 0.0, 2.5)
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


def apply_ldpc_decoding(
    LLR_blocks,
    decoder,
    return_status=False,
    report_unconverged=True,
):
    """
    Decode (35, 1464) soft LLRs → (35, 732) info bits.
    LLR convention: positive = bit 0, negative = bit 1.
    Decoder output app: negative = bit 1 (same convention).
    Info bits are at app[1:733] due to a consistent 1-position offset
    in this decoder implementation.
    """
    info = np.empty((LDPC_BLOCKS_PER_GROUP, LDPC_INFO_BITS), dtype=np.uint8)
    converged = np.zeros(len(LLR_blocks), dtype=bool)
    retry_scales = (0.5, 2.0, 4.0, 0.25, 8.0)
    unconverged = 0
    for i, llr in enumerate(LLR_blocks):
        app, iterations = decoder.decode(llr)
        best_app = app
        best_iterations = iterations

        if iterations >= LDPC_MAX_ITERATIONS:
            for scale in retry_scales:
                candidate_app, candidate_iterations = decoder.decode(llr * scale)
                if candidate_iterations < best_iterations:
                    best_app = candidate_app
                    best_iterations = candidate_iterations
                if candidate_iterations < LDPC_MAX_ITERATIONS:
                    break

        if best_iterations >= LDPC_MAX_ITERATIONS:
            unconverged += 1
        else:
            converged[i] = True
        info[i] = (best_app[:LDPC_INFO_BITS] < 0).astype(np.uint8)

    if unconverged and report_unconverged:
        print(
            f"Warning: {unconverged}/{len(LLR_blocks)} LDPC codewords "
            "did not pass parity checks."
        )
    if return_status:
        return info, converged
    return info


# ── Header parser ─────────────────────────────────────────────────────────────

def parse_header(data):
    if len(data) < 6:
        raise ValueError("Too few bytes to parse header.")
    total_hdr = struct.unpack(">H", data[:2])[0]
    if total_hdr < 6 or len(data) < total_hdr:
        raise ValueError(f"Invalid header length {total_hdr}.")
    file_len  = struct.unpack(">I", data[2:6])[0]
    try:
        filename = data[6:total_hdr].decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("Header filename is not valid UTF-8.") from error
    if not filename or any(ord(character) < 32 for character in filename):
        raise ValueError("Header filename contains invalid characters.")
    return total_hdr, file_len, filename


def frame_layout_from_header(total_header_bytes, file_bytes):
    """Calculate the exact coded frame length described by the header."""
    total_information_bits = 8 * (total_header_bytes + file_bytes)
    information_bits_per_group = LDPC_BLOCKS_PER_GROUP * LDPC_INFO_BITS
    group_count = (
        total_information_bits + information_bits_per_group - 1
    ) // information_bits_per_group
    data_block_count = group_count * DATA_OFDM_SYMBOLS_PER_GROUP
    pilot_block_count = (
        (data_block_count - 1) // (PILOT_BLOCK_PERIOD - 1)
        if data_block_count
        else 0
    )
    return (
        group_count,
        data_block_count,
        pilot_block_count,
        data_block_count + pilot_block_count,
    )


def _equalise_frame_blocks(
    blocks,
    h_active,
    lam,
    phase_common_rate=0.0,
    phase_slope_rate=0.0,
    diagnostics=None,
    constellation_data=None,
    pilot_channel_update=PILOT_CHANNEL_UPDATE,
    reset_phase_rates_on_pilot=True,
    phase_coherence_threshold=PHASE_COHERENCE_THRESHOLD,
):
    """
    Equalise and phase-track one frame with three layers:

    Golay initialises H and phase rates, each QPSK data symbol updates the
    fourth-power tracker, and every periodic pilot resets channel/phase state.
    """
    data_groups = []
    channel_groups = []
    current_data = []
    current_channels = []
    current_H = _smooth_channel(h_active)
    pilot_count = 0
    common = 0.0
    slope = 0.0
    common_rate = float(phase_common_rate)
    slope_rate = float(phase_slope_rate)
    previous_block_index = -1

    for block_index, block in enumerate(blocks):
        if (block_index + 1) % PILOT_BLOCK_PERIOD == 0:
            pilot_H = estimate_channel_from_pilot(block)
            current_H = _blend_channel(
                current_H,
                pilot_H,
                update=pilot_channel_update,
            )
            pilot_count += 1
            if diagnostics is not None:
                diagnostics.append(
                    {
                        "block": block_index + 1,
                        "kind": "pilot",
                    }
                )
            common = 0.0
            slope = 0.0
            if reset_phase_rates_on_pilot:
                common_rate = 0.0
                slope_rate = 0.0
            previous_block_index = block_index
            continue

        H_block = current_H
        equalised = equalise_block_active(block, H_block, lam)
        block_delta = max(1, block_index - previous_block_index)
        predicted_common = common + common_rate * block_delta
        predicted_slope = slope + slope_rate * block_delta
        (
            tracked,
            measured_common,
            measured_slope,
            quality,
        ) = track_qpsk_symbol_phase(
            equalised,
            predicted_common=predicted_common,
            predicted_slope=predicted_slope,
            slope_search_span=0.0015 * block_delta,
        )

        measurement_used = quality >= phase_coherence_threshold
        if measurement_used:
            measured_common_rate = (
                measured_common - common
            ) / block_delta
            measured_slope_rate = (
                measured_slope - slope
            ) / block_delta
            common_rate = 0.8 * common_rate + 0.2 * measured_common_rate
            slope_rate = 0.8 * slope_rate + 0.2 * measured_slope_rate
            common = measured_common
            slope = measured_slope
        else:
            common = predicted_common
            slope = predicted_slope

        if constellation_data is not None:
            constellation_data.append(
                {
                    "block": block_index + 1,
                    "raw": equalised.copy(),
                    "tracked": tracked.copy(),
                }
            )

        if diagnostics is not None:
            diagnostics.append(
                {
                    "block": block_index + 1,
                    "kind": "data",
                    "common": common,
                    "slope": slope,
                    "timing_samples": -slope * OFDM_SIZE / (2.0 * np.pi),
                    "common_rate": common_rate,
                    "slope_rate": slope_rate,
                    "coherence": quality,
                    "used": measurement_used,
                }
            )

        current_data.append(tracked)
        current_channels.append(H_block)
        previous_block_index = block_index

        if len(current_data) == DATA_OFDM_SYMBOLS_PER_GROUP:
            data_groups.append(np.asarray(current_data))
            channel_groups.append(np.mean(np.abs(current_channels), axis=0))
            current_data = []
            current_channels = []

    if current_data:
        valid_symbol_count = len(current_data)
        while len(current_data) < DATA_OFDM_SYMBOLS_PER_GROUP:
            current_data.append(
                np.zeros(DATA_CARRIERS_PER_SYMBOL, dtype=np.complex128)
            )
            current_channels.append(current_channels[-1])
        padded_group = np.zeros(
            (
                DATA_OFDM_SYMBOLS_PER_GROUP,
                DATA_CARRIERS_PER_SYMBOL,
            ),
            dtype=np.complex128,
        )
        padded_group[:valid_symbol_count] = np.asarray(
            current_data[:valid_symbol_count]
        )
        data_groups.append(padded_group)
        channel_groups.append(np.mean(np.abs(current_channels), axis=0))

    return data_groups, channel_groups, pilot_count


def print_phase_diagnostics(diagnostics):
    """Print a compact per-block view of the three-layer phase tracker."""
    if not diagnostics:
        return
    print("Phase tracking diagnostics:")
    print(" block   type      common(rad)  slope(rad/bin)  timing(samples)  coh   used")
    for row in diagnostics:
        if row["kind"] == "pilot":
            print(f" {row['block']:5d}   PILOT RESET")
            continue
        print(
            f" {row['block']:5d}   data   "
            f"{row['common']:+11.4f}  "
            f"{row['slope']:+14.6e}  "
            f"{row['timing_samples']:+15.3f}  "
            f"{row['coherence']:.3f}  "
            f"{'yes' if row['used'] else 'no'}"
        )


def save_constellation_plot(constellation_data, output_path):
    """Save raw/corrected constellation maps before and after the first pilot."""
    if not constellation_data:
        print("Warning: no data symbols available for constellation plot.")
        return

    import matplotlib.pyplot as plt

    ranges = [
        ("Before pilot: blocks 1-19", 1, PILOT_BLOCK_PERIOD - 1),
        (
            "After pilot: blocks 21-39",
            PILOT_BLOCK_PERIOD + 1,
            2 * PILOT_BLOCK_PERIOD - 1,
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)

    for row, (title, first_block, last_block) in enumerate(ranges):
        selected = [
            entry
            for entry in constellation_data
            if first_block <= entry["block"] <= last_block
        ]
        for column, field in enumerate(("raw", "tracked")):
            axis = axes[row, column]
            if selected:
                points = np.concatenate([entry[field] for entry in selected])
                if len(points) > 10_000:
                    point_indices = np.linspace(
                        0,
                        len(points) - 1,
                        10_000,
                        dtype=np.int64,
                    )
                    points = points[point_indices]
                axis.scatter(
                    points.real,
                    points.imag,
                    s=4,
                    alpha=0.22,
                    linewidths=0,
                )
            axis.axhline(0.0, color="0.75", linewidth=0.8)
            axis.axvline(0.0, color="0.75", linewidth=0.8)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(True, alpha=0.2)
            axis.set_xlabel("In-phase")
            axis.set_ylabel("Quadrature")
            stage = "Before phase tracking" if field == "raw" else "After phase tracking"
            axis.set_title(f"{title}\n{stage}")

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Constellation plot saved to: {output_path}")


def _decode_ldpc_groups(
    data_groups,
    channel_groups,
    decoder,
    report_unconverged=True,
):
    all_info_bits = []
    for group_index, (group_symbols, group_H) in enumerate(
        zip(data_groups, channel_groups),
        start=1,
    ):
        llr_blocks = deinterleave_group_soft(group_symbols, H_active=group_H)
        info_blocks, converged = apply_ldpc_decoding(
            llr_blocks,
            decoder,
            return_status=True,
            report_unconverged=False,
        )
        converged_count = int(np.count_nonzero(converged))
        unconverged = len(converged) - converged_count
        if unconverged and report_unconverged:
            print(
                f"LDPC group {group_index}: "
                f"{unconverged}/{len(converged)} codewords "
                "did not pass parity checks."
            )
        all_info_bits.append(info_blocks.reshape(-1))
    return np.concatenate(all_info_bits).astype(np.uint8)


def _bits_to_bytes(bitstream):
    usable = (len(bitstream) // 8) * 8
    return np.packbits(bitstream[:usable], bitorder="big").tobytes()


def _evaluate_header_candidate(blocks, decoder, lam, candidate):
    """Decode and validate one channel/phase hypothesis for the first group."""
    tracking_blocks = min(len(blocks), 2 * PILOT_BLOCK_PERIOD)
    groups, channels, _ = _equalise_frame_blocks(
        blocks[:tracking_blocks],
        candidate["channel"],
        lam,
        phase_common_rate=candidate["common_rate"],
        phase_slope_rate=candidate["slope_rate"],
        pilot_channel_update=candidate["pilot_update"],
        reset_phase_rates_on_pilot=candidate["reset_rates"],
        phase_coherence_threshold=candidate["coherence_threshold"],
    )
    if not groups:
        raise ValueError("Candidate produced no complete LDPC group.")

    llr = deinterleave_group_soft(groups[0], H_active=channels[0])
    info, converged = apply_ldpc_decoding(
        llr,
        decoder,
        return_status=True,
        report_unconverged=False,
    )
    data = _bits_to_bytes(info.reshape(-1))
    total_hdr, file_len, filename = parse_header(data)
    layout = frame_layout_from_header(total_hdr, file_len)
    if layout[3] > len(blocks):
        raise ValueError(
            f"Header requires {layout[3]} blocks, only {len(blocks)} available."
        )

    result = dict(candidate)
    result.update(
        total_hdr=total_hdr,
        file_len=file_len,
        filename=filename,
        layout=layout,
        converged=int(np.count_nonzero(converged)),
    )
    return result


def select_header_hypothesis(
    blocks,
    h_active,
    golay_common_rate,
    golay_slope_rate,
    decoder,
    lam,
):
    """Try several general channel hypotheses and select a valid header."""
    candidates = []
    thresholds = (PHASE_COHERENCE_THRESHOLD, 0.20)
    for threshold in thresholds:
        candidates.append(
            {
                "name": f"Golay, coherence {threshold:.2f}",
                "channel": h_active,
                "common_rate": golay_common_rate,
                "slope_rate": golay_slope_rate,
                "pilot_update": PILOT_CHANNEL_UPDATE,
                "reset_rates": True,
                "coherence_threshold": threshold,
            }
        )

    if len(blocks) >= PILOT_BLOCK_PERIOD:
        first_pilot = _smooth_channel(
            estimate_channel_from_pilot(blocks[PILOT_BLOCK_PERIOD - 1])
        )
        for threshold in thresholds:
            candidates.append(
                {
                    "name": f"first pilot, coherence {threshold:.2f}",
                    "channel": first_pilot,
                    "common_rate": 0.0,
                    "slope_rate": 0.0,
                    "pilot_update": 1.0,
                    "reset_rates": True,
                    "coherence_threshold": threshold,
                }
            )

    if len(blocks) >= 2 * PILOT_BLOCK_PERIOD:
        channel, common_rate, slope_rate = (
            bootstrap_channel_from_periodic_pilots(blocks)
        )
        for threshold in thresholds:
            candidates.append(
                {
                    "name": (
                        f"two-pilot extrapolation, coherence {threshold:.2f}"
                    ),
                    "channel": channel,
                    "common_rate": common_rate,
                    "slope_rate": slope_rate,
                    "pilot_update": 1.0,
                    "reset_rates": False,
                    "coherence_threshold": threshold,
                }
            )

    valid = []
    failures = []
    for candidate in candidates:
        try:
            valid.append(
                _evaluate_header_candidate(
                    blocks,
                    decoder,
                    lam,
                    candidate,
                )
            )
        except ValueError as error:
            failures.append(f"{candidate['name']}: {error}")

    if not valid:
        details = "\n  ".join(failures)
        raise ValueError(
            "No channel hypothesis produced a valid header."
            + (f"\n  {details}" if details else "")
        )

    valid.sort(key=lambda item: item["converged"], reverse=True)
    return valid[0], valid


# ── Main decode pipeline ──────────────────────────────────────────────────────

def decode_signal(
    signal,
    use_ldpc=True,
    lam=1e-6,
    ldpc_decoder=None,
    phase_debug=False,
    constellation_path=None,
):
    signal = np.asarray(signal, dtype=np.float64).squeeze()

    last_chirp_start = find_chirp_end(signal)
    rough_golay_signal_start = (
        last_chirp_start + CHIRP_LENGTH + GUARD_LENGTH
    )
    (
        refined_golay_start,
        peak_delay,
        H0,
        h0,
    ) = refine_sync_with_golay(signal, rough_golay_signal_start)
    data_start = (
        refined_golay_start
        + GOLAY_PAIR_COUNT
        * 2
        * (GOLAY_LENGTH + GOLAY_CP_LENGTH)
        + GUARD_AFTER_GOLAY
    )
    (
        H_rfft,
        h_avg,
        H_estimates,
        h_estimates,
        golay_centres,
        golay_common_rate,
        golay_slope_rate,
    ) = estimate_initial_channel(
        signal,
        refined_golay_start,
        data_start,
    )

    print(
        f"Last chirp found at sample {last_chirp_start}  "
        f"({last_chirp_start/SAMPLE_RATE:.3f} s)"
    )
    print(
        f"Golay refined start: {refined_golay_start}  "
        f"(delay {peak_delay:+d} samples)"
    )
    print(
        f"OFDM data starts at sample {data_start}  "
        f"({data_start/SAMPLE_RATE:.3f} s)"
    )

    h_active = H_rfft[ACTIVE_BINS]
    print(f"Golay H: mean|H|={np.mean(np.abs(h_active)):.3f}  "
          f"min={np.min(np.abs(h_active)):.4f}  max={np.max(np.abs(h_active)):.4f}")
    print(
        "Golay phase prior: "
        f"common={golay_common_rate * BLOCK_LENGTH:+.3e} rad/block, "
        f"slope={golay_slope_rate * BLOCK_LENGTH:+.3e} rad/bin/block"
    )

    data_region = signal[data_start:]
    blocks, block_starts, sample_scale = extract_timing_tracked_blocks(data_region)
    print(f"OFDM blocks available: {len(blocks)}")
    print(
        f"Timing correction: first={block_starts[0]:.2f} samples, "
        f"sample scale={sample_scale:.9f} "
        f"({(sample_scale - 1.0) * 1e6:+.1f} ppm)"
    )
    if phase_debug:
        pilot_rows, pilot_sampling_ppm, pilot_fit = (
            estimate_sampling_mismatch_from_pilots(blocks)
        )
        print_pilot_timing_diagnostics(
            pilot_rows,
            pilot_sampling_ppm,
            pilot_fit,
            title="Preliminary pilot timing diagnostics",
        )

    if not use_ldpc:
        phase_diagnostics = [] if phase_debug else None
        constellation_data = [] if constellation_path else None
        data_groups, _, pilot_count = _equalise_frame_blocks(
            blocks,
            h_active,
            lam,
            phase_common_rate=golay_common_rate * BLOCK_LENGTH,
            phase_slope_rate=golay_slope_rate * BLOCK_LENGTH,
            diagnostics=phase_diagnostics,
            constellation_data=constellation_data,
        )
        print_phase_diagnostics(phase_diagnostics)
        if constellation_path:
            save_constellation_plot(constellation_data, constellation_path)
        print(f"Periodic pilots used: {pilot_count}")
        all_symbols = np.concatenate(
            [group.reshape(-1) for group in data_groups]
        )
        info_bitstream = qpsk_demod(all_symbols)
        print(f"Uncoded groups: {len(data_groups)}")
    else:
        decoder = ldpc_decoder or _load_ldpc_decoder()
        first_group_blocks = frame_layout_from_header(6, 0)[3]
        if len(blocks) < first_group_blocks:
            raise ValueError(
                f"Need at least {first_group_blocks} OFDM blocks to decode "
                f"the header, but only {len(blocks)} are available."
            )

        selected, valid_candidates = select_header_hypothesis(
            blocks,
            h_active,
            golay_common_rate * BLOCK_LENGTH,
            golay_slope_rate * BLOCK_LENGTH,
            decoder,
            lam,
        )
        print(
            f"Header hypothesis: {selected['name']} "
            f"({selected['converged']}/{LDPC_BLOCKS_PER_GROUP} "
            "codewords converged)"
        )
        if len(valid_candidates) > 1:
            print(f"Valid header hypotheses: {len(valid_candidates)}")

        frame_channel = selected["channel"]
        frame_common_rate = selected["common_rate"]
        frame_slope_rate = selected["slope_rate"]
        frame_pilot_update = selected["pilot_update"]
        reset_phase_rates_on_pilot = selected["reset_rates"]
        frame_coherence_threshold = selected["coherence_threshold"]
        total_hdr = selected["total_hdr"]
        file_len = selected["file_len"]
        filename = selected["filename"]

        if phase_debug or constellation_path:
            phase_diagnostics = [] if phase_debug else None
            constellation_data = [] if constellation_path else None
            _equalise_frame_blocks(
                blocks[:min(len(blocks), 2 * PILOT_BLOCK_PERIOD)],
                frame_channel,
                lam,
                phase_common_rate=frame_common_rate,
                phase_slope_rate=frame_slope_rate,
                diagnostics=phase_diagnostics,
                constellation_data=constellation_data,
                pilot_channel_update=frame_pilot_update,
                reset_phase_rates_on_pilot=reset_phase_rates_on_pilot,
                phase_coherence_threshold=frame_coherence_threshold,
            )
            print_phase_diagnostics(phase_diagnostics)
            if constellation_path:
                save_constellation_plot(
                    constellation_data,
                    constellation_path,
                )

        (
            expected_groups,
            expected_data_blocks,
            expected_pilot_blocks,
            expected_total_blocks,
        ) = frame_layout_from_header(total_hdr, file_len)
        print(
            "Header frame length: "
            f"{expected_groups} LDPC groups, "
            f"{expected_data_blocks} data blocks, "
            f"{expected_pilot_blocks} pilots, "
            f"{expected_total_blocks} total OFDM blocks"
        )
        if len(blocks) < expected_total_blocks:
            raise ValueError(
                f"Header requires {expected_total_blocks} OFDM blocks, "
                f"but only {len(blocks)} were received."
            )

        frame_blocks = blocks[:expected_total_blocks]
        if phase_debug:
            pilot_rows, pilot_sampling_ppm, pilot_fit = (
                estimate_sampling_mismatch_from_pilots(frame_blocks)
            )
            print_pilot_timing_diagnostics(
                pilot_rows,
                pilot_sampling_ppm,
                pilot_fit,
                title="In-frame pilot timing diagnostics",
            )
        data_groups, channel_groups, pilot_count = _equalise_frame_blocks(
            frame_blocks,
            frame_channel,
            lam,
            phase_common_rate=frame_common_rate,
            phase_slope_rate=frame_slope_rate,
            pilot_channel_update=frame_pilot_update,
            reset_phase_rates_on_pilot=reset_phase_rates_on_pilot,
            phase_coherence_threshold=frame_coherence_threshold,
        )
        data_groups = data_groups[:expected_groups]
        channel_groups = channel_groups[:expected_groups]
        print(f"Periodic pilots used: {pilot_count}")
        print(f"LDPC groups: {len(data_groups)}")
        info_bitstream = _decode_ldpc_groups(
            data_groups,
            channel_groups,
            decoder,
        )

    data_bytes = _bits_to_bytes(info_bitstream)
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
    #parser.add_argument("output_file", nargs="?", default=None)
    parser.add_argument("--no-ldpc",   action="store_true")
    parser.add_argument("--lam",       type=float, default=1e-6)
    parser.add_argument(
        "--phase-debug",
        action="store_true",
        help="Print per-block common phase, phase slope and coherence.",
    )
    parser.add_argument(
        "--plot-constellation",
        nargs="?",
        const="constellation.png",
        default=None,
        metavar="PNG",
        help="Save raw and phase-corrected constellation maps.",
    )
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
        signal,
        use_ldpc=not args.no_ldpc,
        lam=args.lam,
        phase_debug=args.phase_debug,
        constellation_path=args.plot_constellation,
    )
    out_path = os.path.join(os.getcwd(), filename)

    Path(out_path).write_bytes(payload)
    print(f"Saved '{out_path}'  ({file_len} bytes)")


if __name__ == "__main__":
    main()
