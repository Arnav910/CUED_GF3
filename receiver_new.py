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

import numpy as np
from scipy.io.wavfile import read as wav_read
from scipy.signal import find_peaks

try:
    from .Transmitter import appendix_a_pilot_values
except ImportError:
    from Transmitter import appendix_a_pilot_values


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
LDPC_MAX_ITERATIONS       = 200
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

_APPENDIX_A_VALUES = appendix_a_pilot_values()
if len(_APPENDIX_A_VALUES) != OFDM_SIZE // 2 - 1:
    raise ValueError("Appendix A pilot sequence does not match the OFDM size.")
ACTIVE_PILOT_VALUES = _APPENDIX_A_VALUES[ACTIVE_BINS - 1]


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

def estimate_channel_golay_pairs(signal, chirp_start):
    """Estimate the channel separately from each transmitted Golay pair."""
    a, b = _golay_pair()
    L    = GOLAY_LENGTH
    A    = np.fft.rfft(a, n=L)
    B    = np.fft.rfft(b, n=L)
    denom = np.abs(A)**2 + np.abs(B)**2

    r = signal[chirp_start:]
    pair_stride = 2 * (GOLAY_LENGTH + GOLAY_CP_LENGTH)
    estimates = []
    centres = []
    first_a = CHIRPS_LENGTH + GOLAY_CP_LENGTH

    for pair_index in range(GOLAY_PAIR_COUNT):
        a_off = first_a + pair_index * pair_stride
        b_off = a_off + GOLAY_LENGTH + GOLAY_CP_LENGTH
        if b_off + L > len(r):
            raise ValueError("Not enough samples for all Golay channel estimates.")
        Ya = np.fft.rfft(r[a_off:a_off + L], n=L)
        Yb = np.fft.rfft(r[b_off:b_off + L], n=L)
        estimates.append((np.conj(A) * Ya + np.conj(B) * Yb) / denom)
        centres.append(
            chirp_start
            + a_off
            + GOLAY_LENGTH
            + GOLAY_CP_LENGTH / 2
        )

    return np.asarray(estimates), np.asarray(centres, dtype=np.float64)


def _fit_affine_phase(reference, estimate):
    """
    Robustly fit estimate/reference phase as common phase plus bin slope.

    This avoids directly unwrapping noisy carrier phases. An initial slope is
    obtained from adjacent-bin circular differences, followed by iteratively
    reweighted circular regression that suppresses deep fades and outliers.
    """
    reference = np.asarray(reference, dtype=np.complex128)
    estimate = np.asarray(estimate, dtype=np.complex128)
    ratio = estimate * np.conj(reference)
    weight = np.abs(reference) * np.abs(estimate)
    finite = np.isfinite(ratio) & np.isfinite(weight)
    threshold = 0.02 * np.max(weight[finite])
    valid = finite & (weight > threshold)
    if np.count_nonzero(valid) < 3:
        raise ValueError("Not enough reliable carriers for phase-slope fitting.")

    x = ACTIVE_BINS.astype(np.float64)
    x = x - np.mean(x)
    unit_ratio = ratio / np.maximum(np.abs(ratio), 1e-12)

    adjacent = valid[:-1] & valid[1:]
    if np.count_nonzero(adjacent) < 2:
        raise ValueError("Not enough adjacent reliable carriers for phase fitting.")
    adjacent_phase = np.angle(
        unit_ratio[1:][adjacent] * np.conj(unit_ratio[:-1][adjacent])
    )
    adjacent_weight = np.sqrt(
        weight[1:][adjacent] * weight[:-1][adjacent]
    )
    order = np.argsort(adjacent_phase)
    sorted_phase = adjacent_phase[order]
    cumulative_weight = np.cumsum(adjacent_weight[order])
    phase_slope = sorted_phase[
        np.searchsorted(cumulative_weight, cumulative_weight[-1] / 2)
    ]

    residual_unit = unit_ratio[valid] * np.exp(-1j * phase_slope * x[valid])
    common_phase = np.angle(np.sum(weight[valid] * residual_unit))

    design = np.column_stack([np.ones(np.count_nonzero(valid)), x[valid]])
    base_weight = weight[valid] / np.max(weight[valid])
    for _ in range(8):
        model = common_phase + phase_slope * x[valid]
        residual = np.angle(unit_ratio[valid] * np.exp(-1j * model))
        scale = 1.4826 * np.median(
            np.abs(residual - np.median(residual))
        ) + 0.03
        robust_weight = np.minimum(
            1.0,
            1.5 * scale / np.maximum(np.abs(residual), 1e-12),
        )
        fit_weight = base_weight * robust_weight
        weighted_design = design * np.sqrt(fit_weight[:, None])
        correction = np.linalg.lstsq(
            weighted_design,
            residual * np.sqrt(fit_weight),
            rcond=None,
        )[0]
        common_phase += correction[0]
        phase_slope += correction[1]

    return float(common_phase), float(phase_slope)


def _unwrap_linear_phase_track(phases, sample_positions):
    """Choose 2π branches that make a short phase track most nearly linear."""
    phases = np.asarray(phases, dtype=np.float64)
    sample_positions = np.asarray(sample_positions, dtype=np.float64)
    if len(phases) <= 1:
        return phases.copy()

    centred_time = sample_positions - np.mean(sample_positions)
    best_track = None
    best_error = np.inf
    branch_values = np.arange(-3, 4, dtype=np.int64)

    for branch_indices in np.ndindex(*(len(branch_values),) * (len(phases) - 1)):
        candidate = phases.copy()
        candidate[1:] += 2.0 * np.pi * branch_values[list(branch_indices)]
        fit = np.polyfit(centred_time, candidate, 1)
        residual = candidate - np.polyval(fit, centred_time)
        error = float(np.mean(residual**2))
        if error < best_error:
            best_error = error
            best_track = candidate

    return best_track


def estimate_initial_channel(signal, chirp_start, data_start):
    """
    Use all Golay pairs to estimate the channel and extrapolate their measured
    common-phase/frequency-slope drift to the first OFDM FFT window.
    """
    estimates, centres = estimate_channel_golay_pairs(signal, chirp_start)
    active_estimates = estimates[:, ACTIVE_BINS]
    reference = active_estimates[0]

    common = np.zeros(GOLAY_PAIR_COUNT, dtype=np.float64)
    slope = np.zeros(GOLAY_PAIR_COUNT, dtype=np.float64)
    for index in range(1, GOLAY_PAIR_COUNT):
        common[index], slope[index] = _fit_affine_phase(
            reference, active_estimates[index]
        )
    common = _unwrap_linear_phase_track(common, centres)

    common_rate, common_origin = np.polyfit(centres, common, 1)
    slope_rate, slope_origin = np.polyfit(centres, slope, 1)
    first_fft_centre = data_start + OFDM_CP_LENGTH + OFDM_SIZE / 2
    predicted_common = common_rate * first_fft_centre + common_origin
    predicted_slope = slope_rate * first_fft_centre + slope_origin

    x = ACTIVE_BINS.astype(np.float64)
    x = x - np.mean(x)
    aligned = []
    for index, estimate in enumerate(active_estimates):
        correction = (
            predicted_common - common[index]
            + (predicted_slope - slope[index]) * x
        )
        aligned.append(estimate * np.exp(1j * correction))

    initial = estimates[0].copy()
    initial[ACTIVE_BINS] = np.mean(aligned, axis=0)
    return initial, common_rate, slope_rate


# ── OFDM equaliser ────────────────────────────────────────────────────────────

def equalise_block(block, H_rfft, lam=1e-6):
    Y = np.fft.rfft(block[OFDM_CP_LENGTH:], n=OFDM_SIZE)
    H = H_rfft[ACTIVE_BINS]
    return Y[ACTIVE_BINS] * np.conj(H) / (np.abs(H)**2 + lam)


def equalise_block_active(block, H_active, lam=1e-6):
    """Equalise one block using an active-bin-only channel estimate."""
    Y = np.fft.rfft(block[OFDM_CP_LENGTH:], n=OFDM_SIZE)[ACTIVE_BINS]
    H = np.asarray(H_active, dtype=np.complex128)
    return Y * np.conj(H) / (np.abs(H)**2 + lam)


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
    )
    slope = float(slope_grid[np.argmax(coherence_grid)])

    fine_step = coarse_step / 20.0
    fine_grid = slope + np.arange(-coarse_step, coarse_step + fine_step, fine_step)
    fine_coherence = np.abs(
        np.exp(-4j * np.outer(fine_grid, valid_x))
        @ (valid_weights * valid_fourth)
    )
    slope = float(fine_grid[np.argmax(fine_coherence)])

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


def track_qpsk_group_phase(
    group_symbols,
    block_positions=None,
    initial_common_rate=0.0,
    initial_slope_rate=0.0,
):
    """Track residual phase independently across all symbols in one LDPC group."""
    if block_positions is None:
        block_positions = np.arange(len(group_symbols), dtype=np.float64)
    else:
        block_positions = np.asarray(block_positions, dtype=np.float64)
    if len(block_positions) != len(group_symbols):
        raise ValueError("One physical block position is required per data symbol.")

    tracked = np.empty_like(group_symbols, dtype=np.complex128)
    common = 0.0
    slope = 0.0
    common_rate = float(initial_common_rate)
    slope_rate = float(initial_slope_rate)
    qualities = []

    for index, symbols in enumerate(group_symbols):
        block_delta = (
            block_positions[index] - block_positions[index - 1]
            if index
            else 1.0
        )
        predicted_common = (
            common + common_rate * block_delta if index else common
        )
        predicted_slope = (
            slope + slope_rate * block_delta if index else slope
        )
        tracked[index], measured_common, measured_slope, quality = (
            track_qpsk_symbol_phase(
                symbols,
                predicted_common=predicted_common,
                predicted_slope=predicted_slope,
                slope_search_span=(
                    0.006 if index == 0 else 0.004 * block_delta
                ),
            )
        )

        common_delta = (measured_common - common) / block_delta
        slope_delta = (measured_slope - slope) / block_delta
        if index:
            common_rate = 0.8 * common_rate + 0.2 * common_delta
            slope_rate = 0.8 * slope_rate + 0.2 * slope_delta
        common = measured_common
        slope = measured_slope
        qualities.append(quality)

    return tracked, float(np.median(qualities))


def estimate_channel_from_pilot(block, lam=1e-12):
    """Estimate the channel on active data bins from one periodic pilot."""
    Y = np.fft.rfft(block[OFDM_CP_LENGTH:], n=OFDM_SIZE)[ACTIVE_BINS]
    X = ACTIVE_PILOT_VALUES
    return Y * np.conj(X) / (np.abs(X)**2 + lam)


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
            local_slope, local_intercept = np.polyfit(
                history_indices,
                history_phase,
                1,
                w=np.sqrt(weight[history_indices]),
            )
            local_slope = float(np.clip(local_slope, -0.2, 0.2))
            prediction = local_intercept + local_slope * bin_index
        else:
            prediction = history_phase[-1]

        branch = np.round((prediction - value) / (2.0 * np.pi))
        unwrapped_reliable[position] = value + 2.0 * np.pi * branch

    all_indices = np.arange(len(ratio), dtype=np.float64)
    return np.interp(
        all_indices,
        reliable_indices.astype(np.float64),
        unwrapped_reliable,
    )


def _channel_between_anchors(left_H, right_H, fraction):
    """
    Interpolate magnitude and the fitted affine phase change between anchors.
    Fractions outside [0, 1] extrapolate phase but hold the nearest magnitude.
    """
    phase_change = _intelligent_unwrap_phase_difference(left_H, right_H)
    phase_rotation = np.exp(1j * fraction * phase_change)

    mag_fraction = np.clip(fraction, 0.0, 1.0)
    left_mag = np.maximum(np.abs(left_H), 1e-12)
    right_mag = np.maximum(np.abs(right_H), 1e-12)
    magnitude = np.exp(
        (1.0 - mag_fraction) * np.log(left_mag)
        + mag_fraction * np.log(right_mag)
    )
    left_phase = left_H / left_mag
    return magnitude * left_phase * phase_rotation


def channel_for_block(block_index, anchors):
    """Interpolate or extrapolate the active-bin channel for one OFDM block."""
    positions = [position for position, _ in anchors]
    if block_index <= positions[0] or len(anchors) == 1:
        return anchors[0][1]

    for anchor_index in range(len(anchors) - 1):
        left_position, left_H = anchors[anchor_index]
        right_position, right_H = anchors[anchor_index + 1]
        if block_index <= right_position:
            fraction = (
                (block_index - left_position)
                / (right_position - left_position)
            )
            return _channel_between_anchors(left_H, right_H, fraction)

    left_position, left_H = anchors[-2]
    right_position, right_H = anchors[-1]
    fraction = (
        (block_index - left_position)
        / (right_position - left_position)
    )
    return _channel_between_anchors(left_H, right_H, fraction)


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

    # A recording often continues after the transmitted frame. Keep one weak
    # final block, but stop when low CP correlation is followed by a clear
    # energy drop, instead of fitting the sampling clock to trailing noise.
    baseline_count = min(10, len(block_rms))
    baseline_rms = np.median(block_rms[:baseline_count])
    usable_count = len(measured_starts)
    for index in range(1, len(measured_starts)):
        if (
            cp_metrics[index - 1] < 0.4
            and cp_metrics[index] < 0.4
            and block_rms[index] < 0.5 * baseline_rms
        ):
            usable_count = index
            break

    measured_starts = measured_starts[:usable_count]
    cp_metrics = cp_metrics[:usable_count]
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
    first_start = float(measured_starts[0])
    # Keep FFT windows on the nominal grid while accumulated timing error is
    # well inside the long CP. Fractional resampling from a wandering CP peak
    # distorts every carrier; the per-symbol phase-slope tracker below removes
    # the equivalent timing phase without altering the samples.
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
    filename  = data[6:total_hdr].decode("utf-8", errors="replace")
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
):
    """Build channel anchors and equalise all data blocks in one frame."""
    channel_anchors = [(0, h_active)]
    for block_index, block in enumerate(blocks):
        if (block_index + 1) % PILOT_BLOCK_PERIOD == 0:
            channel_anchors.append(
                (block_index, estimate_channel_from_pilot(block))
            )

    if len(channel_anchors) >= 3:
        first_position, first_H = channel_anchors[1]
        second_position, second_H = channel_anchors[2]
        fraction = -first_position / (second_position - first_position)
        backward_phase = _channel_between_anchors(
            first_H, second_H, fraction
        )
        backward_phase /= np.maximum(np.abs(backward_phase), 1e-12)
        channel_anchors[0] = (0, np.abs(h_active) * backward_phase)

    data_groups = []
    channel_groups = []
    current_data = []
    current_channels = []
    current_positions = []

    for block_index, block in enumerate(blocks):
        if (block_index + 1) % PILOT_BLOCK_PERIOD == 0:
            continue
        H_block = channel_for_block(block_index, channel_anchors)
        current_data.append(equalise_block_active(block, H_block, lam))
        current_channels.append(H_block)
        current_positions.append(block_index)
        if len(current_data) == DATA_OFDM_SYMBOLS_PER_GROUP:
            tracked_group, tracking_quality = track_qpsk_group_phase(
                np.asarray(current_data),
                block_positions=current_positions,
                initial_common_rate=phase_common_rate,
                initial_slope_rate=phase_slope_rate,
            )
            data_groups.append(tracked_group)
            channel_groups.append(np.mean(np.abs(current_channels), axis=0))
            current_data = []
            current_channels = []
            current_positions = []

    if current_data:
        valid_symbol_count = len(current_data)
        while len(current_data) < DATA_OFDM_SYMBOLS_PER_GROUP:
            current_data.append(
                np.zeros(DATA_CARRIERS_PER_SYMBOL, dtype=np.complex128)
            )
            current_channels.append(current_channels[-1])
        tracked_valid, tracking_quality = track_qpsk_group_phase(
            np.asarray(current_data[:valid_symbol_count]),
            block_positions=current_positions,
            initial_common_rate=phase_common_rate,
            initial_slope_rate=phase_slope_rate,
        )
        padded_group = np.zeros(
            (
                DATA_OFDM_SYMBOLS_PER_GROUP,
                DATA_CARRIERS_PER_SYMBOL,
            ),
            dtype=np.complex128,
        )
        padded_group[:valid_symbol_count] = tracked_valid
        data_groups.append(padded_group)
        channel_groups.append(np.mean(np.abs(current_channels), axis=0))

    return data_groups, channel_groups, len(channel_anchors) - 1


def _reference_symbols_from_ldpc(info_blocks, reliable_blocks, encoder):
    """Rebuild interleaved QPSK cells belonging to parity-valid codewords."""
    total_cells = DATA_OFDM_SYMBOLS_PER_GROUP * DATA_CARRIERS_PER_SYMBOL
    reference = np.zeros(
        (DATA_OFDM_SYMBOLS_PER_GROUP, DATA_CARRIERS_PER_SYMBOL),
        dtype=np.complex128,
    )
    reliable_cells = np.zeros(reference.shape, dtype=bool)

    for block_index, is_reliable in enumerate(reliable_blocks):
        if not is_reliable:
            continue
        coded = encoder.encode(info_blocks[block_index]).astype(np.uint8)
        pairs = coded.reshape(-1, 2)
        qpsk = (
            (1.0 - 2.0 * pairs[:, 1].astype(np.float64))
            + 1j * (1.0 - 2.0 * pairs[:, 0].astype(np.float64))
        )
        first_cell = block_index * LDPC_INFO_BITS
        for offset, value in enumerate(qpsk):
            source_cell = first_cell + offset
            target_cell = (APPENDIX_B_STRIDE * source_cell) % total_cells
            symbol_index, bin_index = divmod(
                target_cell,
                DATA_CARRIERS_PER_SYMBOL,
            )
            reference[symbol_index, bin_index] = value
            reliable_cells[symbol_index, bin_index] = True

    return reference, reliable_cells


def _refine_group_from_ldpc(group_symbols, info_blocks, converged, decoder):
    """Use parity-valid LDPC decisions to remove residual per-bin phase."""
    reference, reliable = _reference_symbols_from_ldpc(
        info_blocks,
        converged,
        decoder,
    )
    refined = np.asarray(group_symbols, dtype=np.complex128).copy()

    for symbol_index in range(len(refined)):
        mask = reliable[symbol_index]
        if np.count_nonzero(mask) < DATA_CARRIERS_PER_SYMBOL // 2:
            continue
        left = np.where(mask, reference[symbol_index], 0.0)
        right = np.where(mask, refined[symbol_index], 0.0)
        residual_phase = _intelligent_unwrap_phase_difference(left, right)
        refined[symbol_index] *= np.exp(-1j * residual_phase)

    return refined


def _decode_ldpc_groups(data_groups, channel_groups, decoder):
    all_info_bits = []
    for group_symbols, group_H in zip(data_groups, channel_groups):
        llr_blocks = deinterleave_group_soft(group_symbols, H_active=group_H)
        info_blocks, converged = apply_ldpc_decoding(
            llr_blocks,
            decoder,
            return_status=True,
            report_unconverged=False,
        )

        refined_symbols = np.asarray(group_symbols)
        for _ in range(2):
            if np.all(converged) or np.count_nonzero(converged) < 8:
                break
            refined_symbols = _refine_group_from_ldpc(
                refined_symbols,
                info_blocks,
                converged,
                decoder,
            )
            refined_llr = deinterleave_group_soft(
                refined_symbols,
                H_active=group_H,
            )
            candidate_info, candidate_converged = apply_ldpc_decoding(
                refined_llr,
                decoder,
                return_status=True,
                report_unconverged=False,
            )
            replace = (~converged) | candidate_converged
            info_blocks[replace] = candidate_info[replace]
            converged |= candidate_converged

        unconverged = np.count_nonzero(~converged)
        if unconverged:
            print(
                f"Warning: {unconverged}/{len(converged)} LDPC codewords "
                "did not pass parity checks after phase refinement."
            )
        all_info_bits.append(info_blocks.reshape(-1))
    return np.concatenate(all_info_bits).astype(np.uint8)


def _bits_to_bytes(bitstream):
    usable = (len(bitstream) // 8) * 8
    return np.packbits(bitstream[:usable], bitorder="big").tobytes()


# ── Main decode pipeline ──────────────────────────────────────────────────────

def decode_signal(signal, use_ldpc=True, lam=1e-6, ldpc_decoder=None):
    signal = np.asarray(signal, dtype=np.float64).squeeze()

    chirp_start = find_chirp_start(signal)
    data_start  = chirp_start + CHIRPS_LENGTH + GOLAY_SIGNAL_LENGTH
    print(f"Chirp found at sample {chirp_start}  ({chirp_start/SAMPLE_RATE:.3f} s)")
    print(f"OFDM data starts at sample {data_start}  ({data_start/SAMPLE_RATE:.3f} s)")

    H_rfft, golay_common_rate, golay_slope_rate = estimate_initial_channel(
        signal, chirp_start, data_start
    )
    h_active = H_rfft[ACTIVE_BINS]
    print(f"Golay H: mean|H|={np.mean(np.abs(h_active)):.3f}  "
          f"min={np.min(np.abs(h_active)):.4f}  max={np.max(np.abs(h_active)):.4f}")
    print(
        "Golay phase drift: "
        f"common={golay_common_rate:.3e} rad/sample, "
        f"slope={golay_slope_rate:.3e} rad/bin/sample"
    )

    data_region = signal[data_start:]
    blocks, block_starts, sample_scale = extract_timing_tracked_blocks(data_region)
    print(f"OFDM blocks available: {len(blocks)}")
    print(
        f"Timing correction: first={block_starts[0]:.2f} samples, "
        f"sample scale={sample_scale:.9f} "
        f"({(sample_scale - 1.0) * 1e6:+.1f} ppm)"
    )

    if not use_ldpc:
        data_groups, _, pilot_count = _equalise_frame_blocks(
            blocks,
            h_active,
            lam,
            phase_common_rate=golay_common_rate * BLOCK_LENGTH,
            phase_slope_rate=golay_slope_rate * BLOCK_LENGTH,
        )
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

        header_tracking_blocks = min(
            len(blocks),
            2 * PILOT_BLOCK_PERIOD,
        )
        first_groups, first_channels, _ = _equalise_frame_blocks(
            blocks[:header_tracking_blocks],
            h_active,
            lam,
            phase_common_rate=golay_common_rate * BLOCK_LENGTH,
            phase_slope_rate=golay_slope_rate * BLOCK_LENGTH,
        )
        first_bits = _decode_ldpc_groups(
            first_groups[:1], first_channels[:1], decoder
        )
        first_bytes = _bits_to_bytes(first_bits)
        total_hdr, file_len, filename = parse_header(first_bytes)

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
        data_groups, channel_groups, pilot_count = _equalise_frame_blocks(
            frame_blocks,
            h_active,
            lam,
            phase_common_rate=golay_common_rate * BLOCK_LENGTH,
            phase_slope_rate=golay_slope_rate * BLOCK_LENGTH,
        )
        data_groups = data_groups[:expected_groups]
        channel_groups = channel_groups[:expected_groups]
        print(f"Periodic pilots used: {pilot_count}")
        print(f"LDPC groups: {len(data_groups)}")
        info_bitstream = _decode_ldpc_groups(
            data_groups, channel_groups, decoder
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
