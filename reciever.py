
from utils import make_chirp
import numpy as np
import numpy.typing as npt
import config
from scipy.signal import find_peaks

def find_chirp_in_signal(signal: np.ndarray, num_chirps: int|None=None)->np.ndarray:
    """
    Find chirp locations in signal using cross-correlation and peak detection.
    
    Args:
        signal (ndarray): Input signal to search
        num_chirps: Expected number of chirps (int defaults to none). If provided, returns only top N peaks.
    
    Returns:
        Sorted array of chirp start indices
    """
    reference_chirp = make_chirp(config.CHIRP_LOW, config.CHIRP_HIGH, config.CHIRP_SAMPLES)
    
    # Cross-correlation (gives correlation for each lag)
    correlation = np.correlate(signal, reference_chirp, mode='valid')
    
    # Much stricter threshold and find peaks
    threshold = np.max(correlation) * 0.5
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

def estimte_symbol_offset(chirp_starts: np.ndarray):
    """
    Given the observed locations of the chirp peaks can estimate symbol time offset

    Args:
        chirp_starts (np.ndarray): location of chirp
        signal (np.ndarray): received signal
    """
    chirp_spacing = config.CHIRP_SPACING + config.CHIRP_SAMPLES # Distance between start of 2 chirps
    measured_spacing = np.diff(chirp_starts)
    sto_offset = np.mean(measured_spacing - chirp_spacing)
    return sto_offset

def channel_estimation(chirp_starts: np.ndarray, signal: np.ndarray):
    # obtain chirp from signal
    indices = chirp_starts[:, None] + np.arange(config.CHIRP_SAMPLES)
    chirps = signal[indices]

    reference_chirp = make_chirp(config.CHIRP_LOW, config.CHIRP_HIGH, config.CHIRP_SAMPLES)
    true_chirp_freq = np.fft.fft(reference_chirp)

    
    # convert to frequency domain
    chirps_freq = np.fft.fft(chirps, axis = 1)

    #MLE Estimate
    G_freq = np.sum(np.conjugate(true_chirp_freq)*chirps_freq, axis = 0)/np.sum(np.abs(true_chirp_freq)**2)

    g_time = np.fft.ifft(G_freq)

    return g_time, G_freq