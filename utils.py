from scipy.signal import chirp
import numpy as np

def make_chirp(initial_freq: float, final_freq: float, sample_count: int, sample_rate: float =48000.0, method: str='linear')-> np.ndarray:
    """
    Generate chirp with specified features

    Args:
        initial_freq (float): Initial Chirp Frequency
        final_freq (float): Final Chirp Frequency
        sample_count (int): Number of samples to generate
        sample_rate (float, optional): Sampling rate. Defaults to 48000.0Hz.
        method (str, optional): Chirp generation method. Defaults to 'linear'.

    Returns:
        np.ndarray: Generated chirp signal
    """
    t = np.arange(sample_count) / sample_rate
    chirp_signal = chirp(t, f0=initial_freq, f1=final_freq, t1=t[-1], method=method)
    if isinstance(chirp_signal, np.ndarray):
        return chirp_signal
    else:
        raise ValueError("The generated chirp signal is not a numpy array.")