import numpy as np
from scipy.signal import butter, sosfiltfilt, chirp as scipy_chirp

def apply_tukey_window(signal: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Cosine-taper the edges of a signal to reduce spectral splatter."""
    n = len(signal)
    win = np.ones(n)
    ramp = int(alpha * n / 2)
    if ramp > 0:
        t = np.linspace(0, np.pi, ramp)
        win[:ramp]  = 0.5 * (1 - np.cos(t))
        win[-ramp:] = 0.5 * (1 - np.cos(t[::-1]))
    return signal * win

def make_chirp(duration: float, f0: float, f1: float,
               fs: int, amplitude: float = 1.0,
               method: str = "linear", window=True) -> np.ndarray:
    """
    Generate a single chirp.

    Parameters
    ----------
    duration  : seconds
    f0, f1    : start / end frequency (Hz)
    fs        : sample rate
    amplitude : linear amplitude
    method    : 'linear' | 'logarithmic' | 'hyperbolic'
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    sig = scipy_chirp(t, f0=f0, f1=f1, t1=duration, method=method).astype(np.float32)
    if window:
        sig = apply_tukey_window(sig) * amplitude
    return sig

def make_mls(order: int) -> np.ndarray:
    """
    Generate a binary (±1) Maximum Length Sequence of length 2^order - 1.
    Uses the standard feedback polynomial taps.

    Returns
    -------
    mls : np.ndarray of float32 in {-1, +1}
    """
    # Standard primitive polynomials (taps) for common orders
    TAPS = {
        7:  [7, 6],
        8:  [8, 6, 5, 4],
        9:  [9, 5],
        10: [10, 7],
        11: [11, 9],
        12: [12, 11, 10, 4],
        13: [13, 12, 11, 8],
        14: [14, 13, 12, 2],
        15: [15, 14],
        16: [16, 15, 13, 4],
    }
    if order not in TAPS:
        raise ValueError(f"MLS order {order} not supported. Choose from {list(TAPS)}")

    length = (1 << order) - 1          # 2^order - 1
    register = np.ones(order, dtype=int)  # all-ones seed
    seq = np.empty(length, dtype=np.int8)
    taps = TAPS[order]

    for i in range(length):
        bit = 0
        for t in taps:
            bit ^= register[t - 1]
        seq[i] = register[-1]
        register = np.roll(register, 1)
        register[0] = bit

    return (2 * seq.astype(np.float32) - 1)   # {0,1} → {-1,+1}


def normalise_amplitude(signal: np.ndarray, target: float = 0.5) -> np.ndarray:
    peak = np.max(np.abs(signal))
    if peak == 0:
        return signal
    return signal * (target / peak)


def silence(duration: float, fs: int) -> np.ndarray:
    return np.zeros(int(duration * fs), dtype=np.float32)