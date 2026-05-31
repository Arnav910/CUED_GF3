"""
config.py — Shared parameters for the channel estimation research toolkit.
Tune these to match your physical setup.
"""

# ── Audio ──────────────────────────────────────────────────────────────────
SAMPLE_RATE       = 48_000          # Hz  (match phone + laptop mic)
DTYPE             = "float32"

# ── Frequency band of interest ─────────────────────────────────────────────
F_LOW             = 500             # Hz  lower edge of pilot band
F_HIGH            = 8_000           # Hz  upper edge (< Nyquist = 24 kHz)

# ── Pilot frame parameters ─────────────────────────────────────────────────
PILOT_DURATION    = 0.5             # seconds per pilot frame
PILOT_REPS        = 8               # K — number of repetitions for MLE averaging
GUARD_DURATION    = 0.05            # seconds silence between frames (prevent overlap)

# ── Sync chirp parameters ──────────────────────────────────────────────────
SYNC_CHIRP_DURATION = 0.3           # seconds
SYNC_CHIRP_F0       = 200           # Hz  start freq
SYNC_CHIRP_F1       = 9_000         # Hz  end freq
SYNC_CHIRP_AMPLITUDE= 0.8

# ── MLS (Maximum Length Sequence) ─────────────────────────────────────────
MLS_ORDER         = 13              # PRBS-13 → sequence length = 2^13 - 1 = 8191 chips
MLS_CHIP_RATE     = 8_000           # chips/sec → chip duration = 1/8000 s

# ── Chirp series pilot ─────────────────────────────────────────────────────
CHIRP_N_SWEEPS    = 4               # number of sub-chirps per pilot frame
CHIRP_OVERLAP     = 0.1             # fractional overlap between sub-chirps

# ── White noise pilot ──────────────────────────────────────────────────────
WN_SEED           = 42              # reproducible random sequence

# ── Amplitude / headroom ──────────────────────────────────────────────────
PILOT_AMPLITUDE   = 0.5             # linear (leave headroom for phone speaker)

# ── DFT / channel estimation ──────────────────────────────────────────────
# NFFT >= pilot frame length ensures every chirp sub-band gets DFT coverage.
# Next power-of-2 above 0.5s * 48kHz = 24000 samples → 32768
import math as _math
_PILOT_N = int(PILOT_DURATION * SAMPLE_RATE)
NFFT     = 2 ** _math.ceil(_math.log2(_PILOT_N))   # 32768 at default settings
NOVERLAP = NFFT // 2

# ── Sync detection ────────────────────────────────────────────────────────
SYNC_XCORR_THRESH = 0.1            # normalised cross-correlation threshold
SYNC_SEARCH_WIN   = 2.0             # seconds to search for sync at start of recording

# ── Output paths ──────────────────────────────────────────────────────────
import os
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TX_DIR     = os.path.join(BASE_DIR, "output", "tx")
RX_DIR     = os.path.join(BASE_DIR, "output", "rx")
RESULT_DIR = os.path.join(BASE_DIR, "output", "results")

for _d in [TX_DIR, RX_DIR, RESULT_DIR]:
    os.makedirs(_d, exist_ok=True)
