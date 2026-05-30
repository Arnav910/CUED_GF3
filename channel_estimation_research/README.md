# Audio Modem — OFDM/QPSK over Acoustic Channel

A complete passive audio modem built in pure Python.  
Transmit data as sound, record it with a microphone, and recover the original file with full channel analysis.

---

## System Overview

```
transmitter.py  →  .wav file  →  (phone speaker / air)  →  recorder.py
                                                               ↓
                                                         recorded.wav
                                                               ↓
                                                         receiver.py
                                                               ↓
                                                         received.npz
                                                               ↓
                                                         analyser.py
                                                               ↓
                                          reconstructed.txt + figures
```

---

## Dependencies

```bash
pip install numpy scipy matplotlib
# For recorder — install ONE of:
pip install sounddevice   # requires PortAudio (brew install portaudio / apt install libportaudio2)
pip install pyaudio       # alternative; requires portaudio too
# Fallback: uses system arecord (Linux) or rec/sox (macOS) automatically
```

---

## OFDM Parameters

| Parameter          | Value                  |
|--------------------|------------------------|
| FFT length (N_FFT) | 1024 sub-carriers      |
| Cyclic prefix      | 1024 samples           |
| Total symbol       | 2048 samples           |
| Modulation         | QPSK (Gray coded)      |
| Data sub-carriers  | 511 (indices 1–511)    |
| Bits per symbol    | 1022 bits / OFDM block |

---

## Frame Structure

```
[chirp 0.3s] | [guard 1024 sa] | [pilot×M (each+CP)] | [data blocks with pilot every K]
```

Each pilot/data block = 2048 samples (CP + FFT window).

---

## 1 · Transmitter

```bash
python transmitter.py <input_file.txt> [OPTIONS]
```

| Option           | Default            | Description                          |
|------------------|--------------------|--------------------------------------|
| `--pilot-type`   | `mls`              | `mls` or `noise` pilot sequences     |
| `--seed`         | `42`               | PRNG seed for pilots                 |
| `--pilot-blocks` | `4`                | Leading pilot blocks (M)             |
| `--pilot-every`  | `8`                | Periodic pilot every K data blocks   |
| `--samplerate`   | `48000`            | Sample rate (Hz)                     |
| `--output`       | `transmitted.wav`  | Output WAV filename                  |
| `--amplitude`    | `0.7`              | Output amplitude (0–1)               |

**Example:**
```bash
python transmitter.py my_document.txt --pilot-type mls --pilot-blocks 8 --output tx.wav
```

Outputs: `tx.wav` + `tx_meta.npz` (sidecar with all parameters — needed by receiver)

### Pilot types
- **MLS** — Maximum-Length Sequence (15-bit LFSR, deterministic, low PAPR, good correlation properties)
- **Noise** — Seeded random BPSK on I and Q independently

---

## 2 · Recorder

```bash
python recorder.py [OPTIONS]
```

| Option           | Default          | Description                          |
|------------------|------------------|--------------------------------------|
| `--duration`     | `30`             | Seconds; use `0` for press-Enter mode |
| `--samplerate`   | `48000`          | Must match transmitter               |
| `--channels`     | `1`              | 1=mono, 2=stereo                     |
| `--output`       | `recorded.wav`   | Output WAV filename                  |
| `--list-devices` | —                | Show available devices and exit      |

**Example (timed):**
```bash
python recorder.py --duration 15 --output capture.wav
```

**Example (manual stop):**
```bash
python recorder.py --duration 0 --output capture.wav
# Press Enter to start, Enter again to stop
```

**Tips for best results:**
- Place phone ~30–50 cm from mic, in a quiet room
- Avoid surfaces that cause strong reflections
- Set phone volume to ~70–80% (avoid clipping)
- The recorder normalises the captured audio automatically

---

## 3 · Receiver

```bash
python receiver.py <recorded.wav> [OPTIONS]
```

| Option      | Default                      | Description                         |
|-------------|------------------------------|-------------------------------------|
| `--meta`    | auto-detected                | Path to `_meta.npz` sidecar         |
| `--output`  | `received`                   | Output basename → `received.npz`    |
| `--no-eq`   | off                          | Disable one-tap frequency equaliser |

**Example:**
```bash
python receiver.py capture.wav --meta tx_meta.npz --output received
```

### Synchronisation pipeline
1. **Coarse sync** — cross-correlate squared signal with squared reference chirp (4× downsampled for speed) → finds frame start within a few hundred samples
2. **Fine sync** — CP correlation search ±512 samples around expected pilot position → refines to exact OFDM boundary
3. **CFO estimation** — angle of CP·body correlation → correct carrier frequency offset
4. **CFO correction** — multiply by `exp(-j2π·cfo·n)`
5. **Channel tracking** — one-tap equalisation updated from periodic pilots

### Output: `received.npz`
Contains: raw received sub-carriers, equalized symbols, hard decisions, pilot FD symbols, channel estimate H, recovered bits and bytes, CFO estimate, all metadata.

---

## 4 · Analyser

```bash
python analyser.py <received.npz> [OPTIONS]
```

| Option      | Default | Description                              |
|-------------|---------|------------------------------------------|
| `--no-show` | off     | Skip interactive windows, save PNGs only |
| `--out-dir` | `.`     | Directory for output figures             |

**Example:**
```bash
python analyser.py received.npz --no-show --out-dir ./results/
```

### Analysis performed

#### 1. MLE Channel Estimation
```
H_n = (Σ X_n* · Y_n) / (Σ X_n* · X_n)
```
Estimated per sub-carrier using all M leading pilot OFDM blocks.

#### 2. Text Reconstruction
Recovered payload bytes → UTF-8 text → `reconstructed.txt`

#### 3. QPSK Constellation Cloud  → `constellation.png`
Scatter of all equalised symbols against ideal QPSK points with Gray code labels.

#### 4. Channel FIR  → `channel_fir.png`
- **CIR** (Channel Impulse Response) — IFFT of H_n: |h(t)| linear and dB
- **CFR** (Channel Frequency Response) — |H(f)| dB and phase(°) vs frequency

#### 5. Metrics  → `snr_metrics.png` + `analysis_report.txt`

| Metric | Description |
|--------|-------------|
| SNR per sub-carrier | Signal/noise ratio from pilot deviation |
| Mean / Median SNR (dB) | Aggregate link quality |
| EVM % | Error Vector Magnitude (RMS) |
| MER (dB) | Modulation Error Ratio |
| SER | Symbol Error Rate |
| Est. BER | Estimated Bit Error Rate (SER/2 for Gray-QPSK) |

---

## Full Workflow Example

```bash
# Step 1 — Encode your file
python transmitter.py document.txt --pilot-type mls --output tx.wav

# Step 2 — Play tx.wav on your phone, then record
python recorder.py --duration 20 --output rx_capture.wav

# Step 3 — Decode
python receiver.py rx_capture.wav --meta tx_meta.npz --output received

# Step 4 — Analyse
python analyser.py received.npz --no-show --out-dir results/
```

---

## Architecture Notes

- **Hermitian symmetry** — The IFFT input is mirrored so the time-domain output is real, producing a baseband OFDM signal compatible with any mono audio channel.
- **Cyclic prefix = FFT length** — This unusually long CP (equal to symbol length) gives up to 21.3 ms of multipath protection at 48 kHz, accommodating very reverberant rooms.
- **Normalised IFFT/FFT** — `ifft × √N` and `fft / √N` maintain power consistency.
- **Pilot density** — M=4 leading blocks plus one every K=8 data blocks provide channel tracking for slowly time-varying channels (typical room acoustics).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| High BER, garbled text | Sync offset wrong | Increase `search_range` in `fine_cp_sync`; ensure meta file matches |
| No chirp detected | Low volume or wrong file | Increase phone volume; check WAV plays correctly |
| Recorder error | No PortAudio | Install `libportaudio2` or use `arecord` fallback |
| EVM > 50% | Strong multipath | Add more pilot blocks (`--pilot-blocks 8`) |

---

*Built with: Python 3, NumPy, SciPy, Matplotlib*
