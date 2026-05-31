import sounddevice as sd
import numpy as np
import soundfile as sf

fs = 48000
duration = 5  # seconds

print("Recording...")

audio = sd.rec(
    int(duration * fs),
    samplerate=fs,
    channels=1,
    dtype="float32"
)

sd.wait()

audio = audio[:, 0]  # mono

sf.write("rx_noise_tone.wav", audio.astype(np.float32), fs, subtype="FLOAT")

print("Saved rx_mls_tone.wav (float32)")