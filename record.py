import sounddevice as sd
import numpy as np
import soundfile as sf
import os

def record(file_name: str):
    fs = 48000
    duration = 20  # seconds

    print("Recording...")

    audio = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    audio = audio[:, 0]  # mono\
    directory = os.path.join(os.getcwd(), 'received_signals')
    if not os.path.exists(directory):
        os.makedirs(directory)

    sf.write(os.path.join(directory, file_name), audio.astype(np.float32), fs, subtype="FLOAT")

    print("Saved rx_mls_tone.wav (float32)")

if __name__ == "__main__":
    record('test.wav')