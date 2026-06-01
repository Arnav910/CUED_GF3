import receiver
import config
from scipy.io import wavfile
import numpy as np
import os

def load_wav_mono(path: str) -> tuple[int, np.ndarray]:
    fs, data = wavfile.read(path)

    if data.ndim > 1:
        data = data.mean(axis=1)

    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float64) - 128.0) / 128.0
    else:
        data = data.astype(np.float64)

    return fs, data

def main(path):
    fs, rx = load_wav_mono(path)

    if fs != config.SAMPLE_RATE:
        raise ValueError(f"Expected {config.SAMPLE_RATE} Hz, got {fs} Hz")

    decoded = receiver.decode_signal(
        rx,
        use_channel_estimate=True,
        max_channel_taps=256,
    )

    print("Header length:", decoded.header_length)
    print("File length:", decoded.file_length)
    print("Payload bytes recovered:", len(decoded.payload))
    print("Chirp starts:", decoded.chirp_starts)
    print("Data start:", decoded.data_start)

    with open("recovered_file.bin", "wb") as f:
        f.write(decoded.payload)

if __name__ == "__main__":
    file = 'test.wav'
    dir = os.path.join(os.getcwd(), 'received_signals')
    main(os.path.join(dir, file))