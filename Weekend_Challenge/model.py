import csv
from scipy.io import wavfile
import os
import numpy as np

# Transfer qpsk back to bits
def qpsk_to_bits(symbols, mapping_version=1):
    """
    Convert equalised QPSK symbols to bits using angle decision.
    
    mapping_version=1:
        angles pi/4, 3pi/4, 5pi/4, 7pi/4 correspond to
        00, 01, 11, 10
    
    mapping_version=2:
        angles pi/4, 3pi/4, 5pi/4, 7pi/4 correspond to
        00, 10, 11, 01
    """

    ref_angles = np.array([
        np.pi / 4,
        3 * np.pi / 4,
        5 * np.pi / 4,
        7 * np.pi / 4
    ])

    if mapping_version == 1:
        bit_pairs = [
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 0]
        ]
    else:
        bit_pairs = [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ]

    bits = []

    for z in symbols:
        angle = np.angle(z)

        if angle < 0:
            angle += 2 * np.pi

        # Circular angular distance, get distance right from [-pi,pi]
        distances = np.abs(np.angle(np.exp(1j * (angle - ref_angles))))
        idx = np.argmin(distances)

        bits.extend(bit_pairs[idx])

    return bits

# Bits to bytes
def bits_to_bytes(bits):
    """
    Convert bits to bytes.
    Each byte uses MSB first.
    """
    byte_array = bytearray()
    usable_length = len(bits) // 8 * 8

    for i in range(0, usable_length, 8):
        byte = 0

        for bit in bits[i:i+8]:
            byte = byte*2 + bit

        byte_array.append(byte)
    return bytes(byte_array)

#Parse file correctly
def parse_recovered_bytes(raw_bytes):
    """
    Header format:
        filename + \\0 + filesize + \\0 + raw file data
    """

    first_zero = raw_bytes.find(b"\0")
    second_zero = raw_bytes.find(b"\0", first_zero + 1)

    filename_bytes = raw_bytes[:first_zero]
    filesize_bytes = raw_bytes[first_zero + 1:second_zero]

    try:
        filename = filename_bytes.decode("utf-8")
        filesize_str = filesize_bytes.decode("utf-8")
        filesize = int(filesize_str)
    except:
        return None

    file_data_start = second_zero + 1
    file_data_end = file_data_start + filesize

    if file_data_end > len(raw_bytes):
        return None

    file_data = raw_bytes[file_data_start:file_data_end]

    return filename, filesize, file_data

#try to retrieve file
def recover_file_from_wav(wav_path, output_dir, NLEN, CP, Filter_coff):
    print("\nProcessing:", os.path.basename(wav_path))
    H = Filter_coff
    SYMBOL_LEN = NLEN + CP
    DATA_BINS = np.arange(1,512) #1 to 511
    _, data = wavfile.read(wav_path)

    # If stereo, take one channel
    if data.ndim > 1:
        data = data[:, 0]

    # Convert to float
    data = data.astype(float)

    num_symbols = len(data) // SYMBOL_LEN

    # Remove convolution tail or extra samples
    data = data[:num_symbols * SYMBOL_LEN]

    # Shape into OFDM symbols
    blocks = data.reshape(num_symbols, SYMBOL_LEN)

    # Remove cyclic prefix
    blocks_no_cp = blocks[:, CP:]

    # FFT each OFDM symbol
    Y = np.fft.fft(blocks_no_cp, axis=1)

    # Equalise channel
    X_hat = Y / H

    # Extract useful frequency bins 1 to 511
    data_symbols = X_hat[:, DATA_BINS]

    # Flatten all QPSK symbols into one long sequence
    qpsk_symbols = data_symbols.flatten()

    # Try both possible Gray mappings
    bits = qpsk_to_bits(qpsk_symbols, mapping_version=1)
    raw_bytes = bits_to_bytes(bits)

    parsed = parse_recovered_bytes(raw_bytes)

    if parsed is not None:
        filename, filesize, file_data = parsed
        filename = os.path.basename(filename)
        output_path = os.path.join(output_dir, filename)

        with open(output_path, "wb") as f:
            f.write(file_data)

        print("Success!")
        print("Recovered filename:", filename)
        print("Recovered filesize:", filesize)
        print("Saved to:", output_path)

        return

    print("Failed to parse header for this file.")



def parse_data(data:np.ndarray, CP:int=32,SYMBOL_LEN:int=1056):
    # Convert to float
    data = data.astype(float)

    num_symbols = len(data) // SYMBOL_LEN

    # Remove convolution tail or extra samples
    data = data[:num_symbols * SYMBOL_LEN]

    # Shape into OFDM symbols
    blocks = data.reshape(num_symbols, SYMBOL_LEN)

    # Remove cyclic prefix
    blocks_no_cp = blocks[:, CP:]
    return blocks_no_cp

def get_modulated_message(ofdm_constellation: np.ndarray, channel_fir_freq: np.ndarray, DATA_BINS:np.ndarray):
    # FFT each OFDM symbol
    Y = np.fft.fft(ofdm_constellation, axis=1)

    # Equalise channel
    X_hat = Y / channel_fir_freq

    # Extract useful frequency bins 1 to 511
    data_symbols = X_hat[:, DATA_BINS]

    # Flatten all QPSK symbols into one long sequence
    qpsk_symbols = data_symbols.flatten()

    return qpsk_symbols