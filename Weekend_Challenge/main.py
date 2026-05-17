import csv
from scipy.io import wavfile
import os
import numpy as np
import model

INFO_LENGTH = 1024
CYCLIC_PREFIX = 32
OFDM_SYMBOL_LENGTH = INFO_LENGTH + CYCLIC_PREFIX
DATA_BINS = np.arange(1,512)
# Create/setup relevant repositories
absolute_path = os.path.dirname(__file__)
wav_dir = os.path.join(absolute_path, "wav_files")
output_dir = os.path.join(absolute_path, "recovered_files")
os.makedirs(output_dir, exist_ok=True)

# Prepare FIR
g = [] # TIme domain
with open(os.path.join(absolute_path,'channel.csv')) as file:
    reader = csv.reader(file)
    for row in reader:  
        g.append(float(row[0])) # read filter coeficients


G = np.fft.fft(g,1024) # Convert to Frequency

# read wave files
wav_files = sorted([
    f for f in os.listdir(wav_dir)
    if f.endswith(".wav")
])

for wav_name in wav_files:
    wav_path = os.path.join(wav_dir, wav_name)
    # To reduce abstraction and cleaner workflow, better to separate the process so that a clear workflow is establiched

    # Step 1:  Read the data
    _,data = wavfile.read(wav_path)

    # Step 2: Obtain the data as OFDM Constellation Symbols
    ofdm_constellation = model.parse_data(data, CP=CYCLIC_PREFIX, SYMBOL_LEN=OFDM_SYMBOL_LENGTH)

    # Step 3: Obtain the modulated information in qpsk constellations
    qpsk_form = model.get_modulated_message(ofdm_constellation,channel_fir_freq=G,DATA_BINS=DATA_BINS)
    
    # Step 4: Demodulate and obtain
    demodulated_x_bits = model.qpsk_to_bits(qpsk_form)

    #step 5: Convert to bytes
    demodulated_x_bytes = model.bits_to_bytes(demodulated_x_bits)

    #step 6: Parse result
    x_bytes = model.parse_recovered_bytes(demodulated_x_bytes)

    if x_bytes is not None:
        filename, filesize, file_data = x_bytes
        filename = os.path.basename(filename)
        output_path = os.path.join(output_dir, filename)

        with open(output_path, "wb") as f:
            f.write(file_data)

        print("Success!")
        print("Recovered filename:", filename)
        print("Recovered filesize:", filesize)
        print("Saved to:", output_path) 
    
    #model.recover_file_from_wav(wav_path, output_dir,NLEN =1024,CP=32, Filter_coff=G)


