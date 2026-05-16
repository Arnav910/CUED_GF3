import csv
from scipy.io import wavfile
from FIR import FIR
import os
import numpy as np


# Basic OFDM parameters
NLen = 1024
CP = 32
SYMBOL_LEN = NLen + CP
DATA_BINS = np.arange(1, 512)  # bins 1 to 511

# Prepare FIR
filter_coefs = []
absolute_path = os.path.dirname(__file__)
with open(os.path.join(absolute_path,'channel.csv')) as file:
    reader = csv.reader(file)
    for row in reader:  
        filter_coefs.append(float(row[0]))

relative_path = 'wav_files'
path = os.path.join(absolute_path, relative_path)
files = os.listdir(path)
for file in files:
    data_rate, data = wavfile.read(os.path.join(path,file))
    break


# Frequency response of the channel
g = filter_coefs
G = np.fft.fft(g, NLen)

num_sym = len(data) //SYMBOL_LEN
print(num_sym)

  