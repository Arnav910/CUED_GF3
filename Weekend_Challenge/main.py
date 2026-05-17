import csv
from scipy.io import wavfile
from FIR import FIR
import os
import numpy as np
import model

# Prepare FIR
filter_coefs = []
absolute_path = os.path.dirname(__file__)
with open(os.path.join(absolute_path,'channel.csv')) as file:
    reader = csv.reader(file)
    for row in reader:  
        filter_coefs.append(float(row[0]))

"""
relative_path = 'wav_files'
path = os.path.join(absolute_path, relative_path)
files = os.listdir(path)
file=files[1]
print(file)
data_rate, data = wavfile.read(os.path.join(path,file))
"""

wav_dir = os.path.join(absolute_path, "wav_files")
output_dir = os.path.join(absolute_path, "recovered_files")
G = np.fft.fft(filter_coefs,1024)
os.makedirs(output_dir, exist_ok=True)

wav_files = sorted([
    f for f in os.listdir(wav_dir)
    if f.endswith(".wav")
])

for wav_name in wav_files:
    wav_path = os.path.join(wav_dir, wav_name)
    model.recover_file_from_wav(wav_path, output_dir,NLEN =1024,CP=32, Filter_coff=G)


