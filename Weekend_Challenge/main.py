import csv
from scipy.io import wavfile
from FIR import FIR
import os

def run():
    # Prepare FIR
    filter_coefs = []
    with open('channel.csv') as file:
        reader = csv.reader(file)
        for row in reader:  
            filter_coefs.append(float(row[0]))
    absolute_path = os.path.dirname(__file__)
    relative_path = 'wav_files'
    path = os.path.join(absolute_path, relative_path)
    files = os.listdir(path)
    for file in files:
        data_rate, data = wavfile.read(os.path.join(path,file))
        break
    
    
    

    
run()