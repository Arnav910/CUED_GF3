import csv
import wave
from FIR import FIR

def run():
    # Prepare FIR
    filter_coefs = []
    with open('channel.csv') as file:
        reader = csv.reader(file)
        for row in reader:  
            filter_coefs.append(float(row[0]))

    filter = FIR(filter_coefs)
    
run()