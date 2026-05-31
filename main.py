"""
General Pipeline

Step 1: Read the input file to transmit. 
Step 2: Convert the input file to a bitstream of 0s and 1s
Step 3: Perform q-ary modulation to covnert the bitstream into symbols to get modulated symbols
Step 4: Add a large set of pilot symbols to the modulated symbols 

Need to create a pre stuff for the datastream

Take a chirp created in the frequency domian. Linear/quadratic/anything else (ensure same length as DFT)
Convert to time domain by inverse fft
Add cyclic prefix to the chirp in time domain
Receive at the receiver. The end of the chirp marks the beginning of the start of the signal

Next transmit a OFDM symbol in format [CP Pilot Symbols | Same pilot symbols repeated]
Use this to phase and tiem synchronize. 
 
Step 5: Invert the modulated symbols into time domain (length of FFT whihc is standardized)
Step 6: Add a cyclic prefix 
Step 7: Convert to a wav file and transmit accross the channel
Step 8: Receive the signal and reparse into size of OFDM
Step 9: discard the excess terms from the channel  
Step 10: use pilot to estimate channel 
Step 11: obtain relevant frequency domain params and infer 
Step 12: demodulate
Step 13: error correct

"""

from parse_data import *
from modulation import bits_to_symbols
import os

def main():
    absolute_path = os.path.dirname(__file__)
    relative_path = 'target.txt'
    file_path = os.path.join(absolute_path, relative_path)
    bit_data = read_as_bits(file_path)
    # error_corrected_bits = ...
    symbols = bits_to_symbols(bit_data, 'qpsk', convention=0, scale=1)
