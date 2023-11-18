# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 11:38:06 2023

@author: hkl
"""

from LoRaPHY import LoRaPHY
import numpy as np
from scipy.signal import resample

# Parameters
rf_freq = 470e6
sf = 7
bw = 125e3
fs = 1e6

# Create LoRaPHY instance
phy = LoRaPHY(rf_freq, sf, bw, fs)
phy.has_header = 1
phy.cr = 4
phy.crc = 0
phy.preamble_len = 8

# Encode payload [1 2 3 4 5]
payload = np.array([1, 2, 3, 4, 5])
symbols = phy.encode(payload)
print("[encode] symbols:")
print(symbols)

# Baseband Modulation
sig = phy.modulate(symbols)

# Demodulation
symbols_d, cfo, netid = phy.demodulate(sig)
print("[demodulate] symbols:")
print(symbols_d)

# Resample
resampled_sig = resample(sig, int(2 * bw), fs)

# Decoding
data, checksum = phy.decode(symbols_d)
print("[decode] data:")
print(data)
print("[decode] checksum:")
print(checksum)