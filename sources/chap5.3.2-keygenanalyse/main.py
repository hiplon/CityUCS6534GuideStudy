# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:27:13 2023

@author: hkl
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from func import *
from tqdm import tqdm

# Define start and end indices
startIndex = 99
endIndex = 1199
#endIndex = 30001

# Load data from text files
alice = np.loadtxt('outdoor-walk-alice.txt')
rssi_alice = alice[startIndex:endIndex, 2]

m1 = rssi_alice.shape[0]
n1 = 1

bob = np.loadtxt('outdoor-walk-bob.txt')
rssi_bob = bob[startIndex:endIndex, 2]
m2 = rssi_bob.shape[0]
n2 = 1

# Calculate correlation between Alice and Bob's data
correlation = np.corrcoef(rssi_alice, rssi_bob)[0, 1]

# Calculate time duration and sample rate
time_duration = (alice[endIndex, 1] - alice[startIndex, 1]) / 1000
sample_rate = m1 / time_duration

# Step 1: Linear interpolation
M = 1
x = np.arange(1, m1 + 1)
xq = np.arange(1, m1 + 1, 1/M)
vq1_alice = np.interp(xq, x, rssi_alice)

x2 = np.arange(1, m2 + 1)
xq2 = np.arange(1, m2 + 1, 1/M)
vq1_bob = np.interp(xq2, x2, rssi_bob)

# Plot original and interpolated values
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(x, rssi_alice)
plt.plot(xq, vq1_alice, ':.')
plt.legend(['Original values', 'Interpolated values'])

plt.subplot(2, 1, 2)
plt.plot(x2, rssi_bob)
plt.plot(xq2, vq1_bob, ':.')
plt.legend(['Original values', 'Interpolated values'])

plt.show()

# Define filter parameters
order = 5
framelen = 11

# Apply Savitzky-Golay filter to Alice and Bob's data
alice_afterfilter = savgol_filter(vq1_alice, window_length=framelen, polyorder=order)
bob_afterfilter = savgol_filter(vq1_bob, window_length=framelen, polyorder=order)

# Plot After savgol_filter
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(x, rssi_alice)
plt.plot(xq, vq1_alice, ':.')
plt.plot(xq, alice_afterfilter, '-')
plt.legend(['Original values', 'Interpolated values', 'savgol_filter'])

plt.subplot(2, 1, 2)
plt.plot(x2, rssi_bob)
plt.plot(xq2, vq1_bob, ':.')
plt.plot(xq2, bob_afterfilter, '-')
plt.legend(['Original values', 'Interpolated values', 'savgol_filter'])

plt.show()

# Calculate the lengths of the filtered data
m1 = len(alice_afterfilter)
m2 = len(bob_afterfilter)

# Compute the high-frequency components
alice_high = vq1_alice - alice_afterfilter
bob_high = vq1_bob - bob_afterfilter


# KEY Generation
# Initialize variables
bit_agree_rate_window = []
alice_afterfilter = alice_afterfilter.T  # Transpose alice_afterfilter
bob_afterfilter = bob_afterfilter.T  # Transpose bob_afterfilter

m1 = len(alice_afterfilter)
m2 = len(bob_afterfilter)

# Set window size and calculate window_time
window_size = 10
window_time = (window_size / M) / sample_rate

# Initialize result variables
bit_agree_rate_all = []
bit_agree_rate_all_std = []
bit_agree_rate_all_afterCS = []
bit_agree_rate_all_std_afterCS = []
bit_generation_rate_all = []
bit_generation_rate_all_std = []
bits_all = []



for alpha in tqdm(np.arange(0, 1.1, 0.1)):
    bits_sequence_1 = []
    #aprEntropyXData1 = []
    #aprEntropyXData2 = []
    bit_agree_rate = []
    bit_agree_rate_afterCS = []
    bit_generation_rate = []
    startIndex = 0
    endIndex = window_size
    
    
    numIterations = int(len(alice_afterfilter) / window_size)

    for _ in range(numIterations):
        #aprEntropyXData1.append(entropy1(alice_afterfilter[startIndex:endIndex]))
        startIndex += window_size
        endIndex += window_size

    startIndex = 0
    endIndex = window_size
    numBitsPerSample = 2
    bits_sequence_2 = []
    
    for _ in range(numIterations):
        #aprEntropyXData2.append(entropy1(bob_afterfilter[startIndex:endIndex]))
        startIndex += window_size
        endIndex += window_size

    startIndex = 0
    endIndex = window_size
    

    for _ in tqdm(range(numIterations)):
        sampleArray = alice_afterfilter[startIndex:endIndex]
        Secret_key, validIndices = mAryQuantization(sampleArray, numBitsPerSample, alpha)
        
        #print("Secret_key Alice is ")
        #print(Secret_key) 
        #print(Secret_key)
        #input("wait")
        bits_1 = np.array([int(bit) for bit in Secret_key])
        
        sampleArray = bob_afterfilter[startIndex:endIndex]
        Secret_key, validIndices = mAryQuantization(sampleArray, numBitsPerSample, alpha)
        bits_2 = np.array([int(bit) for bit in Secret_key])
        
        #print("Secret_key Bob is ")
        #print(Secret_key) 

        bit_agree_rate_temp = match_rate(bits_1, bits_2)
        bit_agree_rate.append(bit_agree_rate_temp)

        # Error correction code goes here
        
        # Perform error correction
        len_bits = min(len(bits_1), len(bits_2))
        bits_a = np.array(bits_1[:len_bits])
        bits_b = np.array(bits_2[:len_bits])
        
    
        A = np.random.randn(6, len_bits)
        y1 = np.dot(A, bits_b)
        y2 = np.dot(A, bits_a)
        y = y1 - y2
    
        # Perform error correction using myHomotopy
        mismatch, iterTimes = myHomotopy(A, y)
        
        # Thresholding
        thre1 = 0.5
        thre2 = -0.5
    
        for j in range(len(mismatch)):
            if mismatch[j] > thre1:
                mismatch[j] = 1
            elif mismatch[j] < thre2:
                mismatch[j] = -1
            else:
                mismatch[j] = 0
        
        bits_recover = np.logical_xor(bits_a, mismatch.reshape(len(mismatch))).astype(int)
        #print("bits_recover is ")
        #print(bits_recover) 

        # Calculate bit agreement rate after error correction
        bit_agree_rate_temp = match_rate(bits_recover, bits_b)
        bit_agree_rate_afterCS.append(bit_agree_rate_temp)
    
        bits_sequence_1.extend(bits_recover.tolist())
        bits_sequence_2.extend(bits_b.tolist())
    
        bit_generation_rate_temp = len(bits_a) / window_time
        bit_generation_rate.append(bit_generation_rate_temp)
        
        startIndex += window_size
        endIndex += window_size
    
       
    #input("wait")
    bits_all.extend(bits_sequence_1)
    bit_agree_rate_all.append(np.mean(bit_agree_rate))
    bit_agree_rate_all_std.append(np.std(bit_agree_rate))
    bit_agree_rate_all_afterCS.append(np.mean(bit_agree_rate_afterCS))
    bit_agree_rate_all_std_afterCS.append(np.std(bit_agree_rate_afterCS))
    bit_generation_rate_all.append(len(bits_sequence_1) / time_duration)
    bit_generation_rate_all_std.append(np.std(bit_generation_rate))

# Generate the x values
x = np.arange(0, 1.1, 0.1)

# Create Figure
plt.figure(figsize=(12, 5))

# Subplot 1
plt.subplot(1, 2, 1)
plt.errorbar(x, bit_agree_rate_all, np.array(bit_agree_rate_all_std) / 4, label='Before reconciliation')
plt.errorbar(x, bit_agree_rate_all_afterCS, np.array(bit_agree_rate_all_std_afterCS) / 4, label='After reconciliation')
plt.xlabel('Alpha')
plt.ylabel('Key matching rate(%)')
plt.legend()
plt.title('Key Matching Rate (a)')

# Subplot 2
plt.subplot(1, 2, 2)
plt.errorbar(x, bit_generation_rate_all, np.array(bit_generation_rate_all_std) / 2)
plt.xlabel('Alpha')
plt.ylabel('Key generation rate(bit/sec)')
plt.title('Key Generation Rate (b)')

# Show the plot
plt.tight_layout()
plt.show()