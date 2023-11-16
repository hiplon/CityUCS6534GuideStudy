# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 18:15:00 2023

@author: hkl
"""

import re
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def mAryQuantization(sampleArray, numBitsPerSample, alpha):
    sampleArrayLength = len(sampleArray)
    #entropy_TestData = entropy1(sampleArray)

    # M-ary quantization: M - number of levels
    M = 2**numBitsPerSample

    minVal = min(sampleArray)
    maxVal = max(sampleArray)

    sampleSize = maxVal - minVal

    # Calculate sizes
    ratioStep = (1 - alpha) / M
    ratioGband = alpha / (M - 1)

    stepSize = ratioStep * sampleSize
    gbandSize = ratioGband * sampleSize

    # Start from the bottommost level
    offset = minVal
    levelBase = []
    levelTop = []

    for i in range(1, M + 1):
        levelBase.append(offset)
        levelTop.append(levelBase[i - 1] + stepSize)
        offset = offset + stepSize + gbandSize
    jj = 0
    decimalValArray = []
    validIndices = []
    
    for i in range(sampleArrayLength):
        for j in range(1, M + 1):
            if sampleArray[i] == minVal:
                decimalValArray.append(1)  # Decimal assignment starts from 0
                validIndices.append(i)
                jj += 1
                break
            if sampleArray[i] == maxVal:
                decimalValArray.append(M - 1)  # Decimal assignment starts from 0
                validIndices.append(i)
                jj += 1
                break
            if levelBase[j - 1] <= sampleArray[i] <= levelTop[j - 1]:
                decimalValArray.append(j - 1)  # Decimal assignment starts from 0
                validIndices.append(i)
                jj += 1

    decimalValArrayLen = len(decimalValArray)
    bitString = []
    for i in range(decimalValArrayLen):
        #print("ROUND"+str(i))
        #print(bitString)
        bitString.extend(format(decimalValArray[i], f'0{numBitsPerSample}b'))

    return bitString, validIndices

def myHomotopy(A, y):
    iter_times = 2000
    n = A.shape[0]
    m = A.shape[1]
    x = np.zeros((m, 1))
    act_set = []

    for iter_idx in range(1, iter_times + 1):
        # Compute residual correlations
        c = np.dot(A.T, (y - np.dot(A, x).reshape(6)))
        
        # Compute active set
        lambda_max_idx = np.argmax(np.abs(c))
        lambda_max = np.abs(c[lambda_max_idx])

        act_set = np.where(np.abs(np.abs(c) - lambda_max) < 1e-5)[0]

        
        state = np.zeros(m)
        state[act_set] = 1
        
        
        # Compute direction
        R = np.dot(A[:, act_set].T, A[:, act_set])
        
        #d = np.linalg.solve(R, np.sign(c[act_set]))
        d = np.linalg.pinv(R).dot(np.sign(c[act_set]))
        

        # Compute step
        gamma = 1000
        for idx in range(m-1):
            if state[idx]:
                # Active elements
                my_id = np.where(act_set == idx)[0]
                tmp = max(0, -x[idx] / d[my_id])
            else:
                # Null elements
                av = np.dot(A[:, idx].T, np.dot(A[:, act_set], d))
                tmp1 = max(0, (lambda_max - c[idx]) / (1 - av))
                tmp2 = max(0, (lambda_max + c[idx]) / (1 + av))
                tmp = min(tmp1, tmp2)

            if tmp > 0:
                gamma = min(tmp, gamma)

        # Update x

        
        x[act_set] = x[act_set] + (gamma * d)[0]
        

        #input("wait")

        # Check for convergence
        if np.linalg.norm(y - np.dot(A, x).reshape(6)) < 1e-6:
            break

    return x , iter_idx

# read the RSSI pair
with open('sampling1.txt', 'r') as file:
    input_data = file.read()

# Initailing the list
alice_rssi_values = []
bob_rssi_values = []

# extrat "Alice's RSSI" and "Bob's RSSI"
alice_pattern = re.compile(r"Alice's RSSI: (\S+)")
bob_pattern = re.compile(r"Bob's RSSI: (\S+)")

# 
alice_matches = re.findall(alice_pattern, input_data)
bob_matches = re.findall(bob_pattern, input_data)

# 
alice_rssi_values.extend(map(int, alice_matches))
bob_rssi_values.extend(map(int, bob_matches))


startIndex = 32
endIndex = 64

rssi_alice = np.array(alice_rssi_values[startIndex:endIndex])
rssi_bob = np.array(bob_rssi_values[startIndex:endIndex])

m1 = rssi_alice.shape[0]
n1 = 1

m2 = rssi_bob.shape[0]
n2 = 1


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

# 
print("Array A (Alice's RSSI):", alice_afterfilter)
print("Array B (Bob's RSSI):", bob_afterfilter)

print(len(alice_afterfilter))
print(len(bob_afterfilter))

Secret_key1, validIndices = mAryQuantization(alice_afterfilter, 2, 1)
Secret_key2, validIndices = mAryQuantization(bob_afterfilter, 2, 1)


bits_1 = np.array([int(bit) for bit in Secret_key1])
bits_2 = np.array([int(bit) for bit in Secret_key2])

len_bits = min(len(bits_1), len(bits_2))
bits_a = np.array(bits_1[:len_bits])
bits_b = np.array(bits_2[:len_bits])

A = np.random.randn(6, len_bits)
y1 = np.dot(A, bits_b)
y2 = np.dot(A, bits_a)
y = y1 - y2

# Perform error correction using myHomotopy
mismatch, iterTimes = myHomotopy(A, y)

bits_recover = np.logical_xor(bits_a, mismatch.reshape(len(mismatch))).astype(int)



print("Before error correction")
print(Secret_key1)
print(Secret_key2)
print("After perform error correction for Alice")
print(bits_recover)

