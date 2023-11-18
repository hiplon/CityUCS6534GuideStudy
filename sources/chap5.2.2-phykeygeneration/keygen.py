# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 20:40:42 2023

@author: hkl

"""
import numpy as np
from scipy.signal import savgol_filter

def mAryQuantization(sampleArray, numBitsPerSample):
    sampleArrayLength = len(sampleArray)
    M = 2**numBitsPerSample

    minVal = min(sampleArray)
    maxVal = max(sampleArray)

    offset = minVal
    levelBase = []
    levelTop = []

    for i in range(1, M + 1):
        levelBase.append(offset)
        levelTop.append(levelBase[i - 1])
        offset = offset
    jj = 0
    decimalValArray = []
    validIndices = []
    
    for i in range(sampleArrayLength):
        for j in range(1, M + 1):
            if sampleArray[i] == minVal:
                decimalValArray.append(1)
                validIndices.append(i)
                jj += 1
                break
            if sampleArray[i] == maxVal:
                decimalValArray.append(M - 1) 
                validIndices.append(i)
                jj += 1
                break
            if levelBase[j - 1] <= sampleArray[i] <= levelTop[j - 1]:
                decimalValArray.append(j - 1)
                validIndices.append(i)
                jj += 1

    decimalValArrayLen = len(decimalValArray)
    bitString = []
    for i in range(decimalValArrayLen):
        bitString.extend(format(decimalValArray[i], f'0{numBitsPerSample}b'))

    return bitString

def myHomotopy(A, y):
    iter_times = 2000
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
        d = np.linalg.inv(R).dot(np.sign(c[act_set]))
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
        x[act_set] = x[act_set] + (gamma * d)[0]
        
        if np.linalg.norm(y - np.dot(A, x).reshape(6)) < 1e-6:
            break
    return x

alice_rssi_values = [-39, -33, -33, -33, -33, -34, -33, -33, -33, -33, -34, -34, -33, -33, -33, -33, -33, -33, -33, -34, -39, -34, -39, -33, -32, -33, -33, -33, -33, -33, -33, -34, -34, -33, -33, -33, -33, -33, -33, -29, -40, -37, -28, -31, -31, -32, -31, -31, -30, -31, -30, -28, -25, -25, -37, -37, -29, -27, -27, -29, -29, -29, -30, -32, -33, -31, -31, -30, -30, -31, -26, -28, -29, -29, -28, -28, -28, -28, -30, -30, -31, -26, -26, -27, -26, -31, -30, -32, -32, -32, -32, -33, -33, -28, -28, -29, -29, -29, -31, -32, -29, -31, -31, -31, -26, -28, -28, -28, -28, -27, -27, -33, -27]
bob_rssi_values = [-25, -26, -25, -25, -46, -26, -25, -26, -26, -26, -26, -26, -25, -26, -26, -26, -26, -26, -26, -26, -26, -26, -26, -25, -25, -26, -26, -25, -25, -26, -25, -26, -26, -26, -26, -25, -25, -25, -25, -22, -51, -52, -20, -24, -24, -25, -24, -24, -23, -24, -23, -21, -17, -18, -53, -52, -42, -40, -19, -22, -41, -22, -22, -25, -26, -23, -23, -22, -22, -23, -18, -21, -21, -21, -20, -20, -20, -42, -22, -22, -18, -19, -19, -20, -19, -23, -23, -24, -24, -24, -25, -25, -25, -20, -20, -20, -20, -20, -24, -24, -21, -23, -23, -23, -19, -20, -20, -21, -20, -20, -20, -41, -20]

startIndex = 0
endIndex = 64

rssi_alice = np.array(alice_rssi_values[startIndex:endIndex])
rssi_bob = np.array(bob_rssi_values[startIndex:endIndex])

# Define filter parameters
order = 5
framelen = 11

# Apply Savitzky-Golay filter to Alice and Bob's data
alice_afterfilter = savgol_filter(rssi_alice, window_length=framelen, polyorder=order)
bob_afterfilter = savgol_filter(rssi_bob, window_length=framelen, polyorder=order)
# 
print("Array A (Alice's RSSI):", alice_afterfilter)
print("Array B (Bob's RSSI):", bob_afterfilter)

print(len(alice_afterfilter))
print(len(bob_afterfilter))

#Secret_key1 = mAryQuantization(rssi_alice, 2)
#Secret_key2 = mAryQuantization(rssi_bob, 2)

Secret_key1 = mAryQuantization(alice_afterfilter, 2)
Secret_key2 = mAryQuantization(bob_afterfilter, 2)

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
mismatch = myHomotopy(A, y)

bits_recover = np.logical_xor(bits_a, mismatch.reshape(len(mismatch))).astype(int)

print("Before error correction")
print("Alice's generated key" + str(Secret_key1))
print("Bob's generated key" + str(Secret_key2))
print("After perform error correction for Alice")
print("Alice's generated key" + str(bits_recover))

