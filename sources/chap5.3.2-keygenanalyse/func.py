# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 20:20:04 2023

@author: hkl
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Define the entropy1 function in Python
def entropy1(x):
    x = np.sort(x)
    len_x = len(x)
    y = np.where(np.concatenate(([x[0] - 1], x)) != np.concatenate((x, [x[len_x - 1] + 1])))
    p = np.diff(y) / len_x
    H = -np.sum(p * np.log2(p))
    return H

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

    startIndex = 0
    endIndex = numBitsPerSample
    decimalValArrayLen = len(decimalValArray)
    bitString = []
    for i in range(decimalValArrayLen):
        #print("ROUND"+str(i))
        #print(bitString)
        bitString.extend(format(decimalValArray[i], f'0{numBitsPerSample}b'))

    return bitString, validIndices



def match_rate(bits_1, bits_2):
    num_bitMismatch = 0
    
    if len(bits_1) < len(bits_2):
        length_limit = len(bits_1)
    else:
        length_limit = len(bits_2)

    for i in range(length_limit):
        
        if bits_1[i] != bits_2[i]:
            num_bitMismatch += 1

    bitDisAgreement = (num_bitMismatch / length_limit) * 100
    bitAgreement = 100 - bitDisAgreement

    return bitAgreement

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