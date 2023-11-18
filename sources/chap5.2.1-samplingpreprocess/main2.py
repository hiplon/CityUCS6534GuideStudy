# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 18:15:00 2023

@author: hkl
"""

import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

keylen = 8

callen = 256

listlen = keylen * callen

startIndex = 199
endIndex = startIndex+listlen

alice = np.loadtxt('outdoor-walk-alice.txt')
rssi_alice = alice[startIndex:endIndex, 2]

m1 = rssi_alice.shape[0]
n1 = 1

bob = np.loadtxt('outdoor-walk-bob.txt')
rssi_bob = bob[startIndex:endIndex, 2]
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


# Initailing the list
alice_rssi_values = alice_afterfilter
bob_rssi_values = bob_afterfilter


# 
print("Array A (Alice's RSSI):", alice_rssi_values)
print("Array B (Bob's RSSI):", bob_rssi_values)

print(len(alice_rssi_values))
print(len(bob_rssi_values))



# 将32个值分为4行，每行8个数
rows = [alice_rssi_values[i:i+8] for i in range(0, listlen, callen)]

# 初始化结果列表
result_list = []

# 对每行进行处理
for row in rows:
    trend_values = []
    
    # 判断前后趋势
    for i in range(1, len(row)):
        if row[i] > row[i-1]:
            trend_values.append(1)
        else:
            trend_values.append(0)

    # 统计趋势数量
    count_ones = trend_values.count(1)
    count_zeros = trend_values.count(0)

    # 根据数量判断最终值
    result = 1 if count_ones > count_zeros else 0

    # 添加到结果列表
    result_list.append(result)

# 打印结果
print("Final Result List:", result_list)



rows = [bob_rssi_values[i:i+8] for i in range(0, listlen, callen)]

# 初始化结果列表
result_list = []

# 对每行进行处理
for row in rows:
    trend_values = []
    
    # 判断前后趋势
    for i in range(1, len(row)):
        if row[i] > row[i-1]:
            trend_values.append(1)
        else:
            trend_values.append(0)

    # 统计趋势数量
    count_ones = trend_values.count(1)
    count_zeros = trend_values.count(0)

    # 根据数量判断最终值
    result = 1 if count_ones > count_zeros else 0

    # 添加到结果列表
    result_list.append(result)

# 打印结果
print("Final Result List:", result_list)