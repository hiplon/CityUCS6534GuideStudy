# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 18:15:00 2023

@author: hkl
"""

import re
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


keylen = 4

callen = 8

listlen = keylen * callen

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


startIndex = 0
endIndex = listlen

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

# 将32个值分为4行，每行8个数
rows = [rssi_alice[i:i+8] for i in range(0, listlen, callen)]

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



rows = [rssi_bob[i:i+8] for i in range(0, listlen, callen)]

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