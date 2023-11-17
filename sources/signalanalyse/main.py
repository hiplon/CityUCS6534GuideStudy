# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 21:40:34 2023

@author: hkl
"""


import matplotlib_inline   # setup output image format
matplotlib_inline.backend_inline.set_matplotlib_formats('retina')
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300  # display larger images

from numpy import *
from sklearn import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from matplotlib.ticker import MaxNLocator

import re
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


keylen = 4

callen = 28

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


naxbox2 = [0, listlen*1.1, -55, -15]

plotx_list = pd.Series(range(1, listlen+1)).tolist()


plotx_list = np.array(plotx_list)
# example data
polyX = plotx_list

polyY1 = vq1_alice
polyY2 = vq1_bob

#polyY1 = alice_afterfilter
#polyY2 = bob_afterfilter

polyX = polyX[:,newaxis]

plin1 = {}
plin2 = {}
for d in [1,2,3,4,5,6]:
    # extract polynomial features with degree d
    polyfeats1 = preprocessing.PolynomialFeatures(degree=d)
    polyfeats2 = preprocessing.PolynomialFeatures(degree=d)
    polyXf1 = polyfeats1.fit_transform(polyX)
    polyXf2 = polyfeats2.fit_transform(polyX)

    # fit the parameters
    plin1[d] = linear_model.LinearRegression()
    plin1[d].fit(polyXf1, polyY1)
    
    plin2[d] = linear_model.LinearRegression()
    plin2[d].fit(polyXf2, polyY2)

kernels = [  DotProduct()    + WhiteKernel(), 
             DotProduct()**2 + WhiteKernel(), 
             DotProduct()**3 + WhiteKernel(), 
             RBF()           + WhiteKernel()   ]
gpr1 = {}
gpr2 = {}
for i,k in enumerate(kernels):
    gpr1[i] = gaussian_process.GaussianProcessRegressor(kernel=k, random_state=6534, normalize_y=True)
    gpr1[i].fit(polyX, polyY1)
    
    gpr2[i] = gaussian_process.GaussianProcessRegressor(kernel=k, random_state=6534, normalize_y=True)
    gpr2[i].fit(polyX, polyY2)

kernelnames = ['linear+rbf', 
               'poly2+rbf', 
               'poly3+rbf', 
               'rbf+rbf']
i=0
model = gpr1[i]

model2 = gpr2[i]

X = polyX
Y1 = polyY1
Y2 = polyY2
numx = 150
xr = linspace(naxbox2[0], naxbox2[1], numx)


plt.figure(figsize=(8, 4.8))

feattrans=None

if feattrans != None:
    xrf   = feattrans(xr[:,newaxis])
else:
    xrf = xr[:,newaxis]

if model.__class__.__name__ == "GaussianProcessRegressor":
    Ypred, Ystd = model.predict(xrf, return_std=True)
    Ypred2, Ystd2 = model2.predict(xrf, return_std=True)
    hasstd = True
else:
    Ypred = model.predict(xrf)
    hasstd = False

if hasstd:
    plt.fill_between(xr, Ypred - 2*Ystd, Ypred + 2*Ystd,
                 alpha=0.1, color='k')
    plt.fill_between(xr, Ypred2 - 2*Ystd2, Ypred2 + 2*Ystd2,
                 alpha=0.1, color='g')

#plt.plot(plotx_list, Y1, 'b.')
plt.plot(xr, Ypred, 'r-')
#plt.plot(plotx_list, Y2, 'b.')
plt.plot(xr, Ypred2, 'r-')
plt.axis(naxbox2)
plt.plot(plotx_list, polyY1, label='Alice', marker='*')
#plt.plot(xq, vq1_alice, ':.')
plt.plot(xq, alice_afterfilter,label='Alice-SG',marker = 'o')

plt.plot(plotx_list, polyY2, label='Bob', marker='s')
#plt.plot(xq2, vq1_bob, ':.')
plt.plot(xq2, bob_afterfilter, label='Bob-SG', marker='v')

plt.title('Alice and Bob Signal Analyst')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=8))
plt.ylabel('$RSSI(dBm)$')
# Display the plot
plt.grid(True)
plt.tight_layout()
plt.show()






























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


def plot_regr_trans_1d(model, axbox, X, Y, feattrans=None, numx=100):
    xr = linspace(axbox[0], axbox[1], numx)
    # predict the function
    if feattrans != None:
        xrf   = feattrans(xr[:,newaxis])
    else:
        xrf = xr[:,newaxis]
    if model.__class__.__name__ == "GaussianProcessRegressor":
        Ypred, Ystd = model.predict(xrf, return_std=True)
        hasstd = True
    else:
        Ypred = model.predict(xrf)
        hasstd = False
    if hasstd:
        plt.fill_between(xr, Ypred - 2*Ystd, Ypred + 2*Ystd,
                     alpha=0.2, color='k')
    plt.plot(X, Y, 'b.')
    plt.plot(xr, Ypred, 'r-')
    plt.axis(axbox); plt.grid(True)