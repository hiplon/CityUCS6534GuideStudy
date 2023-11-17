# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 19:55:11 2023

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


data = pd.read_csv('chacha20vsaes128.csv')


plt.figure(figsize=(8, 4.8))  # Adjust the figure size as needed

plotx_list = pd.Series(range(1, 101)).tolist()
 
data_aes = data.where(data['type']=="aes128")

data_aes = data_aes.dropna(axis = 'rows', how='all')

data_chacha = data.where(data['type']=="chacha20").dropna(axis = 'columns', how='all')

data_chacha = data_chacha.dropna(axis = 'rows', how='all')


plt.plot(plotx_list, data_aes['bitrate'], label='AES-128', marker='*')
plt.plot(plotx_list, data_chacha['bitrate'], label='Chacha20', marker='s')

# Add labels and legend
plt.xlabel('Timeline')
plt.ylabel('$Bitrate(Mbits/sec)$')
plt.title('Performance Competition Between ChaCha20 and AES-128 by $iperf3$')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=8))

# Display the plot
plt.grid(True)
plt.tight_layout()
plt.show()

naxbox2 = [0, 150, 30, 60]

plotx_list = np.array(plotx_list)
# example data
polyX = plotx_list
polyY = data_chacha['bitrate']
polyY2 = data_aes['bitrate']
polyX = polyX[:,newaxis]

plin = {}
plin2 = {}
for d in [1,2,3,4,5,6]:
    # extract polynomial features with degree d
    polyfeats = preprocessing.PolynomialFeatures(degree=d)
    polyXf = polyfeats.fit_transform(polyX)

    # fit the parameters
    plin[d] = linear_model.LinearRegression()
    plin[d].fit(polyXf, polyY)
    
    plin2[d] = linear_model.LinearRegression()
    plin2[d].fit(polyXf, polyY2)

kernels = [  DotProduct()    + WhiteKernel(), 
             DotProduct()**2 + WhiteKernel(), 
             DotProduct()**3 + WhiteKernel(), 
             RBF()           + WhiteKernel()   ]
gpr = {}
gpr2 = {}
for i,k in enumerate(kernels):
    gpr[i] = gaussian_process.GaussianProcessRegressor(kernel=k, random_state=6534, normalize_y=True)
    gpr[i].fit(polyX, polyY)
    
    gpr2[i] = gaussian_process.GaussianProcessRegressor(kernel=k, random_state=6534, normalize_y=True)
    gpr2[i].fit(polyX, polyY2)

kernelnames = ['linear+rbf', 
               'poly2+rbf', 
               'poly3+rbf', 
               'rbf+rbf']
i=3
model = gpr[i]

print(model.__class__.__name__)

model2 = gpr2[i]
print(model2.__class__.__name__)

X = polyX
Y = polyY
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


'''
if model.__class__.__name__ == "GaussianProcessRegressor":
    Ypred, Ystd = model.predict(polyX, return_std=True)
    Ypred2, Ystd2 = model2.predict(polyX, return_std=True)
    hasstd = True
else:
    Ypred = model.predict(polyX)
    Ypred2 = model2.predict(polyX)
    hasstd = False
if hasstd:
    plt.fill_between(plotx_list, Ypred - 2*Ystd, Ypred + 2*Ystd,
                 alpha=0.1, color='k')
    plt.fill_between(plotx_list, Ypred2 - 2*Ystd2, Ypred2 + 2*Ystd2,
                 alpha=0.1, color='g')
'''
plt.plot(plotx_list, Y, 'b.')
plt.plot(xr, Ypred, 'r-')
plt.plot(plotx_list, Y2, 'b.')
plt.plot(xr, Ypred2, 'r-')
plt.axis(naxbox2)
plt.plot(plotx_list, data_aes['bitrate'], label='AES-128', marker='*')
plt.plot(plotx_list, data_chacha['bitrate'], label='Chacha20', marker='s')
plt.title('Performance Comparison Between ChaCha20 and AES-128 by $iperf3$')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=8))
plt.xlabel('Timeline')
plt.ylabel('$Bitrate(Mbits/sec)$')
# Display the plot
plt.grid(True)
plt.tight_layout()
plt.show()
'''
gprfig = plt.figure(figsize=(9,4))
for i,k in enumerate(kernels):
    plt.subplot(2,2,i+1)
    plot_regr_trans_1d(gpr[i], naxbox2, polyX, polyY)
    plt.title(kernelnames[i])
    plt.show()
plt.tight_layout()
plt.close()


kernels = [ DotProduct()**2 + WhiteKernel(), 
            DotProduct()**3 + WhiteKernel(), 
            RBF()           + WhiteKernel() ]
kernelnames = ['poly2', 
               'poly3', 
               'rbf']

sfig = plt.figure(figsize=(10,6))
for j,k in enumerate(kernels):
    for i,s in enumerate([20,10,5,2]):
        gpr = gaussian_process.GaussianProcessRegressor(kernel=k, random_state=0, normalize_y=True)
        gpr.fit(polyX[::s], polyY[::s])
        plt.subplot(3,4,i+1+j*4)
        plot_regr_trans_1d(gpr, naxbox2, polyX[::s], polyY[::s])
        if (i==0):
            plt.ylabel(kernelnames[j])
        if (j==0):
            plt.title("{} points".format(len(polyY[::s])))
        plt.show()
plt.close()
'''