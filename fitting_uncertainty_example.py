#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:13:40 2024

@author: trevorreed
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm



#%%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Generate pseudo-data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#--------------- Generate background distribution (second-order polynomial) -----------------#
def background_pdf(x):
    # Coefficients for the polynomial: a*x^2 + b*x + c
    a = -5
    b = 100
    c = 1.0
    return a * x**2 + b * x + c

# Accept-Reject sampling for background data
N_background = 50000
x_min, x_max = 1.0, 1.4
x = np.linspace(x_min, x_max, 1000)
p = background_pdf(x)
p_max = p.max()

N_attempts = int(N_background * 1.5)
x_rand = np.random.uniform(x_min, x_max, N_attempts)
y_rand = np.random.uniform(0, p_max, N_attempts)
mask = y_rand <= background_pdf(x_rand)
background_data = x_rand[mask][:N_background]
#-----------------------------------------------------------------------------------------------#



# Generate signal distribution (Gaussian)
N_signal = 5000
mean_signal = 1.192
sigma_signal = 0.015
signal_data = np.random.normal(mean_signal, sigma_signal, N_signal)

# Combine background and signal data
data = np.concatenate((background_data, signal_data))

# 3. Calculate uncertainties (sqrt of counts per bin)
bin_width = 0.002
bin_edges = np.arange(x_min, x_max + bin_width, bin_width)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
counts, _ = np.histogram(data, bins=bin_edges)
uncertainties = np.sqrt(counts)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#%%



### Fit functions
def gaus_func(x, N_gauss, mean_gauss, sigma_gauss):
    return N_gauss * np.exp(-(x - mean_gauss)**2 / (2 * sigma_gauss**2))# / (sigma_gauss * np.sqrt(2 * np.pi))
    
def fit_function(x, a0, a1, N_gauss, mean_gauss, sigma_gauss):
    #background = a0 + a1 * x + a2 * x**2
    background = a0 + a1 * x
    #signal = N_gauss * np.exp(-(x - mean_gauss)**2 / (2 * sigma_gauss**2)) / (sigma_gauss * np.sqrt(2 * np.pi))# * bin_width
    return background + gaus_func(x, N_gauss, mean_gauss, sigma_gauss)




#%%
#------------------------------------Fit the distribution----------------------------#
# Initial parameter guesses and bounds
p0 = [100, -5, 5000, 1.192, 0.01]
bounds = ([-np.inf, -np.inf, 0, 1.15, 0.001], [100000, np.inf, 100000, 1.25, 0.04])

# Perform the fit
popt, pcov = curve_fit(fit_function, bin_centers, counts, p0=p0, sigma=uncertainties, absolute_sigma=True, bounds=bounds)
#popt, pcov = curve_fit(fit_function, bin_centers, counts, sigma=uncertainties, absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))
#------------------------------------------------------------------------------------#




# ----------------------------------Plot the fit results-----------------------------#
x_fit = np.linspace(x_min, x_max, 1000)
y_fit = fit_function(x_fit, *popt)
plt.errorbar(bin_centers, counts, yerr=uncertainties, fmt='o', markersize=4, label='Data')
plt.plot(x_fit, y_fit, label='Total Fit', color='red')
plt.xlabel('Mass (GeV)')
plt.ylabel('Counts')
plt.show()

# Draw the background and gaussian portions of the fit individually
y_gaus_fit = gaus_func(x_fit, popt[2], popt[3], popt[4])
plt.plot(x_fit, y_gaus_fit, label='Gaussian', color='green', linestyle='--')
plt.legend()
#------------------------------------------------------------------------------------#




### Calculate yield
A_gauss, mean_gauss, sigma_gauss = popt[2], popt[3], popt[4]
integral = A_gauss * np.sqrt(np.pi * 2.0) * sigma_gauss
gaus_yield = integral / bin_width     # Divide by bin width to convert integral to yield
gaus_yield_rounded = int(gaus_yield)


### Get parameter uncertainties
Var_A_gauss = pcov[2, 2]
standard_dev_A_gauss = np.sqrt(Var_A_gauss)
Var_mean_gauss = pcov[3, 3]
standard_dev_mean_gauss = np.sqrt(Var_mean_gauss)
Var_sigma_gauss = pcov[4, 4]
standard_dev_sigma_gauss = np.sqrt(Var_sigma_gauss)
Cov_A_mean = pcov[2, 3]
Cov_A_sigma = pcov[2, 4]
Cov_mean_sigma = pcov[3, 4]


### Use error propagation equation to get uncertainty of the yield
uncertainty_I = np.sqrt((Var_A_gauss * 2.0 * np.pi * sigma_gauss**2) + (Var_sigma_gauss * 2.0 * np.pi * A_gauss**2) + \
                        4.0 * np.pi * Cov_A_sigma * A_gauss * sigma_gauss)
uncertainty_yield = uncertainty_I / bin_width    # Divide by bin width to convert integral uncertainty to yield uncertainty
uncertainty_yield_rounded = int(uncertainty_yield)

# Add fit parameters to the plot
param_text = (
    f"A = {A_gauss:.2f} ± {standard_dev_A_gauss:.2f}\n"
    rf"$\mu$ = {mean_gauss:.5f} ± " + f"{standard_dev_mean_gauss:.5f}" + "\n"
    rf"$\sigma$ = {sigma_gauss:.5f} ± {standard_dev_sigma_gauss:.5f}" + "\n"
    f"Signal Yield = " + "\n" + f"{gaus_yield_rounded} ± {uncertainty_yield_rounded} counts"
)
plt.text(0.05, 0.95, param_text, transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

#print(f"Signal Yield (Integral over Gaussian): {gaus_yield_rounded:.2f} ± {uncertainty_yield_rounded:.2f} counts")
