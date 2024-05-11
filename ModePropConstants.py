# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:04:24 2024

@author: danie
"""

import pyMMF
import numpy as np
import matplotlib.pyplot as plt

plot_beta = 'n'  # plot propagation constant vs. mode number
plot_beta_dif = 'y' # plot the difference in propagation constants
plot_single_mode = 'n' # plot a transverse profile of a single mode
plot_sum_no_abs = 'y' # plot transverse profile with random phases, no absorption
plot_sum_abs = 'y' # plot transverse profile with random phases, mode-dependent absorption

## Parameters
NA = 0.22
radius = 100. # in microns
areaSize = 2.5*radius # 3.5*radius # calculate the field on an area larger than the diameter of the fiber
npoints = 2**7 # resolution of the window
n1 = 1.45
wl = 1.55 # wavelength in microns

# Create the fiber object
profile = pyMMF.IndexProfile(npoints = npoints, areaSize = areaSize)

# Initialize the index profile
profile.initStepIndex(n1=n1,a=radius,NA=NA)

# Instantiate the solver
solver = pyMMF.propagationModeSolver()

# Set the profile to the solver
solver.setIndexProfile(profile)

# Set the wavelength
solver.setWL(wl)

# Estimate the number of modes for a graded index fiber
Nmodes_estim = pyMMF.estimateNumModesSI(wl,radius,NA,pola=1)

print(f"Estimated number of modes using the V number = {Nmodes_estim}")

modes_semianalytical = solver.solve(mode = 'SI', curvature = None)

n_modes = len(modes_semianalytical.betas)
plt_pts = npoints**2

# Sort the modes
modes = {}
idx = np.flip(np.argsort(modes_semianalytical.betas), axis=0)
modes['SA'] = {'betas':np.array(modes_semianalytical.betas)[idx],'profiles':[modes_semianalytical.profiles[i] for i in idx]}

def sort(a):
    return np.flip(np.sort(a),axis=0)

plot_beta = 'y'
if (plot_beta == 'y'):
    plt.plot(sort(np.real(modes_semianalytical.betas)),
             'r--',
             label='Semi-analytical',
             linewidth=2.)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.title('Semi-analytical solutions' ,fontsize = 30)
    plt.ylabel('Propagation constant ', fontsize = 25) #$\beta$ (in $\mu$m$^{-1}$)
    plt.xlabel('Mode index', fontsize = 25)
    plt.legend(fontsize = 22,loc='upper right')
    plt.show()

betas_diff = np.real(modes['SA']['betas'][0:n_modes-1])-np.real(modes['SA']['betas'][1:n_modes])
print('mean betas diff: ',np.max(betas_diff))
print('std betas diff: ',np.std(betas_diff))
print('max betas diff: ',np.max(betas_diff))

if (plot_beta_dif == 'y'):
    plt.plot(betas_diff)
    plt.show()
    
if (plot_single_mode == 'y'):
    imode = 10
    plt.imshow(np.abs(modes['SA']['profiles'][imode].reshape([npoints]*2)))
    plt.show()

# select phases randomly for each mode and sum the fields, no absorption
#
# run this section of code repeatedly to have another random realization
#
phase_exp = np.exp(1.j * np.random.uniform(0, 2 * np.pi, n_modes))
mode_sum = np.zeros(plt_pts,dtype=np.complex_)

for i in range(n_modes):
    mode_sum += phase_exp[idx[i]]*modes_semianalytical.profiles[idx[i]] 

if (plot_sum_no_abs == 'y'):    
    plt.imshow(np.abs(mode_sum.reshape([npoints]*2))**2) # square to get intensity
    plt.show()

print('sum elements no absorption ',np.sum(np.abs(mode_sum)**2)/n_modes)

# absorption term
gamma = 10.e4   # in units of m^{-2}
length = 75.  # in units of m

# find the imaginary part of the propagation constant for each mode
beta_i = np.zeros(n_modes) # save it in case it is needed later
mode_sum_abs = np.zeros(plt_pts,dtype=np.complex_)

for i in range(n_modes):
    beta_i[idx[i]] = gamma/(modes_semianalytical.betas[idx[i]]*1.e6) # convert betas to units of m^{-1}
    mode_sum_abs += np.exp(-length*beta_i[idx[i]])*phase_exp[idx[i]]*modes_semianalytical.profiles[idx[i]]

print('sum elements with absorption ',np.sum(np.abs(mode_sum_abs)**2)/n_modes)

if (plot_sum_no_abs == 'y'):    
    plt.imshow(np.abs(mode_sum_abs.reshape([npoints]*2))**2) # square to get intensity
    plt.show()