# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:04:24 2024

@author: danie
"""

import pyMMF
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import rc

#rc('figure', figsize=(18,9))
#rc('text', usetex=True)

## Parameters
NA = 0.22
radius = 100. # in microns
areaSize = 2.5*radius # 3.5*radius # calculate the field on an area larger than the diameter of the fiber
npoints = 2**8 # resolution of the window
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

#plt.figure(); 
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

imode = 10
plt.imshow(np.abs(modes['SA']['profiles'][imode].reshape([npoints]*2)))
plt.show()

phases = np.exp(1.j * np.random.uniform(0, 2 * np.pi, n_modes))
mode_sum = np.zeros(plt_pts,dtype=np.complex_)

for i in range(n_modes):
    mode_sum += phases[idx[i]]*modes_semianalytical.profiles[idx[i]] 
    
plt.imshow(np.abs(mode_sum.reshape([npoints]*2)))
plt.show()