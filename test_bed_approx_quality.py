# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 19:33:22 2019

@author: Simon
"""

import Glacier
import numpy as np
import matplotlib.pyplot as plt

# You can play with n_samples to see how this influences the difference between
# the exact and approximated concave bed.
n_samples = 10

# Exact Simulation
glac = Glacier.Glacier(mean_gletsjer_w=2930.)
lengths = np.linspace(1., 10e3, 100)

# Approximate Simulation
glac.create_concave_bed(1000., 1800.)
exact = glac.simulate(400, lambda t: 500., plot=True)
sample_x = np.linspace(0., 10000., n_samples)
sample_y = 1000. * np.exp(-sample_x / 1800.)
glac.create_custom_bed(sample_x, sample_y)
approx  = glac.simulate(400, lambda t: 500., plot=True)

times = np.arange(len(approx))
plt.plot(times, approx - exact)
plt.xlabel("Runtime (year)")
plt.ylabel("Difference (m)")
plt.title("$\\Delta L$ between Exact and Approxmate Concave Bed")
plt.show() 