# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 19:21:33 2019

@author: Simon
"""

import Glacier
import numpy as np

glac = Glacier.Glacier(mean_gletsjer_w=2930.)
lengths = np.linspace(1., 10e3, 100)
n_samples = 10

# Linear Bed
glac.create_linear_bed(1000., .1)
glac.generate_profile(lengths, True, "Linear Bed")

# Concave Bed
glac.create_concave_bed(1000., 1800.)
glac.generate_profile(lengths, True, "Concave Bed")

# Custom Bed
sample_x = np.linspace(0., 10000., n_samples)
sample_y = 1000. * np.exp(-sample_x / 1800.)
glac.create_custom_bed(sample_x, sample_y)
glac.generate_profile(lengths, True, "Custom (Concave-like) Bed")