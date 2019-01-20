# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 19:00:23 2019

@author: Simon
"""

import Glacier
import numpy as np
import matplotlib.pyplot as plt

glac = Glacier.Glacier(mean_gletsjer_w=2930.)
lengths = np.linspace(1., 10e3, 100)
equi_vals = np.genfromtxt(".\\ELA_Spitsbergen.txt").T

c_geom= np.genfromtxt(".\\glacierelevationdata.txt", delimiter=',').T
lengths = np.linspace(0.1, c_geom[0,-2], 1000)
glac.create_custom_bed(c_geom[0], c_geom[1])
    
# Do Simulation without buckets
glac.generate_profile(lengths, True, "Veteranen Glacier Bed")
l1 = glac.simulate(len(equi_vals[0]), lambda t : equi_vals[1,-1] + 50,
                  calving_enabled=True,
                  plot=False,
                  plot_forcing=False,
                  plot_title="ESL Data Spitsbergen on a Veteranen Glacier")

# Do Simulation with buckets
glac.create_custom_bed(c_geom[0], c_geom[1])
buckets = np.genfromtxt(".\\Glacierbuckets.csv", delimiter=';', skip_header=1)[:,1:]
for buc in buckets:
    glac.add_bucket(buc[0], buc[1], buc[2], buc[5], buc[6])  
        
l2 = glac.simulate(len(equi_vals[0]), lambda t : equi_vals[1,-1] + 50,
                  calving_enabled=True,
                  plot=False,
                  plot_forcing=False,
                  plot_title="ESL Data Spitsbergen on a Veteranen Glacier")

times = np.arange(len(l1))
plt.plot(times, l1 - l2, label='Bucket Length Impact')
#plt.plot(times, l2, label="With Buckets")
plt.xlabel("Runtime (year)")
plt.ylabel("Length (m)")
plt.title("Impact of Buckets on Veteranen Glacier")
plt.legend()
plt.show()