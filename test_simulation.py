# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 19:25:27 2019

@author: Simon
"""

import Glacier
import numpy as np

glac = Glacier.Glacier(mean_gletsjer_w=2930.)
lengths = np.linspace(1., 10e3, 100)
equi_vals = np.genfromtxt(".\\ELA_Spitsbergen.txt").T

#######################
# E Function          #
#######################

def esl_step_func(t):
    steps = [500., 700., 400.]
    tmod = t % 1500.
    if tmod < 500.:
        return steps[0]
    elif 500. <= tmod <= 1000.:
        return steps[1]
    return steps[2]

def esl_data(t):
    ti = np.argmin(np.abs((equi_vals[0] - equi_vals[0,0]) - t))
    return equi_vals[1,ti] 

#######################
# Test Constant E     #
#######################
# Test Linear Simulation
glac.create_linear_bed(1000., .1)
glac.simulate(400, lambda t: 500., plot=True, plot_ice_development=True)
print "Calculate Steady State"
print glac.calculate_stable_state(1., 500)
print "Calculate e-folding for this set from bottom and top"
print glac.calculate_efolding(1., 500)
print glac.calculate_efolding(16e3, 500) # Reverse
         
# Test Concave Simulation
glac.create_concave_bed(1000., 1800.)
glac.simulate(400, lambda t: 500., plot=True, plot_ice_development=True)
print "Calculate Steady State"
print glac.calculate_stable_state(1., 500)
print "Calculate e-folding for this set from bottom and top"
print glac.calculate_efolding(1., 500)
print glac.calculate_efolding(7e3, 500)

#######################
# Test Step-Wise E    #
#######################
# Test Linear Bed
glac.create_linear_bed(1000., .1)
glac.simulate(4500, esl_step_func, plot=True,  #plot_ice_development=True,
              plot_title="Step-wise equilibrium on a Linear Bed")
# Test Concave Bed
glac.create_concave_bed(1000., 1800.)
glac.simulate(4500, esl_step_func, plot=True,  #plot_ice_development=True,
              plot_title="Step-wise equilibrium on a Concave Bed")

#######################
# Test Prescribed E   #
#######################
# Test Linear Bed
glac.create_linear_bed(1000., .1)
glac.set_length(glac.calculate_stable_state(1., equi_vals[1,0])[1])
glac.simulate(len(equi_vals[0]), esl_data, plot=True,
              plot_title="ESL Data Spitsbergen on a Linear Bed")
# Test Concave Bed
glac.create_concave_bed(1000., 1800.)
glac.set_length(glac.calculate_stable_state(1., equi_vals[1,0])[1])
exact_l = glac.simulate(len(equi_vals[0]), esl_data,
                        plot=True,
                        plot_title="ESL Data Spitsbergen on a Concave Bed")
##################################
# Test Calving with Constant E   #
##################################
# Linear Bed
glac.create_linear_bed(1000., 1.4e3 / 1e4)
glac.simulate(400, lambda t: 500., plot=True, calving_enabled=True,
              plot_forcing=True,
              plot_title='Linear Bed with Calving Flux and Constant E')
# Concave Bed
glac.create_concave_bed(1000., 1800., height_adjustment=-130)
glac.generate_profile(lengths, True, "Concave Bed")
glac.simulate(400, lambda t: 200., plot=True, calving_enabled=True,
              plot_forcing=True,
              plot_title='Concave Bed with Calving Flux and Constant E')