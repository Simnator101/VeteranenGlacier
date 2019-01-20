# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 23:51:22 2018

"""

import numpy as np
import matplotlib.pyplot as plt

__all__ = ["Glacier"]

class Glacier(object):
    def __init__(self, xresolution=100, alpha=3., beta=.007, nu=10., width=1.,
                 calving_param=.1, calving_frac=.3, mean_gletsjer_w=3e3):
        assert(xresolution > 0)
        self.__type = 0
        self.__glaciel_l = 0.
        self.__xres = xresolution
        self.__alpha = alpha
        self.__beta = beta
        self.__nu = nu
        self.__width = width
        self.__cal_p = calving_param
        self.__cal_f = calving_frac
        self.__dxres_dl = 1.
        self.__buckets = np.array([], dtype=np.float)
        self.__mean_gletsjet_w = mean_gletsjer_w
        self.left_store = np.empty(1)
        self.right_store = np.empty(1)
        
    def create_linear_bed(self, height, slope):
        self.__type = 1
        self.__slope = slope
        self.__b0 = height
        self.__glaciel_l = 1.
        self.__buckets = np.array([], dtype=[('length', 'f8'), ('width', 'f8'),
                                  ('height', 'f8'), ('slope', 'f8'),
                                  ('dw_dh', 'f8')])
        
    def create_concave_bed(self, height, scale_length, height_adjustment=0.):
        self.__type = 2
        self.__scale_l = scale_length
        self.__b0 = height
        self.__ba = height_adjustment
        self.__glaciel_l = 1.
        self.__buckets = np.array([], dtype=[('length', 'f8'), ('width', 'f8'),
                                  ('height', 'f8'), ('slope', 'f8'),
                                  ('dw_dh', 'f8')])
        
    def create_custom_bed(self, positions, bed_elev):
        self.__type = 3
        self.__bed_pos = positions
        self.__bed_ele = bed_elev
        self.__glaciel_l = 1.
        self.__buckets = np.array([], dtype=[('length', 'f8'), ('width', 'f8'),
                                  ('height', 'f8'), ('slope', 'f8'),
                                  ('dw_dh', 'f8')])
        # Specific object for a custom bed
#        self.__xfit = np.linspace(0.0, max(positions), self.__xres)
#        self.__hfit = np.interp(self.__xfit, positions, bed_elev)
        
    def add_bucket(self, length, width, height, slope, dw_dh):
        self.__buckets = np.append(self.__buckets,
                                   np.array([
                                           (length, width, height, slope, dw_dh)
                                            ], dtype=self.__buckets.dtype))     
        
    def bed_height(self, x):
        if self.__type is 1:                    # Linear Bed
            return self.__b0 - self.__slope * x
        elif self.__type is 2:                  # Concave Bed
            return self.__ba + self.__b0 * np.exp(-(x / self.__scale_l))
        elif self.__type is 3:                  # Custom Bed
             return np.interp(x, self.__bed_pos, self.__bed_ele)
        return 0
    
    @property
    def wdepth_under_glacier(self):
        bedh_l = self.bed_height(self.__glaciel_l) 
        return -bedh_l if (bedh_l < .0) else 0.0
    
    @property
    def mean_bed_height(self):
        assert(not np.isclose(self.__glaciel_l, 0.0))
        if self.__type is 1:
            return self.__b0 - self.__slope * self.__glaciel_l / 2.
        elif self.__type is 2:
            _exp_p = (1. - np.exp(-self.__glaciel_l / self.__scale_l))
            return self.__ba + self.__b0 * self.__scale_l / self.__glaciel_l *\
                  _exp_p
        elif self.__type is 3:
            xfit = np.linspace(.0, self.__glaciel_l, self.__xres)
            _tmp_h = np.array([self.bed_height(xp) for xp in xfit])
            return np.trapz(_tmp_h, xfit) / self.__glaciel_l
        return 0.
                  
    @property
    def mean_slope(self):
        assert(not np.isclose(self.__glaciel_l, 0.0))
        if self.__type is 1:
            return self.__slope
        elif self.__type is 2:
            _exp_p = (1. - np.exp(-self.__glaciel_l / self.__scale_l))
            return self.__b0 / self.__glaciel_l * _exp_p
        elif self.__type is 3:
            xfit = np.linspace(.0, self.__glaciel_l, self.__xres)
            _tmp_h = np.array([self.bed_height(xp) for xp in xfit])
            _slp = -np.gradient(_tmp_h, xfit[1] - xfit[0])
            return np.trapz(_slp, xfit) / self.__glaciel_l
        return 0.
        
    @property
    def mean_ds_dl(self):
        assert(not np.isclose(self.__glaciel_l, 0.0))
        if self.__type is 1:
            return 0.
        elif self.__type is 2:
            _exp_p = (1. - np.exp(-self.__glaciel_l / self.__scale_l))
            _l_p = -self.__b0 / np.power(self.__glaciel_l, 2) * _exp_p
            _r_p = self.__b0 / self.__glaciel_l / self.__scale_l *\
                   np.exp(-self.__glaciel_l / self.__scale_l)
            return (_l_p + _r_p)
        elif self.__type is 3:
            _old_gl = max(.1, self.__glaciel_l)
            dl = _old_gl / self.__xres
            # Calculate Left and Right mean s
            self.__glaciel_l = self.__glaciel_l - dl
            ms_l = self.mean_slope
            self.__glaciel_l = self.__glaciel_l + 2 * dl
            ms_r = self.mean_slope
            
            self.__glaciel_l = _old_gl
            return (ms_r - ms_l) / 2. / dl
        return 0.
    
    @property
    def mean_ice_height(self):
        return self.__alpha / (1. + self.__nu * self.mean_slope) * np.sqrt(self.__glaciel_l)
    
    @property
    def length(self):
        return self.__glaciel_l
    
    def set_length(self, val):
        assert(not np.isclose(val, 0.))
        self.__glaciel_l = val
    
    def reset_length(self):
        self.set_length(1.0)
        
    def set_glacier_properties(self, alpha=3., beta=.007, nu=10., width=1., 
                               calving_param=.1, calving_frac=.3, mean_gletsjer_w=3e3):
        # self.reset_length() # Reset length because properties have changed
        self.__alpha = alpha
        self.__beta = beta
        self.__nu = nu
        self.__width = width
        self.__cal_p = calving_param
        self.__cal_f = calving_frac
        self.__mean_gletsjet_w = mean_gletsjer_w
        
    def simulate(self, timesteps, equiline_func, dt=1., calving_enabled=False,
                 forcing=0.0, plot=False, plot_title=None, plot_forcing=False,
                 plot_ice_development=False):
        assert(type(forcing) is float or type(forcing) is int)
        timesteps = max(1, timesteps)
        new_l = np.full(timesteps + 1, self.__glaciel_l)
        forcings = np.zeros((timesteps, 2))
        means = np.zeros((timesteps, 2))
        
        # DEMO
        self.right_store = np.empty(timesteps)
        self.left_store= np.empty(timesteps)
        
        for ti in range(timesteps):
            # Calculate dL/dt
            r_v = 3. * self.__alpha / (2. * (1. + self.__nu * self.mean_slope)) * np.sqrt(self.__glaciel_l)
            l_v = self.__alpha * self.__nu / pow(1. + self.__nu * self.mean_slope, 2) *\
                  pow(self.__glaciel_l, 3. / 2.) * self.mean_ds_dl
                  
            # DEMO
            self.right_store[ti] = r_v
            self.left_store[ti] = l_v
            
            mass_b = self.__beta * (self.mean_bed_height + self.mean_ice_height
                                    - equiline_func(ti * dt)) *\
                                   self.__glaciel_l * self.__width
            # If calving is enabled we need some extra forcing
            calv_fx = .0
            if calving_enabled is True:
                wd = self.wdepth_under_glacier
                dens_f = 1000. / 917.
                calv_fx = - self.__cal_p * wd * self.__width *\
                          np.max([self.__cal_f * self.mean_ice_height, dens_f * wd])
                          
            # Calculate Bucket Contribution
            bucket_frc = np.zeros(self.__buckets.size)
            for buc_i in range(bucket_frc.size):
                cbuck = self.__buckets[buc_i]
                bucket_frc[buc_i] = cbuck['width'] * (cbuck['height'] - equiline_func(ti * dt)) * cbuck['length']
                bucket_frc[buc_i] += .5 * (cbuck['slope'] * cbuck['width'] + (cbuck['height'] - equiline_func(ti * dt)) * cbuck['dw_dh']) * cbuck['length'] ** 2
                bucket_frc[buc_i] += 1. / 3. * cbuck['slope'] * cbuck['dw_dh'] * cbuck['length'] ** 3
                bucket_frc[buc_i] *= self.__beta / self.__mean_gletsjet_w / self.mean_ice_height
                # Check Scales
                bucket_frc[buc_i] = max(0.0, bucket_frc[buc_i])
                
            
            # Determine new glacier length
            new_l[ti + 1] = new_l[ti] + pow(r_v - l_v, -1) *\
                            (mass_b + forcing + calv_fx + np.sum(bucket_frc)) * dt
            # Only keep track of this if we need it
            if plot_forcing is True:
                forcings[ti] = [mass_b, calv_fx]
            if plot_ice_development is True:
                means[ti] = [self.mean_bed_height, self.mean_ice_height]
            self.__glaciel_l = new_l[ti + 1]
        
        if plot is True:
            f, ax1 = plt.subplots()
            ax1.plot(dt*np.arange(timesteps + 1, dtype=np.float), 1e-3 * new_l,
                     label="Glacier Length")
            ax1.set_xlabel("Time (yr)")
            ax1.set_ylabel("Length (km)")
            ax1.grid(True)
            if plot_title is not None and type(plot_title) is str:
                plt.title(plot_title)
            plt.legend(loc=4)
            plt.show()
            
        if plot_forcing is True:
            f, ax1 = plt.subplots()
            ax1.plot(dt*np.arange(timesteps, dtype=np.float), forcings.T[0],
                     label="Mass Balance Forcing")
            ax1.plot(dt*np.arange(timesteps, dtype=np.float), forcings.T[1],
                     label="Calving Forcing")
            ax1.set_xlabel("Time (yr)")
            ax1.set_ylabel("Forcing")
            ax1.grid(True)
            if plot_title is not None and type(plot_title) is str:
                plt.title(plot_title)
            plt.legend()
            plt.show()
            
        if plot_ice_development is True:
            f, ax1 = plt.subplots()
            ax1.plot(dt*np.arange(timesteps, dtype=np.float), means.T[0], 'k-',
                     label='Mean Bed Height')
            ax1.plot(dt*np.arange(timesteps, dtype=np.float), means.T[1], 'b-',
                     label='Mean Ice Thickness')
            ax1.plot(dt*np.arange(timesteps, dtype=np.float),
                     np.sum(means.T, axis=0),
                     'b--', label='Mean Surface Height')
            ax1.set_xlabel("Time (yr)")
            ax1.set_ylabel("Height (m)")
            ax1.grid(True)
            if plot_title is not None and type(plot_title) is str:
                plt.title(plot_title)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.show()
        return new_l
    
    def calculate_stable_state(self, starting_length, equi_height, dt=1.):
        starting_length = max(.1, starting_length)
        old_l, prev_l = (self.__glaciel_l, self.__glaciel_l)
        self.__glaciel_l = starting_length
        time_n = 0
        
        while (abs(self.__glaciel_l - prev_l) < .2):
            prev_l = self.__glaciel_l
            self.simulate(1, lambda t: equi_height, dt)
            time_n = time_n + 1
            if time_n *dt > 10e6:                      # Already a stable state
                self.__glaciel_l = old_l
                return (0., old_l)
         
        while (abs(self.__glaciel_l - prev_l) > .1):
            prev_l = self.__glaciel_l
            self.simulate(1, lambda t: equi_height, dt)
            time_n = time_n + 1
            if time_n *dt > 10e6:
                raise ValueError("Unstable State")
        
        prev_l = self.__glaciel_l
        self.__glaciel_l = old_l
        return (dt * time_n, prev_l)
        
    def calculate_efolding(self, starting_length, equi_height, dt=1.):
        starting_length = max(.1, starting_length)
        s_s = self.calculate_stable_state(starting_length, equi_height, dt=1.)
        
        old_l = self.__glaciel_l
        self.__glaciel_l = starting_length
        time_n = 0
        
        while (2. / 3. * (abs(self.__glaciel_l - s_s[1]) > .1)):
            self.simulate(1, lambda t: equi_height, dt)
            time_n = time_n + 1 
            
        prev_l = self.__glaciel_l
        self.__glaciel_l = old_l
        return (dt * time_n, prev_l)
    
    def generate_profile(self, lengths, plot=False, plot_title=None):
        bedp_h = np.array([self.bed_height(l) for l in lengths])
        mean_b = np.empty(len(lengths))
        mean_s = np.empty(len(lengths))
        mds_dt = np.empty(len(lengths))
        _old_gl = self.__glaciel_l
        _tmp_i = 0
        
        for l in lengths:
            self.__glaciel_l = l
            mean_b[_tmp_i] = self.mean_bed_height
            mean_s[_tmp_i] = self.mean_slope
            mds_dt[_tmp_i] = self.mean_ds_dl
            _tmp_i = _tmp_i + 1
        
        self.__glaciel_l = _old_gl
        
        if plot is True:
            f, ax1 = plt.subplots()
            leg1, = ax1.plot(lengths, bedp_h,'b',
                             label='Bed Profile')
            leg2, = ax1.plot(lengths, mean_b,'b--',
                             label='Mean Glacier Height')
            ax2 = ax1.twinx()
            leg3, = ax2.plot(lengths, mean_s,'r', 
                             label='Mean Glacier Slope')
            leg4, = ax2.plot(lengths, mds_dt,'r--',
                             label='Mean $\\mathrm{d}s/\\mathrm{d}L$')
            
            ax1.set_xlabel("Length (m)")
            ax1.set_ylabel("Height (m)")
            ax2.set_ylabel("Slope (straigt) and $\\mathrm{d}s/\\mathrm{d}L$ (m$^{-1}$)")
            plt.legend(handles=[leg1, leg2, leg3, leg4],
                       bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
            ax1.grid(True)
            if plot_title is not None and type(plot_title) is str:
                plt.title(plot_title)
            plt.show()
            
        return (bedp_h, mean_b, mean_s, mds_dt)
    
def esl_data(t):
    ti = np.argmin(np.abs((equi_vals[0] - equi_vals[0,0]) - t))
    return equi_vals[1,ti] 

def esl_step_func(t):
    steps = [500., 700., 400.]
    tmod = t % 1500.
    if tmod < 500.:
        return steps[0]
    elif 500. <= tmod <= 1000.:
        return steps[1]
    return steps[2]
    
if __name__ is "__main__":   
    glac = Glacier()
    lengths = np.linspace(1., 10e3, 100)
    equi_vals = np.genfromtxt(".\\ELA_Spitsbergen.txt").T
    
    # Test custom bed with ESL data
    # This doesn't yet seem to work
    c_geom= np.genfromtxt(".\\glacierelevationdata.txt", delimiter=',').T
    lengths = np.linspace(0.1, c_geom[0,-2], 1000)
#    glac.set_glacier_properties(calving_frac=.7, alpha=.5, mean_gletsjer_w=2930.)
    glac.create_custom_bed(c_geom[0], c_geom[1])
    
    buckets = np.genfromtxt(".\\Glacierbuckets.csv", delimiter=';', skip_header=1)[:,1:]
    for buc in buckets:
        glac.add_bucket(buc[0], buc[1], buc[2], buc[5], buc[6])  
    
    # Calibrate the Winter values for the current Glacier length
    glac.generate_profile(lengths, True, "Veteranen Glacier Bed")
    glac.set_length(10.)
    glac.simulate(len(equi_vals[0]), lambda t : 1130.,
                  calving_enabled=True,
                  plot=True,
                  plot_forcing=True,
                  plot_ice_development=True,
                  plot_title="ESL Data Spitsbergen on a Veteranen Glacier")
    
    #fig = plt.Figure()
    #plt.plot(glac.right_store)
    #plt.show()
    #fig = plt.Figure()
    #plt.plot(glac.left_store)
    #plt.show()