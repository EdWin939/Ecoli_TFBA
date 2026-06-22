# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 08:04:13 2021

@author: p290481
"""
import stochastic.processes.continuous as st
import numpy as np
import pandas as pd

class SimulatedTrajectories:
    def __init__(self, no_particles, length, motion_type):
        self._no_particles  = no_particles
        self._length = length
        self._motion_type = motion_type
        self._trajectories = []
        return
    
    @property
    def motion(self):
        return self._motion_type
    
    @property
    def length(self):
        return self._length
    
    @property
    def no_particles(self):
        return self._no_particles
    
    @property
    def trajectories(self):
        return self._trajectories
    
    
    def create_traj(self, **kwargs):
        
        motion_type = self.motion.lower()
        
        if motion_type == "brownian":
            motion = st.BrownianMotion(**kwargs)
        elif motion_type == "fbm":
            motion = st.FractionalBrownianMotion(**kwargs)
        else:
            print("Choose a valid process!")
            
        for i in range(self.no_particles):
            df_p = pd.DataFrame(columns={"x", "y"}, index=range(0, self.length))
            x = motion.sample(self.length - 1)
            y = motion.sample(self.length - 1)
            df_p.x = x
            df_p.y = y
            
            self._trajectories.append(df_p) 
        return


    def add_noise(self, scale):
        if len(self.trajectories) == 0:
            print("No trajectories to be found!")
            return
        
        for pp in self.trajectories:
            pp.x = pp.x + np.random.normal(scale=scale, size=(self.length,))
            pp.y = pp.y + np.random.normal(scale=scale, size=(self.length,))
        
        return

class SimulatedMovie:
    def __init__(self):
        return
    
    
# # From trackpy's tutorial:

# class SimulatedFrame(object):
    
#     def __init__(self, shape, dtype=np.uint8):
#         self.image = np.zeros(shape, dtype=dtype)
#         self._saturation = np.iinfo(dtype).max
#         self.shape = shape
#         self.dtype =dtype
        
#     def add_spot(self, pos, amplitude, r, ecc=0):
#         "Add a Gaussian spot to the frame."
#         x, y = np.meshgrid(*np.array(list(map(np.arange, self.shape))) - np.asarray(pos))
#         spot = amplitude*np.exp(-((x/(1 - ecc))**2 + (y*(1 - ecc))**2)/(2*r**2)).T
#         self.image += np.clip(spot, 0, self._saturation).astype(self.dtype)
        
#     def with_noise(self, noise_level, seed=0):
#         "Return a copy with noise."
#         rs = np.random.RandomState(seed)
#         noise = rs.randint(-noise_level, noise_level, self.shape)
#         noisy_image = np.clip(self.image + noise, 0, self._saturation).astype(self.dtype)
#         return noisy_image
    
#     def add_noise(self, noise_level, seed=0):
#         "Modify in place with noise."
#         self.image = self.with_noise(noise_level, seed=seed)

        
        
# # My implementation (movie instead of single frame):
# class SimulatedMovie():
#     def __init__(self, snr, length, shape):
#         self._snr = snr
#         self._length = length
#         self._shape = shape

#     def generate_movie(self, amplitude, r, drift, scale, t):
#         movie = np.array([])
        
#         bm = BrownianMotion(drift, scale, t)
#         x = bm.sample(self._length)
#         y = bm.sample(self._length)
        
#         x += self._shape[0]/2
#         y += self._shape[1]/2
        
#         for i in range(self._length):
#             pos = (x[i], y[i])
            
#             frame = SimulatedFrame(self._shape)
# #             print(pos)
#             frame.add_spot(pos, amplitude, r)
#             frame.add_noise(amplitude / self._snr)
#             movie = np.append(movie, frame)
#         return movie
            