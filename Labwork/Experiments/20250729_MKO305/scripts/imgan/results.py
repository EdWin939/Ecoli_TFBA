# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 18:40:06 2021

@author: p290481
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from particle import Particle
import auxiliary as aux
import colicoords as ccoord

class Results():
    def __init__(self, treatment, particle, fps, pixelsize):
        self._treatment = treatment   
        self._particle = particle
        self._fps = fps
        self._pixelsize = pixelsize
        self._no_particles = 0
        
        self._msd = None
        self._dR0_squared = None
        self._vaf = {}
        
        self._particle_list = []
        self._particle_params = None
        self._particle_pairs = None
        
        self._bacteria_list = np.array([])
        self._cellsize_params = None
        self._mapdict = {}
        return
    

    @property
    def treatment(self):
        return self._treatment
    @property
    def particle(self):
        return self._particle
    @property
    def fps(self):
        return self._fps
    @property
    def pixelsize(self):
        return self._pixelsize
    @property
    def no_particles(self):
        self._no_particles = len(self.msd.columns)
        return self._no_particles      
    @property
    def condition(self):
        return "{}, {}, {} fps".format(self.particle, self.treatment, self.fps)
    
    @property
    def particle_params(self):
        return self._particle_params
    @property   
    def cellsize_params(self):
        return self._cellsize_params  
    @property   
    def particle_pairs(self):
        return self._particle_pairs  

    
    @property
    def msd(self):
        return self._msd 
    @property
    def dR0_squared(self):
        return self._dR0_squared 
    @property
    def timedR(self):
        return self._timedR 
    @property
    def vaf(self):
        return self._vaf 
    @property
    def particle_list(self):
        return self._particle_list
    @property
    def bacteria_list(self):
        return self._bacteria_list
    
    ### Add data: 
    def add_data_to_dataframe(self, datatype, inputdata, delta=None):
        """
        Stores time-dependent data in dataframe (rows: time; columns: particles).
        Types of data:
            -Time-averaged MSD (multiple particles)
            -Squared displacements with respect to t=0
            -Velocity autocorrelation function.
        """
        if datatype == "msd":
            data = self.msd  
        elif datatype == "dR0_squared":
            data = self.dR0_squared    
        elif datatype == "vaf":
            if delta is None:
                print("Specify <delta>!")
            else:
                try:
                    data = self.vaf[delta]
                except:
                    self._vaf[delta] = None
                    data = self.vaf[delta]
        else:
            print("Unknown datatype")
        
        if (inputdata is not None) and (not inputdata.empty):
            if data is None:
                data = inputdata
            else:
                data = pd.concat([data, inputdata], axis=1, sort=False)
                data.columns = range(len(data.columns))  
                
        if datatype == "msd":
            self._msd = data
        elif datatype == "dR0_squared":
            self._dR0_squared = data
        elif datatype == "vaf":
            self._vaf[delta] = data
            
        return    

    def add_particle_list(self, img, filtered):
        """ 
        Taking an instance of Image() as input, creates a list of instances of 
        Particle().
        Fills the latter with some relevant data (msd, trajectory, ID).
        """
        movie = img.info["name"]
        
        if filtered is True:
            tracks_in_movie = img.tracks_filtered
            msd_from_movie = img.msd_rows_filtered
        else:
            tracks_in_movie = img.tracks
            msd_from_movie = img.msd_rows
        
        if (tracks_in_movie is not None) and (not tracks_in_movie.empty):
            for group, track_g in tracks_in_movie.groupby(["cell", "particle"]):
                cell_id, pp_id = group
                
                msd_pp = msd_from_movie.loc[(msd_from_movie["particle"]==pp_id) 
                                            & (msd_from_movie["cell"]==cell_id)]
                
                if not msd_pp.empty:
                    pp = Particle(movie, cell_id, pp_id)
                    pp.msd = msd_pp
                    pp.track = track_g
                
                    self._particle_list.append(pp) 
        else:
            print("No particles in movie {}".format(movie))
        
        return

    def add_bacteria_list(self, img):
        """ 
        Taking an instance of Image() as input, creates a np.array of instances of 
        colicoords' Cell().
        Here a numpy array is used instead of a list, since Colicoords' methods 
        for CellList() work with arrays!
        """
        bb = self.bacteria_list
        
        if img.bacteria_list is not None:
            for cell in img.bacteria_list:
                bb = np.append(bb, cell)
                
        self._bacteria_list = bb
        return
   
    def add_cellsize_data(self, df, column_movie="image", column_cell="cell no."):
        if self._cellsize_params is not None:
            print("Already provided cell size values. Will be replaced.")
        
        for pp in self.particle_list:
            movID = pp.id[0]
            cellID = pp.id[1]
            
            idx = [x in movID for x in df[column_movie]]
            df_movie = df[idx]
            df_cell = df_movie.loc[(df_movie[column_cell] == cellID)] 
            df_cell = df_cell.iloc[0]
        
            try:
                pp._cell_area = df_cell["area"]
            except:
                pp._cell_area = np.nan 
                
            try:
                pp._cell_area_mj = df_cell["area_mj"]
            except:
                pp._cell_area_mj = np.nan  
                
            try:
                pp._cell_length_mj = df_cell["length_mj"]
            except:
                pp._cell_length_mj = np.nan
                
            try:
                pp._cell_width_mj = df_cell["width_mj"]
            except:
                pp._cell_width_mj = np.nan  

            try:
                pp._cell_type_mj = df_cell["type_mj"]
            except:
                pp._cell_type_mj = np.nan  
        
        self._cellsize_params = df
        return
            
    def add_contour_data(self, img):     
        for pp in self.particle_list:
            movID = pp.id[0]
            cellID = pp.id[1]
            
            if (img.info["name"] in movID) & (cellID-1 in img.contour.keys()):
                contx, conty = img.contour[cellID - 1]
                x = contx / self.pixelsize
                y = conty / self.pixelsize
                pp.cell_contour = (x, y)
        return
                
            
    
    ### Other methods:   
    def calc_diff_allparticles(self, no_points_fit, fit_type):
        """ 
        Fits MSD=4Dt^a to all the (individual) particles in self.particle_list.
        Saves values of D and a in dataframe self.particle_params.
        """
        for pp in self.particle_list:
            pp.calc_diff_params(no_points_fit, fit_type)
            self.update_single_cell_parameters(pp)
        return
        
    def calc_rg_allparticles(self, um_per_pix=None):
        """ 
        Determines R_g of all the particles in self.particle_list.
        Saves values in dataframe self.particle_params.
        """
        um_per_pix = self.pixelsize if um_per_pix is None else um_per_pix
        
        for pp in self.particle_list:
            pp.calc_rg(um_per_pix)
            self.update_single_cell_parameters(pp)
        return
    
    def calc_dispcorr_allparticles(self, dt, um_per_pix=None, shift_disp=1):
        """
        Calculates the projection of consecutive displacements for all particles.
        """
        um_per_pix = self.pixelsize if um_per_pix is None else um_per_pix
        
        for pp in self.particle_list:
            pp.calc_proj_displacement(dt, um_per_pix, shift_disp)
        return
    
    def calc_dR0squared_allparticles(self, um_per_pix=None, fps=None): 
        """ 
        For the ensemble MSD. Not determined by trackpy.
        """
        um_per_pix = self.pixelsize if um_per_pix is None else um_per_pix
        fps = self.fps if fps is None else fps
        
        for pp in self.particle_list:
            pp.calc_dR0_squared(um_per_pix, fps)
            self.add_data_to_dataframe("dR0_squared", pp.dR0_squared)
        return
    
    def calc_vaf_allparticles(self, delta, um_per_pix=None, fps=None): 
        """ 
        Velocity autocorrelation function. 
        """
        um_per_pix = self.pixelsize if um_per_pix is None else um_per_pix
        fps = self.fps if fps is None else fps
        
        for pp in self._particle_list:
            pp.calc_velocity_autocorrelation(delta, um_per_pix, fps)
            self.add_data_to_dataframe("vaf", pp.vaf, delta)
        return
    
    def calc_timedR_allparticles(self, um_per_pix=None):
        """ 
        Displacements for all time lags. 
        """
        um_per_pix = self.pixelsize if um_per_pix is None else um_per_pix
        
        df = pd.DataFrame({})
        for pp in self.particle_list:
            pp.calc_timedR(um_per_pix)
            df_pp = pp.timedR
            df_pp["particle"] = "{}_{}_{}".format(*pp.id)
            df = pd.concat([df, df_pp])
        self._timedR = df
        return 

    def calc_mobility_param_allparticles(self):
        """ 
        Calculates the parameter that allows to filter out immobile particles.
        """
        for pp in self.particle_list:
            pp.calc_mobility_param()
            self.update_single_cell_parameters(pp)
        return

    def calc_location_allparticles(self, threshold=0.9):
        """
        Obtains the most common location of each particle in the cell. 
        (cell region where the particle spends more than <threshold> fraction of its trajectory).
        This is done individually for all particles.
        """
        for pp in self.particle_list:
            pp.classify_major_locations_track(threshold=threshold)
            self.update_single_cell_parameters(pp)
        return

    def find_bacteria_allparticles_legacy(self, tol=0):
        """
        Assign an instance of ColiCoords.Cell() to each particle, if such cell exists.

        Old version - do not use, as it is very time-consuming.
        """
        bacteria_to_consider = self.get_bacteria_to_consider()

        for pp in self.particle_list:
            pp.find_bacteria(bacteria_to_consider.keys(), tol)
            bacteria_to_consider = self.update_bacteria_to_consider(bacteria_to_consider, pp)
        return

    def find_bacteria_allparticles(self, tol=0):
        bacteria_to_consider = self.get_bacteria_to_consider()
        mapdict = self.get_mapping_dict(bacteria_to_consider)
        for pp in self.particle_list:
            pp.fetch_right_cell(mapdict, tol)
        return

    # def _find_location_in_cell(lc, bb):
    #     lq1 = bb.length / 4
    #     lq3 = 3 * lq1
    #     if (lc <= 0) | (lc >= 0):
    #         return "pole"
    #     elif ((lc <= lq1) & (lc > 0)) | ((lc >= lq3) & (lc < bb.length)):
    #         return "between"
    #     elif (lc >= lq1) & (lc <= lq3):
    #         return "middle"



    def find_particle_pairs(self):
        """ 
        Finds particles that were observed in the same cell.
        """ 
        baseID = ["{}_{}".format(pp.id[0], pp.id[1]) for pp in self.particle_list]

        duplicates = aux.duplicates(baseID)

        self._particle_pairs = duplicates
        
        return

    # def filter_immobile(self, threshold):
    #     for pp in self.particle_list:
    #         x = pp.mobility_param
            
    #         if x is None:
    #             print("Mobility parameter not yet calculated!")
    #         elif x > threshold:
                
    #         else:
                 
    #     return
        
    
    def display_tracks_in_cell_allparticles(self, disptype=None, **kwargs):
        for pp in self.particle_list:
            pp.display_track_in_cell(disptype=disptype, **kwargs)
        return


    def display_msd_allparticles(self, alpha=0.1, ax=None, color="black", label="", legend=False, **kwargs):  
        """ 
        Displays the time-averaged MSD of all the particles.
        Partly adapted from "Image().display_msd()"
        """
        data = self.msd   
        if ax is None:
            fig, ax = plt.subplots()
        data.plot(alpha=alpha, ax=ax, color=color, label=label, legend=legend, **kwargs)
        return        
    
    # TODO: confirm that the coordinates are indeed in pixels
    def get_coord_alltracks(self, coord="x"):
        """ 
        Returns a dataframe where:
            -row: frame number
            -column: values of the <coord> for different particles.
        Values of the <coord> are in [px].
        """
        all_tracks = pd.DataFrame({})
        i = 0   # workaround to avoid getting duplicate indices (because of particles with same number)
         
        for pp in self.particle_list:
            pp_track = pp.track.copy()
            pp_track["particle"] = i
            all_tracks = pd.concat([all_tracks, pp_track])
            i += 1
        
        all_tracks = all_tracks.set_index(["frame", "particle"])[coord].unstack()
        all_tracks.columns = range(i)            #range(len(self._msd.columns))
            
        return all_tracks
           
    def get_ensemble_values_alltracks(self, data_type, points=None, delta=None):
        """ 
        Returns the mean and std - determined by averaging over all particles:
            - Time-averaged MSDs (for the Time-and-ensemble MSD)
            - Squared displacements from the origin (for the Ensemble MSD).
            - Velocity autocorrelation function. Requires that <delta> is specified.
        """
    
        if data_type=="time-ensemble":
            data = self.msd
        elif data_type=="ensemble":
            data = self.dR0_squared.iloc[1::]  # to skip the point (0, 0)
        elif data_type=="vaf":
            if delta is None:
                print("Specify <delta>!")
            else:
                data = self.vaf[delta]                
        else:
            print("Unknown argument.")
    
        points = len(data.index) if points is None else points

        t = data.index.values[0:points]
        mean = data.mean(axis=1).values[0:points]
        std = data.std(axis=1).values[0:points]

        return t, mean, std
    
    
    def get_idx_bacteria_with_particles(self):
        """
        Returns the indices of all "bacteria" (instances of ColiCoords.Cell()) that contain particles.
        """ 
        if self.bacteria_list is not None:
            return [i for (i, x) in enumerate(self.bacteria_list) if x.data.data_dict["storm"].size != 0]
        else:
            print("Instance does not contain any list of Bacteria!")
            return []
      
    def get_bacteria_to_consider(self):
        """
        Obtains a dictionary where the keys are instances of ColiCoords.Cell() and the 
        values are a list of tuples (cellID, particleID), of the particles associated with that cell.
        """
        idx_with_particles = self.get_idx_bacteria_with_particles()

        bacteria_to_consider_list = self.bacteria_list[idx_with_particles]

        bacteria_to_consider = {bb: [(g[0], g[1]) for g, _ in pd.DataFrame(bb.data.storm_storm).groupby(["cell", "particle"])] for bb in bacteria_to_consider_list}
        
        return bacteria_to_consider

    def get_mapping_dict(self, bacteria_list):
        """
        Returns a dictionary where keys are the unique particle IDs (tuple), and the values are lists of potential 
        candidates of ColiCoords.Cell() objects that contain the particle under consideration.
        """
        mapdict = {}
        for bb in bacteria_list:
            df = pd.DataFrame(bb.data.storm_storm)
            unique_ids = df[["movieID", "cell", "particle"]].drop_duplicates().values
            unique_ids_tup = [tuple(x) for x in unique_ids]
            aux.update_dict(mapdict, unique_ids_tup, bb)
        self._mapdict = mapdict
        return mapdict

    def get_particles_in_movie(self, movieID, cellID=None):
        """
        Returns a list of all the particles in a given movie.
        If specified, returns only the particles in a given cell, or list of cells.
        """
        particles_out = []

        for pp in self.particle_list:
            pp_mov, pp_cell, pp_j = pp.id

            if pp_cell is not None:
                if (pp_mov == movieID) & (pp_cell in cellID):
                    particles_out.append(pp)
            else:
                if (pp_mov == movieID):
                    particles_out.append(pp)

        return particles_out

    def update_bacteria_to_consider(self, bacteria_dict, pp):
        """
        Updates the dictionary of "bacteria to consider", containing the instances of ColiCoords.Cell() to be assigned to Particle().

        """
        cell = pp.cell_colicoords

        pp_mov, pp_cell, pp_j = pp.id
        tup_ids = (pp_cell, pp_j)

        if (cell is not None) & (cell in bacteria_dict.keys()):
            bacteria_dict[cell].remove(tup_ids)

            if not bacteria_dict[cell]:
                del bacteria_dict[cell]

        return bacteria_dict

    def update_single_cell_parameters(self, pp):
        """ 
        Saves the values of the parameters determined for each particle (mass, 
        Rg, D, alpha) in a dataframe.
        If values already exist, they will be replaced.
        pp must be an instance of Particle().
        """
        if self.particle_params is not None:
            df = self._particle_params
        else:
            df = pd.DataFrame(columns=["ID", "movieID", "cellID", "particleID", "D", "alpha", "r2", "Rg", "mass", "size", "ecc", "ep", 
                                       "mobparam","cell_area", "cell_area_mj", "cell_length_mj", "cell_width_mj", "cell_type_mj", "major_location"])

        
        pp_full_id = "{}_{}_{}".format(pp.id[0], pp.id[1], pp.id[2])  
        
        df_particle = pd.DataFrame({"ID": pp_full_id,
                                    "movieID": pp.id[0],
                                    "cellID": pp.id[1], 
                                    "particleID": pp.id[2],
                                    "D": pp.diffusion_params[0],
                                    "alpha": pp.diffusion_params[1],
                                    "r2": pp.r2,
                                    "Rg": pp.radius_gyration,
                                    "mass": pp.mass,
                                    "size": pp.size,
                                    "ecc": pp.ecc,
                                    "ep": pp.ep,
                                    "mobparam": pp.mobility_param,
                                    "cell_area": pp.cell_dimensions[0],
                                    "cell_area_mj": pp.cell_dimensions[1],
                                    "cell_length_mj": pp.cell_dimensions[2],
                                    "cell_width_mj": pp.cell_dimensions[3],
                                    "cell_type_mj": pp.cell_type,
                                    "major_location": pp.location,
                                    }, 
                                    index=[0])

        if (df.loc[(df["ID"] == pp_full_id), :].empty):
            df = pd.concat([df, df_particle], axis=0, ignore_index=True)
        else:    
            df.loc[(df["ID"] == pp_full_id), :] = df_particle[:].values
        
        self._particle_params = df
        return