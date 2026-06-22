# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 18:40:35 2021

@author: p290481
"""

import numpy as  np
import pandas as pd
from scipy import optimize, stats
import auxiliary as aux
import colicoords as ccoord
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, movie, cell, particle):
        # IDs
        self._movieID = movie
        self._cellID = cell
        self._particleID = particle
        
        # Data of the corresponding cell:
        self._cell_area = None
        self._cell_area_mj = None
        self._cell_width_mj = None
        self._cell_length_mj = None
        self._cell_type_mj = None
        self._cell_contour = None
        self._cell_colicoords = None
        
        # Data to store
        self._track = None
        self._msd = None
        self._dR0_squared = None
        self._vaf = None
        self._timedR = None
        
        # Calculated parameters
        self._diff = None
        self._alpha = None
        self._no_points_fit = None
        self._fit_stats = None
        self._linear_fit_intercept = None
        self._Rg = None
        self._mass = None
        self._size = None
        self._ecc = None
        self._ep = None
        self._disp = None
        self._mobility_param = None
        self._location = None
        self._location_threshold = None
       
    
    @property    
    def id(self):
        return self._movieID, self._cellID, self._particleID
            
    @property    
    def mass(self):
        self._mass = self._track.iloc[0:5]["mass"].median() 
        return self._mass

    @property    
    def size(self):
        self._size = self._track.iloc[0:5]["size"].median() 
        return self._size

    @property    
    def ecc(self):
        self._ecc = self._track.iloc[0:5]["ecc"].median() 
        return self._ecc

    @property    
    def ep(self):
        self._ep = self._track.iloc[0:5]["ep"].median() 
        return self._ep
    
    ## Input data
    @property
    def track(self):
        return self._track
    
    @track.setter
    def track(self, track):
        self._track = track
        return
    
    @property
    def msd(self):
        return self._msd
    
    @msd.setter
    def msd(self, msd):
        self._msd = msd
        return
    
    
    @property
    def cell_dimensions(self):
        return self._cell_area, self._cell_area_mj, self._cell_length_mj, self._cell_width_mj
    
    @property
    def cell_type(self):
        return self._cell_type_mj


    @property
    def cell_contour(self):
        return self._cell_contour
    
    @cell_contour.setter
    def cell_contour(self, data):
        self._cell_contour = data
        return

    @property
    def cell_colicoords(self):
        return self._cell_colicoords
    
    ## Calculated parameters 
    @property
    def diffusion_params(self):
        return self._diff, self._alpha

    @property
    def r2(self):
        try:
            r, p, stderr = self._fit_stats
            r2 = r**2
        except ValueError:
            r2 = None
        except TypeError:
            r2 = None
        return r2
    
    @property    
    def radius_gyration(self):            
        return self._Rg    
    
    @property
    def displacement_corr(self):
        return self._disp   
    
    @property
    def dR0_squared(self):
        return self._dR0_squared

    @property
    def timedR(self):
        return self._timedR
    
    @property
    def mobility_param(self):
        return self._mobility_param

    @property
    def vaf(self):
        return self._vaf

    @property
    def location(self):
        return self._location
    
    
    def display_track_in_cell(self, disptype=None, axes=None, **kwargs):
        """
        Displays the cell contour and the particle trajectory, relying on the contour and SPT data in self._cell_colicoords.
        """

        if disptype is None:
            print("Specify the type of display: 'colicoords' or 'original'.")


        if disptype=="colicoords":
            if self.cell_colicoords is not None:

                if axes is None:
                    fig, axes = plt.subplots(figsize=(10, 10), sharey=True, sharex=True)

                df = pd.DataFrame(self._cell_colicoords.data.storm_storm)
                df_filt = df.loc[df.particle==self.id[2]]    # filter only the desired particle

                cp = ccoord.CellPlot(self.cell_colicoords)
                cp.imshow("bf", ax=axes)
                cp.plot_outline(ax=axes)
                cp.plot_midline(ax=axes)                
                axes.plot(df_filt["x"], df_filt["y"], **kwargs)
                axes.set_title(self.id)

            else:
                print("Nothing to show.")

        elif disptype=="original":
             if self.cell_contour is not None:

                if axes is None:
                    fig, axes = plt.subplots(figsize=(10, 10), sharey=True, sharex=True)
                    
                cell_x, cell_y = self.cell_contour
                pp_x, pp_y = self.track.x, self.track.y

                axes.plot(cell_x, cell_y, color="green")
                axes.plot(pp_x, pp_y, **kwargs)
                axes.set_title(self.id)          

        else:
            print("Unknown <disptype>. Must be: 'colicoords' or 'original'.") 

        return axes


    def find_bacteria(self, bacteria_list, tol=0):
        """             
        Finds the "bacteria" (instance of ColiCoords.Cell()) in which the particle can be found.
        Only does so if the particle localizations fall inside the boundaries of the cell.

        <bacteria_list>: list of instances of ColiCoords.Cell()
        <tol>: tolerance for considering particles that lie a bit outside the cell contour.
        """
        pp_mov, pp_cell, pp_j = self.id

        for bb in bacteria_list:
            spt_data = pd.DataFrame(bb.data.storm_storm)
            bb_mov = spt_data["movieID"].unique()
            bb_cells = spt_data["cell"].unique()
            bb_particles = spt_data["particle"].unique()

            if (not spt_data.empty) & (pp_mov == bb_mov) & (pp_cell in bb_cells) & (pp_j in bb_particles) & (bb.coords.a2 != 0):
                if self._check_particle_within_contour(bb, tol):
                    self._cell_colicoords = bb
                    return
        return

    def fetch_right_cell(self, mapdict, tol):
        """
        Of the candidate instances of ColiCoords.Cell() selects only the one (if any)
        in which the particle is found within the cell contour.
        """
        try:
            candidates = mapdict[self.id]

            for bb in candidates:
                if self._check_particle_within_contour(bb, tol):
                    self._cell_colicoords = bb
                    return bb
            return None
        except KeyError:
            return None

    def _check_particle_within_contour(self, bb, tol):
        """ 
        Checks if the particle data (stored in an instance of ColiCoords.Cell())
        is within the boundaries defined by the cell contour.

        Needed when the data of a single particle is assigned to multiple instances
        of ColiCoords.Cell().

        <bb>: instance of ColiCoords.Cell().
        <tol>: tolerance for the location of the particle in the cell (if >0, particles that lie a bit outside the cell contour are still considered).

        """

        pp_mov, pp_cell, pp_j = self.id

        df = pd.DataFrame(bb.data.storm_storm)
        pp_x = df.loc[df["particle"]==pp_j, "x"]
        pp_y = df.loc[df["particle"]==pp_j, "y"]

        dist_midline = bb.coords.calc_rc(pp_x, pp_y)
        cell_radius = bb.radius

        return all(dist_midline < cell_radius * (1+tol))

    def classify_all_locations_track(self):
        """
        Determines the location inside the cell ("pole", "middle", "between") of ALL positions a particle
        has been at in its trajectory. 

        Returns a dataframe with the particle trajectory and location in the cell. Coordinates are NOT the absolute
        x and y values in the original movie, but rather the coordinates after the transformation effected by ColiCoords.

        Adapted from colicoords.cell.l_classify():
            https://github.com/Jhsmit/ColiCoords/blob/6c59e5cc827955321dbc705373ac534589783907/colicoords/cell.py#L343
        """

        if self.cell_colicoords is not None:
            bb =  self.cell_colicoords
            spt_data = bb.data.storm_storm
        else:
            print("Particle {} does not have a corresponding instance of ColiCoords.Cell().".format(self.id))
            spt_data = {}

        df = pd.DataFrame(spt_data)

        if len(df) > 0:
            x, y = df["x"], df["y"]
            lc = bb.coords.calc_lc(x, y)
            lq1 = bb.length / 4
            lq3 = 3 * lq1
            
            df["lc"] = lc
            df.loc[(df.lc <= 0) | (df.lc >= 0), "localization"] = "pole"
            df.loc[((df.lc <= lq1) & (df.lc > 0)) | ((df.lc >= lq3) & (df.lc < bb.length)), "localization"] = "between"
            df.loc[(df.lc >= lq1) & (df.lc <= lq3), "localization"] = "middle"
        
        return df

    def classify_major_locations_track(self, threshold=0.9):
        """
        Assigns the major location of the particle within the cell ("pole", "middle", "between"),
        as long as the fraction of all particle localizations in that region is higher than <threshold>.

        If:
        - the particle spends less than <threshold> in any one cell region,
        or
        - the particle has not been assigned an instance of ColiCoords.Cell(),
        the location is said to be "undefined".
        """

        bb = self.cell_colicoords

        df = self.classify_all_locations_track()
        
        try:
            locs = df["localization"]
            locs_rel = locs.value_counts()/locs.count()
            major, = locs_rel.loc[locs_rel>threshold].index.values
        except:
            major = "undefined"

        self._location = major
        self._location_threshold = threshold

        return

    def calc_diff_params(self, no_points_fit, fit_type=None, p0=None): 
        """ 
        Fits MSD=4Dt^a to the time-averaged MSD of each particle (obtained from trackpy).
        Fit can be done to an arbitrary number of points of the MSD curve.
        Stores the optimal parameter values and corresponding covariance matrix.
        """
        msd = self.msd["MSD"].values[0:no_points_fit]              
        tau = self.msd["lag time [s]"].values[0:no_points_fit]     
        
        p0 = [msd[0], 1] if p0 is None else p0
        
        if fit_type == "linear":            # MSD = 4D.t + 2sigma^2
            slope, intercept, r, p, stderr = stats.linregress(tau, msd)
            d = slope/4
            alpha = None
            fit_intercept = intercept
            fit_stats = (r, p, stderr)
            
        elif fit_type == "power-log":       # log(MSD) = alpha*log(tau) + log(4D)
            slope, intercept, r, p, stderr = stats.linregress(np.log(tau), np.log(msd))
            d = np.exp(intercept)/4
            alpha = slope
            fit_intercept = None
            fit_stats = (r, p, stderr)
            
        elif fit_type == "power-linear":    # MSD = 4Dt^alpha
            popt, pcov = optimize.curve_fit(aux.model_msd, tau, msd, p0=p0)
            d = popt[0]
            alpha = popt[1]
            fit_intercept = None
            fit_stats = pcov
        
        self._diff = d
        self._alpha = alpha
        self._no_points_fit = no_points_fit
        self._fit_stats = fit_stats
        self._linear_fit_intercept = fit_intercept
        return

    def calc_rg(self, um_per_pix):
        """ 
        Calculates the radius of gyration of the particle's trajectory.
        The value returned is in um (and not pixels).
        """
        
        x = self.track.x
        y = self.track.y
        
        x_diff_sq = (x - x.mean())**2
        y_diff_sq = (y - y.mean())**2
        r_diff_sq = x_diff_sq + y_diff_sq
        mean_rdiff_sq = r_diff_sq.mean()
        rg = um_per_pix * np.sqrt(mean_rdiff_sq)
        
        self._Rg = rg
        return
    
    def calc_proj_displacement(self, dt, um_per_pix, shift_disp=1):
        """ 
        <dt>: to calculate the spatial displacement, dR_{t} = R(t+dt) - R(t)
        <shift_disp>: delay between the displacements to be used in the correlation. 
                     delta_dR = dR_{t+shiftdisp} - dR_{t}
                     (default: 1 (i.e. consecutive dR))
        
        References:
           Weeks & Weitz (2002) Chemical Physics, 284, 361–367. 
           Munder et al. (2016) eLife, 5, e09347.
            
        """
        df = pd.DataFrame({})
        
        traj = self.track
        deltar = um_per_pix * traj.sub(traj.shift(dt))
        deltar["dr"] = np.sqrt(deltar.x**2 + deltar.y**2) ######### turn previous lines into aux function (?)
        deltar["direction"] = np.arctan2(deltar.y, deltar.x)
        
        delta_deltar = deltar.sub(deltar.shift(shift_disp))
    
        cosines = np.cos(delta_deltar.direction)
        
        projection = deltar.dr * cosines
        
        df["dr_01"] = deltar["dr"].values
        df["Theta_01"] = deltar["direction"].values
        df["dTheta"] = delta_deltar["direction"].values
        df["cos(dThetha)"] = cosines.values
        df["proj_12_over_01"] = projection.shift(-shift_disp).values  
        
        self._disp = df
        return
    
    ##TODO confirm that final values are in um^2
    def calc_dR0_squared(self, um_per_pix, fps):    
        """ 
        Calculate the squared displacement from the intial point of the trajectory:
            dR0(t) = [R(t) - R(0)]^2
        """
        
        traj = self.track

        deltaR0 = traj.sub(traj.iloc[0])        # displacement from first point
    
        deltaR0.x = deltaR0.x * um_per_pix
        deltaR0.y = deltaR0.y * um_per_pix
        deltaR0["dt"] = deltaR0.frame.values/float(fps)   # convert to time
        deltaR0.set_index("dt", inplace=True) 
        
        deltaR0_squared = deltaR0.x**2 + deltaR0.y**2
        
        self._dR0_squared = deltaR0_squared 
        return  

    def calc_timedR(self, um_per_pix):
        """ 
        Calculates the particle displacements for all possible lagtimes and 
        respective 2nd and 4th powers.
        
        Results should be able to recover the MSD values obtained by trackpy:
            df_pp.groupby("dt").dr2.mean()
            
        This implementation is quite slow.
        """
        traj = self.track.copy()
        lagtimes = np.arange(1, len(traj))

        df_pp = pd.DataFrame({})
        for dt in lagtimes:
            deltar_d = um_per_pix * traj.sub(traj.shift(dt))
            deltar_d["dx"] = deltar_d.x
            deltar_d["dy"] = deltar_d.y
            deltar_d["dr"] = np.sqrt(deltar_d.x**2 + deltar_d.y**2)
            deltar_d["dr2"] = deltar_d.dr**2
            deltar_d["dr4"] = deltar_d.dr**4 
            deltar_d["dt"] = dt

            df_pp = pd.concat([df_pp, deltar_d[["dt", "dx", "dy", "dr", "dr2", "dr4"]]])
        self._timedR = df_pp
        return
    

    ##TODO: error messages if attributes are None
    def calc_mobility_param(self):
        """ 
        Reference: 
            Golan & Sherman (2017) Nature Communications, 8(1), 1–15.
        """
        
        if (self.radius_gyration is not None) and (self.timedR is not None):
            rg = self.radius_gyration
            steps = self.timedR.loc[self.timedR["dt"]==1, "dr"]
        else:
            raise RuntimeError("Calculate 'Rg' AND 'timedR' first!")
        
        stepsize = np.nanmean(np.abs(steps))
        mobparam = np.sqrt(np.pi / 2) * (rg/stepsize)
        
        self._mobility_param = mobparam
        
        return
        
    
    ## TODO: confirm that C_delta is correctly determined! (mean or sum?)
    def calc_velocity_autocorrelation(self, delta, um_per_pix, fps):
        """ 
        Calculates the velocity autocorrelation function:
            Cv_delta(dt) = <v_delta(t+dt) . v_delta(t)>, 
            
            where v_delta(t) = [R(t+delta) - R(t)]/ delta
        
        Delta is fixed to determine the velocity vectors.
        The dot-product is determined for all the possible lagtimes, dt. 
        In each case, the value of C is taken as the mean dot-product.
        
        Implementation of the dot-product in terms of vector norms:
            a . b = |a|*|b|*cos(a^b)        
        
        References:
            Weber et al. (2010) Physical Review Letters, 104(23), 238102.
            Bohrer & Xiao (2020) in Physical Microbiology (Vol. 1267, pp. 15–43)
            Weber et al. (2012)
            Golan et al. (2017)
            Lampo et al. (2017) - uses <v(dt).v(0)> !

        """
        track = self.track.copy()
        
        deltar_d = um_per_pix * track.sub(track.shift(delta))
        deltar_d["dr"] = np.sqrt(deltar_d.x**2 + deltar_d.y**2)
        deltar_d["direction"] = np.arctan2(deltar_d.y, deltar_d.x)

        vel_d = deltar_d.dr / delta
        theta = deltar_d.direction

        dt_arr = np.array([]) 
        C_arr = np.array([]) 
        n_arr = np.array([]) 
        df = pd.DataFrame({})

        max_lagtime = len(vel_d) - delta
        lagtimes = np.arange(0, max_lagtime)

        for dt in lagtimes:
            if dt == 0:
                v_t = vel_d.values[delta::]
                C_delta = np.nanmean(v_t**2)   #np.nanmean?!?!
                
                dt_arr = np.append(dt_arr, dt)
                C_arr = np.append(C_arr, C_delta)                
            
            
            else:
                v_t = vel_d.values[delta:-dt]
                theta_t = theta.values[delta:-dt]
    
                v_tdt = vel_d.shift(-dt).values[delta:-dt]
                theta_tdt = theta.shift(-dt).values[delta:-dt]
    
                dtheta = theta_tdt - theta_t
    
                C_delta = np.nanmean(v_t * v_tdt * np.cos(dtheta))  #np.nanmean?!?!
    
                dt_arr = np.append(dt_arr, dt)
                C_arr = np.append(C_arr, C_delta)
                n_arr = np.append(n_arr, len(v_t))

        df["dt"] = dt_arr / float(fps)
        df.set_index("dt", inplace=True) 
        df["C"] = C_arr
#         df["delta"] = delta
#         df["no_points"] = n_arr    
        self._vaf = df
        
        return