# -*- coding: utf-8 -*-
"""
V2: using @property decorators

Created on Sat Nov 14 13:42:14 2020

@author: p290481
"""
import pims
import skimage.io
import skimage.measure
import skimage.morphology
import numpy as np
import trackpy as tp
import colicoords as ccoord
from pystackreg import StackReg
from scipy import ndimage
from matplotlib import animation
import matplotlib.pyplot as plt
import pandas as pd
#
from auxiliary import normalize_intensity, gaussian, fitgaussian, lowercase_dict, draw_contour_cell, convert_df_to_array


class Image():
    def __init__(self):
        self._stack_raw = None
        self._microbej = None
        
        self._info = {"name": [],
                     "channels": {},
                     "um_pixel": [],
                     "shape": [],
                     "no_channels": [],
                     "no_timepoints": [],
                     "fps": []}   
        self.background = None
        self.background_smoothed = None
        self.stack_aligned = None
        self.stack_evenillum = None
        self.stack_bandpass = None
        self.contour = None
        self.mask = None
        self.mask_cells = None
        self.cell_ids = None
        self.tmatrix = None
        self.bacteria_list = None
        self.results = None
        self.msd_cols = None
        self.msd_rows = None
        self.msd_cols_filtered = None
        self.msd_rows_filtered = None
        self.tracks = None
        self.tracks_filtered = None
        self.params_tracking = None
        self.localizations = None
        self.drift = None
 
        return
    
    
    def add_data(self, data, datatype, order=None):
        """
        Stores the input data in the appropriate attributes of Image():
            ._info, ._stack_raw, or ._microbej.
        
        Parameters
        ----------
        data : numpy array or dict
            Data to be stored in the instance of Image(). May be a stack of images,
            a dict of microbeJ results, or a dict of additional information. 
        datatype : str
            Type of input data to be stored.
            Admissible choices are "tif", "info" and "microbej".
        order : str or list
            Must be given whenever inputting a stack of images.
            Specifies what each dimension of the stack is ("x", "y", "t" or "c").
        
        Returns
        -------
        None
        
        Example
        ------- 
            tifstack = skimage.io.imread("... .tif")
            mjres = scipy.io.loadmat("... .mat", simplify_cells=True)
            infodict = {"name": "imagenameX", "fps": 10}
            -
            img = Image()
            img.add_data(tifstack, "tif", order="xyct")
            img.add_data(mjres, "microbej")
            img.add_data(infodict, "info")   
        """
        
        if datatype.lower() == "tif":
            self.stack_raw = (data, order)
            
        elif datatype.lower() == "microbej":
            self.microbej = data
            
        elif datatype.lower() == "info":
            self.info = data  
            
        else:
            raise ValueError("Incorrect data type. Please select: 'tif', 'microbeJ' or 'info'.") 
                   
        return

    @property
    def info(self):
        return self._info
    
    @info.setter
    def info(self, data_in):
        """
        Stores a dict of additional information in Image._info. Alternative to
        add_data(). Warns user if information is being overwritten.
        
        Parameters
        ----------
        data_in : dict
        Dict of information to store.
        
        Returns
        -------
        None
        
        Example
        ------- 
            infodict = {"name": "imagenameX", "fps": 10}
            -
            img = Image()
            img.info = infodict
        
        """ 
        if isinstance(data_in, dict):
            data_lc = lowercase_dict(data_in)
            keys_updated = self._info_to_update(data_lc)  
            if keys_updated:
                print("The following entries already contained information and will be updated:") 
                print(keys_updated)  
            self._info.update(data_lc) 
        else:
            raise TypeError("'info' input must be a dictionary!")        


    @property
    def stack_raw(self):
        return self._stack_raw
    
    @stack_raw.setter
    def stack_raw(self, data_in):
        """
        Stores a stack of images in Image._stack_raw. Alternative to
        add_data(). Also updates relevant information in Image._info 
        (image size, number of channels, number of timepoints).
        
        Parameters
        ----------
        data_in : tuple
        Tuple of two elements, where the first is the stack of images (numpy array)
        and the second is the order (str or list) of the stack.
        
        Returns
        -------
        None
        
        Example
        ------- 
            tifstack = skimage.io.imread("... .tif")
            -
            img = Image()
            img.stack_raw = (tifstack, "xyct")
        
        References
        ------
        https://stackoverflow.com/questions/18714262/property-setter-with-multiple-values
        """ 
        try:
            data, order = data_in
        except ValueError:
            raise ValueError("Pass an iterable with two items. Forgot to specify <order>?")
        except TypeError:
            raise TypeError("Pass an iterable with two items. Forgot to specify <order>?")
        else:
            if self.stack_raw is not None:
                print("""A stack of raw images has already been provided. This stack will be overwritten.""")
            
            sz, ndim = self._get_stack_dims(data)
                              
            if ndim != 2:
                self._check_order(order, ndim)
                data = self._reorder_stack(order, sz, ndim, data)  
            else:
                raise ValueError("Stacks should have more than 2 dimensions.")
            
            self._stack_raw = data                        # what if the shape is 2??
            self._info["no_channels"] = data.shape[0]
            self._info["no_timepoints"] = data.shape[1]
            self._info["shape"] = (data.shape[2], data.shape[3])
               
        return
    
    
    @property
    def microbej(self):
        return self._microbej
    
    @microbej.setter
    def microbej(self, data_in):
        """
        Stores microbeJ results in Image._microbej. Alternative to
        add_data().
        
        Parameters
        ----------
        data_in : dict
        Dict of microbeJ results. Must only contain information of one image.
        
        Returns
        -------
        None
        
        Example
        ------- 
            mjres = scipy.io.loadmat("... .mat", simplify_cells=True)
            -
            i) If the .mat file only contains information of one image:
            img = Image()
            img.microbej = mjres
            
            ii) If the .mat file contains information of multiple images:
            img = Image()
            img.microbej = mjres["Experiment"][i]
            where i is the index of a certain image.
        """
        if self.microbej is not None:
            print("MicrobeJ results have already been provided. The data will be overwritten.")
        
        mj_headers = set(["BACTERIA", "IMAGE", "ANALYSIS_TIME", "NAME", "Bacteria"]) 
        
        if isinstance(data_in, dict):    # this may be converted to self._check_mj_input()
            if "Experiment" in data_in.keys():
                if isinstance(data_in["Experiment"], dict):
                    data = data_in["Experiment"]
                else:
                    raise Warning("Possibly providing multiple datasets to the same image!")  
            elif set(data_in.keys()).issuperset(mj_headers):
                data = data_in
            else:
                raise ValueError("Dictionary doesn't seem to contain microbeJ results!")
        else:
            raise TypeError("'microbeJ' input must be a dictionary!")        
        # try:
        #     data = data_in["Experiment"]    
        # except KeyError:
        #     if set(data_in.keys()).issuperset(mj_headers):
        #         data = data_in 
        #     else:
        #         raise Warning
        # except TypeError:
        #     raise TypeError("'microbeJ' input must be a dictionary.") 
        # except AttributeError:
        #     raise AttributeError("Possibly providing a multiple datasets to the same image.")  
        # except:
        #     raise Warning("Unexpected input")
        
        self._microbej = data
        
        return
        
    def inspect_data(self):
        """ 
        Prints summary of what information is contained in the instance of Image().
        Warns if essntial information (e.g. channels, shape, um_pix) is missing.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        """
        info = self.info
        stack = self.stack_raw
        mjres = self.microbej
        
        if stack is None:
            print(".. No stack of images has been provided.")
        else:
            print(".. Stack provided has {} channel(s) and {} timepoint(s).".format(info["no_channels"], info["no_timepoints"]) )
            
        if mjres is None:
            print(".. No MicrobeJ results have been provided.")
        
        if not info["um_pixel"]:
            print(".. Scale <um_pixel> has not been specified.")    
        
        if (stack is not None) & (len(info["channels"].keys()) != info["no_channels"]):
            print(".. Channels specified do not match the number of channels in the stack ({}).".format(info["no_channels"]))
        
        if (stack is not None) & (not info["fps"]) & (info["no_timepoints"] != 1):
            print(".. This is a movie - you might want to specify the <fps>.")
        
        if not info["name"]:
            print(".. You might want to specify a <name> for the image.")        
        
        return


    def _info_to_update(self, in_lowkey):
        """ 
        Returns the keys of Image._info that will have their values updated.   
        """
        set_pre = set(self.info.keys()) 
        set_input  = set(in_lowkey.keys()) 
           
        keys_common = list(set_pre.intersection(set_input))
        keys_updated = []
        
        for k in keys_common:
            if self.info[k]:
                keys_updated.append(k)
                
        return keys_updated
        

    def _get_stack_dims(self, data):
        """ 
        Returns shape and dimensions of stack of images.
        Stack may be a np.ndarray or a pims.TiffStack.
        """
        if isinstance(data, np.ndarray):
            sz = data.shape
            ndim = len(sz) 
            return sz, ndim
        elif isinstance(data, pims.TiffStack):
            sz = data._tiff.shape
            ndim = len(sz) 
            return sz, ndim
        else:
            raise TypeError("""'tif' type must be an image! Import with skimage.io.imread() or pims.TiffStack().""")   
   
        
    #TODO: implement warning for unaccounted dimension z    
    def _check_order(self, order, ndim): 
        """
        Checks if the order of the stack, as specified by the user, matches the dimensions
        of the input stack.
        """
        if order is not None:
            if len(order) == ndim:
                return
            else:
                raise ValueError("Specified <order> does not match the dimensions of the stack.")
        else:
            raise ValueError("Specify the <order> of the stack: cxyt, txy, xyc, etc.") # this may be a problem via self.add_data()    
        return
            

    def _reorder_stack(self, order, sz, ndim, data):
        """
        Re-orders the stack provided by the user (adding new dimensions if needed)
        to abide by the reference: ctxy.
        """
        ref = ["c", "t", "x", "y"]
        
        order = [x for x in order]
        data = np.array(data)  # convert to numpy array for convenience
        
        if data.ndim != ndim:
            data = np.expand_dims(data, axis=0)
            data = np.reshape(data, sz)
            
        if ndim == 3:  
            data = np.expand_dims(data, axis=0)     # adds one more dimension (1st)
            (missing_dim,) = set(ref) - set(order)  # determines which dim is missing 
            order.insert(0, missing_dim)            # adds missing dim
        
        permutation = tuple([order.index(x) for x in ref])

        return np.transpose(data, axes=permutation)
        

    ## 1. Correct illumination
    def correct_illum(self, background, channel, baseline, stack="raw", percentile=None, 
                      force_zero=False, subtract_from_stack=True):
        """
        Corrects for uneven illumination by: (1) subtracting the baseline fluorescence,
        (2) multiplying each pixel by a correction factor.
        Correction factor is determined by fitting the background fluorescence
        to a 2D-gaussian.
        Applies correction to all timepoints in the specified channel.
        
        Parameters
        ----------
        background : np.ndarray
            Image (2D) representing the background fluorescence, or stack (3D) 
            of images from which to obtain the background and contain only images
            from the specified channel. Stack must be (frame#, x, y).
        channel : str
            Channel on which background correction is to be performed.
        baseline : float or int
            Value of the baseline intensity. Specific to each microscope.
        stack : str, optional
            Stack on which to apply the correction ("raw", "even" or "aligned").
            Default: "raw".
        percentile: float or int
            Percentile of pixels taken from the stack provided as <background>.
            
        Returns
        -------
        None

        Example
        -------
            bg_2d = skimage.io.imread("...tif", "grayscale")
            img.correct_illum(bg_2d, "GFP", 500, stack="raw")
            
            bg_3d = skimage.io.imread("...tif", "grayscale")   # a stack of GFP images, e.g. (5, 512, 512)
            img.correct_illum(bg_3d, "GFP", 500, stack="raw", percentile=50)
            
        """
        # Obtain all the images of the desired channel
        stack = self._get_stack(stack)
        idx_ch, ch = self._get_channel(channel)   
        stack_ch = stack[idx_ch,:,:,:]    # for all the timepoints
        
        # Make sure that background correction has not yet been made for the channel
        if self.stack_evenillum is None:
            self.stack_evenillum = stack.copy()
            self.background = {}
            self.background_smoothed = {}
        else:
            if ch in self.background.keys():
                print("You have already background-corrected the {} channel. This will overwrite the previous attempt.".format(ch.upper()))
                self.background[ch] = []
                self.background_smoothed[ch] = []
        # Obtain 2D-array representing the background fluorescence
        background2d = self._get_background(background, percentile) 
        assert background2d.shape == stack_ch.shape[-2::]    # x and y only
        
        # Obtain smoothed, baseline-corrected background
        background2d = np.subtract(background2d, np.full(background2d.shape, baseline))
        # if force_zero is True:
        #     background2d[background2d<0] = 0  
        params = fitgaussian(background2d)
        fit = gaussian(*params)
        smoothed_bg = fit(*np.indices(background2d.shape))
        
        # Obtain correction factor for pixel intensity
        flatfield2d = np.divide(np.full(smoothed_bg.shape, np.max(smoothed_bg)), smoothed_bg)
        # flatfield3d = np.expand_dims(flatfield2d, axis=0)
        # flatfield3d = np.broadcast_to(flatfield3d, stack_ch.shape)   ### # https://stackoverflow.com/questions/32171917/copy-2d-array-into-3rd-dimension-n-times-python
        flatfield3d = np.repeat(flatfield2d[np.newaxis, :, :], stack_ch.shape[0], axis=0)
        print(stack_ch.shape, flatfield3d.shape)
        assert flatfield3d.shape == stack_ch.shape
        
        # Obtain background-corrected images, by element-wise multiplication
        if subtract_from_stack is True:
            stack_ch = np.subtract(stack_ch, np.full(stack_ch.shape, baseline))
        if force_zero is True:
            stack_ch[stack_ch<0] = 0
        stack_ch = np.multiply(flatfield3d, stack_ch)             
        
        # Store the results
        self.stack_evenillum[idx_ch, :,:,:] = stack_ch
        self.background[ch] = background2d
        self.background_smoothed[ch] = smoothed_bg
        
        return
    
    
    def _get_channel(self, channel):
        """ 
        Returns index and name (lowercase) of the channel.
        """
        ch = channel.lower()
        cc = lowercase_dict(self.info["channels"])
        
        try: 
            idx_ch = cc[ch]
        except KeyError: 
            raise KeyError("{} channel does not exist.".format(ch.upper()))
            
        return idx_ch, ch
    

    def _get_background(self, background, percentile):
        """ 
        Returns the raw background fluorescence.
        If input is a 2D array, assumes it is the real background.
        If input is a 3D array, will determine the n-% percentile of the 
        stack provided as input.

        Parameters
        ----------
        background : np.ndarray
            Image (2D) representing the background fluorescence, or stack (3D) 
            of images from which to obtain the background and contain only images
            from the specified channel. Stack must be (frame#, x, y).
        percentile: float or int
            Percentile of pixels taken from the stack provided as <background>.
            
        Returns
        -------
        np.ndarray (2D)
        """
        ndim = len(background.shape)
        
        if ndim == 2:
            print("Input: example of background.")
            return background
         
        elif ndim == 3:
            print("Input: stack of multiple images.")
            if percentile is None:
                raise ValueError("Provide the percentile!")
                
            return np.percentile(background, percentile, 
                                       axis=0, interpolation="lower")
        else:
            raise ValueError("Input has incorrect dimensions. Provide 2D or 3D array.")        
        return
    
    
    ## 2. Align channels
    def align_channels(self, ref, mov, stack="raw", regtype="rigid", 
                       tmatrix=None, norm=True):
        """ 
        Aligns channels using PystackReg. Transformation matrix can be calculated
        automatically, or provided by the user.

        Parameters
        ----------
        ref : str
            Channel to be taken as reference for image registration.
        mov: str
            Channel to be aligned.
        stack : str, optional
            Stack on which to apply the correction ("raw", "even" or "aligned").
            Default: "raw".
        regtype: str, optional
            Type of registration to be performed ("translation", "rigid", 
            "scaled_rot", "affine", "bilinear"). Default: "rigid".
        tmatrix: np.ndarray, optional
            Matrix specifying the transformation to be applied on the moved image.
        norm: bool, optional
            Determine . Default: True
            
        Returns
        -------
        None
        
        Example
        -------
        
        
        """
        # ref and mov should be a string with the name of the channel taken as reference
        # need to ensure that self.info.channels is not empty
           
        idx_ref, ref = self._get_channel(ref)   
        idx_mov, mov = self._get_channel(mov)           
        stack = self._get_stack(stack)
        sr = self._get_transformation(regtype)
        
        img_ref = stack[idx_ref, 0, :, :]
        img_mov = stack[idx_mov, 0, :, :]
        img_ref_s = stack[idx_ref, :, :, :]
        img_mov_s = stack[idx_mov, :, :, :]
        
        if self.stack_aligned is None:
            self.stack_aligned = stack.copy()
            self.tmatrix = {}
        else:
            if mov in self.tmatrix.keys():
                print("You have already aligned the {} channel. This will overwrite the previous attempt.".format(mov.upper()))
        
        if norm is True:
            img_ref_norm = normalize_intensity(img_ref)
            img_mov_norm = normalize_intensity(img_mov)
            tmatrix = self._calc_matrix(img_ref_norm, img_mov_norm, tmatrix=tmatrix, sr=sr)
        else:
            tmatrix = self._calc_matrix(img_ref, img_mov, tmatrix=tmatrix, sr=sr)     
        
        tmatrix = self._match_matrixdim(tmatrix, img_mov_s)
        out_rot = sr.transform_stack(img_mov_s, tmats=tmatrix)
        
        #self.stack_aligned = self.stack_raw
        self.stack_aligned[idx_ref,:,:,:] = img_ref_s
        self.stack_aligned[idx_mov,:,:,:] = out_rot
        self.tmatrix[mov] = tmatrix
        
        return
 
    
    def _get_transformation(self, regtype):
        """ 
        Initializes PyStackReg according to the desired type of image registration.
        """
        regtype = regtype.lower()
        if regtype == "translation":
            sr = StackReg(StackReg.TRANSLATION)
        elif regtype == "rigid":
            sr = StackReg(StackReg.RIGID_BODY)
        elif regtype == "scaled_rot":
            sr = StackReg(StackReg.SCALED_ROTATION)
        elif regtype == "affine":
            sr = StackReg(StackReg.AFFINE)
        elif regtype == "bilinear":            
            sr = StackReg(StackReg.BILINEAR)
        else:
            raise ValueError("Inexistent choice for image registration. Choose one of the following: 'translation', 'rigid' , 'scaled_rot', 'affine', 'bilinear'.")
        return sr
 
    
    def _calc_matrix(self, *args, tmatrix=None, sr=None):
        """
        Calculates the transformation matrix, if this was not specified before.
        """
        if tmatrix is not None:
            print("Using provided matrix")
            return tmatrix
        else:
            tmatrix = sr.register(*args)
            print("Calculated new matrix")
            return tmatrix
        
        
    def _match_matrixdim(self, tmatrix, stack) :
        """
        Ensures the transformation matrix has dimensions compatible with 
        the stack to be registered.
        """
        if tmatrix.ndim == 2:
            print("Provided 2D matrix with shape {}. This will be repeated across dimension 1.".format(tmatrix.shape))
            tmatrix = np.repeat(tmatrix[np.newaxis, :, :], stack.shape[0], axis=0)
        
        elif tmatrix.ndim == 3:
            
            if tmatrix.shape[0] != stack.shape[0]:
                print("Provided 3D matrix with shape {}. This will be repeated across dimension 1.".format(tmatrix.shape))
                tmatrix = np.repeat(tmatrix[0, :, :], stack.shape[0], axis=0)
            else:
                print("Provided 3D matrix with shape {}".format(tmatrix.shape))
        
        else:
            raise ValueError("Transformation matrix has incorrect dimensions")
        
        return tmatrix
    
   
    ## 3. Generate mask
    def generate_mask(self, shift_pixel=0):
        """
        Generates a labeled mask (Image.mask) where each cell is labeled by an integer number
        based on the microbeJ contours. 
        Also stores the mask of each individual (Image.mask_cells) and a dictionary
        establishing the correspondence between the label and the ID of that cell
        obtained from MicrobeJ (Image.cell_ids).
        Allows for an adjustment of the position of the contours by applying a translation (of <shift_pixel> units)
        along the x- and y- axes.

        Parameters
        ----------
        shift_pixel : float
            Number of pixels the contour will be translated along both x- and y- axis. Can be positive or negative.
            A value of -0.5 is generally necessary for a better fit of the contours obtained with MicrobeJ because the origin is defined differently 
            (Python: CENTER of the upper-left-most pixel vs. ImageJ: CORNER of that same pixel)
            
        Returns
        -------
        None
        
        """
        # stack = self._get_stack(stack)
        mj = self.microbej # _get_mjresults()
        sz = self.info["shape"]
        pxsz = self.info["um_pixel"]
        bacteria_in_img = mj["Bacteria"]
        
        cell_ids = {}  
        mask_cell = {}
        mask = np.zeros(sz)
        contour = {}
        
        if isinstance(bacteria_in_img, list):   # multiple cells per image
            for (idx, b) in enumerate(bacteria_in_img):
                x, y = self._get_contour(b)  # in um 

                x_corr = x + shift_pixel*pxsz
                y_corr = y + shift_pixel*pxsz  
                
                id_cell = b["NAME"]["name"]
            
                mask_cell[idx] = draw_contour_cell(x_corr, y_corr, size=sz, um_px=pxsz)
                contour[idx] = (x_corr, y_corr) 
                cell_ids[idx+1] = id_cell
                mask += mask_cell[idx] * (idx + 1)
                
        elif isinstance(bacteria_in_img, dict):  # single cell per image
            b = bacteria_in_img
            idx = 0
            x, y = self._get_contour(b)  # in um  
            x_corr = x + shift_pixel*pxsz
            y_corr = y + shift_pixel*pxsz          
            id_cell = b["NAME"]["name"]
        
            mask_cell[idx] = draw_contour_cell(x_corr, y_corr, size=sz, um_px=pxsz)
            contour[idx] = (x_corr, y_corr) 
            cell_ids[idx+1] = id_cell
            mask += mask_cell[idx] * (idx + 1)            

        self.contour_translated = shift_pixel    
        self.contour = contour   
        self.mask = mask.astype("int64")
        self.mask_cells = mask_cell
        self.cell_ids = cell_ids
        
        return
    
    def _get_contour(self, mjres):
        """
        Retrieves the coordinates of the contour of a single cell from the 
        MicrobeJ results. 
        Values are in um.
        """
        x = np.array([])
        y = np.array([])

        if isinstance(mjres, dict):
            # try:
            list_coords = mjres["Contour"]
            # except TypeError:
            #     print("Not the right input data")    
        elif isinstance(mjres, list):
            list_coords = mjres           
        else:
            print("incorrect input!!!")
       
        for point in list_coords:
            x_value = point["COORD"]["x"]
            x = np.append(x, x_value)
            
            y_value = point["COORD"]["y"]
            y = np.append(y, y_value)
        
        return x, y 
    
    
    def inspect_mask(self):
        """
        Checks whether the labels start at 1, 
        and displays the minimum and maximum value of the labels.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        
        mask = self.mask.copy()
        
        self._check_mask(mask)
        mask = mask[mask.astype(bool)]
        max_ = np.max(mask)
        min_ = np.min(mask)
        
        if min_ != 1:
            print("First label is not 1!")
        else:
            print("Labels go from {} to {}".format(min_, max_))
        
        return
    
    
    def _check_mask(self, mask):
        """
        Checks whether the mask is an array of integers.
        """
        if mask is not None:
            if not issubclass(mask.dtype.type, np.integer):
                raise TypeError("Mask does not contain integer values!")
        else:
            raise Warning("Instance does not contain a mask!") 
        return
    
    
    # TODO: Should use colicoords.data_to_cell_lists() instead?
    def call_colicoords(self, filtered=True, **kwargs):
        """
        Calls ColiCoords.data_to_cells() to generate an instance of CellList().

        Parameters
        ----------
        stack : str, optional
            Stack on which to call ColiCoords ("raw", "even" or "aligned").
            
        **kwargs : 
            Keyword arguments to be passed to ColiCoords.preprocess.data_to_cells().
            
        Returns
        -------
        None
        
        Example
        -------
            img.call_colicoords("raw", init_coords=True)
        
        Reference:
        -------
        https://github.com/Jhsmit/ColiCoords/blob/master/colicoords/preprocess.py
        https://github.com/Jhsmit/ColiCoords/blob/master/examples/02_batch_processing.ipynb

        """        
        stack = self._get_stack("raw")
        mask = self.mask.copy()
        channels = self.info["channels"]
        ch = "bf"
        
        if filtered is True:
            traj_ = self.tracks_filtered
        else:
            traj_ = self.tracks
        
        ## Add data:
        if  (traj_ is not None) and (not traj_.empty):
            traj = traj_.copy()
            
            data = ccoord.Data()

            # Add binary stack:
            stack_bin = np.repeat(mask[np.newaxis, :, :], stack.shape[1], axis=0)
            data.add_data(stack_bin, "binary")

            # Add brightfield channel stack:
            idx_ch = channels[ch]
            stack_ch = stack[idx_ch,:,:,:]
            data.add_data(stack_ch, "brightfield", name=ch)

            # Add SPT data
            traj["movieID"] = self.info["name"]
            traj["frame_i"] = traj["frame"].copy()
            traj["frame"] = 1  # because of ThuderSTORM's convention (see https://github.com/Jhsmit/ColiCoords/blob/master/examples/06_storm_analysis.ipynb)
            spt_data = convert_df_to_array(traj)
            data.add_data(spt_data, "storm")    
            
            try:    
                blist = ccoord.preprocess.data_to_cells(data, **kwargs)
                self.bacteria_list = blist
            except:
                print("******* Unsuccessful - {} *******".format(self.info["name"]))
        return
        
   
    
    def quantify_fluorescence(self, stack, props, t=0):
        """ 
        Quantification of fluorescence and other geometrical parameters of all 
        cells in a stack. This is done for all channels, at a single timepoint.
        Results stored as a Pandas dataframe (Image.results). 

        Parameters
        ----------
        stack : str, optional
            Stack to be used in the fluorescence quantification ("raw", "even" or "aligned").
            
        props : list
            List of properties to be determined.
            
        t : int
            Index of the time frame on which to perform the quantification. Default: 0.
            
        Returns
        -------
        None
        
        Example
        -------
            img.quantify("raw", ["area", mean_intensity, max_intensity])   # at the 1st timepoint
            
            img.quantify("raw", ["area", mean_intensity, max_intensity], t=9)  # at the 10th timepoint
        
        Reference:
        -------
        https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
        https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops_table

        """
        stack = self._get_stack(stack)
        mask = self.mask.copy()
        name = self.info["name"]
        channels = self.info["channels"]
        results = self.results

        if results is not None:
            print("Results have already been obtained. This will overwrite them.")
        
        df_img = pd.DataFrame({})  # dataframe for a single image
        
        for ch in channels:
            idx_ch, ch = self._get_channel(ch)

            propsT = skimage.measure.regionprops_table(mask,   # [idx_ch, t,:,:], 
                                                       intensity_image=stack[idx_ch, t,:,:],
                                                       properties=props)
            df = pd.DataFrame(propsT)
            df = self._replace_column_names(df, ch, props)   
            df["image"] = name
            df["t"] = t
            
            df_img = pd.concat([df_img, df[df.columns.difference(df_img.columns)]], axis=1)

        self.results = df_img
        
        return
    
    
    def _replace_column_names(self, df, ch, props):
        """ 
        Replaces column names in the results dataframe to avoid ambiguity as 
        to which channel they refer.
        """
        colname = {"max_intensity": "max_" + ch, 
                   "mean_intensity": "mean_" + ch,
                   "min_intensity": "min_" + ch,
                   "label": "cell no."}
        colrepl = {}
        
        for col in df.columns:
            if col in colname.keys():
                colrepl[col] = colname[col]
            else:
                colrepl[col] = col
        
        return df.rename(columns=colrepl)
    
## TODO: docstring!
    def apply_bandpassfilter(self, stack, channel, *args):
        stack = self._get_stack(stack).copy()
        idx, ch = self._get_channel(channel)
        
        stack_ch = stack[idx,:,:,:]
        
        filtered = np.array([tp.bandpass(frame, *args) for frame in stack_ch])
        
        self.stack_bandpass = stack
        self.stack_bandpass[idx,:,:,:] = filtered 
        return 
    
##TODO: docstring! 
## Note: perhaps because of the enforcement of belonging to a cell, some trajectories may end up being shorter than minlength (post-processing). Filtering is required! 
    def calc_track_and_msd2(self, stack, channel, diameter, minmass, gap, mem, maxtau, minlength, 
                            normalized=False, dilate=None, 
                            correct_drift=False, smooth=0, 
                            kwargs_for_batch={}, kwargs_for_link={}):
        pxsz = self.info["um_pixel"]
        fps = self.info["fps"]
        idx, ch = self._get_channel(channel)
        mask = self.mask.copy()
        
        stack = self._get_stack(stack)
        stack_ch = stack[idx, :,:,:]  
        
        if self.tracks is None:
            self.tracks = pd.DataFrame({})  ## need to initialize in __init__().
        else:
            print("Particle tracking has already been done on this image. Results will be overwritten.")
            self.tracks = pd.DataFrame({})
        
        if normalized is True:
            stack_ch = np.array([normalize_intensity(frame) for frame in stack_ch])   ####
        
        ## Using skimage's function (not scipy's!)
        if dilate is not None:
            neighborhood = skimage.morphology.square(dilate)
            mask = skimage.morphology.dilation(mask, neighborhood)
        
        
        ## Localize in all frames
        f = pd.DataFrame({})
        f = tp.batch(stack_ch, diameter, minmass=minmass, **kwargs_for_batch)
        
    
    
        ## Particle tracking:
        # Assign localized to cell
        def _get_cell(x, y, mask):
            x = int(np.rint(x))
            y = int(np.rint(y))
            return mask[y, x]  
            
        t = pd.DataFrame({})
        if not f.empty:
            f["cell"] = f.apply(lambda f: _get_cell(f['x'], f['y'], mask), axis=1)       
            f = f.loc[f["cell"]!=0]
        
            t = tp.link(f, gap, memory=mem, **kwargs_for_link)
            t = tp.filter_stubs(t, threshold=minlength) 
        
        
        ## Drift correction
        if correct_drift is True:
            drift = tp.compute_drift(t, smooth)
            t = tp.subtract_drift(t, drift)
            self.drift = drift
        
        # Data storage:
        msd_all_cols = pd.DataFrame({})
        msd_all_rows = pd.DataFrame({})

        if not t.empty: 
            for cell_no, df_tracks in t.groupby("cell"):
                im, im2 = self._calc_msd_from_tracks(df_tracks, pxsz, fps, maxtau, cell_no)
                
                if msd_all_rows.empty:
                    msd_all_rows = im2
                else:
                    msd_all_rows = pd.concat([msd_all_rows, im2], axis=0, sort=False)   
    
                msd_all_cols = pd.concat([msd_all_cols, im], axis=1, sort=False)
                msd_all_cols.columns = range(len(msd_all_cols.columns))
                
            self.tracks = t        

        self.localizations = f
        self.msd_cols = msd_all_cols
        self.msd_rows = msd_all_rows
        self.params_tracking = {"diameter": diameter, "minmass": minmass, 
                                "gap": gap, "mem": mem, "maxtau": maxtau,
                                "minlength": minlength, "dilate": dilate,
                                "correct_drift": correct_drift, "smooth": smooth}
        
        return f, t
    
    
    def _calc_msd_from_tracks(self, t, pxsz, fps, maxtau, cell_no, dropna=False):
        im = pd.DataFrame({})
        im2 = pd.DataFrame({})
        
        im = tp.imsd(t, mpp=pxsz, fps=fps, max_lagtime=maxtau)
        
        if dropna is True:    
            im = im.dropna(axis=1)  ## will this be consistent with the filtering of tracks?
            im = im[im>0].dropna(axis=1)
        
        im2 = im.reset_index().melt(id_vars="lag time [s]", var_name="particle", value_name="MSD")
        im2["cell"] = cell_no
        
        return im, im2

    

    def filter_tracks(self, condition, maxtau, dropna=True):
        """ 
        Filters the trajectories in all cells according to the provided <condition>.
        
        condition: lambda function
        
        e.g. 
            condition = lambda x: ((x['mass'].mean() > 250) &
                                   (x['ecc'].mean() < 0.1))
        """ 
        pxsz = self.info["um_pixel"]
        fps = self.info["fps"]
        
        if self.tracks is not None:  
            if self.tracks_filtered is None:
                self.tracks_filtered = pd.DataFrame({})
            else:
                print("Tracks have already been filtered. Results will be overwritten.")
                self.tracks_filtered = pd.DataFrame({})
            
            msd_all_cols = pd.DataFrame({})
            msd_all_rows = pd.DataFrame({})
            
            for cell_no in self.tracks["cell"].unique():
                t_filt = tp.filter(self.tracks.loc[self.tracks["cell"]==cell_no], condition)
                
                if not t_filt.empty:
                    im, im2 = self._calc_msd_from_tracks(t_filt, pxsz, fps, maxtau, cell_no, dropna=dropna)
                    
                    if msd_all_rows.empty:
                        msd_all_rows = im2
                    else:
                        msd_all_rows = pd.concat([msd_all_rows, im2], axis=0, sort=False)   
    
                    msd_all_cols = pd.concat([msd_all_cols, im], axis=1, sort=False)
                    msd_all_cols.columns = range(len(msd_all_cols.columns))
                
                    self.tracks_filtered = pd.concat([self.tracks_filtered, t_filt])
        
            self.msd_cols_filtered = msd_all_cols
            self.msd_rows_filtered = msd_all_rows
        return


    
    ## X. Display images/movies
    def create_overlay(self, image, mask=None, w=(1, 1, 1)):
        """ 
        Overlays a mask on an input image.

        Parameters
        ----------
        image : np.ndarray
            Image onto which the mask will be overlayed.
            
        mask : np.ndarray (integers)
            Binary or labeled mask to overlay on the image. If not specified,
            the mask stored in the instance of Image() is used.
            
        w : tuple, optional
            Weights for R, G and B color channels. Default (1, 1, 1).
            
        Returns
        -------
        overlay : np.ndstack
            RGB stack with the mask overlayed.
        
        Example
        -------

        
        """
        
        if mask is not None:
            if not np.dtype(mask) != np.int:    # this may be replaced by self._check_mask()
                raise TypeError("Mask provided should be an array of integers")
            if self.mask is not None:
                print("Mask shown is not the one attached to the image")
        else:
             mask = self.mask.copy()  
        
        assert len(w) == 3         
        mask[mask>0] = 1    
        img_norm = normalize_intensity(image)
        overlay = np.dstack((w[0]*mask, w[1]*img_norm, w[2]*img_norm))
        
        return overlay
    
    
    def display_stack(self, stack, channels, t=0, overlay=True, ax=None, 
                      save=False, *args):   
        
        cc = self.info["channels"]
        stack = self._get_stack(stack)
        mask = self.mask.copy()
        
        if isinstance(channels, list):   # use _get_channels()?????
            n_ch = len(channels)
            chs = set(channels)
            assert set(cc.keys()).issuperset(chs)          
        elif isinstance(channels, str):
            assert channels in cc.keys()
            channels = [channels]
            n_ch = len(channels)
              
        if ax is None:
            f, ax = self._get_axis(n_ch)
                
        if n_ch==1:
            ch, = channels
            idx_ch = cc[ch]
            img = stack[idx_ch, t,:,:]
            if overlay is True:
                img = self.create_overlay(img)    
            ax.imshow(img)
        else:
            for i, ch in enumerate(channels):
                idx_ch = cc[ch]
                img = stack[idx_ch, t,:,:]
                if overlay is True:
                    img = self.create_overlay(img)  
                ax[i].imshow(img)
            
        # if save is True:
        #     plt.savefig(*args)
        
        return
   
## TODO: add argument for yscale in ax[0]; remove figax    
    def display_movie_cell(self, stack, channel, cell_no, 
                           display_crop=False, display_msd=True, display_contour=True, 
                           figax=None, yscale=None, filtered=False,
                           pad=10, raster=5, iterations=None, fps=None):
        
        mask = self.mask.copy()       
        stack = self._get_stack(stack).copy()
        idx, ch = self._get_channel(channel)
        stack_ch = stack[idx,:,:,:]
        
        if filtered is True:
            if self.tracks_filtered is not None:
                tracks = self.tracks_filtered.loc[self.tracks_filtered["cell"]==cell_no]
            else:
                tracks = self.tracks.loc[self.tracks["cell"]==cell_no]
        else:
            tracks = self.tracks.loc[self.tracks["cell"]==cell_no]
        
        particles = tracks["particle"].unique()
        cell = mask==cell_no
        
        # Obtain the cell position in the first frame (same throughout the movie)
        nonzeros = np.nonzero(cell)
        
        # Axes:
        if figax is None:
            if display_msd is True:
                fig, ax = plt.subplots(1,2)
            else:
                fig, ax = plt.subplots()
        else:
            fig, ax = figax
        
        # Number of iterations and fps:
        if iterations is None:
            iterations = stack_ch.shape[0]
            assert iterations == self.info["no_timepoints"]
        if fps is None:
            fps = self.info["fps"]         # speed after saving
            int_frame = int(1/fps * 1000)  # in ms - speed for display
        
        # Generate lines 
        lines = []
        x = {}
        y = {}
        for p in particles:   
            lines.append(ax[1].plot([], [], "ro-", markerfacecolor="white", markevery=[-1], linewidth=2)[0])
            x[p] = []
            y[p] = []
            coords = tracks.loc[tracks["particle"]==p]
    
            for i in np.arange(0, iterations):
                if i in np.array(coords["frame"]):
                    x[p].append(coords.loc[coords["frame"]==i, "x"].values[0])
                    y[p].append(coords.loc[coords["frame"]==i, "y"].values[0])
                else:
                    x[p].append(np.nan)
                    y[p].append(np.nan)    
        
        if display_contour is True:
            contx, conty = self.contour[cell_no - 1]  # indexes differ!
            pxsz = self.info["um_pixel"]
            ax[1].plot(contx/pxsz, conty/pxsz, color="green", alpha=0.7)
        
        ax[1].set_xlim((min(nonzeros[1])-pad, max(nonzeros[1])+pad))
        ax[1].set_ylim((max(nonzeros[0])+pad, min(nonzeros[0])-pad))

        ## Display MSD curves:
        self.display_msd(cell_no, ax=ax[0], yscale=yscale, filtered=filtered)
        tt = " \n".join(": ".join(map(str, _)) for _ in self.params_tracking.items())
        ax[0].text(0.05, 0.98, tt, 
                   verticalalignment="top", horizontalalignment="left",
                   transform=ax[0].transAxes,
                   color="grey", fontsize=7)
        
        def _animation(i): 
            background = ax[1].imshow(stack_ch[i], cmap="gray", 
                                      vmin=stack_ch[i][cell].min(), 
                                      vmax=stack_ch[i][cell].max()+10)
              
            for j, line in enumerate(lines): 
                p = list(x.keys())[j]
                start = max((i - raster, 0))
                line.set_data(x[p][start:i+1], y[p][start:i+1])   
            
            ax[1].set_title("Cell {}, frame {}, {} fps".format(cell_no, i, fps), fontsize=9)
        
            return lines, background,
        
        anim = animation.FuncAnimation(fig, _animation, np.arange(0, iterations), interval=int_frame, blit=True)    
        plt.suptitle("Movie: '{}.tif'".format(self.info["name"]))

        return anim    
    
## TODO: docstring
    def display_msd(self, cell_no=None, ax=None, yscale=None, filtered=False):
        
        if filtered is True:
            msd = self.msd_rows_filtered
            msd2 = self.msd_cols_filtered
        else:
            msd = self.msd_rows
            msd2 = self.msd_cols
        
        if ax is None:
            fig, ax = plt.subplots()         

        # All particles in the image
        msd2.plot(alpha=0.1, 
                  ax=ax, 
                  label="", 
                  legend=False, 
                  color="black")
        # All particles in the cell
        if cell_no is not None:
            msd.loc[msd["cell"] == cell_no].pivot(index="lag time [s]", 
                                                  columns="particle", 
                                                  values="MSD").plot(loglog=True, 
                                                                     alpha=0.8, 
                                                                     color="red", 
                                                                     ax=ax, 
                                                                     legend=False,
                                                                     label="Cell {}".format(cell_no))

        # if yscale is not None:
        #     lo, up = yscale
        #     ax.set_yscale(lo, up)
        ax.set_ylabel(r"$MSD_{\tau}$ $(\mu m^2)$")
        ax.set_xlabel(r"$\tau$ $(s)$")
        ax.grid(alpha=0.3) 
        ax.tick_params(axis="both", which="both")
        return
    
    
    def display_movie(self, stack, channel, cell_no=None, ax=None, pad=10, **kwargs):    
        name = self.info["name"]
        mask = self.mask.copy()
        stack = self._get_stack(stack)
        idx, ch = self._get_channel(channel)
        stack_ch = stack[idx,:,:,:]        
  
        fps = self.info["fps"]              # speed after saving
        int_frame = int(1/fps * 1000)       # speed for display
        iterations = stack_ch.shape[0]
        
        if ax is None:
            fig, ax = plt.subplots()
              
        
        if cell_no is not None:
            cell = mask==cell_no
            cropped = [frame * cell for frame in stack_ch]
            
            mm = cropped[0]
            nonzeros = np.nonzero(mm)

            ax.set_xlim((min(nonzeros[1])-pad, max(nonzeros[1])+pad))
            ax.set_ylim((max(nonzeros[0])+pad, min(nonzeros[0])-pad))
        else:
            cell_no = "all"
        
        def _animation(i): 
            background = ax.imshow(stack_ch[i], cmap="gray", **kwargs)            
            ax.set_title("Cell {}, frame {}".format(cell_no, i))
            
            return background, #cont, 
        
        anim = animation.FuncAnimation(fig, _animation, np.arange(0, iterations), interval=int_frame, blit=False)   
        
        return anim
    
    
    
    def _get_axis(self, n_ch):  
        if n_ch == 1:
            return plt.subplots()
        else:
            return plt.subplots(1, n_ch)
    
        
    def _get_stack(self, stack):
        stack = stack.lower()
        if stack == "raw":
            stack = self.stack_raw
        elif stack == "even":
            stack = self.stack_evenillum
        elif stack == "aligned":
            stack = self.stack_aligned
        elif stack == "bandpass":
            stack = self.stack_bandpass
        else:
            raise Warning("Unknown choice for stack")   
            
        return stack
    
    
    def _get_mjresults(self):
        if self.microbej is not None:
            return self.microbej
        else:
            raise Warning("No MicrobeJ results have been provided yet.")
    


    