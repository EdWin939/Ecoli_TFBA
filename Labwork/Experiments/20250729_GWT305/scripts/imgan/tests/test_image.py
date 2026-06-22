# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 15:18:18 2020

@author: p290481
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import image_v2 as image
# import image
import glob
# import os
import pims
# import skimage

import numpy as np
# import pandas as pd
# import seaborn as sns
import scipy.io as sio
# import matplotlib.pyplot as plt

from skimage import exposure, measure, segmentation, io, transform
# from pystackreg import StackReg


all_img = glob.glob(os.path.join("example_images", "*.tif"))
all_mj = glob.glob(os.path.join("example_images", "*.mat"))

# Load images:
im1 = io.imread(all_img[0])
print("Shape of im1:  "+ str(im1.shape))
im2 = io.imread(all_img[1])
print("Shape of im2:  "+ str(im2.shape))
im3 = io.imread(all_img[2])
print("Shape of im3:  "+ str(im3.shape))
#
im4 = pims.open(all_img[0])
print("Shape of im4:  "+ str(im4._tiff.shape))
im5 = pims.open(all_img[1])
print("Shape of im5:  "+ str(im5._tiff.shape))
im6 = pims.open(all_img[2])
print("Shape of im6:  "+ str(im6._tiff.shape))

# # Load MicrobeJ data:
df_mj = sio.loadmat(all_mj[0], simplify_cells=True)
df_mj1 = df_mj["Experiment"]
df_mj2 = df_mj["Experiment"][0]

# Load background:
bg1 = io.imread(all_img[3], "grayscale")
print("Shape of bg1:  "+ str(bg1.shape))
bg2 = io.imread(all_img[4], "grayscale")
print("Shape of bg2:  "+ str(bg2.shape))


class TestAddData:

    def test_add_data_wrongtype(self):
        c = image.Image()
        with pytest.raises(TypeError):
            c.add_data(im1, "microbeJ")


    def test_add_data_mj_standard(self):
        c = image.Image()
        c.add_data(df_mj["Experiment"][0], "microbeJ")
            
            
    def test_add_data_mj_capitalletters(self):
        c = image.Image()
        c.add_data(df_mj["Experiment"][0], "MiCrobEJ")
          
            
    def test_add_data_mj_orderasinput(self):
        c = image.Image()    
        c.add_data(df_mj["Experiment"][0], "MiCrobEJ", order="xyz")


    def test_add_data_mj_listofresults(self):
        c = image.Image()        
        with pytest.raises(TypeError): 
            c.add_data(df_mj["Experiment"], "MiCrobEJ")

    # TODO: correct test?
    def test_add_data_mj_experimentdict(self):
        c = image.Image()  
        with pytest.raises(Warning): 
            c.add_data(df_mj, "MiCrobEJ")
    
        
    def test_add_data_mj_wrongdata(self):
        c = image.Image()  
        dictex ={"ex": 1, "ex3": 3} 
        with pytest.raises(ValueError):
            c.add_data(dictex, "MiCrobEJ")   
    
        
    def test_add_data_mj_wrongdatatype(self):
        c = image.Image() 
        with pytest.raises(ValueError):
            c.add_data(df_mj["Experiment"], "MiCrobEZZzz")  
       

    def test_add_data_info_standard(self):
        c = image.Image()
        c.add_data(im4, "tif", order="txy")    
        infodict = {"shape": (128, 128), "channels": {"cfp": 0, "yfp": 1}, "new entry": 123456798}
        c.add_data(infodict, "info")


    def test_add_data_info_wrongdata(self):
        c = image.Image()
        c.add_data(im4, "tif", order="txy")     
        with pytest.raises(TypeError):
            infodict = ["a", "b"]
            c.add_data(infodict, "info")


    def test_add_data_tif_skimage_order1(self):
        c = image.Image()
        c.add_data(im1, "tif", order="txy")


    def test_add_data_tif_skimage_order2(self):
        c = image.Image()
        c.add_data(im2, "tif", order=["x", "y", "c"])


    def test_add_data_tif_skimage_order3(self):
        c = image.Image()
        c.add_data(im3, "tif", order="txyc")

    
    def test_add_data_tif_pims_order1(self):
        c = image.Image()
        c.add_data(im4, "tif", order="txy")


    def test_add_data_tif_pims_order2(self):
        c = image.Image()
        c.add_data(im5, "tif", order=["c", "x", "y"])


    def test_add_data_tif_pims_order3(self):
        c = image.Image()
        c.add_data(im6, "tif", order="tcxy")


#
#

class TestSetter:
    
    def test_set_info_standard(self):
        c = image.Image()
        c.info = {"no_channels": 3}


    def test_set_stack_standard(self):
        c = image.Image()
        c.add_data(bg1, "tif", order="xyc")
    
    
    def test_set_microbej_listimages(self):
        c = image.Image()
        with pytest.raises(Warning): 
            c.microbej = df_mj


    def test_set_microbej_listimages2(self):
        c = image.Image()
        with pytest.raises(TypeError): 
            c.microbej = df_mj1  
  
            
    def test_set_microbej_standard(self):
        c = image.Image()
        c.microbej = df_mj2            



#
#

baseline = 0    # the images used as input are already 0 - 1 (! in general, this would not be the case)
channel = "GFP"
infodict_im6 = {"channels": {"GFP": 1, "BF": 0, "rfp": 2, "phluorin": 3}}
s = "raw"
p = 50

class TestCorrIllum:
    def test_corr_illum_stack_ok(self):
        c = image.Image()
        c.add_data(im6, "tif", order="tcxy")
        c.add_data(infodict_im6, "info")
        c.correct_illum(bg1, channel, baseline, stack=s, percentile=p)

            
    def test_corr_illum_singleimage_ok(self):  
        c = image.Image()
        c.add_data(im6, "tif", order="tcxy")
        c.add_data(infodict_im6, "info")
        c.correct_illum(bg2, channel, baseline, stack=s)
       
        
    def test_corr_illum_stack_without_percentile(self):   
        c = image.Image()
        c.add_data(im6, "tif", order="tcxy")
        c.add_data(infodict_im6, "info") 
        with pytest.raises(ValueError): 
            c.correct_illum(bg1, channel, baseline, stack=s)
   
    
    def test_corr_illum_stack_wrong_dims(self):    
        c = image.Image()
        c.add_data(im6, "tif", order="tcxy")
        c.add_data(infodict_im6, "info")
        bg3 = np.expand_dims(bg1, axis=2)
        with pytest.raises(ValueError): 
            c.correct_illum(bg3, channel, baseline, stack=s, percentile=p)
   
        
    def test_corr_illum_stack_twice(self):      
            c = image.Image()
            c.add_data(im6, "tif", order="tcxy")
            c.add_data(infodict_im6, "info")
            c.correct_illum(bg1, channel, baseline, stack=s, percentile=p)
            c.correct_illum(bg1, channel, baseline, stack=s, percentile=p)

            
    def test_corr_illum_singleimage_twice(self):    
            c = image.Image()
            c.add_data(im6, "tif", order="tcxy")
            c.add_data(infodict_im6, "info")
            c.correct_illum(bg2, channel, baseline, stack=s, percentile=p)
            c.correct_illum(bg2, channel, baseline, stack=s, percentile=p)



# #
# #
infodict_im2 = {"channels": {"BF": 0, "YFP": 1, "CFP": 2}}

class TestAlignChannels:
    

    def test_align_channels_standard(self):
        c = image.Image()
        c.add_data(im2, "tif", order="xyc")
        c.add_data(infodict_im2, "info")
        c.align_channels(ref="cfp", mov="yfp", stack="raw")

                       
    def test_align_channels_different_registration(self):
        c = image.Image()
        c.add_data(im2, "tif", order="xyc")
        c.add_data(infodict_im2, "info")
        c.align_channels(ref="cfp", mov="yfp", stack="raw", regtype="bilinear")      

     
    def test_align_channels_inexistent_registration(self):    
        c = image.Image()
        c.add_data(im2, "tif", order="xyc")
        c.add_data(infodict_im2, "info")
        with pytest.raises(ValueError): 
            c.align_channels(ref="cfp", mov="yfp", stack="raw", regtype="Xxxxx")      

          
            
    def test_align_channels_4dmatrix(self):    
        mat = np.array([[[[ 0.99190672, -0.12696869,  6.59710315],
                              [ 0.12696869,  0.99190672, 45.9438708 ],
                              [ 0.,          0.,          1.        ]]]])
        c = image.Image()
        c.add_data(im2, "tif", order="xyc")
        c.add_data(infodict_im2, "info")
        with pytest.raises(ValueError): 
            c.align_channels(ref="cfp", mov="yfp", stack="raw", tmatrix=mat)
  
    
    
    def test_align_channels_3dmatrix_okay(self):  
        mat = np.array([[[ 0.99190672, -0.12696869,  6.59710315],
                              [ 0.12696869,  0.99190672, 45.9438708 ],
                              [ 0.,          0.,          1.        ]]])
        c = image.Image()
        c.add_data(im2, "tif", order="xyc")
        c.add_data(infodict_im2, "info")
        c.align_channels(ref="cfp", mov="yfp", stack="raw", tmatrix=mat)
   
    
    def test_align_channels_2dmatrix_okay(self): 
        mat = np.array([[ 0.99190672, -0.12696869,  6.59710315],
                              [ 0.12696869,  0.99190672, 45.9438708 ],
                              [ 0.,          0.,          1.        ]])
        c = image.Image()
        c.add_data(im2, "tif", order="xyc")
        c.add_data(infodict_im2, "info")
        c.align_channels(ref="cfp", mov="yfp", stack="raw", tmatrix=mat)

    
    
    def test_align_channels_inexistentchannel(self): 
        mat = np.array([[ 0.99190672, -0.12696869,  6.59710315],
                              [ 0.12696869,  0.99190672, 45.9438708 ],
                              [ 0.,          0.,          1.        ]])
        c = image.Image()
        c.add_data(im2, "tif", order="xyc")
        c.add_data(infodict_im2, "info")
        with pytest.raises(KeyError): 
            c.align_channels(ref="cfp", mov="gfp", stack="raw", tmatrix=mat)

    
    def test_align_channels_twice_okay(self): 
        mat = np.array([[ 0.99190672, -0.12696869,  6.59710315],
                              [ 0.12696869,  0.99190672, 45.9438708 ],
                              [ 0.,          0.,          1.        ]])
        c = image.Image()
        c.add_data(im2, "tif", order="xyc")
        c.add_data(infodict_im2, "info")
        c.align_channels(ref="cfp", mov="yfp", stack="raw")
        c.align_channels(ref="cfp", mov="yfp", stack="raw", tmatrix=mat)
 
    
#
#

class TestGenerateMask:
    def test_generate_mask_okay(self):
        infodict = {"channels": {"BF": 0, "YFP": 1, "CFP": 2}, "um_pixel": 0.16}
        c = image.Image()
        c.add_data(im2, "tif", order="xyc")
        c.add_data(infodict, "info")
        c.add_data(df_mj["Experiment"][0], "microbeJ")
        c.generate_mask()



# #
# #

# class TestInspectMask:
#     def test_inspect_mask():
#         return



# #
# #
# class TestCallColiCoords:
#     def test_call_colicoords():
#         # try:
#         #     print("--Trial 1: movie, single channel")  
#         #     infodict = {"channels": {"GFP": 0}, "um_pixel": 0.16}
#         #     c = image.Image()
#         #     c.add_data(im1, "tif", order="txy")
#         #     c.add_data(infodict, "info")
#         #     c.add_data(df_mj["Experiment"][0], "microbeJ")
#         #     c.generate_mask("raw") 
#         #     c.call_colicoords("raw")
#         #     print("Trial 1: OK")
#         #     print("")
#         # except:
#         #     print("--Trial 1: unexpected outcome!") 
#         #     print("") 
            
#         # try:
#         #     print("--Trial 2: frame, multichannel")  
#         #     infodict = {"channels": {"BF": 0, "YFP": 1, "CFP": 2}, "um_pixel": 0.16}
#         #     c = image.Image()
#         #     c.add_data(im2, "tif", order="xyc")
#         #     c.add_data(infodict, "info")
#         #     c.add_data(df_mj["Experiment"][0], "microbeJ")
#         #     c.generate_mask("raw") 
#         #     c.call_colicoords("raw")
#         #     print("Trial 2: OK")
#         #     print("")
#         # except:
#         #     print("--Trial 2: unexpected outcome!") 
#         #     print("") 
    
    
#         print("--Trial 3: movie, multichannel") 
#         infodict = {"channels": {"BF": 0, "GFP": 1, "RFP": 2, "phluorin": 3, "rfp": 4}}
#         c = image.Image()
#         c.add_data(im3, "tif", order="txyc")
#         c.add_data(infodict, "info")
#         # c.add_data(df_mj["Experiment"][0], "microbeJ")
#         c.inspect_data()
#         c.generate_mask("raw") 
#         c.call_colicoords("raw")
    
#         # try:
#         #     print("--Trial 3: movie, multichannel")  
#         #     infodict = {"channels": {"BF": 0, "GFP": 1, "RFP": 2, "phluorin": 2}, "um_pixel": 0.16}
#         #     c = image.Image()
#         #     c.add_data(im3, "tif", order="txyc")
#         #     c.add_data(infodict, "info")
#         #     c.add_data(df_mj["Experiment"][0], "microbeJ")
#         #     c.generate_mask("raw") 
#         #     c.call_colicoords("raw")
#         #     print("Trial 3: OK")
#         #     print("")
#         # except:
#         #     print("--Trial 3: unexpected outcome!") 
#         #     print("") 
    
#         return c
