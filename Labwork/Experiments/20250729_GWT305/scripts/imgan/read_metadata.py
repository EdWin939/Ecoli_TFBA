# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:19:53 2021

@author: P290481
"""

import tifffile

def read_metadata_from_tif(path_to_tif):
    """ 
    Reads a .tif file, and returns the metadata stored in it.
    Metadata is a single string.
    """
    with tifffile.TiffFile(path_to_tif) as tif:
        data = tif.asarray()
        metadata = tif.imagej_metadata
        
    return metadata


def convert_metadata_to_dict(metadata_str):
    """ 
    Converts metadata in a string into a dictionary.
    Assumes that:
        1) each key-value pair is separated by "\n"
        2) the values are separated from the keys by " = "
    """
    
    metadata_dict = {}

    for item in metadata_str.split('\n'):
        try:
            key, val = item.split(' = ')
            key = key.lstrip(' ')
            metadata_dict[key] = val
        except:
            print(item)
            print("error")
            
    return metadata_dict


def get_xy_position(metadata_dict):
    """ 
    Retrieves the (x, y) coordinates of the imaging location from the metadata.
    """
    x0 = metadata_dict["m_dXYPositionX0"]
    y0 = metadata_dict["m_dXYPositionY0"]
    return x0, y0


def get_time(metadata_dict):
    """ 
    Retrieves the timestamp from the metadata.
    """
    all_keys = list(metadata_dict.keys())
    time_key = [i for i in all_keys if i.find("TextInfoItem_9")!= -1]
    t = metadata_dict[time_key[0]]
    return t