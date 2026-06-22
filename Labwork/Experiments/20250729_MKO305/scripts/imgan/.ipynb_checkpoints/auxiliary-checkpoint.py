# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 15:25:09 2020

@author: p290481
"""

import numpy as np
from scipy import optimize
import skimage.draw 
import os
import collections

def get_basename(path):
    fname = os.path.basename(path)
    fname = os.path.splitext(fname)[0]  
    return fname

def normalize_intensity(image):
    """ 
    Normalizes the intensity values of an image, setting them to the range [0, 1]. 
    """
    
    return (image - image.min())/(image.max() - image.min())


def moments(data):
    """
    Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments.
    """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y


def gaussian(height, center_x, center_y, width_x, width_y):
    """
    Returns a gaussian function with the given parameters.
    """
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)


def fitgaussian(data):
    """
    Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit.
    """
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p


def draw_contour_cell(x, y, size=(512,512), um_px=0.16):
    """
    Given the x and y coordinates of a contour (both numpy arrays), draws the polygon
    corresponding to each cell on an image of size defined by the tuple <size>.
    
    Note that the x  values correspond to the columns of the image matrix, 
    and the y values correspond to the rows. 
    
    The coordinates, given in um, are converted to pixels, according to the parameter
    <um_px>.
    """
    img = np.zeros(size)
    
    x_rescaled = x/um_px
    y_rescaled = y/um_px
    
    rr, cc = skimage.draw.polygon(y_rescaled, x_rescaled, size)
    img[rr, cc] = 1
    
    return img


def model_msd(tau, D, alpha):
    return (4*D)*(tau**alpha)


def model_general_powerlaw(x, beta, K):
    return K*(x**beta)


def sort_two_arrays(x, y):
    """ Order two arrays simultaneously, taking x as reference. """ 
    return x[x.argsort()], y[x.argsort()]


def bin_sorted_array(x, size=50):
    """ 
    Compute the mean and stdev of slices of the input array (defined size.
    Last slice may contain less points than all others.
    """
    x_out = np.array([np.mean(x[size*i:min(size*(i+1), len(x))]) for i in range(round(len(x)/size))])
    x_std = np.array([np.std(x[size*i:min(size*(i+1), len(x))]) for i in range(round(len(x)/size))])
    return x_out, x_std


def convert_df_to_array(df):
    """
    Thanks to 
    https://stackoverflow.com/questions/40554179/how-to-keep-column-names-when-converting-from-pandas-to-numpy
    https://pandas.pydata.org/pandas-docs/version/0.25.1/reference/api/pandas.DataFrame.as_matrix.html
    """
    arr_ip = [tuple(i) for i in df.values]
    dtyp = np.dtype(list(zip(df.dtypes.index, df.dtypes)))
    arr = np.array(arr_ip, dtype=dtyp)
    return arr


def update_dict(dict_in, keylist, value):
    """
    Updates a dictionary where the values of each key are lists.
    Appends to an existing entry, or creates a new entry.
    """
    for key in keylist:
        if key in dict_in.keys():
            dict_in[key].append(value)
        else:
            dict_in[key] = [value]
    return dict_in


def duplicates(n):
    """ 
    Create a dictionary with {value_repeated: [indices where the value is found]}
    
    https://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list """
    counter=collections.Counter(n) 
    
    dups=[i for i in counter if counter[i]!=1] 
    
    result={}
    for item in dups:
             result[item]=[i for i,j in enumerate(n) if j==item] 
    return result

def load_mat():
    
    return



def lowercase_dict(data):
    """ returns a dictionary with lowercase keys """
    return {k.lower(): v for k, v in data.items()}


# def reorder_stack(refernce, real):
    
# def save_list_images():


# def parse_name(name, *args):
#     name = name.split(*args)[i]
#     #name = x["NAME"]["fullname"]
#     name = name.replace(*args)
#     return print(name)


# parse_name("aaaa_vvvv 3333", " " , -2, "_", "")