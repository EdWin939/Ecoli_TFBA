# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:32:00 2021

@author: P290481
"""

from scipy import stats
import random
import itertools 
import pandas as pd
import numpy as np

def ttest_pairs(data, group, fixed, compare, var, **kwargs):
    """ 
    Applies a t-test to a dataframe with multiple columns.
    
    <data>: dataframe to be analysed
    
    <group>: variable that will be fixed during the t-test
        
    <fixed>: value of the variable that will be fixed (<group>)
        
    <compare>: categorical variable describing the subset of data to be compared
          
    <var>: numerical variable to be compared in the t-test
    
    
    e.g.: We want to assess the effect of different growth conditions/'treatments'
    on the mean 'fluorescence' of the cells. If we have different 'fluorescent markers',
    we may want to fix our attention on just one of them, say, 'GFP'. Thus:
        - group: "fluorescent marker"
        - fixed: "GFP"
        - compare: "treatment"
        - var: "fluorescence"
    
    """
    
    ## Select subset of data to analyse:
    df_g = data.loc[data[group]==fixed]
    
    ## Generate all possible combinations:
    possibilities = df_g[compare].unique()
    comp = itertools.combinations(possibilities, 2)
    
    df_out = pd.DataFrame({})
    
    print("***{} = {}***".format(group, fixed))
    for c in comp:
        df_c = pd.DataFrame({}, index=[0])
        test = stats.ttest_ind(df_g.loc[df_g[compare]==c[0], var], 
                              df_g.loc[df_g[compare]==c[1], var], 
                              equal_var=False,
                              **kwargs)
        df_c["pair"] = "{}, {}".format(*c)
        df_c["t-stat"] = test[0]
        df_c["p-value"] = test[1]
        
        df_out = pd.concat([df_out, df_c], axis=0, ignore_index=True)

    return df_out



def mean_confidence_interval(data, confidence=0.95):
    """
    Returns the mean, <m>, and the confidence interval, <h>, with a given confidence level.
    
    https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data 
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h



def get_tvalue(no_point, no_params, alpha):
    """ 
    http://kitchingroup.cheme.cmu.edu/blog/2013/02/12/Nonlinear-curve-fitting-with-parameter-confidence-intervals/
    """ 
    try:
        dof = max(0, no_point - no_params)
        return stats.distributions.t.ppf(1-alpha/2, dof)
    except:
        return 0
    
    
def correlation_from_covariance_matrix(pcov):
    """ 
    https://math.stackexchange.com/questions/186959/correlation-matrix-from-covariance-matrix/300775
    https://nedcharles.com/regression/Nonlinear_Regression.html
    """ 
    cov_diag = np.sqrt(np.diag(np.diag(pcov)))
    cov_diag_inv = np.linalg.inv(cov_diag)
    corr_mat = cov_diag_inv * pcov * cov_diag_inv
    return corr_mat