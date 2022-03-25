# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 14:05:03 2019

@author: Jerry Xing
"""
import numpy as np
def preprocess(xTr,xTe):
# function [xTr,xTe,u,m]=preprocess(xTr,xTe);
#
# Preproces the data to make the training features have zero-mean and
# standard-deviation 1
# input:
# xTr - raw training data as d by n_train numpy ndarray 
# xTe - raw test data as d by n_test numpy ndarray
    
# output:
# xTr - pre-processed training data 
# xTe - pre-processed testing data
#
# u,m - any other data should be pre-processed by x-> u*(x-m)
#       where u is d by d ndnumpy array and m is d by 1 numpy ndarray

    ## << Remove 2 lines above and insert your solution here
    dTr, nTr =  np.shape(xTr)
    dTe, nTe = np.shape(xTe)

    m = np.mean(xTr, axis=1)[:, None]
    m_xTr = np.repeat(np.mean(xTr, axis=1)[:, None], nTr, axis=1)
    m_xTe = np.repeat(np.mean(xTr, axis=1)[:, None], nTe, axis=1)

    u = np.ones((dTr, 1)) / np.repeat(np.std(xTr, axis=1)[:, None], dTr, axis=1)
    u = np.diag(np.diag(u))

    xTr = u @ (xTr - m_xTr)
    xTe = u @ (xTe - m_xTe)
    ## >>
    return xTr, xTe, u, m