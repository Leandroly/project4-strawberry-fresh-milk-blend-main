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
    u = np.multiply(np.eye(dTr),1/(np.std(xTr,axis=1).T))

    xTr = np.dot(u, xTr - m)
    xTe = np.dot(u, xTe - m)
    ## >>
    return xTr, xTe, u, m