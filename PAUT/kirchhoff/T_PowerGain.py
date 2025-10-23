#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:43:17 2020

@author: nephilim
"""
import numpy as np
from matplotlib import pyplot,cm

def tpowGain(data,twtt,power):
    factor = np.reshape(twtt**(float(power)),(len(twtt),1))
    factmat = np.tile(factor,(1,data.shape[1]))  
    return np.multiply(data,factmat)

if __name__=='__main__':
    aa=tpowGain(data,np.arange(512)/4,1.5)
   