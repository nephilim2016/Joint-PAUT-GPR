#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 05:30:10 2025

@author: nephilim
"""
import numpy as np
from matplotlib import pyplot,cm
from scipy.signal import butter, filtfilt, hilbert, fftconvolve
# from config_parameters import config

def robust_norm(x):
    mad = np.median(np.abs(x - np.median(x))) + 1e-12
    return x / mad

def trim_and_pad(arr: np.ndarray, n_trim: int) -> np.ndarray:
    res = np.empty_like(arr)
    res[:-n_trim] = arr[n_trim:]
    res[-n_trim:] = np.mean(arr)
    return res

def pick_takeoff_single(x, ref, k_ray, n_consec):
    x_nrm = robust_norm(x)
    # coarse via xcorr
    xc = fftconvolve(x_nrm, ref[::-1], mode='full')[len(ref)-1:]
    pk  = int(np.argmax(xc))
    # envelope
    env = np.abs(hilbert(x_nrm))
    noise_end = len(x)
    sigma_hat = np.sqrt(np.mean(env[:noise_end]**2) / (4/np.pi))
    theta     = k_ray * sigma_hat
    # backtrack
    below = 0
    j = pk
    while j >= 0 and j < len(env)-1:
        below = below + 1 if env[j] < theta else 0
        if below >= n_consec:
            # linear sub‑sample interp
            y0, y1 = env[j+n_consec-1], env[j+n_consec]
            frac   = (theta - y0)/(y1-y0+1e-12)
            return j+n_consec-1 + np.clip(frac,0,1), env, theta, pk
        j -= 1
    return j, env, theta, pk

def align_trace(data, k_ray, n_consec):
    new_data=[]
    for index,profile in enumerate(data):
        new_profile=np.zeros_like(profile)
        new_profile[:,0]=profile[:,0]
        for idx in range(profile.shape[1]):
            p,e,t,pk=pick_takeoff_single(profile[:,idx], profile[:,index], k_ray, n_consec)
            if int(p)==0:
                new_profile[:,idx]=profile[:,idx]
            else:
                x_new=trim_and_pad(profile[:,idx],int(p))
                new_profile[:,idx]=x_new
        new_data.append(new_profile)
    return new_data


def extract_direct_wave(data, P_FA = 1e-6, N_CONSEC = 10):
    if data.ndim==2:
        data=data[np.newaxis,...]
    K_ray  = np.sqrt(-2*np.log(P_FA)) 
    new_data=align_trace(data, K_ray, N_CONSEC)
            
    return new_data
    # # # index=20
    # # # pyplot.figure()
    # # # pyplot.imshow(data[index],extent=[0,1,0.4,0],vmin=-0.05,vmax=0.05,cmap=cm.jet)
    
    # # # pyplot.figure()
    # # # pyplot.imshow(new_data[index],extent=[0,1,0.4,0],vmin=-0.05,vmax=0.05,cmap=cm.jet)
    
    # for index in range(5):
    #     pyplot.figure()
    #     pyplot.plot(ref_profile[index])
        
    #     pyplot.figure()
    #     pyplot.imshow(new_data[index],extent=[0,1,0.4,0],vmin=-0.05,vmax=0.05,cmap=cm.gray)
