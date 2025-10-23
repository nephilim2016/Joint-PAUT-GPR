#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 06:03:34 2025

@author: nephilim
"""

from config_parameters import config
from clutter_removal import ClutterRemoval
from forward_modeling import ForwardModeling
import numpy as np
import time
import skimage
import imaging_condition
from matplotlib import pyplot,cm
import T_PowerGain

if __name__=='__main__': 
    # Forward modeling for the synthetic model
    # epsilon_=np.ones((config.xl,config.zl))*4
    # epsilon_[20:30,40:60]=1
    # epsilon=np.zeros((config.xl+20,config.zl+20))
    # epsilon[10:-10,10:-10]=epsilon_
    # epsilon[:10,:]=epsilon[10,:]
    # epsilon[-10:,:]=epsilon[-10-1,:]
    # epsilon[:,:10]=epsilon[:,10].reshape((len(epsilon[:,10]),-1))
    # epsilon[:,-10:]=epsilon[:,-10-1].reshape((len(epsilon[:,-10-1]),-1))
    # epsilon[:10+config.air_layer,:]=1
    # sigma=np.zeros_like(epsilon)
    # true_profile=ForwardModeling(sigma,epsilon.copy(),config.wavelet_type,config.frequency)._forward_2d()
    # air_trace=ForwardModeling(np.zeros((config.xl+20,config.zl+20)),np.ones((config.xl+20,config.zl+20)),config.wavelet_type,config.frequency)._forward_2d_air()
    # true_profile=true_profile-np.tile(air_trace,(true_profile.shape[1],1)).T
    # true_profile=skimage.transform.resize(true_profile,(config.k_max,len(config.source_site)))
    # config.true_profile=true_profile-ClutterRemoval(true_profile,max_iter=1000,rank=1,lam=1e-4,method='GoDec').clutter_removal()
    
    true_profile=np.load('FieldNoRef2025.npy')
    
    # true_profile=T_PowerGain.tpowGain(true_profile,np.arange(true_profile.shape[0])/4,0.5)
    config.true_profile=skimage.transform.resize(true_profile,(config.k_max,len(config.source_site)))
    
    start_time=time.time() 

    # # Explosion
    # config.imaging_condition='explosion'
    # # iepsilon_=np.load('OverThrust.npy')
    # iepsilon_=np.ones((config.xl,config.zl))*3
    # iepsilon_=skimage.filters.gaussian(iepsilon_,sigma=30)
    # iepsilon=np.zeros((config.xl+20,config.zl+20))
    # iepsilon[10:-10,10:-10]=iepsilon_
    # iepsilon[:10,:]=iepsilon[10,:]
    # iepsilon[-10:,:]=iepsilon[-10-1,:]
    # iepsilon[:,:10]=iepsilon[:,10].reshape((len(iepsilon[:,10]),-1))
    # iepsilon[:,-10:]=iepsilon[:,-10-1].reshape((len(iepsilon[:,-10-1]),-1))
    # iepsilon[:10+config.air_layer,:]=1
    # iepsilon*=4
    # I=imaging_condition.imaging_condition(iepsilon)
    # print(time.time()-start_time)
    # np.save('explosion_rtm.npy',I)
    
    start_time=time.time() 
    #Correlation
    # config.imaging_condition='explosion'
    # config.imaging_condition='correlation'
    config.imaging_condition='poynting'
    # iepsilon_=np.load('field_model1.npy')
    # iepsilon_=skimage.transform.resize(iepsilon_,(600,500))
    iepsilon_=np.ones((config.xl,config.zl))*6
    # iepsilon_=skimage.filters.gaussian(iepsilon_,sigma=5)+0.5
    iepsilon=np.zeros((config.xl+20,config.zl+20))
    iepsilon[10:-10,10:-10]=iepsilon_
    iepsilon[:10,:]=iepsilon[10,:]
    iepsilon[-10:,:]=iepsilon[-10-1,:]
    iepsilon[:,:10]=iepsilon[:,10].reshape((len(iepsilon[:,10]),-1))
    iepsilon[:,-10:]=iepsilon[:,-10-1].reshape((len(iepsilon[:,-10-1]),-1))
    iepsilon[:10+config.air_layer,:]=1
    # iepsilon*=4
    I=imaging_condition.imaging_condition(iepsilon)
    print(time.time()-start_time)
    # np.save('correlation_rtm_result.npy',I[0])
    # np.save('correlation_rtm_source.npy',I[1])
    # np.save('poynting_rtm_result.npy',I[0])
    # np.save('poynting_rtm_source.npy',I[1])
    # np.save('explosion_rtm_result.npy',I)
    
    pyplot.figure()
    pyplot.imshow(I[0][10:-10,10:-10]/I[1][10:-10,10:-10], extent=[0,800,150,0], cmap='jet')
    
    # from scipy.signal import hilbert
    # env = np.abs(hilbert(I[0][10:-10,10:-10], axis=0))
    # pyplot.figure()
    # pyplot.imshow(env, extent=[0,600,500,0], cmap='jet')
    
    # env=env[]
    # # pyplot.savefig('kirchhoff_m.png',dpi=1000)
    
    # # env = np.abs(hilbert(img, axis=0))
    # # pyplot.figure()
    # # pyplot.imshow(env, extent=[0,2,1,0], cmap='jet')