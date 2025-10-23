#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 07:09:10 2025

@author: nephilim
"""
from config_parameters import config
import numpy as np
import time
import skimage
import imaging_condition
from matplotlib import pyplot,cm
from scipy.signal import hilbert
import align_gather
import reference_guided
import reference_rank_svd 
from T_PowerGain import tpowGain
import skimage.transform
import scipy.io as scio

def reverse_time_migration(config,index,rec_num=200,method='poynting'):
    start_time=time.time() 
    
    config.true_profile=[]
    data_=scio.loadmat('../%s_1.mat'%index)
    data_=data_['data_all'].T
    data_=data_.astype('float')
    
    data_tmp=[]
    for idx in range(16):
        data=data_[:1024,16*idx:16*(idx+1)]
        data_tmp.append(data)

    data=[]
    for data_ in data_tmp:
        for idx in range(data_.shape[1]):
            data_[:,idx]-=np.mean(data_[:,idx])
            data_[:,idx]=align_gather.bandpass(data_[:,idx], fs=1/1e-6/2, fmin=5e4*0.3, fmax=5e4*1.8, order=4)
        data.append(data_)
    
    align_data=[]
    shifts_data=[]
    pad_data=[]
    for idx,data_ in enumerate(data):
        if idx==0:
            ref=data_[:,1]
        else:
            ref=data_[:,idx-1]
        aligned, shifts, pad = align_gather.align_by_ref_pad(data_, ref, allow_frac=True)
        aligned=tpowGain(aligned,np.arange(aligned.shape[0])/4,1.)
        align_data.append(aligned)
        shifts_data.append(shifts)
        pad_data.append(pad)
            
    # clean_data=[]
    # for idx,data_ in enumerate(align_data):
    #     if idx==0:
    #         REF_ID=1
    #     else:
    #         REF_ID=idx-1
    #     clean=reference_rank_svd.extract_first_wave_full_gather(data_, REF_ID, WIN_LEN=240, GUARD=60)
    #     clean_data.append(clean)
    
    # data=clean_data[1]
    # pyplot.figure()
    # pyplot.imshow(data,extent=(0,3,1,0),vmin=np.min(data)/2,vmax=np.max(data)/2,cmap=cm.jet)
    
    clean_data=[]
    for idx,data_ in enumerate(align_data):
        if idx==0:
            ref=data_[:220,1]
        else:
            ref=data_[:220,idx-1]
        clean=reference_guided.extract_first_wave_full_gather(data_, ref)
        clean_data.append(clean)
    
    restored_data=[]
    for aligned,shifts,pad in zip(clean_data,shifts_data,pad_data):
        restored = align_gather.restore_gather_pad(aligned, shifts, pad, nt_orig=1024, allow_frac=True)
        restored_data.append(restored)
    
    for data in restored_data:
        data=skimage.transform.resize(data,(config.k_max,rec_num))
        config.true_profile.append(data)

    irho=np.ones((config.xl+20,config.zl+20))*2000
    ivp=np.ones((config.xl+20,config.zl+20))*2200
    
    if method=='poynting':
        I=imaging_condition.imaging_condition(irho,ivp,'poynting')
        np.save('%s_poynting_rtm_result.npy'%index,I[0])
        np.save('%s_poynting_rtm_source.npy'%index,I[1])  
    elif method == 'correlation':
        I=imaging_condition.imaging_condition(irho,ivp,'correlation')
        np.save('%s_correlation_rtm_result.npy'%index,I[0])
        np.save('%s_correlation_rtm_source.npy'%index,I[1])  

if __name__=='__main__':
    for index in range(9,16):
        reverse_time_migration(config,index,rec_num=200,method='correlation')

  