#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 22:35:20 2025

@author: nephilim
"""

import numpy as np
from typing import Tuple
from scipy.signal import hilbert
from matplotlib import pyplot,cm
import scipy.io as scio
import align_gather
import reference_guided
import reference_rank_svd 
from T_PowerGain import tpowGain
import skimage.transform
from wigb import wiggle

def kirchhoff_prestack(data: np.ndarray,     # (ns,nr,nt)
                       shots: np.ndarray,    # (ns,)
                       recs: np.ndarray,     # (nr,)
                       t: np.ndarray,        # (nt,)
                       v: float,
                       aperture: float | None = None,
                       weight: bool = True
                       ) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Constant-velocity 2-D pre-stack Kirchhoff migration."""
    ns,nr,nt = data.shape
    dt = t[1] - t[0]

    x_img = recs.copy()
    z_img = v * t / 2.0 
    nx, nz = len(x_img), len(z_img)
    image  = np.zeros((nz,nx), np.float32)

    Xr = recs[:,None,None]
    Xi = x_img[None,:]
    Zi = z_img[:,None]
    tr_tab = np.sqrt((Xr-Xi)**2 + Zi**2)/v

    full_mask = np.ones((1,nx), bool)

    for ishot, xs in enumerate(shots):
        ts = np.sqrt((xs - Xi)**2 + Zi**2)/v     # (nz,nx)
        mask_line = full_mask if aperture is None else (np.abs(x_img-xs)<=aperture)[None,:]

        for ir in range(nr):
            ttot   = ts + tr_tab[ir]             # (nz,nx)
            it_int = np.round(ttot/dt).astype(int)
            valid  = (it_int>=0)&(it_int<nt)&mask_line

            if not np.any(valid):
                continue

            trace = data[ishot, ir]
            samp  = np.zeros_like(it_int, np.float32)
            samp[valid] = trace[it_int[valid]]

            if weight:
                w = 1.0/np.sqrt(np.maximum(ts*tr_tab[ir],1e-9))
                samp *= w
            image += samp
    image /= ns
    return image, x_img, z_img


if __name__=='__main__':
    index=15
    data_=scio.loadmat('../%s_1.mat'%index)
    data_=data_['data_all'].T
    data_=data_.astype('float')
    
    data_tmp=[]
    for idx in range(16):
        data=data_[:1024,16*idx:16*(idx+1)]
        data_tmp.append(data)
    
    pyplot.figure()
    pyplot.imshow(data_tmp[0],vmin=-50,vmax=50,extent=(0,1,3,0),cmap=cm.jet)
    
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
        
    # data=align_data[1]
    # pyplot.figure()
    # pyplot.imshow(data,extent=(0,3,1,0),vmin=np.min(data)/2,vmax=np.max(data)/2,cmap=cm.jet)
    
    
    clean_data=[]
    for idx,data_ in enumerate(align_data):
        if idx==0:
            REF_ID=1
        else:
            REF_ID=idx-1
        clean=reference_rank_svd.extract_first_wave_full_gather(data_, REF_ID, WIN_LEN=240, GUARD=60)
        clean_data.append(clean)
    
    data=clean_data[1]
    pyplot.figure()
    pyplot.imshow(data,extent=(0,1,3,0),vmin=np.min(data)/2,vmax=np.max(data)/2,cmap=cm.jet)
    
    # clean_data=[]
    # for idx,data_ in enumerate(align_data):
    #     if idx==0:
    #         ref=data_[:220,1]
    #     else:
    #         ref=data_[:220,idx-1]
    #     clean=reference_guided.extract_first_wave_full_gather(data_, ref)
    #     clean_data.append(clean)
    
    # data=clean_data[1]
    # pyplot.figure()
    # pyplot.imshow(data,extent=(0,1,3,0),vmin=np.min(data)/2,vmax=np.max(data)/2,cmap=cm.jet)
    
    restored_data=[]
    for aligned,shifts,pad in zip(clean_data,shifts_data,pad_data):
        restored = align_gather.restore_gather_pad(aligned, shifts, pad, nt_orig=1024, allow_frac=True)
        restored_data.append(restored)
        
    data=restored_data[1]
    pyplot.figure()
    pyplot.imshow(data,extent=(0,1,3,0),vmin=np.min(data)/2,vmax=np.max(data)/2,cmap=cm.jet)
    
    rec_num=32
    data_clean=[]
    for data in restored_data:
        data=skimage.transform.resize(data,(1024,rec_num))
        data_clean.append(data.T)
    data_clean=np.array(data_clean)

    # recsx  = np.array([rc[1] for rc in config.receiver_site]) * config.dx
    # shotsx = np.arange(10, config.zl + 10, 10)*2e-3
    recsx  = np.linspace(0, 0.4, rec_num)
    shotsx = np.linspace(0, 0.4, 16)
    
    # t_axis = config.t 
    t_axis = np.linspace(0, 1e-6*1024, 1024)
    v_const = 2.2e3 
    dt = t_axis[1]-t_axis[0]
    f0 = 5e4
    # data_clean = wedge_mute_auto(data,
    #                              shotsx, recsx,
    #                              v_mute=v_const,
    #                              dt=dt,
    #                              f0=f0,
    #                              gamma=4.0,
    #                              N_cycle=4,
    #                              M_cont=5)
    

    
    img, xi, zi = kirchhoff_prestack(data_clean, shotsx, recsx, t_axis, v_const, aperture=1)

    # np.save('./%s_kirchhoff.npy'%index,img)
    # # pyplot.figure()
    # # pyplot.imshow(data[0].T,extent=[0,1,0.4,0], cmap='jet')
    
    pyplot.figure()
    pyplot.imshow(data_clean[0].T,extent=[0,1,0.4,0], cmap='jet')
    
    pyplot.figure()
    pyplot.imshow(img, extent=[0,40,120,0], cmap='jet')
    
    env = np.abs(hilbert(img, axis=0))
    pyplot.figure()
    pyplot.imshow(env, extent=[0,40,120,0], cmap='jet')
    # # pyplot.savefig('kirchhoff_m.png',dpi=1000)
    
    # env = np.abs(hilbert(img, axis=0))
    # pyplot.figure()
    # pyplot.imshow(env, extent=[0,40,120,0], cmap='jet')
    # # pyplot.savefig('kirchhoff.png',dpi=1000)

    