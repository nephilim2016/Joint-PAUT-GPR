#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 06:03:20 2025

@author: nephilim
"""

from multiprocessing import Pool
import numpy as np
import time
import scipy.ndimage as ndi
from config_parameters import config
from add_cpml import Add_CPML
from wavelet_creation import WaveletType
import update_field_forward
import update_field_reverse 
    
def imaging_condition_correlation(rho,vp,index,CPML_Params):
    #Get Forward Params
    f=WaveletType(config.t, config.frequency, config.wavelet_type).create_wavelet()
    #Get Forward Data ----> <Generator>
    forward_generator=update_field_forward.time_loop(config.xl,config.zl,config.dx,config.dz,config.dt,
                                                     rho.copy(),vp.copy(),CPML_Params,f,config.k_max,
                                                     config.source_site[index])
    #Get Generator Data
    forward_field=[]
    for idx in range(config.k_max):
        tmp=forward_generator.__next__()
        forward_field.append(np.array(tmp[0]))
    

    #Get Reversion Data ----> <Generator>
    reverse_generator=update_field_reverse.reverse_time_loop(config.xl,config.zl,config.dx,config.dz,
                                                             config.dt,rho.copy(),vp.copy(),
                                                             CPML_Params,config.k_max,
                                                             config.receiver_site,config.true_profile[index])
    #Get Generator Data
    reverse_field=[]
    for idx in range(config.k_max):
        tmp=reverse_generator.__next__()
        reverse_field.append(np.array(tmp[0]))
    reverse_field.reverse()
    
    time_sum=np.zeros((config.xl+2*CPML_Params.npml,config.zl+2*CPML_Params.npml))
    time_sum_source=np.zeros((config.xl+2*CPML_Params.npml,config.zl+2*CPML_Params.npml))

    for k in range(config.k_max):
        f_f=forward_field[k]
        r_f=reverse_field[k]
        time_sum+=f_f*r_f
        time_sum_source+=f_f**2
        
    I_image=time_sum
    I_source=time_sum_source
    
    return I_image,I_source

def poynting_weight(u_now, u_prev, dx, dz, dt, smooth=3, eps=1e-12):
    gradx = (np.roll(u_now,-1,axis=1)-np.roll(u_now,1,axis=1))/(2*dx)
    gradz = (np.roll(u_now,-1,axis=0)-np.roll(u_now,1,axis=0))/(2*dz)
    dpt   = (u_now - u_prev) / dt
    Px, Pz = -dpt*gradx, -dpt*gradz
    if smooth>0:
        Px = ndi.gaussian_filter(Px, smooth, mode="nearest")
        Pz = ndi.gaussian_filter(Pz, smooth, mode="nearest")
    norm = np.sqrt(Px*Px + Pz*Pz) + eps
    cos_th = Pz / norm
    w_down = 0.5*(1 + cos_th)
    return w_down

def split_wavefield(u_now, u_prev, dx, dz, dt):
    w_down = poynting_weight(u_now, u_prev, dx, dz, dt)
    return w_down * u_now, (1-w_down) * u_now, w_down 

def imaging_condition_correlation_poynting(rho,vp,index,CPML_Params):
    #Get Forward Params
    f=WaveletType(config.t, config.frequency, config.wavelet_type).create_wavelet()
    #Get Forward Data ----> <Generator>
    forward_generator=update_field_forward.time_loop(config.xl,config.zl,config.dx,config.dz,config.dt,
                                                     rho.copy(),vp.copy(),CPML_Params,f,config.k_max,
                                                     config.source_site[index])
    #Get Generator Data
    forward_field=[]
    for idx in range(config.k_max):
        tmp=forward_generator.__next__()
        forward_field.append(np.array(tmp[0]))
    
    #Get Reversion Data ----> <Generator>
    reverse_generator=update_field_reverse.reverse_time_loop(config.xl,config.zl,config.dx,config.dz,
                                                             config.dt,rho.copy(),vp.copy(),
                                                             CPML_Params,config.k_max,
                                                             config.receiver_site,config.true_profile[index])
    #Get Generator Data
    reverse_field=[]
    for idx in range(config.k_max):
        tmp=reverse_generator.__next__()
        reverse_field.append(np.array(tmp[0]))
    reverse_field.reverse()
    
    time_sum_phys=np.zeros((config.xl+2*CPML_Params.npml,config.zl+2*CPML_Params.npml))
    illum=np.zeros((config.xl+2*CPML_Params.npml,config.zl+2*CPML_Params.npml))

    for k in range(config.k_max):
        f_f=forward_field[k]
        r_f=reverse_field[k]
        
        if k == 0:
            f_prev = np.zeros_like(f_f)
            r_prev = np.zeros_like(r_f)
        else:
            f_prev = forward_field[k-1]
            r_prev = reverse_field[k-1]        
        
        f_f_down, f_f_up, wS = split_wavefield(f_f, f_prev, config.dx, config.dz, config.dt)
        r_f_down, r_f_up, wR = split_wavefield(r_f, r_prev, config.dx, config.dz, config.dt)


        cos_prod = (wS-0.5)*(wR-0.5)*4        
        mask = cos_prod < -0.2             

        time_sum_phys += (f_f_down * r_f_up + f_f_up * r_f_down) * mask
        illum += f_f_down**2 + f_f_up**2

    return time_sum_phys, illum
            


def imaging_condition(irho,ivp,condition='correlation'): 
    start_time=time.time()  
    CPML_Params=Add_CPML(config.xl,config.zl,irho.copy(),ivp.copy(),config.dx,config.dz,config.dt)    
    
    pool=Pool(processes=128)
    res_l=[]
    if condition=='correlation':
        imaging_result=0
        source_result=0
        for index,value in enumerate(config.source_site):
            res=pool.apply_async(imaging_condition_correlation,args=(irho.copy(),ivp.copy(),index,CPML_Params))
            res_l.append(res)
        pool.close()
        pool.join()
    
        for res in res_l:
            result=res.get()
            imaging_result+=result[0]
            source_result+=result[1]
            del result
        pool.terminate() 
        print('Misfit elapsed time is %s seconds !'%str(time.time()-start_time))
        return imaging_result,source_result
    elif condition=='poynting':
        imaging_result=0
        source_result=0
        for index,value in enumerate(config.source_site):
            res=pool.apply_async(imaging_condition_correlation_poynting,args=(irho.copy(),ivp.copy(),index,CPML_Params))
            res_l.append(res)
        pool.close()
        pool.join()
    
        for res in res_l:
            result=res.get()
            imaging_result+=result[0]
            source_result+=result[1]
            del result
        pool.terminate() 
        print('Misfit elapsed time is %s seconds !'%str(time.time()-start_time))
        return imaging_result,source_result