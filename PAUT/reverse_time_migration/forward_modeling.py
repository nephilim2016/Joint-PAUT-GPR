#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 01:53:44 2025

@author: nephilim
"""

import numpy as np
from multiprocessing import Pool
from add_cpml import Add_CPML
from wavelet_creation import WaveletType
import shutil
import os
import update_field_forward
from config_parameters import config

class ForwardModeling(Add_CPML,WaveletType):
    def __init__(self,rho,vp,wavelet_type,freq):
        self.rho=rho
        self.vp=vp
        self.wavelet_type=wavelet_type
        self.freq=freq
        # Initialize the WaveletType part of this class
        WaveletType.__init__(self, config.t, freq, wavelet_type)
        self.f = self.create_wavelet()  # Use the inherited method to create the wavelet
        # Initialize the AddCPML part of this class
        Add_CPML.__init__(self, config.xl, config.zl, self.rho, self.vp, config.dx, config.dz, config.dt)    
    #Forward modelling ------ timeloop
    def _time_loop(self,value_source,value_receiver):
        Uv=np.zeros((config.xl+2*self.npml,config.zl+2*self.npml))
        Wv=np.zeros((config.xl+2*self.npml,config.zl+2*self.npml))
        Pp=np.zeros((config.xl+2*self.npml,config.zl+2*self.npml))
            
        memory_dPp_dx=np.zeros((2*self.npml,config.zl+2*self.npml))
        memory_dPp_dz=np.zeros((config.xl+2*self.npml,2*self.npml))
        memory_dUv_dx=np.zeros((2*self.npml,config.zl+2*self.npml))
        memory_dWv_dz=np.zeros((config.xl+2*self.npml,2*self.npml))
        
        record=np.zeros((config.k_max,len(value_receiver)))
        receiver_array = np.array(config.receiver_site)
        receiver_site_x = receiver_array[:, 0]
        receiver_site_z = receiver_array[:, 1]

        for tt in range(config.k_max):
            Uv,Wv=update_field_forward.update_uw(config.xl,config.zl,config.dx,config.dz,config.dt,
                                                self.rho,self.vp,self.npml,
                                                self.a_x_half,self.a_z_half,
                                                self.b_x_half,self.b_z_half,
                                                self.k_x_half,self.k_z_half,
                                                Uv,Wv,Pp,memory_dPp_dx,memory_dPp_dz)
            
            Pp=update_field_forward.update_p(config.xl,config.zl,config.dx,config.dz,config.dt,
                                             self.rho,self.vp,self.npml,self.a_x,self.a_z,
                                             self.b_x,self.b_z,self.k_x,self.k_z,
                                             Uv,Wv,Pp,memory_dUv_dx,memory_dWv_dz)
            Pp[value_source[0]][value_source[1]]+=self.f[tt]
            record[tt,:]=Pp[receiver_site_x,receiver_site_z]
            # pyplot.imshow(Ey,vmin=-50,vmax=50)
            # pyplot.pause(0.01)
        return value_source[1],np.array(record)
        
    def _forward_2d(self):
        #Create Folder
        if not os.path.exists('./%sHz_forward_data_file'%self.freq):
            os.makedirs('./%sHz_forward_data_file'%self.freq)
        else:
            shutil.rmtree('./%sHz_forward_data_file'%self.freq)
            os.makedirs('./%sHz_forward_data_file'%self.freq)
        pool=Pool(processes=128)
        res_l=[]
        for source_position in config.source_site:
            res=pool.apply_async(self._time_loop,args=(source_position,config.receiver_site))
            res_l.append(res)
        pool.close()
        pool.join()
        for res in res_l:
            result=res.get()
            np.save('./%sHz_forward_data_file/%s_record.npy'%(self.freq,result[0]),result[1])
            del result
        del res_l
        pool.terminate() 
        
    def _forward_2d_background(self):
        #Create Folder
        if not os.path.exists('./%sHz_forward_back_data_file'%self.freq):
            os.makedirs('./%sHz_forward_back_data_file'%self.freq)
        else:
            shutil.rmtree('./%sHz_forward_back_data_file'%self.freq)
            os.makedirs('./%sHz_forward_back_data_file'%self.freq)
        pool=Pool(processes=128)
        res_l=[]
        for source_position in config.source_site:
            res=pool.apply_async(self._time_loop,args=(source_position,config.receiver_site))
            res_l.append(res)
        pool.close()
        pool.join()
        for res in res_l:
            result=res.get()
            np.save('./%sHz_forward_back_data_file/%s_back_record.npy'%(self.freq,result[0]),result[1])
            del result
        del res_l
        pool.terminate() 

