#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 01:53:44 2025

@author: nephilim
"""

import numpy as np
from matplotlib import pyplot
from multiprocessing import Pool
from add_cpml import Add_CPML
from wavelet_creation import WaveletType
import time
import shutil
import os
import update_field_forward
from config_parameters import config

class ForwardModeling(Add_CPML,WaveletType):
    def __init__(self,sigma,epsilon,wavelet_type,Freq):
        self.sigma=sigma
        self.epsilon=epsilon
        self.wavelet_type=wavelet_type
        self.Freq=Freq
        # Initialize the WaveletType part of this class
        WaveletType.__init__(self, config.t, Freq, wavelet_type)
        self.f = self.create_wavelet()  # Use the inherited method to create the wavelet
        # Initialize the AddCPML part of this class
        Add_CPML.__init__(self, config.xl, config.zl, self.sigma, self.epsilon, config.dx, config.dz, config.dt)    
    #Forward modelling ------ timeloop
    def _time_loop(self,value_source,value_receiver):
        Ey=np.zeros((config.xl+2*self.npml,config.zl+2*self.npml))
        Hz=np.zeros((config.xl+2*self.npml,config.zl+2*self.npml))
        Hx=np.zeros((config.xl+2*self.npml,config.zl+2*self.npml))
            
        memory_dEy_dx=np.zeros((2*self.npml,config.zl+2*self.npml))
        memory_dEy_dz=np.zeros((config.xl+2*self.npml,2*self.npml))
        memory_dHz_dx=np.zeros((2*self.npml,config.zl+2*self.npml))
        memory_dHx_dz=np.zeros((config.xl+2*self.npml,2*self.npml))
        

        for tt in range(config.k_max):
            Hz,Hx=update_field_forward.update_H(config.xl,config.zl,config.dx,config.dz,config.dt,
                                                self.sigma,self.epsilon,self.mu,self.npml,
                                                self.a_x_half,self.a_z_half,
                                                self.b_x_half,self.b_z_half,
                                                self.k_x_half,self.k_z_half,
                                                Hz,Hx,Ey,memory_dEy_dx,memory_dEy_dz)
            
            Ey=update_field_forward.update_E(config.xl,config.zl,config.dx,config.dz,config.dt,
                                             self.ca,self.cb,self.npml,self.a_x,self.a_z,
                                             self.b_x,self.b_z,self.k_x,self.k_z,
                                             Hz,Hx,Ey,memory_dHz_dx,memory_dHx_dz)
            Ey[value_source[0]][value_source[1]]+=-self.cb[value_source[0]][value_source[1]]*self.f[tt]*config.dt/config.dx/config.dz
            # pyplot.imshow(Ey,vmin=-50,vmax=50)
            # pyplot.pause(0.01)
            yield Ey[value_receiver[0],value_receiver[1]],

    def _forward_2d(self):
        #Create Folder
        if not os.path.exists('./%sHz_forward_data_file'%self.Freq):
            os.makedirs('./%sHz_forward_data_file'%self.Freq)
        else:
            shutil.rmtree('./%sHz_forward_data_file'%self.Freq)
            os.makedirs('./%sHz_forward_data_file'%self.Freq)
        pool=Pool(processes=128)
        profile=np.empty((config.k_max,len(config.source_site)))
        res_l=[]
        for index,data_position in enumerate(zip(config.source_site,config.receiver_site)):
            res=pool.apply_async(self._forward2d,args=(data_position[0],data_position[1],index))
            res_l.append(res)
        pool.close()
        pool.join()
        for res in res_l:
            result=res.get()
            profile[:,result[0]]=result[1]
            del result
        del res_l
        pool.terminate() 
        np.save('./%sHz_forward_data_file/record.npy'%self.Freq,profile)
        return profile

    def _forward2d(self,value_source,value_receiver,index):
        forward_data=self._time_loop(value_source,value_receiver)
        profile=np.empty((config.k_max))
        for idx in range(config.k_max):
            tmp=forward_data.__next__()
            profile[idx]=tmp[0]
        return index,profile
    
    def _forward_2d_air(self):
        Proifle=self._forward2d_air(config.source_site[0],config.receiver_site[0])
        return Proifle

    def _forward2d_air(self,value_source,value_receiver):
        forward_data=self._time_loop(value_source,value_receiver)
        profile=np.empty((config.k_max))
        for idx in range(config.k_max):
            tmp=forward_data.__next__()
            profile[idx]=tmp[0]
        return profile
