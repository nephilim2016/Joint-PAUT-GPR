#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:52:32 2018

@author: nephilim
"""
import numpy as np

#Add CPML condition
class Add_CPML():
    def __init__(self,xl,zl,sigma,epsilon,dx,dz,dt):
        self.npml=10
        self.Npower=4
        self.k_max_CPML=5
        self.alpha_max_CPML=0.008
        self.Rcoef=1e-8
        self.x_len=xl+2*self.npml
        self.z_len=zl+2*self.npml      
        self.k_x=np.ones(self.x_len+0)
        self.a_x=np.zeros(self.x_len+0)
        self.b_x=np.zeros(self.x_len+0)
        self.k_x_half=np.ones(self.x_len+0)
        self.a_x_half=np.zeros(self.x_len+0)
        self.b_x_half=np.zeros(self.x_len+0)
        
        self.k_z=np.ones(self.z_len+0)
        self.a_z=np.zeros(self.z_len+0)
        self.b_z=np.zeros(self.z_len+0)
        self.k_z_half=np.ones(self.z_len+0)
        self.a_z_half=np.zeros(self.z_len+0)
        self.b_z_half=np.zeros(self.z_len+0)
        self.add_CPML(sigma,epsilon,dx,dz,dt)    
    
    def add_CPML(self,sigma,epsilon,dx,dz,dt):
        ep0 = 8.841941282883074e-12
        sig_x_tmp=np.zeros(self.x_len)
        k_x_tmp=np.ones(self.x_len)
        alpha_x_tmp=np.zeros(self.x_len)
        a_x_tmp=np.zeros(self.x_len)
        b_x_tmp=np.zeros(self.x_len)
        sig_x_half_tmp=np.zeros(self.x_len)
        k_x_half_tmp=np.ones(self.x_len)
        alpha_x_half_tmp=np.zeros(self.x_len)
        a_x_half_tmp=np.zeros(self.x_len)
        b_x_half_tmp=np.zeros(self.x_len)
        
        sig_z_tmp=np.zeros(self.z_len)
        k_z_tmp=np.ones(self.z_len)
        alpha_z_tmp=np.zeros(self.z_len)
        a_z_tmp=np.zeros(self.z_len)
        b_z_tmp=np.zeros(self.z_len)
        sig_z_half_tmp=np.zeros(self.z_len)
        k_z_half_tmp=np.ones(self.z_len)
        alpha_z_half_tmp=np.zeros(self.z_len)
        a_z_half_tmp=np.zeros(self.z_len)
        b_z_half_tmp=np.zeros(self.z_len)
        
        thickness_CPML_x=self.npml*dx
        thickness_CPML_z=self.npml*dz
        sig0_x= (self.Npower+1)/(150*np.pi*dx)
        sig0_z= (self.Npower+1)/(150*np.pi*dz)
        
        xoriginleft=thickness_CPML_x
        xoriginright=(self.x_len-1)*dx-thickness_CPML_x
        for i in range(self.x_len):
            xval=dx*i
            abscissa_in_CPML=xoriginleft-xval
            if abscissa_in_CPML>=0:
                abscissa_normalized=abscissa_in_CPML/thickness_CPML_x
                sig_x_tmp[i]=sig0_x*abscissa_normalized**self.Npower
                k_x_tmp[i]=1+(self.k_max_CPML-1)*abscissa_normalized**self.Npower
                alpha_x_tmp[i]=self.alpha_max_CPML*(1-abscissa_normalized)+0.1*self.alpha_max_CPML
                
            abscissa_in_CPML=xoriginleft-xval-dx/2
            if abscissa_in_CPML>=0:
                abscissa_normalized=abscissa_in_CPML/thickness_CPML_x
                sig_x_half_tmp[i]=sig0_x*abscissa_normalized**self.Npower
                k_x_half_tmp[i]=1+(self.k_max_CPML-1)*abscissa_normalized**self.Npower
                alpha_x_half_tmp[i]=self.alpha_max_CPML*(1-abscissa_normalized)+0.1*self.alpha_max_CPML
                
            abscissa_in_CPML=xval-xoriginright
            if abscissa_in_CPML>=0:
                abscissa_normalized=abscissa_in_CPML/thickness_CPML_x
                sig_x_tmp[i]=sig0_x*abscissa_normalized**self.Npower
                k_x_tmp[i]=1+(self.k_max_CPML-1)*abscissa_normalized**self.Npower
                alpha_x_tmp[i]=self.alpha_max_CPML*(1-abscissa_normalized)+0.1*self.alpha_max_CPML
            
            abscissa_in_CPML=xval+dx/2-xoriginright
            if abscissa_in_CPML>=0:
                abscissa_normalized=abscissa_in_CPML/thickness_CPML_x
                sig_x_half_tmp[i]=sig0_x*abscissa_normalized**self.Npower
                k_x_half_tmp[i]=1+(self.k_max_CPML-1)*abscissa_normalized**self.Npower
                alpha_x_half_tmp[i]=self.alpha_max_CPML*(1-abscissa_normalized)+0.1*self.alpha_max_CPML
    
            b_x_tmp[i]=np.exp(-(sig_x_tmp[i]/k_x_tmp[i]+alpha_x_tmp[i])*dt/ep0)
            b_x_half_tmp[i]=np.exp(-(sig_x_half_tmp[i]/k_x_half_tmp[i]+alpha_x_half_tmp[i])*dt/ep0)
            if abs(sig_x_tmp[i]>1e-6):
                a_x_tmp[i]=sig_x_tmp[i]*(b_x_tmp[i]-1)/(k_x_tmp[i]*(sig_x_tmp[i]+k_x_tmp[i]*alpha_x_tmp[i]))
            if abs(sig_x_half_tmp[i]>1e-6):
                a_x_half_tmp[i]=sig_x_half_tmp[i]*(b_x_half_tmp[i]-1)/(k_x_half_tmp[i]*(sig_x_half_tmp[i]+k_x_half_tmp[i]*alpha_x_half_tmp[i]))

        zoriginbottom=thickness_CPML_z
        zorigintop=(self.z_len-1)*dz-thickness_CPML_z
        for i in range(self.z_len):
            zval=dz*i
            abscissa_in_CPML=zoriginbottom-zval
            if abscissa_in_CPML>=0:
                abscissa_normalized=abscissa_in_CPML/thickness_CPML_z
                sig_z_tmp[i]=sig0_z*abscissa_normalized**self.Npower
                k_z_tmp[i]=1+(self.k_max_CPML-1)*abscissa_normalized**self.Npower
                alpha_z_tmp[i]=self.alpha_max_CPML*(1-abscissa_normalized)+0.1*self.alpha_max_CPML
            
            abscissa_in_CPML=zoriginbottom-zval-dz/2
            if abscissa_in_CPML>=0:
                abscissa_normalized=abscissa_in_CPML/thickness_CPML_z
                sig_z_half_tmp[i]=sig0_z*abscissa_normalized**self.Npower
                k_z_half_tmp[i]=1+(self.k_max_CPML-1)*abscissa_normalized**self.Npower
                alpha_z_half_tmp[i]=self.alpha_max_CPML*(1-abscissa_normalized)+0.1*self.alpha_max_CPML
    
            abscissa_in_CPML=zval-zorigintop
            if abscissa_in_CPML>=0:
                abscissa_normalized=abscissa_in_CPML/thickness_CPML_z
                sig_z_tmp[i]=sig0_z*abscissa_normalized**self.Npower
                k_z_tmp[i]=1+(self.k_max_CPML-1)*abscissa_normalized**self.Npower
                alpha_z_tmp[i]=self.alpha_max_CPML*(1-abscissa_normalized)+0.1*self.alpha_max_CPML
            
            abscissa_in_CPML=zval+dz/2-zorigintop
            if abscissa_in_CPML>=0:
                abscissa_normalized=abscissa_in_CPML/thickness_CPML_z
                sig_z_half_tmp[i]=sig0_z*abscissa_normalized**self.Npower
                k_z_half_tmp[i]=1+(self.k_max_CPML-1)*abscissa_normalized**self.Npower
                alpha_z_half_tmp[i]=self.alpha_max_CPML*(1-abscissa_normalized)+0.1*self.alpha_max_CPML
    
            b_z_tmp[i]=np.exp(-(sig_z_tmp[i]/k_z_tmp[i]+alpha_z_tmp[i])*dt/ep0)
            b_z_half_tmp[i]=np.exp(-(sig_z_half_tmp[i]/k_z_half_tmp[i]+alpha_z_half_tmp[i])*dt/ep0)
            if abs(sig_z_tmp[i]>1e-6):
                a_z_tmp[i]=sig_z_tmp[i]*(b_z_tmp[i]-1)/(k_z_tmp[i]*(sig_z_tmp[i]+k_z_tmp[i]*alpha_z_tmp[i]))
            if abs(sig_z_half_tmp[i]>1e-6):
                a_z_half_tmp[i]=sig_z_half_tmp[i]*(b_z_half_tmp[i]-1)/(k_z_half_tmp[i]*(sig_z_half_tmp[i]+k_z_half_tmp[i]*alpha_z_half_tmp[i]))
        epsilon*=ep0
        ca=(1-sigma*dt/2/epsilon)/(1+sigma*dt/2/epsilon)
        cb=1/epsilon/(1+sigma*dt/2/epsilon)
        ca_r=(epsilon*2)/(epsilon*2+sigma*dt)
        self.a_x=a_x_tmp
        self.b_x=b_x_tmp
        self.k_x=k_x_tmp
        self.a_z=a_z_tmp
        self.b_z=b_z_tmp
        self.k_z=k_z_tmp
        self.a_x_half=a_x_half_tmp
        self.b_x_half=b_x_half_tmp
        self.k_x_half=k_x_half_tmp
        self.a_z_half=a_z_half_tmp
        self.b_z_half=b_z_half_tmp
        self.k_z_half=k_z_half_tmp
        self.ca=ca
        self.cb=cb
        self.ca_r=ca_r
        self.mu=np.ones_like(epsilon)*1.2566370614359173e-06
