#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 01:13:15 2025

@author: nephilim
"""

from numba import jit
import numpy as np

@jit(nopython=True)            
def update_uw(xl,zl,dx,dz,dt,rho,vp,npml,a_x,a_z,b_x,b_z,k_x,k_z,Uv,Wv,Pp,memory_dPp_dx,memory_dPp_dz):
    x_len=xl+2*npml
    z_len=zl+2*npml

    for j in range(1,z_len-1):
        for i in range(1,x_len-1):
            value_dPp_dx=(Pp[i+1][j]-Pp[i][j])/dx
                         
            if (i>=npml) and (i<x_len-npml):
                Uv[i][j]-=value_dPp_dx*dt/rho[i][j]
                
            elif i<npml:
                memory_dPp_dx[i][j]=b_x[i]*memory_dPp_dx[i][j]+a_x[i]*value_dPp_dx
                value_dPp_dx=value_dPp_dx/k_x[i]+memory_dPp_dx[i][j]
                Uv[i][j]-=value_dPp_dx*dt/rho[i][j]
                
            elif i>=x_len-npml:
                memory_dPp_dx[i-xl][j]=b_x[i]*memory_dPp_dx[i-xl][j]+a_x[i]*value_dPp_dx
                value_dPp_dx=value_dPp_dx/k_x[i]+memory_dPp_dx[i-xl][j]
                Uv[i][j]-=value_dPp_dx*dt/rho[i][j]
                      
    for j in range(1,z_len-1):
        for i in range(1,x_len-1):
            value_dPp_dz=(Pp[i][j+1]-Pp[i][j])/dz
                         
            if (j>=npml) and (j<z_len-npml):
                Wv[i][j]-=value_dPp_dz*dt/rho[i][j]
                
            elif j<npml:
                memory_dPp_dz[i][j]=b_z[j]*memory_dPp_dz[i][j]+a_z[j]*value_dPp_dz
                value_dPp_dz=value_dPp_dz/k_z[j]+memory_dPp_dz[i][j]
                Wv[i][j]-=value_dPp_dz*dt/rho[i][j]
                
            elif j>=z_len-npml:
                memory_dPp_dz[i][j-zl]=b_z[j]*memory_dPp_dz[i][j-zl]+a_z[j]*value_dPp_dz
                value_dPp_dz=value_dPp_dz/k_z[j]+memory_dPp_dz[i][j-zl]
                Wv[i][j]-=value_dPp_dz*dt/rho[i][j]

    return Uv,Wv

@jit(nopython=True)            
def update_p(xl,zl,dx,dz,dt,rho,vp,npml,a_x,a_z,b_x,b_z,k_x,k_z,Uv,Wv,Pp,memory_dUv_dx,memory_dWv_dz):
    x_len=xl+2*npml
    z_len=zl+2*npml
    kappa=rho*vp**2
    for j in range(1,z_len-1):
        for i in range(1,x_len-1):
            value_dUv_dx=(Uv[i][j]-Uv[i-1][j])/dx
         
            value_dWv_dz=(Wv[i][j]-Wv[i][j-1])/dz                        

            if (i>=npml) and (i<x_len-npml) and (j>=npml) and (j<z_len-npml):
                Pp[i][j]-=kappa[i][j]*(value_dUv_dx+value_dWv_dz)*dt
                
            elif (i<npml) and (j>=npml) and (j<z_len-npml):
                memory_dUv_dx[i][j]=b_x[i]*memory_dUv_dx[i][j]+a_x[i]*value_dUv_dx
                value_dUv_dx=value_dUv_dx/k_x[i]+memory_dUv_dx[i][j]
                Pp[i][j]-=kappa[i][j]*(value_dUv_dx+value_dWv_dz)*dt
                
            elif (i>=x_len-npml) and (j>=npml) and (j<z_len-npml):
                memory_dUv_dx[i-xl][j]=b_x[i]*memory_dUv_dx[i-xl][j]+a_x[i]*value_dUv_dx
                value_dUv_dx=value_dUv_dx/k_x[i]+memory_dUv_dx[i-xl][j]
                Pp[i][j]-=kappa[i][j]*(value_dUv_dx+value_dWv_dz)*dt
                
            elif (j<npml) and (i>=npml) and (i<x_len-npml):
                memory_dWv_dz[i][j]=b_z[j]*memory_dWv_dz[i][j]+a_z[j]*value_dWv_dz
                value_dWv_dz=value_dWv_dz/k_z[j]+memory_dWv_dz[i][j]
                Pp[i][j]-=kappa[i][j]*(value_dUv_dx+value_dWv_dz)*dt
                
            elif (j>=z_len-npml) and (i>=npml) and (i<x_len-npml):
                memory_dWv_dz[i][j-zl]=b_z[j]*memory_dWv_dz[i][j-zl]+a_z[j]*value_dWv_dz
                value_dWv_dz=value_dWv_dz/k_z[j]+memory_dWv_dz[i][j-zl]
                Pp[i][j]-=kappa[i][j]*(value_dUv_dx+value_dWv_dz)*dt
                
            elif (i<npml) and (j<npml):
                memory_dUv_dx[i][j]=b_x[i]*memory_dUv_dx[i][j]+a_x[i]*value_dUv_dx
                value_dUv_dx=value_dUv_dx/k_x[i]+memory_dUv_dx[i][j]
                
                memory_dWv_dz[i][j]=b_z[j]*memory_dWv_dz[i][j]+a_z[j]*value_dWv_dz
                value_dWv_dz=value_dWv_dz/k_z[j]+memory_dWv_dz[i][j]
                
                Pp[i][j]-=kappa[i][j]*(value_dUv_dx+value_dWv_dz)*dt
                
            elif (i<npml) and (j>=z_len-npml):
                memory_dUv_dx[i][j]=b_x[i]*memory_dUv_dx[i][j]+a_x[i]*value_dUv_dx
                value_dUv_dx=value_dUv_dx/k_x[i]+memory_dUv_dx[i][j]
                
                memory_dWv_dz[i][j-zl]=b_z[j]*memory_dWv_dz[i][j-zl]+a_z[j]*value_dWv_dz
                value_dWv_dz=value_dWv_dz/k_z[j]+memory_dWv_dz[i][j-zl]
                
                Pp[i][j]-=kappa[i][j]*(value_dUv_dx+value_dWv_dz)*dt
                
            elif (i>=x_len-npml) and (j<npml):
                memory_dUv_dx[i-xl][j]=b_x[i]*memory_dUv_dx[i-xl][j]+a_x[i]*value_dUv_dx
                value_dUv_dx=value_dUv_dx/k_x[i]+memory_dUv_dx[i-xl][j]
                
                memory_dWv_dz[i][j]=b_z[j]*memory_dWv_dz[i][j]+a_z[j]*value_dWv_dz
                value_dWv_dz=value_dWv_dz/k_z[j]+memory_dWv_dz[i][j]
               
                Pp[i][j]-=kappa[i][j]*(value_dUv_dx+value_dWv_dz)*dt
                
            elif (i>=x_len-npml) and (j>=z_len-npml):
                memory_dUv_dx[i-xl][j]=b_x[i]*memory_dUv_dx[i-xl][j]+a_x[i]*value_dUv_dx
                value_dUv_dx=value_dUv_dx/k_x[i]+memory_dUv_dx[i-xl][j]
                
                memory_dWv_dz[i][j-zl]=b_z[j]*memory_dWv_dz[i][j-zl]+a_z[j]*value_dWv_dz
                value_dWv_dz=value_dWv_dz/k_z[j]+memory_dWv_dz[i][j-zl]
                
                Pp[i][j]-=kappa[i][j]*(value_dUv_dx+value_dWv_dz)*dt
    return Pp

#Reverse modelling ------ timeloop
def reverse_time_loop(xl,zl,dx,dz,dt,rho,vp,CPML_Params,k_max,ref_pos,record):
    npml=CPML_Params.npml        
    Uv=np.zeros((xl+2*npml,zl+2*npml))
    Wv=np.zeros((xl+2*npml,zl+2*npml))
    Pp=np.zeros((xl+2*npml,zl+2*npml))
      
    memory_dPp_dx=np.zeros((2*npml,zl+2*npml))
    memory_dPp_dz=np.zeros((xl+2*npml,2*npml))
    memory_dUv_dx=np.zeros((2*npml,zl+2*npml))
    memory_dWv_dz=np.zeros((xl+2*npml,2*npml))
    
    a_x=CPML_Params.a_x
    b_x=CPML_Params.b_x
    k_x=CPML_Params.k_x
    a_z=CPML_Params.a_z
    b_z=CPML_Params.b_z
    k_z=CPML_Params.k_z
    a_x_half=CPML_Params.a_x_half
    b_x_half=CPML_Params.b_x_half
    k_x_half=CPML_Params.k_x_half
    a_z_half=CPML_Params.a_z_half
    b_z_half=CPML_Params.b_z_half
    k_z_half=CPML_Params.k_z_half
    
    receiver_array = np.array(ref_pos)
    receiver_site_x = receiver_array[:, 0]
    receiver_site_z = receiver_array[:, 1]

            
    for tt in range(k_max):
        Pp[receiver_site_x,receiver_site_z]+=record[k_max-tt-1,:]
        Uv,Wv=update_uw(xl,zl,dx,dz,dt,rho,vp,npml,a_x_half,a_z_half,b_x_half,b_z_half,k_x_half,k_z_half,Uv,Wv,Pp,memory_dPp_dx,memory_dPp_dz)
        Pp=update_p(xl,zl,dx,dz,dt,rho,vp,npml,a_x,a_z,b_x,b_z,k_x,k_z,Uv,Wv,Pp,memory_dUv_dx,memory_dWv_dz)
        # pyplot.imshow(Ey,vmin=-50,vmax=50)
        # pyplot.pause(0.01)
        yield Pp.tolist(),