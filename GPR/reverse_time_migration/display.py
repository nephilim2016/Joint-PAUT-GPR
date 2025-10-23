#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 16:45:42 2025

@author: nephilim
"""
import numpy as np
from matplotlib import pyplot,cm
from scipy.signal import hilbert

if __name__=='__main__':
    img=np.load('correlation_rtm_result.npy')
    ilu=np.load('correlation_rtm_source.npy')
    img=img[10:-10,10:-10]
    ilu=ilu[10:-10,10:-10]
    
    #%%
    pyplot.figure(figsize=(8,6))
    pyplot.rcParams.update({'font.size': 15})
    gci=pyplot.imshow(img,extent=(0,500,600,0),cmap=cm.gray,vmin=-1e6,vmax=1e6)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,500,6))
    ax.set_xticklabels([0,0.2,0.4,0.6,0.8,1.0])
    ax.set_yticks(np.linspace(0,600,7))
    ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0,1.2])
    pyplot.xlabel('Distance (m)')
    pyplot.ylabel('Depth (m)')
    pyplot.savefig('correlation_rtm.png',dpi=1000)
    #%%  
    
    env = np.abs(hilbert(img/ilu, axis=0))
    np.save('env_gc.npy',env)
    
    #%%
    pyplot.figure(figsize=(8,6))
    pyplot.rcParams.update({'font.size': 15})
    gci=pyplot.imshow(env,extent=(0,500,600,0),cmap=cm.jet)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,500,6))
    ax.set_xticklabels([0,0.2,0.4,0.6,0.8,1.0])
    ax.set_yticks(np.linspace(0,600,7))
    ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0,1.2])
    pyplot.xlabel('Distance (m)')
    pyplot.ylabel('Depth (m)')
    pyplot.savefig('correlation_env.png',dpi=1000)
    #%%
    
    img=np.load('poynting_rtm_result.npy')
    ilu=np.load('poynting_rtm_source.npy')
    img=img[10:-10,10:-10]
    ilu=ilu[10:-10,10:-10]
    
    env = np.abs(hilbert(img/ilu, axis=0))
    np.save('env_gp.npy',env)

    #%%
    pyplot.figure(figsize=(8,6))
    pyplot.rcParams.update({'font.size': 15})
    gci=pyplot.imshow(img,extent=(0,500,600,0),cmap=cm.gray,vmin=-1e6,vmax=1e6)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,500,6))
    ax.set_xticklabels([0,0.2,0.4,0.6,0.8,1.0])
    ax.set_yticks(np.linspace(0,600,7))
    ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0,1.2])
    pyplot.xlabel('Distance (m)')
    pyplot.ylabel('Depth (m)')
    pyplot.savefig('poynting_rtm.png',dpi=1000)
    #%%  
    
    env = np.abs(hilbert(img/ilu, axis=0))
    np.save('env_gc.npy',env)
    
    #%%
    pyplot.figure(figsize=(8,6))
    pyplot.rcParams.update({'font.size': 15})
    gci=pyplot.imshow(env,extent=(0,500,600,0),cmap=cm.jet)
    ax=pyplot.gca()
    ax.set_xticks(np.linspace(0,500,6))
    ax.set_xticklabels([0,0.2,0.4,0.6,0.8,1.0])
    ax.set_yticks(np.linspace(0,600,7))
    ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0,1.2])
    pyplot.xlabel('Distance (m)')
    pyplot.ylabel('Depth (m)')
    pyplot.savefig('poynting_env.png',dpi=1000)
    
    # img=np.load('explosion_rtm_result.npy')
    # img=img[20:,10:-10]
    # #%%
    # pyplot.figure(figsize=(8,6))
    # pyplot.rcParams.update({'font.size': 15})
    # gci=pyplot.imshow(img,extent=(0,500,600,0),cmap=cm.gray)
    # ax=pyplot.gca()
    # ax.set_xticks(np.linspace(0,500,6))
    # ax.set_xticklabels([0,0.2,0.4,0.6,0.8,1.0])
    # ax.set_yticks(np.linspace(0,600,7))
    # ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0,1.2])
    # pyplot.xlabel('Distance (m)')
    # pyplot.ylabel('Depth (m)')
    # pyplot.savefig('poynting_rtm.png',dpi=1000)
    # #%%    
    # env = np.abs(hilbert(img, axis=0))
    # np.save('env_ge.npy',env)

    # #%%
    # pyplot.figure(figsize=(8,6))
    # pyplot.rcParams.update({'font.size': 15})
    # gci=pyplot.imshow(env,extent=(0,500,600,0),cmap=cm.jet)
    # ax=pyplot.gca()
    # ax.set_xticks(np.linspace(0,500,6))
    # ax.set_xticklabels([0,0.2,0.4,0.6,0.8,1.0])
    # ax.set_yticks(np.linspace(0,600,7))
    # ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0,1.2])
    # pyplot.xlabel('Distance (m)')
    # pyplot.ylabel('Depth (m)')
    # pyplot.savefig('poynting_env.png',dpi=1000)
    # #%%