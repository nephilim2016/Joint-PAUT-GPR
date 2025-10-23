#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 07:06:54 2025

@author: nephilim
"""

import numpy as np
from scipy.signal import hilbert
import skimage.transform
from matplotlib import pyplot as plt, cm

def tukey(n, alpha=0.1):
    w = np.ones(n)
    x = np.arange(n)/(n-1)
    first = x < alpha/2
    last  = x > 1-alpha/2
    w[first] = .5*(1+np.cos(2*np.pi*(x[first]/alpha - .5)))
    w[last]  = .5*(1+np.cos(2*np.pi*((x[last]-1)/alpha + .5)))
    return w

def stack_with_taper(blocks, step, alpha=0.2):
    """
    用汉宁窗对每个 block 做水平 taper，再叠加并归一化。
    blocks: list of 2D arrays, shape = (nz, nx_block)
    step  : 水平移动步长
    """
    nz, nxb = blocks[0].shape
    nblocks = len(blocks)
    total_length = (nblocks-1)*step + nxb
    img_sum    = np.zeros((nz, total_length), dtype=float)
    weight_sum = np.zeros((nz, total_length), dtype=float)

    # 生成汉宁窗，一端至另一端都是 0 → 1 → 0
    # w1d = np.hanning(nxb)[None, :]      # shape (1, nxb)
    w1d    = tukey(nxb, alpha=alpha)[None, :]
    W2d = np.tile(w1d, (nz, 1))         # shape (nz, nxb)

    for i, blk in enumerate(blocks):
        start = i * step
        img_sum[:, start:start+nxb]    += blk * W2d
        weight_sum[:, start:start+nxb] += W2d

    # 防止除零
    mask = weight_sum <= 0
    weight_sum[mask] = 1.0

    img_fused = img_sum / weight_sum
    return img_fused

if __name__ == '__main__':
    zl = 1200
    xl = 400
    step = 100

    # 模拟你的 load + resize 流程
    blocks = []
    for idx in range(15, 8, -1):
        raw = np.load(f'./{idx}_kirchhoff.npy')
        blk = skimage.transform.resize(raw, (zl, xl), preserve_range=True)
        blocks.append(blk)

    # 用 taper 融合
    img = stack_with_taper(blocks, step)

    plt.figure(figsize=(8,6))
    plt.rcParams.update({'font.size': 15})
    gci=plt.imshow(img,extent=(0,500,600,0),cmap=cm.gray)
    ax=plt.gca()
    ax.set_xticks(np.linspace(0,500,6))
    ax.set_xticklabels([0,0.2,0.4,0.6,0.8,1.0])
    ax.set_yticks(np.linspace(0,600,6))
    ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0])
    plt.xlabel('Distance (m)')
    plt.ylabel('Time (ms)')
    plt.savefig('gather_rtm.png',dpi=1000)
    # 包络图
    env = np.abs(hilbert(img, axis=0))
    
    plt.figure(figsize=(8,6))
    plt.rcParams.update({'font.size': 15})
    gci=plt.imshow(env,extent=(0,500,600,0),cmap=cm.jet)
    ax=plt.gca()
    ax.set_xticks(np.linspace(0,500,6))
    ax.set_xticklabels([0,0.2,0.4,0.6,0.8,1.0])
    ax.set_yticks(np.linspace(0,600,6))
    ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0])
    plt.xlabel('Distance (m)')
    plt.ylabel('Time (ms)')
    plt.savefig('gather_env.png',dpi=1000)