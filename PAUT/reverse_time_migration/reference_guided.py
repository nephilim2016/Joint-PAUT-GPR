#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 03:44:01 2025

@author: nephilim
"""

import numpy as np
from scipy.signal import correlate, hilbert
from typing import Tuple

def locate_ref_window(trace: np.ndarray,
                      ref:   np.ndarray,
                      guard: int = 50) -> Tuple[int,int,float]:
    """
    在 trace 中找到与 ref 最相似的区间
    ----------------------------------
    guard : 允许在互相关峰两侧最多再扩多少样点，用来包住拖尾
    返回   : (i_start, i_end, ncc_max)
    """
    n_ref = len(ref)
    # 步 1) 归一化互相关 (NCC)
    xc  = correlate(trace, ref, mode='valid')
    denom = (np.convolve(trace**2, np.ones(n_ref), 'valid')
             * np.sum(ref**2))**0.5 + 1e-12
    ncc = xc/denom
    shift = int(np.argmax(ncc))

    # 步 2) 以峰值为中心，两侧再扩 guard 样点
    i_start = max(0,            shift - guard)
    i_end   = min(len(trace)-1, shift + n_ref - 1 + guard)

    return i_start, i_end, float(ncc[shift])


def extract_first_wave(trace: np.ndarray,
                       ref:   np.ndarray,
                       guard:int = 50):
    """返回 (first_wave, remainder, i0, i1)"""
    i0,i1,_ = locate_ref_window(trace, ref, guard)
    first   = trace[i0:i1+1].copy()
    remain  = trace.copy(); remain[i0:i1+1] = 0.0
    return first, remain, i0, i1

def extract_first_wave_full_gather(aligned,ref):
    # aligned = np.load('aligned.npy')   # (nt, nr) = (1158, 16)

    # # ① 任选一条能量最强的参考子波（这里取第 2 道前 200 点）
    # ref = aligned[:150, 1]

    # ② 对每条道提取首波
    first_waves = []
    cleaned     = aligned.copy()
    bounds      = []

    for r in range(aligned.shape[1]):
        first, remain, i0, i1 = extract_first_wave(aligned[:, r], ref, guard=50)
        first_waves.append(first)
        cleaned[:, r] = remain
        bounds.append((i0,i1))
    return cleaned
        
if __name__=='__main__':
    import numpy as np, matplotlib.pyplot as plt, matplotlib.cm as cm

    aligned = np.load('aligned.npy')   # (nt, nr) = (1158, 16)

    # ① 任选一条能量最强的参考子波（这里取第 2 道前 200 点）
    ref = aligned[:150, 1]

    # ② 对每条道提取首波
    first_waves = []
    cleaned     = aligned.copy()
    bounds      = []

    for r in range(aligned.shape[1]):
        first, remain, i0, i1 = extract_first_wave(aligned[:, r], ref, guard=50)
        first_waves.append(first)
        cleaned[:, r] = remain
        bounds.append((i0,i1))

    # ③ 可视化检查
    plt.figure(figsize=(12,4))
    plt.subplot(121); plt.imshow(aligned,  cmap=cm.jet, aspect='auto'); plt.title('aligned raw')
    plt.subplot(122); plt.imshow(cleaned,  cmap=cm.jet, aspect='auto'); plt.title('after removing first wave')
    plt.tight_layout(); plt.show()
    print('每道首波区间前 8 道：', bounds[:8])