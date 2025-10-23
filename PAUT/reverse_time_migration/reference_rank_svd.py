#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 04:01:18 2025

@author: nephilim
"""

import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt, matplotlib.cm as cm

# ---------- 工具 ----------
def locate_ref(trace, ref):
    n = len(ref)
    xc  = correlate(trace, ref, mode='valid')
    den = (np.convolve(trace**2, np.ones(n), 'valid') * np.sum(ref**2))**.5+1e-12
    ncc = xc/den
    return int(np.argmax(ncc))

def pad_or_cut(seg, L):
    if len(seg) < L:
        return np.pad(seg, (0, L-len(seg)))
    if len(seg) > L:
        return seg[:L]
    return seg

# ---------- 主函数 ----------
def svd_ref_clean_v2(g, ref_id=0, win_len=260, guard=40, n_iter=2):
    nt,nr = g.shape
    L     = win_len + 2*guard
    ref0  = g[:win_len, ref_id] - np.mean(g[:win_len, ref_id])

    # 预先定位 + 截窗
    segs   = np.zeros((nr, L))
    bounds = []
    for r in range(nr):
        sh  = locate_ref(g[:,r], ref0)
        i0  = max(0, sh-guard)
        i1  = min(nt, sh+win_len+guard)
        seg = pad_or_cut(g[i0:i1, r], L)
        segs[r] = seg
        bounds.append((i0, i1))

    # 迭代：自适应幅度 + 权重
    segs_work = segs.copy()
    weight    = np.ones(nr)
    for it in range(n_iter):
        # --- Rank-1 SVD ---
        U,s,Vt = np.linalg.svd(segs_work*weight[:,None], full_matrices=False)
        rank1  = np.outer(U[:,0], s[0]*Vt[0]) / weight[:,None]

        # --- 更新幅度系数 ---
        for r in range(nr):
            a = np.dot(segs[r], rank1[r]) / (np.dot(rank1[r], rank1[r]) + 1e-12)
            segs_work[r] = segs[r]/max(a,1e-6)

        # --- 更新权重 (基于残差) ---
        resid = segs_work - rank1
        rms   = np.sqrt(np.mean(resid**2, axis=1))
        rms  /= np.median(rms)+1e-12
        weight = 1/(1+10*rms)       # 残差大⇒权重小，范围约 [~0.1,1]

    # -------- 应用 Rank-1 去除 --------
    cleaned = g.copy()
    for r,(i0,i1) in enumerate(bounds):
        seg_len = i1 - i0
        cleaned[i0:i1, r] -= rank1[r, :seg_len]
    return cleaned

def extract_first_wave_full_gather(aligned,REF_ID,WIN_LEN,GUARD,ITER=2):
    D = aligned
    clean = svd_ref_clean_v2(D, ref_id=REF_ID,
                             win_len=WIN_LEN, guard=GUARD,
                             n_iter=ITER)
    return clean

# FILE_IN   = './aligned.npy'
# REF_ID    = 1          # 选能量最强的一道
# WIN_LEN   = 340
# GUARD     = 40
# ITER      = 2          # 迭代次数 (1 or 2)

# ---------- 脚本执行 ----------


# if SAVE_NPY:
#     np.save('aligned_clean_svd_v2.npy', clean)

# if SHOW_FIG:
#     nt,nr = D.shape
#     ext = [0, nr-1, nt*1e-3, 0]
#     plt.figure(figsize=(12,4))
#     plt.subplot(121); plt.imshow(D,    extent=ext,cmap=cm.jet,aspect='auto'); plt.title('Aligned raw'); plt.ylabel('Time (ms)')
#     plt.subplot(122); plt.imshow(clean,extent=ext,cmap=cm.jet,aspect='auto'); plt.title('After SVD_REF v2'); plt.xlabel('Trace #')
#     plt.tight_layout(); plt.show()
    
# # plt.figure(figsize=(12,4))
# # plt.imshow(clean,extent=ext,cmap=cm.jet,aspect='auto',vmin=-50,vmax=50); plt.title('Aligned raw');