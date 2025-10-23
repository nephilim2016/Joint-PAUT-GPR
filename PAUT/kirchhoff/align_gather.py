#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 04:07:17 2025

@author: nephilim
"""

import numpy as np
from scipy.signal import fftconvolve, hilbert
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

# ---------- 工具 ----------
def robust_norm(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return (x - med) / mad

def pick_takeoff_single(trace, ref, k_ray=4.0, n_consec=5):
    tr_nrm, ref_nrm = robust_norm(trace), robust_norm(ref)
    xc = fftconvolve(tr_nrm, ref_nrm[::-1], mode="full")[len(ref)-1:]
    pk = int(np.argmax(xc))
    env = np.abs(hilbert(tr_nrm))
    sigma = np.sqrt(np.mean(env**2) / (4/np.pi))
    theta = k_ray * sigma
    below = 0; j = pk
    while j > 0:
        below = below + 1 if env[j] < theta else 0
        if below >= n_consec:
            y0, y1 = env[j+n_consec-1], env[j+n_consec]
            frac = (theta - y0)/(y1-y0+1e-12)
            return (j+n_consec-1)+np.clip(frac,0,1)
        j -= 1
    return float(pk)

# ---------- 对齐 ----------
def align_by_ref_pad(gather, ref,
                     k_ray=4.0, n_consec=5, smooth_win=5,
                     allow_frac=True, fill_value=0.0):
    """
    gather : (nt, nr), 时间在前
    ref    : 1-D 参考子波
    ----------------------------------------
    返回:
        aligned  : (nt+2*pad, nr)
        shifts   : (nr,)  正值=向后移
        pad      : int    上下各补 pad
    """
    nt, nr = gather.shape
    picks = np.array([pick_takeoff_single(gather[:, r], ref,
                                          k_ray, n_consec) for r in range(nr)])
    picks_s = median_filter(picks, size=smooth_win)
    t0 = np.median(picks_s)
    shifts = t0 - picks_s

    pad = int(np.ceil(np.abs(shifts).max())) + 2      # 双向留裕量
    nt_new = nt + 2*pad
    t_new  = np.arange(nt_new)
    aligned = np.full((nt_new, nr), fill_value, dtype=gather.dtype)

    for r in range(nr):
        f = interp1d(np.arange(nt), gather[:, r],
                     kind='linear', bounds_error=False,
                     fill_value=fill_value)
        aligned[:, r] = f(t_new - pad - shifts[r])    # 核心公式

    return aligned, shifts, pad

# ---------- 还原 ----------
def restore_gather_pad(aligned, shifts, pad,
                       nt_orig, allow_frac=True, fill_value=0.0):
    """
    aligned : (nt_orig+2*pad, nr)
    shifts  : (nr,)
    pad     : 对齐时用的同一个 pad
    ----------------------------------------
    返回 restored : (nt_orig, nr)
    """
    nt_new, nr = aligned.shape
    assert nt_new == nt_orig + 2*pad
    t_new = np.arange(nt_new)
    t_ori = np.arange(nt_orig)
    restored = np.full((nt_orig, nr), fill_value, dtype=aligned.dtype)

    for r in range(nr):
        f = interp1d(t_new, aligned[:, r],
                     kind='linear', bounds_error=False,
                     fill_value=fill_value)
        restored[:, r] = f(t_ori + pad + shifts[r])   # 核心公式

    return restored

def bandpass(trace, fs, fmin, fmax, order=4):
    nyq = 0.5*fs
    b, a = butter(order, [fmin/nyq, fmax/nyq], btype='band')
    return filtfilt(b, a, trace)