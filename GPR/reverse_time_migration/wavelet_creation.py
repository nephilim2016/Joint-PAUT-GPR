#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:59:51 2018

@author: nephilim
"""
import numpy as np

class WaveletType(object):
    def __init__(self, t, f, wavelet_type):
        self.t = t
        self.f = f
        self.wavelet_type = wavelet_type

    def create_wavelet(self):
        if self.wavelet_type == 'gaussian':
            return self._gaussian()
        elif self.wavelet_type == 'blackmanharris':
            return self._blackmanharris()
        else:
            return self._ricker()

    def _gaussian(self):
        # Example implementation of a Gaussian-type wavelet.
        a = np.array([0.3532222, -0.488, 0.145, -0.010222222])
        T = 1.14 / self.f
        w = np.zeros(len(self.t))
        for n in range(len(a)):
            w += a[n] * np.cos(2 * n * np.pi * self.t / T)
        w[self.t >= T] = 0  # Zero out values after time T
        return w

    def _blackmanharris(self):
        a = np.array([0.3532222, -0.488, 0.145, -0.010222222])
        T = 1.14 / self.f
        w = np.zeros(len(self.t))
        for n in range(len(a)):
            w += a[n] * np.cos(2 * n * np.pi * self.t / T)
        w[self.t >= T] = 0
        # Create a discrete derivative to simulate a pulse shape
        w_tmp = w[1:]
        w_tmp = np.append(w_tmp, 0)
        srcpulse = w_tmp - w
        srcpulse /= np.max(np.abs(srcpulse))
        return srcpulse

    def _ricker(self):
        # Ricker wavelet (Mexican hat) implementation.
        w = -1 * (2 * np.pi**2 * (self.f * self.t - 1)**2 - 1) * np.exp(-np.pi**2 * (self.f * self.t - 1)**2)
        return w

# Example usage:
if __name__ == '__main__':
    wave = WaveletType(t=np.arange(1000) * 2e-11, f=4e8, wavelet_type='ricker').create_wavelet()