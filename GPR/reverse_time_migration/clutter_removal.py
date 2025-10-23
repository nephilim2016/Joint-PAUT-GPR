#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 06:53:21 2025

@author: nephilim
"""

import numpy as np
import numpy.linalg as la
from matplotlib import pyplot,cm
import pywt

def wavelet_denoise(signal, wavelet='db4', mode='soft', level=None):
    """
    Wavelet-based denoising using universal thresholding (VisuShrink).
    
    Parameters
    ----------
    signal : 1D array-like
        The input signal to be denoised.
    wavelet : str
        Type of wavelet to use (e.g., 'db4', 'sym5', etc.).
    mode : str
        Threshold mode ('soft' or 'hard').
    level : int or None
        Decomposition level. If None, uses the maximum level possible.
        
    Returns
    -------
    denoised_signal : 1D numpy array
        The denoised signal.
    """
    # 1) Wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # 2) Threshold the detail coefficients (skip the approximation coefficients at index 0)
    thresholded_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        # Estimate noise sigma using the median absolute deviation
        sigma = np.median(np.abs(c)) / 0.6745
        # Compute universal threshold
        uthresh = sigma * np.sqrt(2 * np.log(len(c)))
        # Apply thresholding
        c_denoised = pywt.threshold(c, value=uthresh, mode=mode)
        thresholded_coeffs.append(c_denoised)
    
    # 3) Reconstruct the signal from the thresholded coefficients
    denoised_signal = pywt.waverec(thresholded_coeffs, wavelet)
    return denoised_signal

class ClutterRemoval(object):
    def __init__(self, X_data, max_iter=1000, rank=1, lam=1e-4, method='GoDec'):
        self.max_iter=max_iter
        self.rank=rank
        self.lam=lam
        self.method=method
        self.X_data=X_data
        
    def clutter_removal(self) -> np.ndarray:
        if self.method == 'GoDec':
            return self._PreProcessGPRGoDec()
        elif self.method == 'RNMF':
            return self._PreProcessGPRRNM()
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
    def _nmf_init(self, V: np.ndarray, rank: int) -> (np.ndarray, np.ndarray):
        """
        Randomly initialize W, H for standard NMF.

        Args:
            V   : (m x n) data matrix
            rank: desired rank

        Returns:
            W   : (m x rank)
            H   : (rank x n)
        """
        m, n = V.shape
        # Use a reproducible random state if desired
        W = np.random.rand(m, rank)
        H = np.random.rand(rank, n)
        return W, H

    def _basic_nmf_update(self, V: np.ndarray, W: np.ndarray, H: np.ndarray, max_iter: int = 1000) -> (np.ndarray, np.ndarray):
        """
        Basic multiplicative NMF updates to factor V ~ W @ H (both nonnegative).

        Args:
            V       : (m x n) data matrix
            W, H    : initial guesses for factorization
            max_iter: number of iterations

        Returns:
            W, H
        """
        eps = 1e-8
        for _ in range(max_iter):
            # Update W
            #   numerator_W = V @ H.T
            #   denominator_W = W @ (H @ H.T)
            #   W <- W * (num/den)
            WHHT = W @ (H @ H.T) + eps  # shape (m x rank)
            W *= (V @ H.T + eps) / WHHT

            # Update H
            #   numerator_H = W.T @ V
            #   denominator_H = (W.T @ W) @ H
            WTWH = (W.T @ W) @ H + eps  # shape (rank x n)
            H *= (W.T @ V + eps) / WTWH

        return W, H

    def _shrink(self, M: np.ndarray, tau: float) -> np.ndarray:
        """
        Soft-threshold (L1) shrinkage operator: sign(M)*max(abs(M)-tau, 0).
        """
        return np.sign(M) * np.maximum(np.abs(M) - tau, 0)

    def _robust_update_WH(self, X: np.ndarray, W: np.ndarray, H: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Update step for (W, H) in robust scenario:
          S = X - W@H
          We define R = S - X = (X - W@H) - X = -W@H

        W <- W * [ (|R @ H.T| - R @ H.T) ] / [2*(W@(H@H.T)) + eps]
        H <- H * [ (|W.T@R| - W.T@R) ] / [2*((W.T@W)@H) + eps]

        Then rescale W, H so that ||W||_F = 1. (Or keep that scale step.)
        """
        eps = 1e-8
        # R = S - X = -(W @ H)
        R = - (W @ H)

        # Numerator for W
        RhT = R @ H.T  # shape (m x rank)
        num_W = np.abs(RhT) - RhT

        # Denominator for W
        # 2*(W@(H@H.T))
        HHt = H @ H.T  # shape (rank x rank)
        den_W = 2 * (W @ HHt) + eps

        W *= num_W / den_W
        # Next, update H
        wTR = W.T @ R  # shape (rank x n)
        num_H = np.abs(wTR) - wTR

        # Denominator for H
        # 2*((W.T@W)@H)
        WtW = W.T @ W  # shape (rank x rank)
        den_H = 2 * (WtW @ H) + eps

        H *= num_H / den_H

        # Normalize W, re-scale H accordingly
        normW = np.linalg.norm(W)
        if normW > 1e-12:
            W /= normW
            H *= normW

        return W, H

    def _robust_nmf(self, X: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Robust NMF decomposition:
          X ~ W@H + S
        with S = shrink(X - W@H, lam/2) at each iteration.

        Args:
            X       : (m x n) data matrix
            lam     : L1 shrinkage parameter
            max_iter: number of outer iterations
            rank    : rank for W,H

        Returns:
            W, H, S
        """
        # 1) Initialize W, H via basic NMF
        W, H = self._nmf_init(X, self.rank)
        W, H = self._basic_nmf_update(X, W, H, max_iter=50)  # small warm start

        for i in range(self.max_iter):
            # Recompute M = W@H
            M = W @ H
            # 1) Sparse update: S = shrink(X - M, lam/2)
            S = self._shrink(X - M, self.lam * 0.5)

            # 2) Update W, H
            W, H = self._robust_update_WH(X, W, H)
            # Optional: if changes are small, break early
            # if i % 10 == 0:
            #     diff = np.linalg.norm(X - W@H - S, 'fro')
            #     if diff < 1e-6:
            #         print(f"Stopping early at iteration {i}, diff={diff}")
            #         break

        return W, H, S

    def _PreProcessGPRRNM(self) -> np.ndarray:
        """
        Pre-process GPR data by scaling, robust NMF, and un-scaling.

        Args:
            X_data : (m x n) data
            lam    : L1 shrinkage for the robust NMF
            max_iter: max iteration for robust_nmf
            rank   : rank for W,H

        Returns:
            RefData: X_data with outliers removed (X - S)
        """
        mn = self.X_data.min()
        mx = self.X_data.max()
        scale = mx - mn

        # Scale data to [0, 1]
        X_scaled = (self.X_data - mn) / (scale + 1e-12)

        # Perform robust NMF
        W, H, S = self._robust_nmf(X_scaled)

        # Unscale data
        S_unscaled = S * scale
        RefData = self.X_data - S_unscaled
        return RefData


    def _find_top_k_entries(self, M, k):
        """
        Return a boolean mask for the k largest-magnitude entries in M.
        The mask has True for those k entries, False otherwise.
        """
        if k >= M.size:
            # If k >= total elements, no need to mask
            return np.ones(M.shape, dtype=bool)
        absM = np.abs(M).ravel()
        # Partition around the (length-k)th largest value
        idx_partition = np.argpartition(absM, -k)
        cutoff = absM[idx_partition[-k]]
        return (np.abs(M) >= cutoff)

    def _randomized_svd(self, M, oversampling=10, power_iter=2):
        """
        A simple implementation of randomized SVD for matrix M.
        rank: target rank
        oversampling: extra basis vectors
        power_iter: number of power iterations for better accuracy

        Returns: U, s, Vt such that M ~ U * diag(s) * Vt
        """
        m, n = M.shape
        r = self.rank + oversampling
        # Random Gaussian test matrix
        G = np.random.randn(n, r)

        # Y = M * G
        Y = M @ G
        # Power iteration to reduce errors in the low-rank subspace
        for _ in range(power_iter):
            Y = M @ (M.T @ Y)

        # QR factorization
        Q, _ = np.linalg.qr(Y, mode='reduced')  # Q is m x r

        # Project M into this subspace
        B = Q.T @ M  # shape r x n

        # Now do an SVD on B
        Ub, s, Vt = la.svd(B, full_matrices=False)
        # Truncate to desired rank
        U_trunc = Ub[:, :self.rank]
        s_trunc = s[:self.rank]
        Vt_trunc = Vt[:self.rank, :]

        # Map back to original space
        U = Q @ U_trunc
        return U, s_trunc, Vt_trunc

    def _godec_randSVD(self, X, card, tol=1e-7, oversampling=10, power_iter=2):
        """
        GoDec with Randomized SVD:
          Decompose X ~ L + S, where
            rank(L) = 'rank'
            nnz(S) <= 'card'

        Args:
            X          : (m x n) matrix
            rank       : desired rank for L
            card       : max number of nonzero entries in S
            max_iter   : maximum number of iterations
            tol        : convergence threshold on ||X - L - S||_F
            oversampling: extra basis vectors for randomized SVD
            power_iter : power iterations for randomized SVD

        Returns:
            L, S
        """
        m, n = X.shape
        # Initialize
        L = X.copy()
        S = np.zeros_like(X)

        normX = la.norm(X, 'fro') + 1e-9

        for it in range(self.max_iter):
            # 1) Low-rank approximation of (X - S) by rank=r
            X_minus_S = X - S
            # Use randomized SVD to get top 'rank' components
            U, sigma, Vt = self._randomized_svd(X_minus_S, oversampling=oversampling, power_iter=power_iter)
            # Reconstruct the rank-r approximation
            # L_new = U * diag(sigma) * Vt
            L_new = (U * sigma) @ Vt

            # 2) Sparse approximation: keep 'card' largest entries of (X - L_new)
            R = X - L_new
            mask = self._find_top_k_entries(R, card)
            S_new = np.zeros_like(X)
            S_new[mask] = R[mask]

            # Check convergence
            diff = la.norm(L_new - L, 'fro') / normX

            # Update
            L, S = L_new, S_new
            if diff < tol:
                break

        return L, S

    def _PreProcessGPRGoDec(self):
        mn = self.X_data.min()
        mx = self.X_data.max()
        scale = mx - mn

        # Scale data to [0, 1]
        X_scaled = (self.X_data - mn) / (scale + 1e-12)

        # Now we want to force rank=2
        # Suppose we guess card=500 outliers
        L_gd, S_gd = self._godec_randSVD(X_scaled, card=X_scaled.size, tol=1e-7, oversampling=100, power_iter=1)
        X_data=X_scaled*scale+mn
        S_gd=S_gd*scale
        L_gd=L_gd*scale+mn
        # Evaluate
        err = la.norm(X_data - L_gd - S_gd, 'fro') / la.norm(X_data, 'fro')
        
        for idx in range(L_gd.shape[1]):
            L_gd[:,idx] = wavelet_denoise(L_gd[:,idx], wavelet='db4', mode='soft', level=None)
        
        # print("GoDec Randomized SVD:")
        print(f"Relative reconstruction error = {err:.6e}")
        return L_gd
        

if __name__=='__main__':
    X_data=np.load('record.npy')
    max_iter=50
    rank=1
    lam=1e-4
    method='GoDec'
    
    PreProc=ClutterRemoval(X_data,max_iter,rank,lam,method)
    RefData=PreProc.clutter_removal()
    pyplot.figure()
    pyplot.imshow(X_data,extent=(0,1,0,1),vmin=-10,vmax=10)
    pyplot.figure()
    pyplot.imshow(RefData,extent=(0,1,0,1),vmin=-10,vmax=10)
    pyplot.figure()
    pyplot.imshow(X_data-RefData,extent=(0,1,0,1),vmin=-10,vmax=10)