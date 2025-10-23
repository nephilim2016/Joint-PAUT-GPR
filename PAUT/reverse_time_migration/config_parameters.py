#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 04:16:32 2025

@author: nephilim
"""
from dataclasses import dataclass, field
import numpy as np

class PartiallyFrozen:
    # List the field names that should be immutable once set.
    _frozen_fields = {'xl', 'zl', 'dx', 'dz', 'dt', 'k_max', 't'}

    def __setattr__(self, key, value):
        # If the field is in our frozen set and it already exists, prevent modification.
        if key in self._frozen_fields and key in self.__dict__:
            raise AttributeError(f"Field '{key}' is frozen and cannot be modified.")
        super().__setattr__(key, value)
        
@dataclass
class ConfigParameters(PartiallyFrozen):
    xl: int = 600   
    zl: int = 200
    dx: float = 2e-3
    dz: float = 2e-3
    dt: float = 4e-7
    k_max: int = 2500

    wavelet_type: str ='gaussian'
    frequency: float = 1e5
    
    ## Dynamic fields initialized post-construction.
    source_site: list = field(init=False)
    receiver_site: list = field(init=False)
    t: np.ndarray = field(init=False)
    true_profile: np.ndarray = field(default_factory=lambda: np.array([]), init=False)

    def __post_init__(self):
        # Compute t based on k_max and dt.
        self.t = np.arange(self.k_max) * self.dt
        self.source_site = [(10, idx) for idx in np.linspace(10, self.zl + 10, 16).astype('int')]
        self.receiver_site = [(10, idx) for idx in np.linspace(10, self.zl + 10, 200).astype('int')]

config = ConfigParameters()