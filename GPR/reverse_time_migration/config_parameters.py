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
    _frozen_fields = {'xl', 'zl', 'dx', 'dz', 'dt', 'k_max', 'air_layer', 't'}

    def __setattr__(self, key, value):
        # If the field is in our frozen set and it already exists, prevent modification.
        if key in self._frozen_fields and key in self.__dict__:
            raise AttributeError(f"Field '{key}' is frozen and cannot be modified.")
        super().__setattr__(key, value)
        
@dataclass
class ConfigParameters(PartiallyFrozen):
    xl: int = 75   
    zl: int = 400
    dx: float = 2e-2
    dz: float = 2e-2
    dt: float = 4e-11
    k_max: int = 500
    air_layer: int = 2
    wavelet_type: str ='gaussian'
    frequency: float = 4e8
    
    # imaging_condition='explosion'
    # imaging_condition='correlation'
    imaging_condition: str = field(init=False)
    
    ## Dynamic fields initialized post-construction.
    source_site: list = field(init=False)
    receiver_site: list = field(init=False)
    t: np.ndarray = field(init=False)
    true_profile: np.ndarray = field(default_factory=lambda: np.array([]), init=False)

    def __post_init__(self):
        # Compute t based on k_max and dt.
        self.t = np.arange(self.k_max) * self.dt
        self.source_site = [(10, idx) for idx in range(10, self.zl + 10, 2)]
        self.receiver_site = [(10, idx) for idx in range(10, self.zl + 10, 2)]

config = ConfigParameters()