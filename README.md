# Joint-PAUT-GPR

### Multi-Geophysical Imaging Framework for Concrete Defect Detection Using Phased Array Ultrasonics Testing and Ground Penetrating Radar

**Authors:** Xiangyu Wang, Hai Liu*, Heming Peng, Xu Meng, Jie Cui  
**Affiliation:** School of Civil Engineering, Guangzhou University, China  
**Corresponding Author:** Hai Liu (hliu@gzhu.edu.cn)

---

## 🧠 Overview

This repository provides the **implementation codes and experimental datasets** used in the paper:

> **Wang, X., Liu, H., Peng, H., Meng, X., Cui, J. (2025).**  
> *Multi-Geophysical Imaging Framework for Concrete Defect Detection Using Phased Array Ultrasonics Testing and Ground Penetrating Radar.*  

The repository contains all scripts and raw laboratory data supporting the **multi-geophysical joint imaging framework**, which integrates **Phased Array Ultrasonic Testing (PAUT)** and **Ground Penetrating Radar (GPR)** for high-precision non-destructive testing (NDT) of concrete structures.

---

## 🧩 Framework Description

The framework achieves **mutual verification and complementary enhancement** between ultrasonic and electromagnetic modalities. It consists of:

1. **Pre-processing:**  
   - DC drift and time-zero correction  
   - Band-pass filtering and gain compensation  
   - First-arrival wave suppression using *Robust Cross-Correlation with Rayleigh-Noise Adaptive Thresholding* and *GoDec low-rank sparse decomposition*  

2. **Imaging Algorithms:**  
   - Total Focusing Method (TFM) reformulated as a Kirchhoff migration  
   - Reverse Time Migration (RTM) with **Poynting-vector-based directional constraint** for artifact suppression  
   - Unified implementation for both PAUT and GPR wavefields

3. **Multi-Geophysical Joint Imaging:**  
   - Morphological co-localization algorithm for cross-validation of anomalies  
   - Integration of RTM-PAUT and RTM-GPR images for defect confirmation and enhancement

---

## 🧪 Experimental Data

The `laboratory/` folder includes raw PAUT and GPR data collected from a **1.0 m × 0.6 m concrete block** containing embedded cavities of varying sizes.  
All experiments were conducted at **Guangzhou University’s NDT Laboratory** using:
- **PAUT:** A1040 MIRA 3D Pro (16-element transducer array, 50 kHz)  
- **GPR:** Leica C-Thrue (2 GHz antenna)


